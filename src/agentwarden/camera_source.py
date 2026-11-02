"""Frame sources for the vision pipeline. Every source yields Frame(t, jpeg, w, h) with width <= 640.

t is seconds from the first frame (a clip's media time, or wall time for a live camera).
"""
import bisect
import io
import ipaddress
import json
import subprocess
import time
from pathlib import Path
from typing import Callable, Iterable, Iterator, NamedTuple
from urllib.parse import urlsplit

from PIL import Image

MAX_W = 640
# is_private also covers documentation/reserved ranges, so list the LAN ranges explicitly. Link-local
# (169.254/16 incl. the 169.254.169.254 cloud metadata address, fe80::/10) is deliberately NOT here: a phone on
# Wi-Fi has an RFC1918 address.
_LAN_NETS = [ipaddress.ip_network(n) for n in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "fc00::/7")]
MAX_RESPONSE_BYTES = 5_000_000


class Frame(NamedTuple):
    t: float
    jpeg: bytes
    w: int
    h: int


def _prep(data: bytes) -> tuple[bytes, int, int]:
    """Downscale to MAX_W wide (never upscale); returns (jpeg bytes, w, h)."""
    im = Image.open(io.BytesIO(data))
    w, h = im.size
    if w <= MAX_W and data.startswith(b"\xff\xd8\xff"):
        return data, w, h
    if w > MAX_W:
        im = im.resize((MAX_W, max(1, round(h * MAX_W / w))), Image.LANCZOS)
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=90)
    return buf.getvalue(), im.size[0], im.size[1]


def _frame(t: float, data: bytes) -> Frame:
    jpg, w, h = _prep(data)
    return Frame(float(t), jpg, w, h)


class FakeSource:
    """Flat synthetic frames for tests and the fake-camera demo: n frames at fps, or explicit times."""

    def __init__(self, n: int | None = None, fps: float = 1.0, times: Iterable[float] | None = None,
                 w: int = 160, h: int = 120):
        if times is None and n is None:
            raise ValueError("FakeSource needs n or times")
        self.times = list(times) if times is not None else [i / fps for i in range(n)]
        buf = io.BytesIO()
        Image.new("RGB", (w, h), (90, 90, 90)).save(buf, "JPEG")
        self._jpeg, self._w, self._h = buf.getvalue(), w, h

    def __iter__(self) -> Iterator[Frame]:
        for t in self.times:
            yield Frame(float(t), self._jpeg, self._w, self._h)


class StillsSource:
    """A list of image files, one per period_s."""

    def __init__(self, paths: Iterable[str | Path], period_s: float = 1.0):
        self.paths, self.period_s = [Path(p) for p in paths], period_s

    def __iter__(self) -> Iterator[Frame]:
        for i, p in enumerate(self.paths):
            yield _frame(i * self.period_s, p.read_bytes())


class ReplayFrames:
    """The real frames of a recorded replay: one Frame per record, from record["photo"] (e.g. "t031.jpg") under
    `root` (default: <replay json dir>/pour_frames), at the record's own time. overlay(t) is the pre-drawn
    "ov_<photo>" picture (mask tint, box, surface line) of the record at or before t, or None when it is missing."""

    def __init__(self, replay_json: str | Path, root: str | Path | None = None):
        path = Path(replay_json)
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = next((data[k] for k in ("readings", "results", "frames") if isinstance(data.get(k), list)), [])
        self.root = Path(root) if root else path.parent / "pour_frames"
        self._recs = [(float(r.get("t", i)), r["photo"]) for i, r in enumerate(data) if r.get("photo")]
        if not self._recs:
            raise ValueError("replay JSON names no photos")

    @classmethod
    def available(cls, replay_json: str | Path, root: str | Path | None = None) -> bool:
        try:
            rf = cls(replay_json, root)
        except (OSError, ValueError, KeyError, TypeError):
            return False
        return (rf.root / rf._recs[0][1]).is_file()

    def times(self) -> list[float]:
        return [t for t, _ in self._recs]

    def __iter__(self) -> Iterator[Frame]:
        for t, photo in self._recs:
            yield _frame(t, (self.root / photo).read_bytes())

    def overlay(self, t: float) -> bytes | None:
        i = max(bisect.bisect_right(self.times(), t) - 1, 0)
        p = self.root / f"ov_{self._recs[i][1]}"
        return p.read_bytes() if p.is_file() else None


class ClipSource:
    """Frames from a video file at `fps`, decoded by an ffmpeg subprocess (MJPEG pipe)."""

    def __init__(self, path: str | Path, fps: float = 1.0):
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.path, self.fps = Path(path), fps
        if not self.path.is_file():
            raise FileNotFoundError(self.path)

    def __iter__(self) -> Iterator[Frame]:
        cmd = ["ffmpeg", "-v", "error", "-i", str(self.path),
               "-vf", f"fps={self.fps},scale='min({MAX_W},iw)':-2",
               "-f", "image2pipe", "-c:v", "mjpeg", "-q:v", "3", "-"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        buf, n = b"", 0
        try:
            while True:
                chunk = proc.stdout.read(1 << 16)
                if not chunk:
                    break
                buf += chunk
                while True:  # a JPEG runs SOI..EOI; 0xFF is byte-stuffed inside entropy data
                    start = buf.find(b"\xff\xd8")
                    end = buf.find(b"\xff\xd9", start + 2) if start >= 0 else -1
                    if start < 0 or end < 0:
                        break
                    yield _frame(n / self.fps, buf[start : end + 2])
                    buf, n = buf[end + 2 :], n + 1
            if n == 0:
                raise RuntimeError(f"ffmpeg produced no frames from {self.path.name}")
        finally:
            proc.kill()
            proc.wait()


def _check_private_host(url: str) -> str:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise ValueError("camera URL must be http or https")
    if parts.username or parts.password:
        raise ValueError("pass credentials as user/password, not in the URL")
    host = parts.hostname or ""
    if host != "localhost":
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            raise ValueError(f"camera host must be localhost or a private IP, got {host!r}") from None
        if not (ip.is_loopback or any(ip in net for net in _LAN_NETS if ip.version == net.version)):
            raise ValueError(f"camera host must be localhost or a private IP, got {host!r}")
    return host


class HttpShotSource:
    """Phone IP-Webcam `/shot.jpg`: one GET per sample. Loopback/RFC1918/ULA hosts only (no link-local), no
    redirects, no proxies, responses capped at max_bytes. Errors name the host's HTTP status or exception type
    only: never the URL, path or credentials."""

    def __init__(self, url: str, user: str | None = None, password: str | None = None,
                 period_s: float = 1.0, n: int | None = None, timeout_s: float = 5.0,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
                 max_bytes: int = MAX_RESPONSE_BYTES):
        self._host = _check_private_host(url)
        self._url, self._auth = url, (user, password or "") if user else None
        self.period_s, self.n, self.timeout_s, self.max_bytes = period_s, n, timeout_s, max_bytes
        self._clock, self._sleep = clock, sleep

    def _get(self, client) -> bytes:
        import httpx

        try:
            with client.stream("GET", self._url) as r:
                if r.status_code >= 300:  # 3xx too: a redirect could leave the private network
                    raise RuntimeError(f"camera returned HTTP {r.status_code}")
                declared = r.headers.get("content-length", "")
                if declared.isdigit() and int(declared) > self.max_bytes:
                    raise ValueError("camera response too large")
                body = bytearray()
                for chunk in r.iter_bytes():
                    body += chunk
                    if len(body) > self.max_bytes:
                        raise ValueError("camera response too large")
                return bytes(body)
        except httpx.HTTPError as e:  # its message would carry the full URL
            raise RuntimeError(f"camera request failed: {type(e).__name__}") from None

    def __repr__(self) -> str:  # never includes the URL path, query or credentials
        return f"HttpShotSource(host={self._host!r})"

    __str__ = __repr__

    def __iter__(self) -> Iterator[Frame]:
        import httpx  # lazy: only live cameras need it

        t0, i = self._clock(), 0
        with httpx.Client(auth=self._auth, follow_redirects=False, timeout=self.timeout_s, trust_env=False) as c:
            while self.n is None or i < self.n:
                t = self._clock() - t0
                yield _frame(t, self._get(c))
                i += 1
                if self.n is None or i < self.n:
                    self._sleep(self.period_s)
