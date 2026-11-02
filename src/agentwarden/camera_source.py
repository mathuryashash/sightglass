"""Frame sources for the vision pipeline. Every source yields Frame(t, jpeg, w, h) with width <= 640.

t is seconds from the first frame (a clip's media time, or wall time for a live camera).
"""
import io
import ipaddress
import subprocess
import time
from pathlib import Path
from typing import Callable, Iterable, Iterator, NamedTuple
from urllib.parse import urlsplit

from PIL import Image

MAX_W = 640
# is_private also covers documentation/reserved ranges, so list the LAN ranges explicitly
_LAN_NETS = [ipaddress.ip_network(n) for n in
             ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "169.254.0.0/16", "fc00::/7", "fe80::/10")]


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
    """Phone IP-Webcam `/shot.jpg`: one GET per sample. Private/localhost hosts only, no redirects."""

    def __init__(self, url: str, user: str | None = None, password: str | None = None,
                 period_s: float = 1.0, n: int | None = None, timeout_s: float = 5.0,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep):
        self._host = _check_private_host(url)
        self._url, self._auth = url, (user, password or "") if user else None
        self.period_s, self.n, self.timeout_s = period_s, n, timeout_s
        self._clock, self._sleep = clock, sleep

    def __repr__(self) -> str:  # never includes the URL path, query or credentials
        return f"HttpShotSource(host={self._host!r})"

    __str__ = __repr__

    def __iter__(self) -> Iterator[Frame]:
        import httpx  # lazy: only live cameras need it

        t0, i = self._clock(), 0
        with httpx.Client(auth=self._auth, follow_redirects=False, timeout=self.timeout_s) as c:
            while self.n is None or i < self.n:
                t = self._clock() - t0
                r = c.get(self._url)
                r.raise_for_status()  # 3xx raises too: a redirect could leave the private network
                yield _frame(t, r.content)
                i += 1
                if self.n is None or i < self.n:
                    self._sleep(self.period_s)
