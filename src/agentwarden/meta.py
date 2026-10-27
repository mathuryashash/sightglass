"""Thin Meta Responses API client: ledger-guarded, cassette-backed, transport injected."""
from __future__ import annotations

import base64
import json
import logging
import math
from typing import Any, Callable

from .cassette import Cassette
from .cost import Ledger

log = logging.getLogger(__name__)

Transport = Callable[[str, dict, dict], "tuple[int, Any]"]
MODES = ("replay", "record", "live")
SAM_MODEL = "sam-3.1"
SPARK_MODEL = "muse-spark-1.3"
SPARK_MAX_OUTPUT_TOKENS = 1024
_CHARS_PER_TOKEN = 3  # conservative: over-estimates tokens, so the reserve is a true worst case
_HEADERS = {"Content-Type": "application/json"}  # auth is added inside httpx_transport only


class MetaError(Exception):
    """Non-2xx response; carries the status code (never the body or any header)."""

    def __init__(self, status: int) -> None:
        super().__init__(f"Meta API returned HTTP {status}")
        self.status = status


def httpx_transport(api_key: str, base_url: str) -> Transport:
    """Build the real transport. The key lives only in this closure; never called in tests."""
    def send(url: str, headers: dict, json_body: dict) -> tuple[int, Any]:
        import httpx  # lazy: only needed for live calls

        r = httpx.post(f"{base_url.rstrip('/')}/{url.lstrip('/')}", json=json_body, timeout=120.0,
                       headers={**headers, "Authorization": f"Bearer {api_key}"})
        try:
            return r.status_code, r.json()
        except ValueError:
            return r.status_code, None
    return send


class MetaClient:
    """Reserve -> transport -> settle (-> store in record mode); replay reads the cassette."""

    def __init__(self, transport: Transport, ledger: Ledger, cassette: Cassette, run_id: str,
                 cap_usd: float, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.transport, self.ledger, self.cassette = transport, ledger, cassette
        self.run_id, self.cap_usd, self.mode = run_id, cap_usd, mode

    def _call(self, model: str, body: dict, run_id: str, reserve: Callable[[], str],
              settle: Callable[[str, Any], None]) -> Any:
        if self.mode == "replay":
            hit = self.cassette.get(model, body, run_id)
            if hit is None:
                raise LookupError(f"cassette miss for {model} (run {run_id!r})")
            return hit
        rid = reserve()
        try:
            status, resp = self.transport("responses", dict(_HEADERS), body)
        except BaseException:
            self.ledger.release(rid)
            raise
        log.info("meta call model=%s status=%s", model, status)
        if not 200 <= status < 300:
            self.ledger.release(rid)
            raise MetaError(status)
        settle(rid, resp)
        if self.mode == "record":
            self.cassette.put(model, body, run_id, resp)
        return resp

    def sam_image(self, image_bytes: bytes, concept: str, run_id: str) -> Any:
        """Segment `concept` in one image; charges 1 image."""
        url = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
        body = {"model": SAM_MODEL,
                "input": [{"role": "user", "content": [
                    {"type": "input_text", "text": concept},
                    {"type": "input_image", "image_url": url}]}],
                "metadata": {"mask_encoding": "one_bit", "include_confidence": "true"}}
        return self._call(
            SAM_MODEL, body, run_id,
            lambda: self.ledger.reserve_media(SAM_MODEL, "image", 1, run_id, self.cap_usd),
            lambda rid, _resp: self.ledger.settle_media(rid, 1))

    def spark_json(self, prompt: str, schema: dict, run_id: str) -> Any:
        """Ask muse-spark for JSON matching `schema`; returns the parsed object."""
        body = {"model": SPARK_MODEL, "input": prompt,
                "max_output_tokens": SPARK_MAX_OUTPUT_TOKENS,
                "text": {"format": {"type": "json_schema", "name": "result",
                                    "schema": schema, "strict": True}}}
        est_in = math.ceil((len(prompt) + len(json.dumps(schema))) / _CHARS_PER_TOKEN)

        def settle(rid: str, resp: Any) -> None:
            usage = (resp or {}).get("usage") or {}
            # no usage reported: charge the worst case rather than nothing
            self.ledger.settle(rid, usage.get("input_tokens", est_in),
                               usage.get("output_tokens", SPARK_MAX_OUTPUT_TOKENS))

        resp = self._call(
            SPARK_MODEL, body, run_id,
            lambda: self.ledger.reserve(SPARK_MODEL, est_in, SPARK_MAX_OUTPUT_TOKENS,
                                        run_id, self.cap_usd),
            settle)
        return json.loads(_output_text(resp))


def _output_text(resp: Any) -> str:
    """Concatenate output_text parts of a Responses API reply."""
    parts = [c["text"] for item in resp.get("output", []) if item.get("type") == "message"
             for c in item.get("content", []) if c.get("type") == "output_text"]
    return "".join(parts)
