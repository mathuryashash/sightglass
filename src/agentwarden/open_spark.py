"""Open-weights path: the same advisory diagnoser, run on Muse Spark Open hosted by Baseten.

`OpenSparkClient` exposes the one method the diagnoser needs, `spark_json(prompt, schema, run_id)`, over Baseten's
OpenAI-compatible Model API (model id `meta-models/muse-spark-open`). It is advisory only, exactly like the hosted
Spark path: the diagnoser's static tests forbid any route from its output to the permit, interlock or reset.

Every call is written to a cassette (request hash -> response) so a judge can replay the recorded open-weights runs
with no key. The API key is passed in by the caller, never read from the environment here and never logged.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from agentwarden.meta import _loads_lenient

OPEN_MODEL = "meta-models/muse-spark-open"
BASE_URL = "https://inference.baseten.co/v1"
MAX_TOKENS = 8192  # the model reasons first (hundreds to thousands of hidden tokens)

Transport = Callable[[dict], dict]


def httpx_transport(api_key: str, base_url: str = BASE_URL, timeout: float = 180.0) -> Transport:
    import httpx

    def send(body: dict) -> dict:
        with httpx.Client(timeout=timeout, trust_env=False) as c:
            r = c.post(base_url + "/chat/completions", json=body,
                       headers={"Authorization": "Api-Key " + api_key})
        if r.status_code != 200:
            raise RuntimeError(f"Baseten returned HTTP {r.status_code}")
        return r.json()

    return send


class OpenSparkClient:
    def __init__(self, transport: Transport | None, cassette_dir: str | Path, mode: str = "record"):
        if mode not in ("record", "replay"):
            raise ValueError("mode must be record or replay")
        if mode == "record" and transport is None:
            raise ValueError("record mode needs a transport")
        self.transport, self.mode = transport, mode
        self.dir = Path(cassette_dir)
        self.calls = 0
        self.usage: list[dict] = []

    @staticmethod
    def _body(prompt: str, schema: dict) -> dict:
        # Schema goes in the prompt (portable across OpenAI-compatible servers) plus json_object mode.
        return {"model": OPEN_MODEL, "max_tokens": MAX_TOKENS,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "user",
                              "content": prompt + "\n\nJSON schema for your reply:\n" + json.dumps(schema, sort_keys=True)}]}

    def _path(self, body: dict) -> Path:
        h = hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()
        return self.dir / f"{h}.json"

    def spark_json(self, prompt: str, schema: dict, run_id: str) -> Any:
        body = self._body(prompt, schema)
        p = self._path(body)
        if p.exists():
            resp = json.loads(p.read_text(encoding="utf-8"))["response"]
        elif self.mode == "replay":
            raise LookupError("open-weights cassette miss")
        else:
            resp = self.transport(body)
            self.dir.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"model": OPEN_MODEL, "run_id": run_id, "request": body, "response": resp},
                                    indent=1), encoding="utf-8")
        self.calls += 1
        self.usage.append(resp.get("usage") or {})
        choice = (resp.get("choices") or [{}])[0]
        if choice.get("finish_reason") not in (None, "stop"):
            raise ValueError(f"open model did not finish (finish_reason={choice.get('finish_reason')!r})")
        return _loads_lenient((choice.get("message") or {}).get("content") or "")
