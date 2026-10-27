"""Record/replay store for model calls, one JSON file per key."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

_KEY_RE = re.compile(r"sk-[A-Za-z0-9]{16,}")
_BAD_SUBSTRINGS = ("Authorization", "Bearer ")


def key_for(model: str, request: Any, run_id: str) -> str:
    """sha256 over canonical JSON of {model, request, run_id}."""
    blob = json.dumps({"model": model, "request": request, "run_id": run_id},
                      sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _check(obj: Any) -> None:
    """Raise ValueError if any string (key or value) looks like auth material."""
    if isinstance(obj, str):
        if any(s in obj for s in _BAD_SUBSTRINGS) or _KEY_RE.search(obj):
            raise ValueError("refusing to store auth material")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _check(k)
            _check(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _check(v)


class Cassette:
    """Directory-backed store keyed by model + request + run_id."""

    def __init__(self, directory: str | Path) -> None:
        self.dir = Path(directory)

    def _path(self, model: str, request: Any, run_id: str) -> Path:
        return self.dir / f"{key_for(model, request, run_id)}.json"

    def get(self, model: str, request: Any, run_id: str) -> Any | None:
        """Return the stored response, or None if absent."""
        p = self._path(model, request, run_id)
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def put(self, model: str, request: Any, run_id: str, response: Any) -> Path:
        """Store a response; ValueError if request or response holds auth material."""
        _check(model)
        _check(run_id)
        _check(request)
        _check(response)
        self.dir.mkdir(parents=True, exist_ok=True)
        p = self._path(model, request, run_id)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(response, sort_keys=True), encoding="utf-8", newline="\n")
        os.replace(tmp, p)
        return p
