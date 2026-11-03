# Third-party software

SightGlass is MIT licensed (see `LICENSE`). It depends on the packages below. Versions are the ones pinned in `pyproject.toml` (where there is no pin, `uv.lock` sets the version). Check each package's own license file before you redistribute.

| Package | Used for | Pin |
|---|---|---|
| meta-sam-parser | Parses SAM 3.1 output (Meta; released under the SAM License) | 0.1.0 |
| fastapi | Dashboard server | 0.162.0 |
| uvicorn | Runs the dashboard server | 0.54.0 |
| httpx | HTTP client for live Meta and Baseten calls and the phone camera | uv.lock |
| numpy | Mask and level arithmetic | >=2.4.6 |
| pillow | Frame decoding, overlays | >=13.0.0 |
| pymodbus | Simulated Modbus link | 3.16.1 |
| pytest, hypothesis | Tests (dev only) | >=9.1.1, >=6.170.12 |

## Models and services

- Meta SAM 3.1, Muse Spark and Muse Spark Open are used under their own terms. No model weights are in this repository. Recorded model responses are in `demo/real/spark_cassette/` and `demo/real/open_spark_cassette/`.
- Muse Spark Open is called through Baseten's hosted API.

## Media

The images in `demo/real/` and `docs/` are photos and frames recorded by the author. They show a plastic bottle and an orange liquid.

## Tools

Claude Code (Anthropic) was used as a coding assistant. See `docs/AI_ASSISTANTS.md`.
