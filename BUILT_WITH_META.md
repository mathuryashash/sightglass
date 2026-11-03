# Built with Meta

SightGlass uses three Meta models: SAM 3.1, Muse Spark (hosted), and Muse Spark Open (through Baseten). This file says what each one does, the exact request shape, what worked, and what hurt.

Prices below are the ones in `src/agentwarden/cost.py`, which the spend ledger uses.

## SAM 3.1

**Job.** Find the liquid and the bottle in each camera frame, and find a hand if one is in the way. The level estimator turns the masks into a percentage. SAM does not decide anything about the pump.

**Request.** One image per call, one concept per call (`src/agentwarden/meta.py:113`, `sam_image`):

```
POST {base_url}/responses
{
  "model": "sam-3.1",
  "input": [{"type": "message", "role": "user", "content": [
      {"type": "input_text",  "text": "<concept>"},
      {"type": "input_image", "image_url": "data:image/jpeg;base64,..."}]}],
  "metadata": {"mask_encoding": "one_bit", "include_confidence": "true"}
}
```

The concepts are "orange liquid", "plastic bottle" and "hand" (`src/agentwarden/vision.py:27-28`). The three calls are at `vision.py:58`, `:59` and `:66`. The reply is parsed with Meta's `meta-sam-parser` package (pinned to 0.1.0). Our code checks that the mask size matches the frame size and raises if it does not (`src/agentwarden/sam.py`).

**What worked.**

- Plain-language concepts were enough. "orange liquid" found the liquid and "plastic bottle" found the container on a hand-held phone clip, with no training and no per-tank setup.
- On the 54-frame pour, SAM readings went backwards by at most 1.8 points after the hand left the shot (`eval/RESULTS.md` section 1). A colour-threshold reader on the same frames went backwards by up to 7.9 points (section 7). That is one clip.
- Against 7 still photos with eyeballed reference levels, the mean difference was 0.2 points (section 1). That is agreement with a rough reference, not an accuracy figure.
- The "hand" concept made a usable occlusion detector. On the hand clip it turned 21 frames marked OK, some far off, into 16 `OCCLUDED` frames and 6 clean ones (section 8).

**Pain points and feedback.**

- **Video has a 50-frame minimum.** The price table has a 50-frame floor for video (`cost.py:29`, 0.0002 USD per frame, minimum 50). A short clip costs the same as 50 frames. We used per-image calls instead, at 0.0025 USD per image (`cost.py:21`). A lower or no minimum would make short video calls usable for this kind of check.
- **One concept per call.** Reading a tank needs the liquid and the container, and a safe system also wants a hand check. That is three image calls per frame. The hand check alone adds 50 % to the image cost over two calls. A multi-concept call would cut this.
- **Output parsing.** The reply carries boxes as text tokens. We read box width and height from each box token with a regex, because the parser package does not expose them, and compare them with the frame size. A typed reply would remove that step.
- **Hand masks cut the liquid mask.** When a hand crosses the bottle, SAM still reports a liquid mask, but a cut one. Before the occlusion check, readings on the hand clip were up to 88.6 points off while marked OK. This is not a SAM bug, but anyone using the masks needs to know it.

## Muse Spark (hosted)

**Job.** After a trip, or after the camera has been unverified for more than 10 s, read a 120 s incident window and return a JSON diagnosis for a person. It is advisory. A static test forbids any code path from it to the pump.

**Request** (`src/agentwarden/meta.py:127`, `spark_json`; calls at `src/agentwarden/diagnoser.py:341` and `:344`):

```
POST {base_url}/responses
{
  "model": "muse-spark-1.3",
  "input": "<prompt with INCIDENT_JSON>",
  "max_output_tokens": 8192,
  "text": {"format": {"type": "json_schema", "name": "result", "schema": {...}, "strict": false}}
}
```

The schema has five keys: `label` (SPOOF, LEAK, CAMERA_FAULT, BENIGN, UNSURE), `confidence`, `evidence`, `operator_note` and `suggested_check`. The model may ask for up to two read-only lookups through an optional `tool_request` field, so there are at most two calls per incident. Our code checks every field, and checks that every number quoted in `evidence` appears in the incident data. A diagnosis that fails a check is shown as "diagnosis unavailable", never as advice.

**What worked.**

- On 6 recorded incidents (freeze, drift, replay at two fault starts) the model labelled all 6 SPOOF, quoted only numbers that were in the data, and agreed with the plain rule label (`eval/RESULTS.md` section 6).
- The recorded calls are stored in `demo/real/spark_cassette`, so the dashboard shows real Spark text with no key.

**Pain points and feedback.**

- **Reasoning tokens ate the output budget.** With `max_output_tokens` at 1024, the reply was cut off in the middle of the JSON. The model reasons before it answers, using roughly 250 to 1,500 hidden tokens (comment at `meta.py:19`). We raised the cap to 8192 (commit `2f7aa63`). It would help to see the reasoning token count separately in `usage`, and to be able to cap it.
- **Prose before the JSON.** Even with a JSON schema, the reply sometimes started with a sentence. The client now parses the whole text, and if that fails, the first balanced `{...}` object in it (`meta.py:150`, `_loads_lenient`). We left `strict` false.
- **Tool use.** We did not know of a tool-calling format to rely on, so the optional lookup is a JSON field in the schema and our code runs it. It works, but it is not native tool use.

## Muse Spark Open (Baseten)

**Job.** The same diagnoser, same prompt, same checks, on open weights. This shows the advice layer does not depend on one hosted endpoint.

**Request** (`src/agentwarden/open_spark.py:54`):

```
POST https://inference.baseten.co/v1/chat/completions
{
  "model": "meta-models/muse-spark-open",
  "max_tokens": 8192,
  "response_format": {"type": "json_object"},
  "messages": [{"role": "user", "content": "<prompt>\n\nJSON schema for your reply:\n<schema>"}]
}
```

The schema is placed in the prompt, so the request stays portable across OpenAI-compatible servers. The auth header is `Authorization: Api-Key <key>`.

**What worked.** 3 of 3 recorded runs (freeze, drift, replay at fault start 20) labelled SPOOF, with evidence lines that quote the data (`demo/real/open_spark_runs.json`). Each call took between 7.7 and 13.5 s (`secs` in that file). The note for the freeze run reads: "Controller froze near 41% while camera showed tank steadily filling past 65%." The requests and responses are in `demo/real/open_spark_cassette/`, so they replay without a key.

**Pain points and feedback.**

- **Model id mismatch.** The id in the guide we followed ended in "-1026" and returned HTTP 404. The id that worked was `meta-models/muse-spark-open`. Please keep the guide and the catalogue in step.
- **Same reasoning budget issue.** The model reasons first, so `max_tokens` needs to be large (8192 here). The client rejects any reply whose `finish_reason` is not `stop`, so a truncated reply is an error and not a guess.

## What this cost

Measured from the local spend ledger on 3 Nov 2026: SAM 3.1 0.72 USD, Muse Spark 0.24 USD (0.96 USD on the Meta API), plus about 0.045 USD on Baseten for the three Muse Spark Open calls (computed from their token usage at the listed 1.25/4.25 USD per million input/output tokens). Every live path goes through the spend ledger in `src/agentwarden/cost.py`, which refuses a call that could pass its cap, and the live Spark path in the dashboard is capped at 1 USD per process.

Keyless replay costs nothing. The recorded SAM readings, the Spark cassette (`demo/real/spark_cassette`) and the Baseten cassette (`demo/real/open_spark_cassette`) are in the repository.
