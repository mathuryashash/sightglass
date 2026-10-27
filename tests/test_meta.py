import base64
import logging
import sys
import types

import pytest

from agentwarden.cassette import Cassette
from agentwarden.cost import Ledger, SpendRefused
from agentwarden.meta import MetaClient, MetaError, httpx_transport

IMG = b"\x89PNG-fake-bytes"


class FakeTransport:
    def __init__(self, status=200, body=None):
        self.status, self.body, self.calls = status, body if body is not None else {}, []

    def __call__(self, url, headers, json_body):
        self.calls.append((url, dict(headers), json_body))
        return self.status, self.body


def spark_body(text='{"ok": true}', i=1000, o=200):
    return {"output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
            "usage": {"input_tokens": i, "output_tokens": o}}


@pytest.fixture
def parts(tmp_path):
    return Ledger(tmp_path / "l.jsonl", 100.0), Cassette(tmp_path / "cas")


def client(parts, transport, mode="live", cap=1.0, run_id="r1"):
    ledger, cas = parts
    return MetaClient(transport, ledger, cas, run_id, cap, mode)


def stored(parts):
    return list(parts[1].dir.glob("*.json")) if parts[1].dir.exists() else []


def test_bad_mode_rejected(parts):
    with pytest.raises(ValueError):
        client(parts, FakeTransport(), mode="yolo")


def test_sam_image_request_shape_and_charge(parts):
    t = FakeTransport(body={"output": []})
    client(parts, t).sam_image(IMG, "red mug", "r1")
    (url, headers, body), = t.calls
    assert url == "responses"
    assert body["model"] == "sam-3.1"
    (msg,) = body["input"]
    assert msg["role"] == "user"
    kinds = {p["type"]: p for p in msg["content"]}
    assert kinds["input_text"]["text"] == "red mug"
    assert kinds["input_image"]["image_url"] == "data:image/png;base64," + base64.b64encode(IMG).decode()
    assert body["metadata"] == {"mask_encoding": "one_bit", "include_confidence": "true"}
    assert parts[0].spent_total() == 0.0025


def test_sam_image_returns_response(parts):
    t = FakeTransport(body={"output": [{"id": "x"}]})
    assert client(parts, t).sam_image(IMG, "mug", "r1") == {"output": [{"id": "x"}]}


def test_two_sam_images_charge_0_005(parts):
    c = client(parts, FakeTransport(body={"output": []}))
    c.sam_image(IMG, "a", "r1")
    c.sam_image(IMG, "b", "r1")
    assert parts[0].spent_total() == 0.0050


def test_spark_json_charges_usage_tokens(parts):
    t = FakeTransport(body=spark_body(i=1000, o=200))
    out = client(parts, t).spark_json("list items", {"type": "object"}, "r1")
    assert out == {"ok": True}
    body = t.calls[0][2]
    assert body["model"] == "muse-spark-1.3"
    assert body["max_output_tokens"] > 0
    # 1000 * 1.25 + 200 * 4.25 = 2100 micro-dollars
    assert parts[0].spent_total() == 0.0021


def test_spark_json_missing_usage_settles_at_reserve(parts):
    body = spark_body()
    del body["usage"]
    client(parts, FakeTransport(body=body)).spark_json("p", {"type": "object"}, "r1")
    # est_in = ceil((1 + 18 bytes) / 3) + 256 = 263; 263 * 1.25 + 1024 * 4.25 = 4680.75 -> 4681
    assert parts[0].spent_total() == 0.004681


def test_spark_estimate_uses_utf8_bytes(parts):
    body = spark_body()
    del body["usage"]
    client(parts, FakeTransport(body=body)).spark_json("é" * 31, {"type": "object"}, "r1")
    # bytes = 62 + 18 = 80; est_in = ceil(80 / 3) + 256 = 283; 283 * 1.25 + 4352 = 4705.75 -> 4706
    assert parts[0].spent_total() == 0.004706


def test_spark_format_is_not_strict(parts):
    t = FakeTransport(body=spark_body())
    schema = {"type": "object"}
    client(parts, t).spark_json("p", schema, "r1")
    assert t.calls[0][2]["text"] == {
        "format": {"type": "json_schema", "name": "result", "schema": schema, "strict": False}}


def test_sam_input_item_has_message_type(parts):
    t = FakeTransport(body={"output": []})
    client(parts, t).sam_image(IMG, "mug", "r1")
    assert t.calls[0][2]["input"][0]["type"] == "message"


PNG = b"\x89PNG-bytes"
JPEG = b"\xff\xd8\xff-bytes"
WEBP = b"RIFF\x00\x00\x00\x00WEBP-bytes"


@pytest.mark.parametrize("data,mime", [(PNG, "image/png"), (JPEG, "image/jpeg"), (WEBP, "image/webp")])
def test_sam_mime_from_magic_bytes(parts, data, mime):
    t = FakeTransport(body={"output": []})
    client(parts, t).sam_image(data, "mug", "r1")
    url = t.calls[0][2]["input"][0]["content"][1]["image_url"]
    assert url == f"data:{mime};base64," + base64.b64encode(data).decode()


@pytest.mark.parametrize("data", [b"GIF89a-bytes", b"", b"RIFF\x00\x00\x00\x00WAVE-bytes", b"plain text"])
def test_sam_unknown_type_refused_before_reserve(parts, monkeypatch, data):
    t = FakeTransport(body={"output": []})
    c = client(parts, t)
    monkeypatch.setattr(parts[0], "reserve_media", lambda *a, **k: pytest.fail("reserved"))
    with pytest.raises(ValueError):
        c.sam_image(data, "mug", "r1")
    assert t.calls == []


def test_sam_settles_on_images_processed(parts):
    t = FakeTransport(body={"output": [], "usage": {"images_processed": 3}})
    client(parts, t, cap=1.0).sam_image(IMG, "mug", "r1")
    assert parts[0].spent_total() == 0.0075  # 3 * 0.0025


def test_sam_settle_failure_releases_reservation(parts, monkeypatch):
    def bad_settle(rid, count):
        raise RuntimeError("disk full")

    monkeypatch.setattr(parts[0], "settle_media", bad_settle)
    with pytest.raises(RuntimeError, match="disk full"):
        client(parts, FakeTransport(body={"output": []}), cap=0.0025).sam_image(IMG, "mug", "r1")
    monkeypatch.undo()
    # reservation was freed: a second call fits the 0.0025 cap again
    client(parts, FakeTransport(body={"output": []}), cap=0.0025).sam_image(IMG, "mug", "r1")


def test_incomplete_status_raises_refusal_no_retry(parts):
    from agentwarden.meta import MetaRefusal

    t = FakeTransport(body={"status": "incomplete", "output": []})
    with pytest.raises(MetaRefusal) as e:
        client(parts, t).sam_image(IMG, "mug", "r1")
    assert isinstance(e.value, MetaError)
    assert len(t.calls) == 1


def test_refusal_part_raises_and_is_not_recorded(parts):
    from agentwarden.meta import MetaRefusal

    body = {"status": "completed", "usage": {"input_tokens": 1000, "output_tokens": 200},
            "output": [{"type": "message", "content": [{"type": "refusal", "refusal": "no"}]}]}
    t = FakeTransport(body=body)
    with pytest.raises(MetaRefusal):
        client(parts, t, mode="record").spark_json("p", {"type": "object"}, "r1")
    assert len(t.calls) == 1
    assert stored(parts) == []
    assert parts[0].spent_total() == 0.0021  # tokens were consumed, so they are charged


def test_completed_status_is_not_refusal(parts):
    body = spark_body()
    body["status"] = "completed"
    assert client(parts, FakeTransport(body=body)).spark_json("p", {"type": "object"}, "r1") == {"ok": True}


def test_non_2xx_raises_with_status_and_charges_nothing(parts):
    t = FakeTransport(status=429, body={"error": "slow down"})
    with pytest.raises(MetaError) as e:
        client(parts, t).sam_image(IMG, "mug", "r1")
    assert e.value.status == 429
    assert "429" in str(e.value)
    assert parts[0].spent_total() == 0.0
    parts[0].reserve_media("sam-3.1", "image", 400, "r1", 1.0)  # reservation was released


def test_transport_exception_releases(parts):
    def boom(url, headers, body):
        raise ConnectionError("down")

    with pytest.raises(ConnectionError):
        client(parts, boom, cap=0.0025).sam_image(IMG, "mug", "r1")
    client(parts, FakeTransport(body={}), cap=0.0025).sam_image(IMG, "mug", "r1")


def test_cap_refusal_skips_transport(parts):
    t = FakeTransport(body={})
    with pytest.raises(SpendRefused):
        client(parts, t, cap=0.001).sam_image(IMG, "mug", "r1")
    assert t.calls == []


def test_record_stores_then_replay_hits(parts):
    t = FakeTransport(body={"output": [{"id": "m1"}]})
    client(parts, t, mode="record").sam_image(IMG, "mug", "r1")
    assert len(t.calls) == 1
    assert len(stored(parts)) == 1

    t2 = FakeTransport()
    got = client(parts, t2, mode="replay").sam_image(IMG, "mug", "r1")
    assert got == {"output": [{"id": "m1"}]}
    assert t2.calls == []
    assert parts[0].spent_total() == 0.0025  # replay adds no spend


def test_replay_miss_raises_and_never_calls(parts):
    t = FakeTransport()
    with pytest.raises(LookupError):
        client(parts, t, mode="replay").sam_image(IMG, "mug", "r1")
    assert t.calls == []


def test_replay_spark_hit(parts):
    client(parts, FakeTransport(body=spark_body()), mode="record").spark_json("p", {"type": "object"}, "r1")
    t2 = FakeTransport()
    assert client(parts, t2, mode="replay").spark_json("p", {"type": "object"}, "r1") == {"ok": True}
    assert t2.calls == []


def test_live_does_not_store(parts):
    client(parts, FakeTransport(body={"output": []}), mode="live").sam_image(IMG, "mug", "r1")
    assert stored(parts) == []


def test_record_failure_stores_nothing(parts):
    with pytest.raises(MetaError):
        client(parts, FakeTransport(status=500), mode="record").sam_image(IMG, "mug", "r1")
    assert stored(parts) == []


def test_different_concepts_do_not_collide(parts):
    client(parts, FakeTransport(body={"n": 1}), mode="record").sam_image(IMG, "a", "r1")
    with pytest.raises(LookupError):
        client(parts, FakeTransport(), mode="replay").sam_image(IMG, "b", "r1")


class FakeHttpx(types.ModuleType):
    """Stands in for httpx: records requests, replies with a canned status/body or raises."""

    def __init__(self, status=200, body=None, exc=None):
        super().__init__("httpx")
        self.status, self.body, self.exc, self.requests = status, body or {"output": []}, exc, []

    def post(self, url, json=None, timeout=None, headers=None):
        self.requests.append({"url": url, "json": json, "headers": dict(headers or {})})
        if self.exc:
            raise self.exc
        fake = self

        class Resp:
            status_code = fake.status

            def json(self):
                return fake.body

        return Resp()


def test_key_reaches_transport_but_never_cassette_logs_or_errors(parts, caplog, monkeypatch):
    fake_key = "sk-" + "Z9" * 12  # built at runtime, never a literal
    fake = FakeHttpx()
    monkeypatch.setitem(sys.modules, "httpx", fake)
    transport = httpx_transport(fake_key, "https://example.invalid/v1")
    with caplog.at_level(logging.DEBUG):
        client(parts, transport, mode="record").sam_image(IMG, "mug", "r1")
        fake.status, fake.body = 401, {"error": "bad"}
        with pytest.raises(MetaError) as bad_status:
            client(parts, transport).sam_image(IMG, "x", "r1")
        fake.exc = RuntimeError("connection failed")
        with pytest.raises(RuntimeError) as boom:
            client(parts, transport).sam_image(IMG, "y", "r1")
    # the Authorization header reached the fake transport...
    assert fake.requests[0]["url"] == "https://example.invalid/v1/responses"
    assert fake.requests[0]["headers"]["Authorization"] == "Bearer " + fake_key
    # ...and the key is nowhere else
    blob = "".join(p.read_text() for p in stored(parts))
    assert blob and fake_key not in blob and "authorization" not in blob.lower()
    assert fake_key not in caplog.text and "authorization" not in caplog.text.lower()
    assert fake_key not in str(bad_status.value) and fake_key not in str(boom.value)


def test_httpx_transport_is_a_lazy_factory():
    tr = httpx_transport("sk-" + "A1" * 12, "https://example.invalid/v1")  # never called
    assert callable(tr)
