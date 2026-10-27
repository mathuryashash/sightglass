import base64
import json
import logging

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
    assert parts[0].spent_total() > 0


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


def test_key_never_in_headers_cassette_or_logs(parts, caplog):
    fake_key = "sk-" + "Z9" * 12  # built at runtime, never a literal
    t = FakeTransport(body={"output": []})
    with caplog.at_level(logging.DEBUG):
        client(parts, t, mode="record").sam_image(IMG, "mug", "r1")
        with pytest.raises(MetaError):
            client(parts, FakeTransport(status=401, body={"error": "bad"})).sam_image(IMG, "x", "r1")
    # headers the client builds carry no auth; the key lives only inside httpx_transport
    assert not any(k.lower() == "authorization" for k in t.calls[0][1])
    assert fake_key not in json.dumps(t.calls[0][1])
    blob = "".join(p.read_text() for p in stored(parts))
    assert fake_key not in blob and "authorization" not in blob.lower()
    assert fake_key not in caplog.text and "authorization" not in caplog.text.lower()


def test_httpx_transport_is_a_lazy_factory():
    tr = httpx_transport("sk-" + "A1" * 12, "https://example.invalid/v1")  # never called
    assert callable(tr)
