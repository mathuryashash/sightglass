import threading

import pytest

from agentwarden.cassette import Cassette, key_for


def test_cassette_key_includes_run_id():
    req = {"a": 1, "b": [1, 2]}
    assert key_for("m", req, "r1") != key_for("m", req, "r2")
    assert key_for("m", req, "r1") == key_for("m", req, "r1")
    assert key_for("m", {"a": 1, "b": [1, 2]}, "r1") == key_for("m", {"b": [1, 2], "a": 1}, "r1")


def test_cassette_roundtrip(tmp_path):
    c = Cassette(tmp_path)
    assert c.get("m", {"x": 1}, "r1") is None
    c.put("m", {"x": 1}, "r1", {"text": "hi"})
    assert c.get("m", {"x": 1}, "r1") == {"text": "hi"}
    assert c.get("m", {"x": 1}, "r2") is None


@pytest.mark.parametrize("bad", [
    "Authorization: x", "Bearer abc", "sk-" + "Z" * 20,
])
def test_cassette_refuses_auth_material(tmp_path, bad):
    c = Cassette(tmp_path)
    with pytest.raises(ValueError):
        c.put("m", {"h": bad}, "r1", {"ok": 1})
    with pytest.raises(ValueError):
        c.put("m", {"h": 1}, "r1", {"nested": [{"v": bad}]})
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("bad", [
    "authorization: x", "AUTHORIZATION: x", "bearer abc", "BEARER abc",
    b"Authorization: x", b"bearer abc", b"sk-" + b"Z" * 20,
])
def test_cassette_refuses_case_insensitive_and_bytes(tmp_path, bad):
    c = Cassette(tmp_path)
    with pytest.raises(ValueError):
        c.put("m", {"h": bad}, "r1", {"ok": 1})
    with pytest.raises(ValueError):
        c.put("m", {"h": 1}, "r1", {"nested": [bad]})
    assert list(tmp_path.iterdir()) == []


def test_cassette_concurrent_same_key(tmp_path):
    c = Cassette(tmp_path)
    errors = []

    def work() -> None:
        try:
            for _ in range(25):
                c.put("m", {"x": 1}, "r1", {"text": "hi"})
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    ts = [threading.Thread(target=work) for _ in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert errors == []
    assert c.get("m", {"x": 1}, "r1") == {"text": "hi"}
    assert len(list(tmp_path.iterdir())) == 1  # no leftover temp files
