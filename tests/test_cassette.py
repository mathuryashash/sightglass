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
    "Authorization: x", "Bearer abc", "sk-ABCDEFGHIJKLMNOP1234",
])
def test_cassette_refuses_auth_material(tmp_path, bad):
    c = Cassette(tmp_path)
    with pytest.raises(ValueError):
        c.put("m", {"h": bad}, "r1", {"ok": 1})
    with pytest.raises(ValueError):
        c.put("m", {"h": 1}, "r1", {"nested": [{"v": bad}]})
    assert list(tmp_path.iterdir()) == []
