import os


def test_conftest_strips_key_1(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "fake")
    monkeypatch.setenv("META_TOKEN", "fake")
    assert os.environ["MODEL_API_KEY"] == "fake"


def test_conftest_strips_key_2():
    assert "MODEL_API_KEY" not in os.environ
    assert not [k for k in os.environ if k.startswith("META_")]
