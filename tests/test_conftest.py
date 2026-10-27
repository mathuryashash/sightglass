import os


def test_conftest_strips_key_1():
    os.environ["MODEL_API_KEY"] = "fake"  # set directly, no monkeypatch cleanup
    os.environ["META_TOKEN"] = "fake"
    os.environ["SG_ENV_FILE"] = "fake"
    os.environ["SG_OTHER"] = "fake"
    assert os.environ["MODEL_API_KEY"] == "fake"


def test_conftest_strips_key_2():
    assert "MODEL_API_KEY" not in os.environ
    assert not [k for k in os.environ if k.startswith("META_")]
    assert not [k for k in os.environ if k.startswith("SG_")]
