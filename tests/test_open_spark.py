"""Open-weights path: offline tests with a fake transport (no network, no key)."""
import ast
import json
from pathlib import Path

import pytest

from agentwarden import diagnoser as D
from agentwarden.open_spark import OPEN_MODEL, OpenSparkClient

SRC = Path(__file__).resolve().parents[1] / "src" / "agentwarden" / "open_spark.py"


def _incident():
    plc = [(float(t), 24.0) for t in range(0, 30)]
    cam = [(float(t), 24.0 + 1.5 * t) for t in range(0, 30)]
    ev = [{"t": 29.0, "status": "OK", "pump_on": False, "tripped": True, "trip_reason": "DIVERGENCE",
           "actions": ["WRITE_PERMIT_FALSE"]}]
    return D.build_incident(ev, plc, cam, limits={"divergence_pts": 15.0, "divergence_s": 5.0, "high_high_pct": 95.0})


def _reply(obj, finish="stop"):
    return {"choices": [{"finish_reason": finish, "message": {"content": json.dumps(obj)}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20}}


GOOD = {"label": "SPOOF", "confidence": 0.8,
        "evidence": ["plc held 24.0 while camera rose to 67.5"],
        "operator_note": "Controller stuck while camera saw filling.",
        "suggested_check": "Read the sight glass."}


def test_open_client_records_then_replays_keyless(tmp_path):
    sent = []

    def fake(body):
        sent.append(body)
        return _reply(GOOD)

    c = OpenSparkClient(fake, tmp_path, "record")
    out = D.diagnose(_incident(), c, "t")
    assert isinstance(out, D.Diagnosis) and out.label == "SPOOF"
    assert sent and sent[0]["model"] == OPEN_MODEL
    r = OpenSparkClient(None, tmp_path, "replay")
    out2 = D.diagnose(_incident(), r, "t")
    assert isinstance(out2, D.Diagnosis) and out2.label == "SPOOF"


def test_open_client_replay_miss_is_unavailable(tmp_path):
    out = D.diagnose(_incident(), OpenSparkClient(None, tmp_path, "replay"), "t")
    assert isinstance(out, D.DiagnosisError)


def test_open_client_truncated_output_is_unavailable(tmp_path):
    c = OpenSparkClient(lambda b: _reply(GOOD, finish="length"), tmp_path, "record")
    assert isinstance(D.diagnose(_incident(), c, "t"), D.DiagnosisError)


def test_open_client_ungrounded_rejected(tmp_path):
    bad = dict(GOOD, evidence=["plc held 31.7 while camera rose to 99.9"])
    c = OpenSparkClient(lambda b: _reply(bad), tmp_path, "record")
    out = D.diagnose(_incident(), c, "t")
    assert isinstance(out, D.DiagnosisError) and out.reason == "ungrounded"


def test_open_client_requires_transport_in_record_mode(tmp_path):
    with pytest.raises(ValueError):
        OpenSparkClient(None, tmp_path, "record")


def test_open_spark_has_no_path_to_actuation():
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | \
            {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    mods = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names} | \
           {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not mods & {"agentwarden.permit", "agentwarden.interlock", "agentwarden.modbus_io", "agentwarden.pipeline"}
    assert not names & {"clear_permit", "rearm_from_reset", "reset_with_grant", "reset", "redeem"}
    assert "environ" not in names and "getenv" not in names  # never reads keys from env vars
