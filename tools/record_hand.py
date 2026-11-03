# Recording script (run once by the author with a local key file; not needed for replay). Kept for provenance.

# Record SAM 3.1 "hand" detections on the hand and lights clips (occlusion signal). Never prints the key.
import io, sys, json, pathlib
from PIL import Image
sys.path.insert(0, r"D:/hackathons/meta-ai-dev/agentwarden/src")
from agentwarden.cost import Ledger
from agentwarden.cassette import Cassette
from agentwarden.meta import MetaClient, httpx_transport, _output_text
from agentwarden.sam import parse_sam
ROOT = pathlib.Path(r"D:/hackathons/meta-ai-dev")
key = next(l.split("=", 1)[1].strip() for l in open(ROOT / "secrets/.env", encoding="utf-8") if l.startswith("MODEL_API_KEY="))
client = MetaClient(httpx_transport(key, "https://api.meta.ai/v1"), Ledger(ROOT / "runs/ledger.jsonl", 100.0),
                    Cassette(ROOT / "runs/cassettes"), "clips-hand", 0.30, "record")
for name in ["hand", "lights"]:
    RJ = ROOT / f"agentwarden/demo/real/{name}_replay.json"
    recs = json.loads(RJ.read_text())
    for rec in recs:
        img = Image.open(ROOT / f"agentwarden/demo/real/{name}_frames" / rec["photo"]).convert("RGB")
        W = 478; H = round(img.height * W / img.width); img = img.resize((W, H))
        buf = io.BytesIO(); img.save(buf, "JPEG", quality=90)
        dets = parse_sam(_output_text(client.sam_image(buf.getvalue(), "hand", "clips-hand")), W, H)
        hb = [[round(v) for v in d.box] for d in dets if (d.confidence or 0) >= 0.5]
        rec["hand_boxes"] = hb
        box = rec.get("box")
        ov = 0.0
        if box:
            bx0, by0, bx1, by1 = box; ba = max(1, (bx1 - bx0) * (by1 - by0))
            for h in hb:
                ix = max(0, min(bx1, h[2]) - max(bx0, h[0])); iy = max(0, min(by1, h[3]) - max(by0, h[1]))
                ov = max(ov, ix * iy / ba)
        rec["hand_overlap"] = round(ov, 3)
        print(name, rec["t"], rec["status"], rec.get("level"), "hands", len(hb), "overlap", rec["hand_overlap"], flush=True)
    RJ.write_text(json.dumps(recs, indent=2))
