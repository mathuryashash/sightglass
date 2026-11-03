# Recording script (run once by the author with a local key file; not needed for replay). Kept for provenance.
# Go/no-go spike on the user's REAL bottle photos (7 levels). Never prints the key.
# Per photo: SAM "orange liquid" + "bottle" -> select.pick -> level.estimate with a calibration
# built from the bottle box (bottom of box = 0 %, shoulder ~ 25 % down from top = 100 %).
import io, sys, time, json, pathlib
from PIL import Image, ImageDraw
sys.path.insert(0, r"D:/hackathons/meta-ai-dev/agentwarden/src")
from agentwarden.cost import Ledger
from agentwarden.cassette import Cassette
from agentwarden.meta import MetaClient, httpx_transport, _output_text
from agentwarden.sam import parse_sam
from agentwarden.select import pick
from agentwarden.level import estimate, Calibration

ROOT = pathlib.Path(r"D:/hackathons/meta-ai-dev")
NAME = sys.argv[1]
SP = ROOT / f"agentwarden/demo/real/{NAME}_frames"
OUT = ROOT / "agentwarden/demo/real"
key = next(l.split("=", 1)[1].strip() for l in open(ROOT / "secrets/.env", encoding="utf-8")
           if l.startswith("MODEL_API_KEY="))
client = MetaClient(httpx_transport(key, "https://api.meta.ai/v1"), Ledger(ROOT / "runs/ledger.jsonl", 100.0),
                    Cassette(ROOT / "runs/cassettes"), "clips-replay", 0.60, "record")
W = 478  # downscale 3072x4096 -> 768x1024 (SAM cost is per image, smaller = faster)
results = []
for jpg in sorted(SP.glob("t*.jpg")):
    img = Image.open(jpg).convert("RGB")
    H = round(img.height * W / img.width)
    img = img.resize((W, H))
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=90); data = buf.getvalue()
    dets = {}
    t = time.time()
    for concept in ["orange liquid", "plastic bottle"]:
        try:
            resp = client.sam_image(data, concept, "clips-replay")
            dets[concept] = parse_sam(_output_text(resp), W, H)
        except Exception as e:
            print(jpg.name, concept, "ERROR", type(e).__name__, str(e)[:200]); dets[concept] = []
    dt = time.time() - t
    sel = pick(dets["plastic bottle"], dets["orange liquid"])
    row = {"t": (int(jpg.stem[1:]) - 1) / 2.0, "photo": jpg.name, "secs": round(dt, 1),
           "n_bottle": len(dets["plastic bottle"]), "n_liquid": len(dets["orange liquid"])}
    if sel is None:
        row["status"] = "CAMERA_BLIND"
    else:
        box, mask, conf = sel
        l, t_, r, b = box
        # calibration: 0 % at bottle bottom, 100 % at shoulder (~22 % of bottle height below its top)
        shoulder = t_ + 0.22 * (b - t_)
        cal = Calibration.from_points([(b, 0.0), (shoulder, 100.0)])
        rd = estimate(mask, box, cal, conf, max(d.confidence or 0 for d in dets["plastic bottle"]))
        row.update(status=rd.status, level=None if rd.level_pct is None else round(rd.level_pct, 1),
                   conf=round(rd.confidence, 2), top_row=rd.top_row, box=[round(v) for v in box])
        ov = img.copy(); d = ImageDraw.Draw(ov)
        d.rectangle(box, outline=(0, 255, 0), width=4)
        if mask is not None:
            m = Image.fromarray((mask * 120).astype("uint8"))
            ov.paste(Image.new("RGB", ov.size, (0, 80, 255)), (0, 0), m)
        if rd.top_row is not None:
            d.line([(l, rd.top_row), (r, rd.top_row)], fill=(255, 0, 0), width=4)
        d.text((10, 10), f"{rd.status} {row.get('level')}% c={row['conf']}", fill=(255, 0, 0))
        ov.save(SP / f"ov_{jpg.stem}.jpg")
        row.update(level_pct=row["level"], confidence=row["conf"])
    print(json.dumps(row), flush=True); results.append(row)
(OUT / f"{NAME}_replay.json").write_text(json.dumps(results, indent=2))
