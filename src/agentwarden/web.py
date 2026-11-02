"""HMI dashboard: FastAPI on 127.0.0.1 only. No CDN, no external JS; server-rendered HTML, PNG and SVG.

SIMULATED CONTROLLER ON LOCALHOST - NO REAL SYSTEM. This module only displays state. The two POST routes
(human reset, demo fault buttons) call Pipeline methods and need a per-process random token plus an allowed
Host header. Nothing here can enable the pump: a reset is only ever the pipeline's human_reset(), which the
safety logic may refuse.

    python -m agentwarden.web --replay demo --scenario freeze --fault-start 20
"""
import argparse
import hmac
import html
import io
import secrets
import threading
from urllib.parse import parse_qs, urlsplit

import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response
from PIL import Image, ImageDraw

from agentwarden.advisor import add_diagnoser_args, build_advisor
from agentwarden.fault_injector import SCENARIOS
from agentwarden.modbus_io import ModbusClient, ModbusPlcServer
from agentwarden.pipeline import BANNER, Pipeline, add_common_args, drive, resolve
from agentwarden.vision import ReplayVision

ALLOWED_HOSTS = ("127.0.0.1", "localhost")
TREND_WINDOW_S = 300.0
SVG_W, SVG_H, PAD_L, PAD_B, PAD_T = 640, 220, 34, 22, 8
PLC_COLOR, CAM_COLOR = "#0072B2", "#D55E00"  # colour-blind-safe pair


def _host_ok(host_header: str | None) -> bool:
    host = (host_header or "").lower()
    if host.startswith("["):  # IPv6 literals are never allowed
        return False
    return host.rsplit(":", 1)[0] in ALLOWED_HOSTS if ":" in host else host in ALLOWED_HOSTS


def render_frame(jpeg: bytes | None, mask, box, top_row) -> bytes:
    """Latest camera frame as PNG, with the liquid mask tinted, the container box and the surface line."""
    if not jpeg:
        im = Image.new("RGB", (320, 240), (60, 60, 60))
        ImageDraw.Draw(im).text((110, 112), "no frame yet", fill=(220, 220, 220))
    else:
        im = Image.open(io.BytesIO(jpeg)).convert("RGB")
        if mask is not None and mask.shape == (im.height, im.width):
            arr = np.asarray(im).copy()
            arr[mask] = (0.55 * arr[mask] + 0.45 * np.array([255, 140, 0])).astype(np.uint8)
            im = Image.fromarray(arr)
        d = ImageDraw.Draw(im)
        if box is not None:
            d.rectangle(box, outline=(0, 200, 255), width=2)
            if top_row is not None:
                d.line([(box[0], top_row), (box[2], top_row)], fill=(255, 0, 0), width=2)
    out = io.BytesIO()
    im.save(out, "PNG")
    return out.getvalue()


def _runs(points):
    """Split [(x, y|None)] into runs of consecutive non-None points."""
    run = []
    for x, y in points:
        if y is None:
            if run:
                yield run
            run = []
        else:
            run.append((x, y))
    if run:
        yield run


def render_trend(history) -> str:
    """Inline SVG of PLC and camera level (%) over the last 5 minutes."""
    hist = list(history)
    t1 = hist[-1][0] if hist else 0.0
    t0 = t1 - TREND_WINDOW_S
    pw, ph = SVG_W - PAD_L - 6, SVG_H - PAD_B - PAD_T

    def x_of(t):
        return round(PAD_L + (t - t0) / TREND_WINDOW_S * pw, 1)

    def y_of(v):
        return round(PAD_T + (100 - min(100.0, max(0.0, v))) / 100 * ph, 1)

    def line(idx, color, label):
        pts = [(x_of(r[0]), None if r[idx] is None else y_of(r[idx])) for r in hist if r[0] >= t0]
        return "".join(
            f'<polyline fill="none" stroke="{color}" stroke-width="2" data-series="{label}" '
            f'points="{" ".join(f"{x},{y}" for x, y in run)}"/>' for run in (list(_runs(pts)) or [[]]))

    grid = "".join(
        f'<line x1="{PAD_L}" x2="{SVG_W - 6}" y1="{y_of(v)}" y2="{y_of(v)}" stroke="#888" stroke-width="1" '
        f'stroke-dasharray="{"4 3" if v == 95 else "1 0"}" opacity="0.5"/>'
        f'<text x="2" y="{y_of(v) + 4}" font-size="11" fill="#888">{v}</text>' for v in (0, 50, 95, 100))
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SVG_W} {SVG_H}" role="img" '
        f'aria-label="PLC and camera level, last 5 minutes">'
        f'<title>PLC and camera level (%), last 5 minutes</title>{grid}'
        f'{line(1, PLC_COLOR, "plc")}{line(2, CAM_COLOR, "camera")}'
        f'<text x="{PAD_L + 6}" y="{SVG_H - 6}" font-size="12" fill="{PLC_COLOR}">PLC (reported)</text>'
        f'<text x="{PAD_L + 130}" y="{SVG_H - 6}" font-size="12" fill="{CAM_COLOR}">camera</text>'
        f'<text x="{SVG_W - 170}" y="{SVG_H - 6}" font-size="11" fill="#888">last 5 min, 95 = high-high</text></svg>')


CSS = """
body{font:15px system-ui,sans-serif;margin:0;background:#14171c;color:#e8eaed}
.banner{background:#b00020;color:#fff;font-weight:700;text-align:center;padding:8px;letter-spacing:.04em}
main{display:grid;grid-template-columns:minmax(280px,1fr) minmax(300px,1.4fr);gap:16px;padding:16px}
section{background:#1d222a;border-radius:8px;padding:12px 16px}
h2{margin:8px 0;font-size:14px;text-transform:uppercase;color:#9aa4b2}
dl{display:grid;grid-template-columns:auto 1fr;gap:4px 14px;margin:0}dt{color:#9aa4b2}dd{margin:0;font-weight:600}
.bad{color:#ff6b6b}.good{color:#5fd38d}.warn{background:#4a3b00;color:#ffd95a;padding:8px;border-radius:6px;margin-top:8px}
.hidden{display:none}ul{margin:4px 0;padding-left:20px}
img{max-width:100%;border-radius:6px}svg{width:100%;height:auto;background:#fff;border-radius:6px}
button{font:inherit;padding:6px 12px;margin:2px;border-radius:6px;border:1px solid #556;background:#2b3340;color:#e8eaed;cursor:pointer}
form{display:inline}
.wide{grid-column:1/-1}.note{color:#9aa4b2;font-size:13px}
"""

JS = """
const $=id=>document.getElementById(id);
const fmt=v=>v===null||v===undefined?'-':Number(v).toFixed(1)+' %';
const relayText=r=>!r?'-':(r.pump_enabled?'CLOSED - pump allowed':'OPEN - pump cut')+' ('+r.driver+(r.relay_fault?', FAULT':'')+')';
async function poll(){
 try{
  const s=await (await fetch('/api/state',{cache:'no-store'})).json();
  $('clock').textContent=s.clock;
  $('plc').textContent=fmt(s.plc_level);
  $('cam').textContent=fmt(s.camera.level_pct);
  $('conf').textContent=Math.round(s.camera.confidence*100)+' %';
  $('status').textContent=s.camera.status;
  $('pump').textContent=s.pump_on?'RUNNING':'STOPPED';
  $('permit').textContent=s.permit?'ON':'OFF';
  $('trip').textContent=s.tripped?('TRIPPED ('+s.trip_reason+')'):'not tripped';
  $('trip').className=s.tripped?'bad':'good';
  $('scenario').textContent=s.scenario;
  const ul=$('alarms');ul.textContent='';
  for(const a of s.alarms){const li=document.createElement('li');li.textContent=a;ul.appendChild(li);}
  $('blind').textContent=s.camera_blind_notice?'camera blinded - no trip':'';
  $('blind').className=s.camera_blind_notice?'warn':'hidden';
  $('frame').src='/frame.png?'+Date.now();
  $('relay').textContent=relayText(s.relay);
  $('trend').innerHTML=await (await fetch('/trend.svg',{cache:'no-store'})).text();
  $('diag').innerHTML=await (await fetch('/diagnosis',{cache:'no-store'})).text();
 }catch(e){$('clock').textContent='disconnected';}
}
setInterval(poll,1000);
"""


def relay_text(r: dict | None) -> str:
    if not r:
        return "-"
    state = "CLOSED - pump allowed" if r.get("pump_enabled") else "OPEN - pump cut"
    return f'{state} ({r.get("driver", "?")}{", FAULT" if r.get("relay_fault") else ""})'


def render_diagnosis(d: dict | None) -> str:
    """The advisory card. The model text is untrusted: every string goes through html.escape."""
    e = html.escape
    d = d or {}
    state = d.get("state", "off")
    rule = d.get("rule_label")
    rule_line = f"<p>Rule baseline (no model): <b>{e(str(rule))}</b></p>" if rule else ""
    if state == "ready":
        try:
            conf = f'{round(float(d.get("confidence", 0)) * 100)} %'
        except (TypeError, ValueError):
            conf = "-"
        evidence = "".join(f"<li>{e(str(x))}</li>" for x in d.get("evidence") or [])
        cross = ""
        if d.get("agrees") is False:
            cross = (f'<p class="bad">DISAGREES with the rule baseline: model {e(str(d.get("label")))}, '
                     f'rule {e(str(rule))}</p>')
        elif d.get("agrees") is True:
            cross = f'<p class="good">Agrees with the rule baseline ({e(str(rule))})</p>'
        body = (f'<p>Label: <b>{e(str(d.get("label")))}</b> &middot; confidence {e(conf)}</p>'
                f'<ul>{evidence}</ul><p>{e(str(d.get("operator_note", "")))}</p>'
                f'<p>Suggested check: {e(str(d.get("suggested_check", "")))}</p>{cross}')
        if not cross:
            body += rule_line
    elif state == "pending":
        body = f"<p>diagnosing... ({e(str(d.get('trigger', '')))})</p>{rule_line}"
    elif state == "unavailable":
        body = (f'<p class="warn">diagnosis unavailable ({e(str(d.get("reason", "")))}). The safety rule is not '
                f"affected.</p>{rule_line}")
    else:
        body = "<p>No diagnosis requested.</p>"
    return (f'<h2>Diagnosis (advisory - Muse Spark)</h2>{body}'
            f'<p class="note">advisory only - does not control the pump</p>')


def render_page(s: dict, token: str, nonce: str, trend_svg: str) -> str:
    """Every dynamic string goes through html.escape."""
    e = html.escape
    cam = s["camera"]

    def pct(v):
        return "-" if v is None else f"{v:.1f} %"

    blind = "camera blinded - no trip" if s["camera_blind_notice"] else ""
    alarms = "".join(f"<li>{e(str(a))}</li>" for a in s["alarms"])
    trip = f'TRIPPED ({e(str(s["trip_reason"]))})' if s["tripped"] else "not tripped"
    inject = "".join(
        f'<form method="post" action="/inject/{e(sc)}"><input type="hidden" name="token" value="{e(token)}">'
        f'<button type="submit">inject: {e(sc)}</button></form>' for sc in SCENARIOS)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>SightGlass HMI (simulated)</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style nonce="{nonce}">{CSS}</style></head><body>
<div class="banner" role="alert">{e(BANNER)}</div>
<main>
<section><h2>Process <span id="clock">{e(s["clock"])}</span></h2><dl>
<dt>PLC level</dt><dd id="plc">{pct(s["plc_level"])}</dd>
<dt>Camera level</dt><dd id="cam">{pct(cam["level_pct"])}</dd>
<dt>Camera confidence</dt><dd id="conf">{round(cam["confidence"] * 100)} %</dd>
<dt>Camera status</dt><dd id="status">{e(cam["status"])}</dd>
<dt>Pump</dt><dd id="pump">{"RUNNING" if s["pump_on"] else "STOPPED"}</dd>
<dt>Permit</dt><dd id="permit">{"ON" if s["permit"] else "OFF"}</dd>
<dt>Relay</dt><dd id="relay">{e(relay_text(s.get("relay")))}</dd>
<dt>Trip</dt><dd id="trip" class="{"bad" if s["tripped"] else "good"}">{trip}</dd>
<dt>Fault scenario</dt><dd id="scenario">{e(s["scenario"])}</dd></dl>
<div id="blind" class="{"warn" if blind else "hidden"}">{blind}</div>
<h2>Alarms</h2><ul id="alarms">{alarms}</ul>
<h2>Controls</h2>
<form method="post" action="/reset"><input type="hidden" name="token" value="{e(token)}">
<button type="submit">Human reset (refused while the condition holds)</button></form>
<div>{inject}</div></section>
<section><h2>Camera frame (mask + surface line)</h2>
<img id="frame" src="/frame.png" alt="latest camera frame with liquid mask overlay" width="640">
<h2>Trend</h2><div id="trend">{trend_svg}</div></section>
<section class="wide" id="diag">{render_diagnosis(s.get("diagnosis"))}</section>
</main><script nonce="{nonce}">{JS}</script></body></html>"""


def create_app(pl: Pipeline) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.state.token = secrets.token_urlsafe(32)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        if not _host_ok(request.headers.get("host")):  # also blocks DNS rebinding
            return PlainTextResponse("bad host", status_code=400)
        origin = request.headers.get("origin")
        if request.method == "POST" and origin is not None and urlsplit(origin).hostname not in ALLOWED_HOSTS:
            return PlainTextResponse("bad origin", status_code=403)
        resp = await call_next(request)
        resp.headers.setdefault("Cache-Control", "no-store")
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        return resp

    async def authorized(request: Request) -> bool:
        sent = request.headers.get("x-token")
        if sent is None:
            body = (await request.body()).decode("utf-8", "replace")
            sent = (parse_qs(body).get("token") or [""])[0]
        return hmac.compare_digest(sent.encode(), app.state.token.encode())

    @app.get("/", response_class=HTMLResponse)
    def index():
        nonce = secrets.token_urlsafe(16)
        page = render_page(pl.snapshot(), app.state.token, nonce, render_trend(pl.history))
        csp = (f"default-src 'none'; img-src 'self'; connect-src 'self'; style-src 'nonce-{nonce}'; "
               f"script-src 'nonce-{nonce}'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'")
        return HTMLResponse(page, headers={"Content-Security-Policy": csp})

    @app.get("/api/state")
    def state():
        return JSONResponse(pl.snapshot())

    @app.get("/frame.png")
    def frame():
        with pl.lock:
            a = pl.last_analysis
            png = render_frame(pl.last_jpeg, a.mask if a else None, a.box if a else None,
                               a.reading.top_row if a else None)
        return Response(png, media_type="image/png")

    @app.get("/trend.svg")
    def trend():
        with pl.lock:
            svg = render_trend(pl.history)
        return Response(svg, media_type="image/svg+xml")

    @app.get("/diagnosis", response_class=HTMLResponse)
    def diagnosis():
        return HTMLResponse(render_diagnosis(pl.snapshot().get("diagnosis")))

    @app.post("/reset")
    async def reset(request: Request):
        if not await authorized(request):
            return PlainTextResponse("forbidden", status_code=403)
        pl.human_reset()
        return RedirectResponse("/", status_code=303)

    @app.post("/inject/{scenario}")
    async def inject(scenario: str, request: Request):
        if not await authorized(request):
            return PlainTextResponse("forbidden", status_code=403)
        if scenario not in SCENARIOS:
            return PlainTextResponse("unknown scenario", status_code=404)
        pl.inject(scenario)
        return RedirectResponse("/", status_code=303)

    return app


def make_server(app, host: str = "127.0.0.1", port: int = 8765) -> uvicorn.Server:
    """A uvicorn server bound to loopback. Any other host is refused."""
    if host not in ALLOWED_HOSTS:
        raise ValueError(f"dashboard is bound to 127.0.0.1 only, refusing host {host!r}")
    return uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="SightGlass dashboard (SIMULATED, 127.0.0.1 only)")
    add_common_args(p)
    add_diagnoser_args(p)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--pace", type=float, help="wall seconds per tick (default 1/fps; 0 for live cameras)")
    a = p.parse_args(argv)
    server = make_server(None, a.host, a.port)  # refuse a bad host before doing anything else
    source, vision = resolve(a)
    try:
        advice = build_advisor(a.diagnoser, a.run_id, key_file=a.key_file, base_url=a.base_url,
                               readings=vision.records() if isinstance(vision, ReplayVision) else None)
    except ValueError as err:
        p.error(str(err))
    live_camera = bool(a.source and a.source.startswith("http:"))
    pace = a.pace if a.pace is not None else (0.0 if live_camera else 1.0 / a.fps)
    mb_server = mb_client = None
    if a.modbus:
        mb_server = ModbusPlcServer(port=0).start()
        mb_client = ModbusClient("127.0.0.1", mb_server.port)
    pl = Pipeline(vision, a.scenario, a.fault_start, None, a.seed, a.sigma, mb_server, mb_client, advice=advice)
    server.config.app = create_app(pl)
    stop = threading.Event()
    threading.Thread(target=drive, args=(pl, source, live_camera, pace, stop), daemon=True).start()
    print(f"{BANNER}\nHMI on http://127.0.0.1:{a.port}/  (Ctrl-C to stop)")
    try:
        server.run()
    finally:
        stop.set()
        pl.close()
        if mb_client:
            mb_client.close()
        if mb_server:
            mb_server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
