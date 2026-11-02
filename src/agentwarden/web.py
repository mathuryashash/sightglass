"""HMI dashboard: FastAPI on 127.0.0.1 only. No CDN, no external fonts or JS; server-rendered HTML, PNG and SVG.

SIMULATED CONTROLLER ON LOCALHOST - NO REAL SYSTEM. This module only displays state. The POST routes (human reset,
recalibrate, demo fault buttons) call Pipeline methods and need a per-process random token plus an allowed Host
header. Nothing here can enable the pump: a reset is only ever the pipeline's human_reset(), which the safety logic
may refuse.

Hardening: every response carries nosniff, Cross-Origin-Resource-Policy: same-origin and a CSP; cross-site requests
(Sec-Fetch-Site, Origin, Referer) are refused; POST bodies are capped at 4 KB; the frame and the trend are rendered
OUTSIDE the pipeline's tick lock from a copied snapshot, and at most once per tick.

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
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response
from PIL import Image, ImageDraw

from agentwarden.advisor import add_diagnoser_args, build_advisor
from agentwarden.fault_injector import SCENARIOS
from agentwarden.modbus_io import ModbusClient, ModbusPlcServer
from agentwarden.pipeline import BANNER, Pipeline, add_common_args, drive, resolve
from agentwarden.vision import ReplayVision

ALLOWED_HOSTS = ("127.0.0.1", "localhost")
MAX_BODY = 4096
TREND_WINDOW_S = 300.0
MIN_SPAN_S = 60.0  # a short run still fills the chart
SVG_W, SVG_H, PAD_L, PAD_R, PAD_B, PAD_T = 1000, 340, 46, 14, 34, 14
PLC_COLOR, CAM_COLOR = "#56B4E9", "#F0E442"  # colour-blind-safe pair on a dark ground
CHART_BG, GRID_COLOR, TRIP_COLOR, FAULT_COLOR = "#10151c", "#2c3644", "#ff4d4d", "#d9a3ff"
TINT = np.array([255, 140, 0])
CSP_BASE = "default-src 'none'; frame-ancestors 'none'; base-uri 'none'"


def _host_ok(host_header: str | None) -> bool:
    host = (host_header or "").lower()
    if host.startswith("["):  # IPv6 literals are never allowed
        return False
    return host.rsplit(":", 1)[0] in ALLOWED_HOSTS if ":" in host else host in ALLOWED_HOSTS


def _same_origin_only(request: Request) -> bool:
    """False for anything a browser marks as coming from another site, or that names a foreign Origin/Referer."""
    if request.headers.get("sec-fetch-site") not in (None, "same-origin", "none"):
        return False
    origin = request.headers.get("origin")
    if origin is not None and urlsplit(origin).hostname not in ALLOWED_HOSTS:
        return False
    referer = request.headers.get("referer")
    return not referer or urlsplit(referer).hostname in ALLOWED_HOSTS


def _tint(region):
    return (0.55 * region + 0.45 * TINT).astype(np.uint8)


def render_frame(jpeg: bytes | None, mask, box, top_row) -> bytes:
    """Camera frame as PNG: liquid tinted, container box, red surface line. Without a mask but with a recorded box and
    surface row (replay), the liquid is the box below the surface row."""
    if not jpeg:
        im = Image.new("RGB", (320, 240), (60, 60, 60))
        ImageDraw.Draw(im).text((110, 112), "no frame yet", fill=(220, 220, 220))
    else:
        im = Image.open(io.BytesIO(jpeg)).convert("RGB")
        arr = np.asarray(im).copy()
        if mask is not None and mask.shape == (im.height, im.width):
            arr[mask] = _tint(arr[mask])
        elif mask is None and box is not None and top_row is not None:
            x0, y0, x1, y1 = (max(0, int(round(v))) for v in box)
            r0 = max(int(top_row), y0)
            arr[r0:y1, x0:x1] = _tint(arr[r0:y1, x0:x1])
        im = Image.fromarray(arr)
        d = ImageDraw.Draw(im)
        if box is not None:
            d.rectangle(box, outline=(0, 200, 255), width=3)
            if top_row is not None:
                d.line([(box[0], top_row), (box[2], top_row)], fill=(255, 0, 0), width=3)
    out = io.BytesIO()
    im.save(out, "PNG")
    return out.getvalue()


def overlay_png(data: bytes) -> bytes:
    """A pre-drawn overlay picture (JPEG) as PNG, pixel for pixel."""
    out = io.BytesIO()
    Image.open(io.BytesIO(data)).convert("RGB").save(out, "PNG")
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


def render_trend(history, trip_t: float | None = None, fault_t: float | None = None) -> str:
    """Dark inline SVG of PLC and camera level (%): 3 px lines, a red vertical line at the trip, a dotted line where
    the fault was injected. The window is the last 5 minutes, but never narrower than 60 s."""
    hist = list(history)
    t_last = hist[-1][0] if hist else 0.0
    t0 = max(hist[0][0], t_last - TREND_WINDOW_S) if hist else 0.0
    span = max(MIN_SPAN_S, t_last - t0)
    pw, ph = SVG_W - PAD_L - PAD_R, SVG_H - PAD_B - PAD_T

    def x_of(t):
        return round(PAD_L + (t - t0) / span * pw, 1)

    def y_of(v):
        return round(PAD_T + (100 - min(100.0, max(0.0, v))) / 100 * ph, 1)

    def line(idx, color, label):
        pts = [(x_of(r[0]), None if r[idx] is None else y_of(r[idx])) for r in hist if r[0] >= t0]
        return "".join(
            f'<polyline fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round" data-series="{label}" '
            f'points="{" ".join(f"{x},{y}" for x, y in run)}"/>' for run in (list(_runs(pts)) or [[]]))

    grid = "".join(
        f'<line x1="{PAD_L}" x2="{SVG_W - PAD_R}" y1="{y_of(v)}" y2="{y_of(v)}" stroke="{GRID_COLOR}" stroke-width="1"'
        f'{" stroke-dasharray=" + chr(34) + "6 5" + chr(34) if v == 95 else ""}/>'
        f'<text x="{PAD_L - 6}" y="{y_of(v) + 5}" font-size="15" fill="#8d99aa" text-anchor="end">{v}</text>'
        for v in (0, 25, 50, 75, 95, 100))
    ticks = "".join(
        f'<text x="{x_of(t0 + span * i / 4)}" y="{SVG_H - 12}" font-size="15" fill="#8d99aa" '
        f'text-anchor="middle">t={round(t0 + span * i / 4)}s</text>' for i in range(5))

    def marker(t, kind, color, label, dotted):
        if t is None or not t0 <= t <= t0 + span:
            return ""
        x = x_of(t)
        anchor, dx = ("end", -8) if x > SVG_W * 0.7 else ("start", 8)
        dash = ' stroke-dasharray="2 7" stroke-linecap="round"' if dotted else ""
        return (f'<line data-marker="{kind}" x1="{x}" x2="{x}" y1="{PAD_T}" y2="{SVG_H - PAD_B}" stroke="{color}" '
                f'stroke-width="3"{dash}/>'
                f'<text x="{x + dx}" y="{PAD_T + (40 if dotted else 20)}" font-size="20" font-weight="700" '
                f'fill="{color}" text-anchor="{anchor}">{label}</text>')

    marks = (marker(fault_t, "fault", FAULT_COLOR, "fault injected", True)
             + marker(trip_t, "trip", TRIP_COLOR, "TRIPPED" + ("" if trip_t is None else f" t={round(trip_t)}s"),
                      False))
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SVG_W} {SVG_H}" role="img" '
        f'aria-label="PLC and camera level over time">'
        f'<title>PLC and camera level (%)</title><rect width="{SVG_W}" height="{SVG_H}" fill="{CHART_BG}" rx="10"/>'
        f'{grid}{ticks}{line(1, PLC_COLOR, "plc")}{line(2, CAM_COLOR, "camera")}{marks}'
        f'<text x="{PAD_L + 12}" y="{SVG_H - 44}" font-size="20" font-weight="700" fill="{PLC_COLOR}">'
        f'CONTROLLER (PLC)</text>'
        f'<text x="{PAD_L + 210}" y="{SVG_H - 44}" font-size="20" font-weight="700" fill="{CAM_COLOR}">CAMERA</text>'
        f'<text x="{SVG_W - PAD_R - 6}" y="{y_of(95) - 6}" font-size="15" fill="#8d99aa" text-anchor="end">'
        f'95 = high-high</text></svg>')


CSS = """
:root{--bg:#0b0e13;--card:#151a22;--line:#2a3340;--text:#eef1f5;--mute:#8d99aa;--plc:#56B4E9;--cam:#F0E442;--green:#2fd27a;--amber:#ffb020;--red:#ff4d4d;--blue:#4f6f96}
html,body{height:100vh;margin:0;overflow:hidden}
body{display:grid;grid-template-rows:auto auto auto minmax(0,1fr);background:var(--bg);color:var(--text);font:20px/1.25 system-ui,"Segoe UI",Roboto,sans-serif}
.banner{background:#b00020;color:#fff;font-weight:800;font-size:22px;text-align:center;padding:7px;letter-spacing:.06em}
.big{font-size:150px;font-weight:800;line-height:1;font-variant-numeric:tabular-nums}
.hero{display:grid;grid-template-columns:1fr 500px 1fr;gap:16px;padding:12px 16px 0}
.hcard{background:var(--card);border-radius:14px;padding:6px 24px 10px;text-align:center}
.lbl{font-size:30px;font-weight:700;letter-spacing:.08em;color:var(--mute)}
.plc .big{color:var(--plc)}.cam .big{color:var(--cam)}
.hgap{font-size:72px;font-weight:800;display:flex;align-items:center;justify-content:center;border-radius:14px;border:4px solid var(--line);background:var(--card);white-space:nowrap}
.hgap.ok{color:var(--green);border-color:#1f5e3d}.hgap.amber{color:#1f1500;background:var(--amber);border-color:var(--amber)}.hgap.red{color:#fff;background:var(--red);border-color:var(--red)}
.strip{display:grid;grid-template-columns:auto 1fr auto;gap:24px;align-items:center;padding:12px 16px}
.pill{font-size:52px;font-weight:800;padding:4px 36px;border-radius:60px;white-space:nowrap}
.pill.normal{background:var(--green);color:#04210f}.pill.watching{background:var(--amber);color:#2a1d00}.pill.tripped{background:var(--red);color:#fff}.pill.blocked{background:var(--blue);color:#fff}
.pump{display:flex;align-items:center;gap:14px}.pump svg{width:64px;height:64px}
.pumptxt{font-size:44px;font-weight:800;line-height:1}.permittxt{font-size:22px;color:var(--mute)}
.pc{stroke-width:4}.pt{stroke-width:2}
.pump.on .pc{fill:#12452c;stroke:var(--green)}.pump.on .pt{fill:var(--green);stroke:var(--green)}.pump.on .pumptxt{color:var(--green)}
.pump.off .pc{fill:#3a1518;stroke:var(--red)}.pump.off .pt{fill:var(--mute);stroke:var(--mute)}.pump.off .pumptxt{color:var(--red)}
.meta{text-align:right;font-size:20px;color:var(--mute)}
main{display:grid;grid-template-columns:360px 460px minmax(0,1fr);gap:16px;padding:0 16px 16px;min-height:0}
section{background:var(--card);border-radius:14px;padding:10px 16px;min-height:0;overflow:hidden}
h2{margin:8px 0 6px;font-size:16px;text-transform:uppercase;letter-spacing:.06em;color:var(--mute)}
dl{display:grid;grid-template-columns:auto 1fr;gap:3px 12px;margin:0;font-size:18px}dt{color:var(--mute)}dd{margin:0;font-weight:600}
.bad{color:var(--red)}.good{color:var(--green)}.warn{background:#4a3b00;color:#ffd95a;padding:6px 10px;border-radius:8px;margin-top:8px;font-size:18px;font-weight:700}
.hidden{display:none}ul{margin:2px 0;padding-left:20px;font-size:18px}
.frame{display:flex;flex-direction:column;padding:10px}
.frame img{flex:1;min-height:0;width:100%;object-fit:contain;border-radius:8px}
.rightcol{display:grid;grid-template-rows:auto minmax(0,1fr);gap:16px;min-height:0}
.chart{padding:0;background:none}svg{display:block;width:100%;height:auto}
.lower{display:grid;grid-template-columns:1.15fr 1fr;gap:16px;min-height:0}
ol{list-style:none;margin:0;padding:0;font-size:22px}li{padding:2px 0}.tt{display:inline-block;min-width:92px;color:var(--cam);font-weight:700}
.advisory{background:#11151b;border:2px dashed var(--line);color:var(--mute);font-size:16px}.advisory h2{margin-top:2px}
button{font:inherit;font-size:16px;padding:6px 10px;margin:2px 0;border-radius:8px;border:1px solid #556;background:#2b3340;color:var(--text);cursor:pointer;width:100%;text-align:left}
form{display:block}.note{color:var(--mute);font-size:14px}
"""

JS = """
const $=id=>document.getElementById(id);
const fmt=v=>v===null||v===undefined?'-':Number(v).toFixed(1)+' %';
const relayText=r=>!r?'-':(r.pump_enabled?'CLOSED - pump allowed':'OPEN - pump cut')+' ('+r.driver+(r.relay_fault?', FAULT':'')+')';
let lastTick=-1;
async function poll(){
 try{
  const s=await (await fetch('/api/state',{cache:'no-store'})).json(),v=s.view;
  $('clock').textContent=s.clock;
  $('loop').textContent='loop last tick '+s.loop_age_s.toFixed(1)+' s ago';
  $('loop').className=s.alarms.includes('LOOP_STALLED')?'bad':'';
  $('h-plc').textContent=v.plc_txt;$('h-cam').textContent=v.cam_txt;
  $('h-gap').textContent=v.gap_txt;$('h-gap').className='hgap '+v.gap_class;
  $('pill').textContent=v.pill_text;$('pill').className='pill '+v.pill;
  $('pump').className='pump '+v.pump_class;$('pump-txt').textContent=v.pump_txt;$('permit-txt').textContent=v.permit_txt;
  $('pump-state').textContent=s.pump_on?'RUNNING':'STOPPED';$('permit').textContent=s.permit?'ON':'OFF';
  $('plc').textContent=fmt(s.plc_level);
  $('cam').textContent=fmt(s.camera.level_pct);
  $('conf').textContent=Math.round(s.camera.confidence*100)+' %';
  $('status').textContent=s.camera.status;
  $('trip').textContent=s.tripped?('TRIPPED ('+s.trip_reason+')'):'not tripped';
  $('trip').className=s.tripped?'bad':'good';
  $('scenario').textContent=s.scenario;
  $('relay').textContent=relayText(s.relay);
  const ul=$('alarms');ul.textContent='';
  for(const a of s.alarms){const li=document.createElement('li');li.textContent=a;ul.appendChild(li);}
  $('blind').textContent=s.camera_notice;
  $('blind').className=s.camera_notice?'warn':'hidden';
  const ol=$('timeline');ol.textContent='';
  for(const e of s.events){const li=document.createElement('li'),tt=document.createElement('span');tt.className='tt';tt.textContent='t='+Math.round(e.t)+'s';li.appendChild(tt);li.appendChild(document.createTextNode(e.text));ol.appendChild(li);}
  if(s.n_ticks!==lastTick){lastTick=s.n_ticks;$('frame').src='/frame.png?'+s.n_ticks;$('trend').innerHTML=await (await fetch('/trend.svg',{cache:'no-store'})).text();}
  $('diag').innerHTML=await (await fetch('/diagnosis',{cache:'no-store'})).text();
 }catch(e){$('clock').textContent='disconnected';}
}
setInterval(poll,500);
"""

PUMP_ICON = ('<svg viewBox="0 0 64 64" aria-hidden="true"><circle class="pc" cx="32" cy="32" r="26"/>'
             '<path class="pt" d="M24 20 L46 32 L24 44 Z"/></svg>')


def relay_text(r: dict | None) -> str:
    if not r:
        return "-"
    state = "CLOSED - pump allowed" if r.get("pump_enabled") else "OPEN - pump cut"
    return f'{state} ({r.get("driver", "?")}{", FAULT" if r.get("relay_fault") else ""})'


def _pct(v) -> str:
    return "--" if v is None else f"{round(v)}%"


def hmi_view(s: dict) -> dict:
    """What the hero row, the status pill and the pump block say, derived from one snapshot. The only place the
    NORMAL / WATCHING / TRIPPED / CAMERA BLOCKED decision is made (the page and /api/state both use it)."""
    cam = s.get("camera") or {}
    plc, lvl = s.get("plc_level"), cam.get("level_pct")
    t_pts = (s.get("limits") or {}).get("T_pts", 15.0)
    tripped = bool(s.get("tripped"))
    blocked = bool(s.get("camera_blind_notice")) or bool(cam.get("camera_moved"))
    gap = abs(plc - lvl) if plc is not None and lvl is not None and (tripped or not blocked) else None
    amber = gap is not None and gap > t_pts / 2
    if tripped:
        pill, text = "tripped", "TRIPPED - PUMP LOCKED OFF"
    elif blocked:
        pill, text = "blocked", "CAMERA BLOCKED - NO TRIP"
    elif amber or s.get("alarms"):
        pill, text = "watching", "WATCHING"
    else:
        pill, text = "normal", "NORMAL"
    pump = bool(s.get("pump_on"))
    return {
        "pill": pill, "pill_text": text, "plc_txt": _pct(plc), "cam_txt": _pct(lvl),
        "gap_txt": "GAP -" if gap is None else f"GAP {round(gap)} pts",
        "gap_class": "red" if tripped else "amber" if amber else "ok",
        "pump_txt": "PUMP ON" if pump else "PUMP OFF", "pump_class": "on" if pump else "off",
        "permit_txt": "PERMIT ON" if s.get("permit") else "PERMIT OFF",
    }


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
    return (f'<h2>Diagnosis (advisory - Muse Spark)</h2>'
            f'<p class="note">Secondary information. Muse Spark cannot control the pump.</p>{body}'
            f'<p class="note">advisory only - does not control the pump</p>')


def render_page(s: dict, token: str, nonce: str, trend_svg: str) -> str:
    """Every dynamic string goes through html.escape."""
    e = html.escape
    cam, v = s["camera"], hmi_view(s)

    def pct(x):
        return "-" if x is None else f"{x:.1f} %"

    alarms = "".join(f"<li>{e(str(a))}</li>" for a in s["alarms"])
    trip = f'TRIPPED ({e(str(s["trip_reason"]))})' if s["tripped"] else "not tripped"
    notice = e(s.get("camera_notice", ""))
    events = "".join(f'<li><span class="tt">t={round(ev["t"])}s</span>{e(str(ev["text"]))}</li>'
                     for ev in s.get("events", []))
    inject = "".join(
        f'<form method="post" action="/inject/{e(sc)}"><input type="hidden" name="token" value="{e(token)}">'
        f'<button type="submit">inject: {e(sc)}</button></form>' for sc in SCENARIOS)
    age = s.get("loop_age_s", 0.0)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>SightGlass HMI (simulated)</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style nonce="{nonce}">{CSS}</style></head><body>
<div class="banner" role="alert">{e(BANNER)}</div>
<div class="hero">
<div class="hcard plc"><div class="lbl">CONTROLLER SAYS</div>
<div class="big" id="h-plc">{e(v["plc_txt"])}</div></div>
<div id="h-gap" class="hgap {v["gap_class"]}">{e(v["gap_txt"])}</div>
<div class="hcard cam"><div class="lbl">CAMERA SEES</div>
<div class="big" id="h-cam">{e(v["cam_txt"])}</div></div>
</div>
<div class="strip">
<div id="pill" class="pill {v["pill"]}">{e(v["pill_text"])}</div>
<div id="pump" class="pump {v["pump_class"]}">{PUMP_ICON}<div><div class="pumptxt" id="pump-txt">{e(v["pump_txt"])}</div>
<div class="permittxt" id="permit-txt">{e(v["permit_txt"])}</div></div></div>
<div class="meta"><div id="loop" class="{"bad" if "LOOP_STALLED" in s["alarms"] else ""}">loop last tick {age:.1f} s ago</div>
<div id="clock">{e(s["clock"])}</div></div>
</div>
<main>
<section><h2>Process</h2><dl>
<dt>PLC level</dt><dd id="plc">{pct(s["plc_level"])}</dd>
<dt>Camera level</dt><dd id="cam">{pct(cam["level_pct"])}</dd>
<dt>Camera confidence</dt><dd id="conf">{round(cam["confidence"] * 100)} %</dd>
<dt>Camera status</dt><dd id="status">{e(cam["status"])}</dd>
<dt>Pump</dt><dd id="pump-state">{"RUNNING" if s["pump_on"] else "STOPPED"}</dd>
<dt>Permit</dt><dd id="permit">{"ON" if s["permit"] else "OFF"}</dd>
<dt>Relay</dt><dd id="relay">{e(relay_text(s.get("relay")))}</dd>
<dt>Trip</dt><dd id="trip" class="{"bad" if s["tripped"] else "good"}">{trip}</dd>
<dt>Fault scenario</dt><dd id="scenario">{e(s["scenario"])}</dd></dl>
<div id="blind" class="{"warn" if notice else "hidden"}">{notice}</div>
<h2>Alarms</h2><ul id="alarms">{alarms}</ul>
<h2>Controls</h2>
<form method="post" action="/reset"><input type="hidden" name="token" value="{e(token)}">
<button type="submit">Human reset (refused while the condition holds)</button></form>
<form method="post" action="/recalibrate"><input type="hidden" name="token" value="{e(token)}">
<button type="submit">Camera was moved: recalibrate</button></form>
{inject}</section>
<section class="frame"><h2>Camera frame (mask + surface line)</h2>
<img id="frame" src="/frame.png" alt="latest camera frame with liquid mask overlay"></section>
<div class="rightcol">
<section class="chart" id="trend">{trend_svg}</section>
<div class="lower">
<section><h2>Event timeline</h2><ol id="timeline">{events}</ol></section>
<section id="diag" class="advisory">{render_diagnosis(s.get("diagnosis"))}</section>
</div></div>
</main><script nonce="{nonce}">{JS}</script></body></html>"""


class _RenderCache:
    """The last render per name, keyed by the pipeline's tick number: a flood of GETs costs one render per tick.
    Its lock is not the pipeline's, so a slow render can never hold up a tick."""

    def __init__(self):
        self._lock, self._last = threading.Lock(), {}

    def get(self, name, key, build):
        with self._lock:
            hit = self._last.get(name)
            if hit is None or hit[0] != key:
                hit = self._last[name] = (key, build())
            return hit[1]


def create_app(pl: Pipeline) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.state.token = secrets.token_urlsafe(32)
    cache = _RenderCache()

    @app.middleware("http")
    async def guard(request: Request, call_next):
        resp = None
        if not _host_ok(request.headers.get("host")):  # also blocks DNS rebinding
            resp = PlainTextResponse("bad host", status_code=400)
        elif not _same_origin_only(request):  # a page on another site must not drive or read this one
            resp = PlainTextResponse("cross-site request refused", status_code=403)
        elif request.method == "POST" and int(request.headers.get("content-length") or 0) > MAX_BODY:
            resp = PlainTextResponse("body too large", status_code=413)
        if resp is None:
            resp = await call_next(request)
        resp.headers["Cache-Control"] = "no-store"
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        if "content-security-policy" not in resp.headers:
            resp.headers["Content-Security-Policy"] = CSP_BASE
        return resp

    async def authorized(request: Request) -> bool:
        sent = request.headers.get("x-token")
        if sent is None:
            body = bytearray()
            async for chunk in request.stream():  # capped: a chunked body has no Content-Length to check
                body += chunk
                if len(body) > MAX_BODY:
                    raise HTTPException(status_code=413, detail="body too large")
            sent = (parse_qs(body.decode("utf-8", "replace")).get("token") or [""])[0]
        return hmac.compare_digest(sent.encode(), app.state.token.encode())

    def trend_svg(inp) -> str:
        return cache.get("trend", inp["key"], lambda: render_trend(inp["history"], inp["trip_t"], inp["fault_t"]))

    @app.get("/", response_class=HTMLResponse)
    def index():
        nonce = secrets.token_urlsafe(16)
        page = render_page(pl.snapshot(), app.state.token, nonce, trend_svg(pl.render_inputs()))
        csp = (f"default-src 'none'; img-src 'self'; connect-src 'self'; style-src 'nonce-{nonce}'; "
               f"script-src 'nonce-{nonce}'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'")
        return HTMLResponse(page, headers={"Content-Security-Policy": csp})

    @app.get("/api/state")
    def state():
        snap = pl.snapshot()
        return JSONResponse({**snap, "view": hmi_view(snap)})

    @app.get("/frame.png")
    def frame():
        inp = pl.render_inputs()  # copied under the pipeline lock; drawn outside it, once per tick

        def build():
            if inp["overlay"]:
                return overlay_png(inp["overlay"])
            a = inp["analysis"]
            return render_frame(inp["jpeg"], a.mask if a else None, a.box if a else None,
                                a.reading.top_row if a else None)

        return Response(cache.get("frame", inp["key"], build), media_type="image/png")

    @app.get("/trend.svg")
    def trend():
        return Response(trend_svg(pl.render_inputs()), media_type="image/svg+xml")

    @app.get("/diagnosis", response_class=HTMLResponse)
    def diagnosis():
        return HTMLResponse(render_diagnosis(pl.snapshot().get("diagnosis")))

    @app.post("/reset")
    async def reset(request: Request):
        if not await authorized(request):
            return PlainTextResponse("forbidden", status_code=403)
        pl.human_reset()
        return RedirectResponse("/", status_code=303)

    @app.post("/recalibrate")
    async def recalibrate(request: Request):
        if not await authorized(request):
            return PlainTextResponse("forbidden", status_code=403)
        pl.recalibrate()
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
    if a.loop and (a.modbus or (a.source and a.source.startswith("http:"))):
        p.error("--loop is for in-process replays (no --modbus, no live camera)")
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
    pl.overlay_for = getattr(source, "overlay", None)  # real replay frames come with their pre-drawn overlays
    server.config.app = create_app(pl)
    stop = threading.Event()
    threading.Thread(target=drive, args=(pl, source, live_camera, pace, stop),
                     kwargs={"loop": a.loop, "start_at": a.start_at}, daemon=True).start()
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
