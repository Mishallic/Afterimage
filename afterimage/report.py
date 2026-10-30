"""The match page: a self-contained HTML results page for one session.

Open out/<session>/index.html in a browser. Everything runs client-side, including the
scoreboard card download, so it works straight from disk with no server.
"""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ · Afterimage</title>
<style>
  :root { --bg: #0b0d12; --panel: #151922; --line: #262c38; --text: #eef1f6; --muted: #8a93a6; --gold: #f2c14e; }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text); font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { max-width: 1100px; margin: 0 auto; padding: 32px 16px 64px; }
  header h1 { margin: 0; font-size: 44px; font-weight: 900; letter-spacing: -0.5px; }
  header p { margin: 4px 0 0; color: var(--muted); }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 24px 0; }
  .tile { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; }
  .tile b { display: block; font-size: 24px; font-weight: 800; }
  .tile span { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 2px; }
  .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 16px; margin-top: 16px; }
  .panel h2 { margin: 0 0 4px; font-size: 13px; color: var(--muted); text-transform: uppercase; letter-spacing: 2px; }
  .hint { margin: 0 0 12px; color: var(--muted); }
  .two { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 16px; align-items: start; }
  .proof { position: relative; cursor: crosshair; }
  .moment { position: relative; }
  .proof img { width: 100%; display: block; border-radius: 8px; }
  .dot { position: absolute; width: 18px; height: 18px; margin: -9px 0 0 -9px; border: 3px solid var(--gold); border-radius: 50%; pointer-events: none; display: none; }
  video { width: 100%; display: block; border-radius: 8px; background: #000; }
  .stack { position: relative; }
  .stack video + video { position: absolute; inset: 0; }
  input[type=range] { width: 100%; accent-color: var(--gold); }
  label { color: var(--muted); }
  button { background: var(--gold); color: #111; border: 0; border-radius: 8px; padding: 10px 18px; font-weight: 800; cursor: pointer; }
  footer { margin-top: 32px; color: var(--muted); font-size: 13px; }
  @media (max-width: 720px) { .two { grid-template-columns: 1fr; } header h1 { font-size: 32px; } }
</style>
</head>
<body>
<main>
  <header><h1>__TITLE__</h1><p>Made by hand. The painting stays. Every pixel is copied from a real frame: aligned and colour-matched, never invented.</p></header>
  <section class="grid" id="tiles"></section>

  <section class="panel">
    <h2>Pixel receipt</h2>
    <p class="hint">Tap anywhere on the painting to play the camera's footage of the last time that spot changed, at natural speed, sound on.</p>
    <div class="two">
      <div>
        <div class="proof" id="proof"><img id="painting" src="__FINAL__" alt="The finished painting"><i class="dot" id="dot"></i></div>
        <label><input type="checkbox" id="showmap"> Colour by when it was painted</label>
      </div>
      <div><div class="moment"><video id="moment" src="moments.mp4" playsinline></video><i class="dot" id="camdot"></i></div><p class="hint" id="when"></p></div>
    </div>
  </section>

  <section class="panel">
    <h2>Underneath</h2>
    <p class="hint">Drag to peel the painting back one layer of changes at a time, down to the bare canvas. Every layer is what the camera saw there before.</p>
    <img id="under" src="__FINAL__" alt="The painting, peeled back" style="width: 100%; display: block; border-radius: 8px">
    <input id="peel" type="range" min="0" max="1" value="0">
    <p class="hint" id="underlabel"></p>
  </section>

  <section class="panel">
    <h2>Ghost</h2>
    <p class="hint">Drag to bring the painter back.</p>
    <div class="stack"><video id="clean" src="timelapse.mp4" muted playsinline loop></video><video id="raw" src="raw.mp4" muted playsinline loop></video></div>
    <input id="ghost" type="range" min="0" max="100" value="0">
  </section>

  <section class="panel two">
    <div><h2>Reel</h2><video src="reel.mp4" controls playsinline></video></div>
    <div><h2>Camera vs Afterimage</h2><video src="compare.mp4" controls playsinline></video></div>
  </section>

  <section class="panel">
    <h2>Check it yourself</h2>
    <p class="hint">Every frame above can be re-derived from the raw footage and the cached SAM 3.1 masks.
      No step in the pipeline generates pixels.</p>
    <p id="chain"></p>
    <p class="hint"><code>python scripts/afterimage_verify.py receipt.json --video &lt;raw footage&gt;</code> · <a href="receipt.json" style="color: var(--gold)">receipt.json</a></p>
  </section>

  <section class="panel"><h2>Share</h2><button id="download">Download card</button></section>
  <footer>__FOOTER__</footer>
</main>
<script>
const stats = __STATS__;
const receipt = __RECEIPT__;
const proof = __PROOF__;
document.getElementById("chain").textContent =
  `Footage ${proof.source.sha256.slice(0, 16)}… · ${proof.output.frames} frames · ` +
  `output chain ${proof.output.canvas_sha256.slice(0, 16)}… · ` +
  `${proof.timeline.vetoed} transient changes vetoed · ${proof.generative_steps.length} generative steps`;
const images = { final: "__FINAL__", map: "__MAP__" };
const clock = s => { s = Math.round(s); const h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60), x = s % 60;
  return h ? `${h}:${String(m).padStart(2, "0")}:${String(x).padStart(2, "0")}` : `${m}:${String(x).padStart(2, "0")}`; };
const tiles = [
  ["Session", clock(stats.duration_s)],
  ["Painting", clock(stats.active_s)],
  ["First stroke", stats.first_paint_s == null ? "—" : clock(stats.first_paint_s)],
  ["Most worked", `${stats.longest_fight} · ${Math.round(stats.longest_fight_minutes)} min`],
  ["AI-generated pixels", "0"],
];
document.getElementById("tiles").innerHTML = tiles.map(([k, v]) => `<div class="tile"><b>${v}</b><span>${k}</span></div>`).join("");

const painting = document.getElementById("painting"), dot = document.getElementById("dot");
const moment = document.getElementById("moment"), when = document.getElementById("when"), camdot = document.getElementById("camdot");
let stopAt = null;
document.getElementById("showmap").onchange = e => { painting.src = e.target.checked ? images.map : images.final; };
document.getElementById("proof").onclick = e => {
  const r = painting.getBoundingClientRect();
  const fx = (e.clientX - r.left) / r.width, fy = (e.clientY - r.top) / r.height;
  dot.style.left = `${fx * 100}%`; dot.style.top = `${fy * 100}%`; dot.style.display = "block";
  const gx = Math.min(receipt.width - 1, Math.floor(fx * receipt.width));
  const gy = Math.min(receipt.height - 1, Math.floor(fy * receipt.height));
  const frame = receipt.grid[gy * receipt.width + gx];
  if (frame < 0) { when.textContent = "Bare canvas: never painted here."; moment.pause(); camdot.style.display = "none"; return; }
  // The change was first seen at `seconds`; the paint landed while the hand was still there, so start earlier.
  const seconds = frame / receipt.sample_fps, start = Math.max(seconds - receipt.lead_s, 0);
  const [a, b, c, d, e2, f, g, h, i] = receipt.to_camera, w = g * fx + h * fy + i;
  camdot.style.left = `${(a * fx + b * fy + c) / w * 100}%`; camdot.style.top = `${(d * fx + e2 * fy + f) / w * 100}%`;
  camdot.style.display = "block";
  when.textContent = `Last changed ${clock(seconds)} into the session. Playing the camera from ${clock(start)}.`;
  moment.currentTime = start;
  stopAt = seconds + receipt.tail_s;
  moment.muted = false;
  moment.play();
};
moment.ontimeupdate = () => { if (stopAt !== null && moment.currentTime >= stopAt) { moment.pause(); stopAt = null; } };

const under = __UNDER__;
const underImg = document.getElementById("under"), peel = document.getElementById("peel"), underLabel = document.getElementById("underlabel");
peel.max = under.length;
const setPeel = () => {
  const n = Number(peel.value);
  underImg.src = n === 0 ? images.final : under[n - 1].file;
  underLabel.textContent = n === 0 ? "The finished painting." : n === under.length ? "Bare canvas: every change undone."
    : `${under[n - 1].k} layer${under[n - 1].k === 1 ? "" : "s"} of changes peeled away.`;
};
peel.oninput = setPeel; setPeel();

const clean = document.getElementById("clean"), raw = document.getElementById("raw"), ghost = document.getElementById("ghost");
const setGhost = () => { raw.style.opacity = ghost.value / 100; };
ghost.oninput = setGhost; setGhost();
const sync = () => { if (Math.abs(raw.currentTime - clean.currentTime) > 0.05) raw.currentTime = clean.currentTime; requestAnimationFrame(sync); };
document.querySelector(".stack").onclick = () => { if (clean.paused) { clean.play(); raw.play(); } else { clean.pause(); raw.pause(); } };
requestAnimationFrame(sync);

document.getElementById("download").onclick = () => {
  const c = document.createElement("canvas"); c.width = 1080; c.height = 1350;
  const g = c.getContext("2d");
  g.fillStyle = "#0b0d12"; g.fillRect(0, 0, c.width, c.height);
  g.fillStyle = "#eef1f6"; g.font = "900 64px system-ui, sans-serif"; g.fillText(document.title.split(" · ")[0], 70, 130);
  g.fillStyle = "#8a93a6"; g.font = "600 30px system-ui, sans-serif"; g.fillText("Made by hand. The painting stays.", 72, 180);
  const img = new Image();
  img.onload = () => {
    const box = { x: 70, y: 230, w: 940, h: 760 };
    const s = Math.min(box.w / img.width, box.h / img.height), w = img.width * s, h = img.height * s;
    g.drawImage(img, box.x + (box.w - w) / 2, box.y + (box.h - h) / 2, w, h);
    tiles.slice(0, 4).forEach(([k, v], i) => {
      const x = 70 + (i % 2) * 480, y = 1060 + Math.floor(i / 2) * 105;
      g.fillStyle = "#8a93a6"; g.font = "700 24px system-ui, sans-serif"; g.fillText(k.toUpperCase(), x, y);
      g.fillStyle = "#eef1f6"; g.font = "800 40px system-ui, sans-serif"; g.fillText(String(v), x, y + 46);
    });
    g.fillStyle = "#f2c14e"; g.font = "700 26px system-ui, sans-serif"; g.fillText("No AI-generated pixels · Afterimage", 70, 1310);
    const a = document.createElement("a"); a.download = "afterimage-card.png"; a.href = c.toDataURL("image/png"); a.click();
  };
  img.src = images.final;
};
</script>
</body>
</html>
"""


def _data_uri(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()


def footer(models: dict | None) -> str:
    """Which models did the work. Muse Spark Open's licence asks for its line verbatim."""
    used = [value for value in ((models or {}).get("masks"), (models or {}).get("spark")) if value]
    text = f"Built with {' and '.join(used)}. " if used else ""
    if any("muse-spark-open" in value for value in used):
        text += "Built with Muse Spark Open. "
    return text + "No generative fill."


def write_report(
    out_dir: Path, stats: dict, title: str, final_image: Path, time_map: Path, receipt: dict, proof: dict,
    underneath: list[dict] = (), models: dict | None = None,
) -> Path:
    """`receipt` is the tap-to-replay lookup grid; `proof` is receipt.json; `underneath` lists the peeled stills."""
    page = (
        TEMPLATE.replace("__TITLE__", html.escape(title))
        .replace("__FOOTER__", html.escape(footer(models)))
        .replace("__STATS__", json.dumps(stats))
        .replace("__RECEIPT__", json.dumps(receipt))
        .replace("__PROOF__", json.dumps(proof))
        .replace("__UNDER__", json.dumps(list(underneath)))
        .replace("__FINAL__", _data_uri(final_image))
        .replace("__MAP__", _data_uri(time_map))
    )
    path = out_dir / "index.html"
    path.write_text(page, encoding="utf-8")
    return path
