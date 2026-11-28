"""Interactive live web demo server for DeadLock (zero extra dependencies).

Streams 4-panel DOOM gameplay (First-Person RGB + Action HUD, 360° Ray Radar,
EgoSpatialMemory heatmap, and Top-Down Trajectory) over MJPEG + JSON API.

Usage:
    PYTHONPATH=. .venv/bin/python -m training.web_demo --port 7860
"""

from __future__ import annotations

import argparse
import io
import json
import os
import tempfile
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import vizdoom as vd
from doom_env import make_env

from submission.agent import DoomAgent
from training.env_wrapper import FRONT_RAY_IDX, RAY_DOOR_ID, RAY_LOCKED_ID
from training.visualize import render_frame
from training.wad_gen import write_procedural_wad

ROOT = Path(__file__).resolve().parents[1]
FREEDOOM2_IWAD = os.path.join(os.path.dirname(vd.__file__), "freedoom2.wad")
CELL_SIZE = 96.0


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>DeadLock — Live Autonomous DOOM Agent</title>
<style>
  :root {
    --bg: #0b0e14;
    --panel: #121722;
    --border: #232b3d;
    --text: #e2e8f0;
    --muted: #94a3b8;
    --accent: #10b981;
    --cyan: #06b6d4;
    --amber: #f59e0b;
    --red: #ef4444;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", sans-serif;
    padding: 18px 22px;
    max-width: 1240px;
    margin: 0 auto;
  }
  header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
    margin-bottom: 14px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
  }
  h1 { font-size: 1.35rem; font-weight: 800; letter-spacing: -0.02em; }
  h1 span { color: var(--accent); }
  .subtitle { font-size: 0.85rem; color: var(--muted); margin-top: 2px; }
  .pill-row { display: flex; gap: 8px; flex-wrap: wrap; }
  .pill {
    font-size: 0.75rem;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 999px;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.4);
    color: #a7f3d0;
  }
  .pill.cyan {
    background: rgba(6, 182, 212, 0.12);
    border-color: rgba(6, 182, 212, 0.4);
    color: #a5f3fc;
  }
  .layout {
    display: grid;
    grid-template-columns: 1fr 290px;
    gap: 16px;
    align-items: start;
  }
  @media (max-width: 960px) {
    .layout { grid-template-columns: 1fr; }
  }
  .stage {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px;
  }
  .stage img {
    width: 100%;
    height: auto;
    display: block;
    border-radius: 6px;
    border: 1px solid #2b354b;
    background: #0e1016;
  }
  .stats-bar {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 8px;
    margin-top: 10px;
  }
  .stat {
    background: #0d111a;
    border: 1px solid var(--border);
    border-radius: 7px;
    padding: 8px 10px;
  }
  .stat .k { font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }
  .stat .v { font-size: 1.05rem; font-weight: 700; margin-top: 2px; font-variant-numeric: tabular-nums; }
  .status-badge {
    display: inline-block;
    padding: 3px 9px;
    border-radius: 5px;
    font-size: 0.76rem;
    font-weight: 700;
  }
  .st-run { background: rgba(6,182,212,0.18); color: #67e8f9; }
  .st-exit { background: rgba(16,185,129,0.22); color: #6ee7b7; }
  .st-dead { background: rgba(239,68,68,0.22); color: #fca5a5; }
  .st-cap { background: rgba(245,158,11,0.22); color: #fcd34d; }

  .controls {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .ctrl-group label {
    display: block;
    font-size: 0.76rem;
    color: var(--muted);
    font-weight: 600;
    margin-bottom: 5px;
  }
  select, input[type="number"] {
    width: 100%;
    background: #0d111a;
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 7px 9px;
    font-size: 0.86rem;
  }
  input[type="range"] { width: 100%; accent-color: var(--accent); }
  .seed-row { display: grid; grid-template-columns: 1fr auto; gap: 6px; }
  .btn-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  button {
    background: #1e293b;
    color: var(--text);
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 8px 10px;
    font-size: 0.82rem;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.15s;
  }
  button:hover { background: #293548; }
  button.primary {
    background: var(--accent);
    color: #042f2e;
    border-color: #059669;
    font-weight: 700;
  }
  button.primary:hover { background: #34d399; }
  .info-box {
    font-size: 0.76rem;
    color: var(--muted);
    line-height: 1.45;
    background: #0d111a;
    border: 1px solid var(--border);
    border-radius: 7px;
    padding: 10px;
  }
  .info-box strong { color: var(--text); }
</style>
</head>
<body>
<header>
  <div>
    <h1><span>DeadLock</span> — Live Autonomous DOOM Agent</h1>
    <div class="subtitle">Zero-Shot PyTorch Policy · Hybrid RGB+Depth + Circular Rays + Persistent Frontier Memory</div>
  </div>
  <div class="pill-row">
    <span class="pill">940,720 Params · 3.6 MB</span>
    <span class="pill cyan" id="lat-pill">Latency: -- ms</span>
    <span class="pill" id="state-pill">RUNNING</span>
  </div>
</header>

<div class="layout">
  <div class="stage">
    <img id="stream" src="/stream.mjpg" alt="DeadLock Live Stream" />
    <div class="stats-bar">
      <div class="stat"><div class="k">Step</div><div class="v" id="st-step">0</div></div>
      <div class="stat"><div class="k">Health</div><div class="v" id="st-hp">100</div></div>
      <div class="stat"><div class="k">Kills</div><div class="v" id="st-kills">0</div></div>
      <div class="stat"><div class="k">Doors</div><div class="v" id="st-doors">0</div></div>
      <div class="stat"><div class="k">Keycards</div><div class="v" id="st-keys">0</div></div>
      <div class="stat"><div class="k">Cells</div><div class="v" id="st-cells">1</div></div>
    </div>
  </div>

  <aside class="controls">
    <div class="ctrl-group">
      <label for="suite">Map / Difficulty Tier</label>
      <select id="suite">
        <option value="tier3" selected>Tier 3 — Keycard + Locked Door</option>
        <option value="tier4">Tier 4 — Full Gauntlet</option>
        <option value="tier2">Tier 2 — Multi-Room Exploration</option>
        <option value="tier1">Tier 1 — Combat &amp; Survival</option>
        <option value="two_rooms">Starter — two_rooms.wad</option>
        <option value="fd_map01">Freedoom2 — MAP01</option>
        <option value="fd_map03">Freedoom2 — MAP03</option>
        <option value="fd_map06">Freedoom2 — MAP06</option>
      </select>
    </div>

    <div class="ctrl-group">
      <label for="seed">Map Seed (Unseen Procedural)</label>
      <div class="seed-row">
        <input id="seed" type="number" value="99305" min="0" max="9999999" />
        <button onclick="randomSeed()">Random</button>
      </div>
    </div>

    <div class="ctrl-group">
      <label>Playback Speed: <span id="fps-val">14</span> FPS</label>
      <input id="fps" type="range" min="4" max="30" value="14" oninput="document.getElementById('fps-val').textContent=this.value; sendCtrl({fps: Number(this.value)})" />
    </div>

    <div class="ctrl-group">
      <label>Sampling Temp: <span id="temp-val">0.40</span></label>
      <input id="temp" type="range" min="0" max="70" value="40" oninput="const t=(this.value/100).toFixed(2); document.getElementById('temp-val').textContent=t; sendCtrl({temp: Number(t)})" />
    </div>

    <div class="btn-grid">
      <button class="primary" onclick="restartEpisode()">Restart Map</button>
      <button id="pause-btn" onclick="togglePause()">Pause</button>
    </div>
    <button onclick="sendCtrl({action:'step'})">Single Step (+1 tick)</button>

    <div class="info-box">
      <div><strong>4-Panel Live View:</strong></div>
      <div>• <strong>Top-Left:</strong> First-person DOOM + action pills</div>
      <div>• <strong>Mid-Top:</strong> 360° rays &amp; visible entities</div>
      <div>• <strong>Mid-Bot:</strong> Learned <code>16×25×25</code> spatial map</div>
      <div>• <strong>Right:</strong> Ground-truth trajectory &amp; doors</div>
      <div style="margin-top:6px;">Door shots this episode: <strong id="st-dshots">0</strong></div>
    </div>
  </aside>
</div>

<script>
let isPaused = false;
async function sendCtrl(payload) {
  await fetch('/api/control', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload),
  });
}
function restartEpisode() {
  isPaused = false;
  document.getElementById('pause-btn').textContent = 'Pause';
  sendCtrl({
    action: 'restart',
    suite: document.getElementById('suite').value,
    seed: Number(document.getElementById('seed').value) || 0,
    fps: Number(document.getElementById('fps').value),
    temp: Number(document.getElementById('temp').value) / 100.0,
  });
}
function randomSeed() {
  document.getElementById('seed').value = 80000 + Math.floor(Math.random() * 19000);
  restartEpisode();
}
function togglePause() {
  isPaused = !isPaused;
  document.getElementById('pause-btn').textContent = isPaused ? 'Resume' : 'Pause';
  sendCtrl({action: isPaused ? 'pause' : 'resume'});
}
async function pollState() {
  try {
    const r = await fetch('/api/state');
    const s = await r.json();
    document.getElementById('st-step').textContent = s.step;
    document.getElementById('st-hp').textContent = Math.round(s.health);
    document.getElementById('st-kills').textContent = s.kills;
    document.getElementById('st-doors').textContent = s.doors;
    document.getElementById('st-keys').textContent = s.keys;
    document.getElementById('st-cells').textContent = s.cells;
    document.getElementById('st-dshots').textContent = s.door_shots;
    document.getElementById('lat-pill').textContent = 'p50: ' + s.p50_ms.toFixed(2) + ' ms';
    const sp = document.getElementById('state-pill');
    sp.textContent = s.status;
    sp.className = 'pill ' + (s.status === 'EXIT REACHED' ? '' : (s.status === 'DIED' ? 'cyan' : 'cyan'));
  } catch (e) {}
}
setInterval(pollState, 220);
</script>
</body>
</html>
"""


class LiveEpisodeRunner:
    """Thread-safe ViZDoom + DoomAgent runner producing MJPEG frames and live telemetry."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="deadlock_demo_"))
        dummy = make_env(ROOT / "maps" / "two_rooms.wad", "MAP01", episode_ticks=700, render_buffers=False)
        self.agent = DoomAgent(dummy.observation_space, dummy.action_space, device=device)
        dummy.close()

        self._lock = threading.Lock()
        self._frame_cond = threading.Condition()
        self._latest_jpeg: bytes = b""
        self._frame_seq: int = 0
        self._stop = False

        self.suite = "tier3"
        self.seed = 99305
        self.fps = 14.0
        self.temp = 0.40
        self.paused = False
        self._step_once = False
        self._restart_req = True

        self.status_dict = {
            "step": 0,
            "health": 100.0,
            "kills": 0,
            "doors": 0,
            "keys": 0,
            "cells": 1,
            "door_shots": 0,
            "p50_ms": 0.0,
            "status": "RUNNING",
        }

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _resolve_wad(self, suite: str, seed: int) -> tuple[str | Path, str, int]:
        if suite == "two_rooms":
            return ROOT / "maps" / "two_rooms.wad", "MAP01", 2100
        if suite.startswith("fd_map") and os.path.isfile(FREEDOOM2_IWAD):
            return FREEDOOM2_IWAD, "MAP" + suite[-2:], 2800
        tier = {"tier1": 1, "tier2": 2, "tier3": 3, "tier4": 4}.get(suite, 3)
        ticks = {1: 1400, 2: 2100, 3: 2800, 4: 3500}[tier]
        wad_path = write_procedural_wad(self.tmp_dir / f"live_t{tier}_{seed}.wad", tier=tier, seed=seed)
        return wad_path, "MAP01", ticks

    def control(self, data: dict) -> None:
        with self._lock:
            if "fps" in data:
                self.fps = max(1.0, min(60.0, float(data["fps"])))
            if "temp" in data:
                self.temp = max(0.0, min(1.5, float(data["temp"])))
            if "suite" in data:
                self.suite = str(data["suite"])
            if "seed" in data:
                self.seed = int(data["seed"])
            act = data.get("action")
            if act == "restart":
                self.paused = False
                self._restart_req = True
            elif act == "pause":
                self.paused = True
            elif act == "resume":
                self.paused = False
            elif act == "step":
                self.paused = True
                self._step_once = True

    def get_state(self) -> dict:
        with self._lock:
            return dict(self.status_dict)

    def wait_next_frame(self, last_seq: int, timeout: float = 1.0) -> tuple[int, bytes]:
        with self._frame_cond:
            self._frame_cond.wait_for(lambda: self._frame_seq != last_seq or self._stop, timeout=timeout)
            return self._frame_seq, self._latest_jpeg

    def stop(self) -> None:
        self._stop = True
        with self._frame_cond:
            self._frame_cond.notify_all()

    def _encode_jpeg(self, img) -> bytes:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()

    def _run_loop(self) -> None:
        env = None
        try:
            while not self._stop:
                with self._lock:
                    self._restart_req = False
                    suite, seed, temp = self.suite, self.seed, self.temp

                wad, map_name, ticks = self._resolve_wad(suite, seed)
                if env is not None:
                    env.close()
                env = make_env(wad, map_name, episode_ticks=ticks, render_buffers=True)
                obs, _ = env.reset(seed=seed)
                self.agent.reset()
                self.agent.deterministic = temp <= 0.0
                self.agent.temperature = temp if temp > 0.0 else 0.40

                sectors = env.sector_polygons()
                all_pts = np.concatenate(
                    [s[:, :2] for _, s in sectors if len(s)] + [s[:, 2:4] for _, s in sectors if len(s)], axis=0
                )
                bounds = (
                    float(all_pts[:, 0].min()),
                    float(all_pts[:, 0].max()),
                    float(all_pts[:, 1].min()),
                    float(all_pts[:, 1].max()),
                )
                ps0 = env.privileged_state()
                prev_ceil = ps0["sector_ceiling"].copy() if ps0["sector_ceiling"] is not None else None
                visited = {(int(ps0["x"] // CELL_SIZE), int(ps0["y"] // CELL_SIZE))}
                opened_doors: set[int] = set()
                traj = [(ps0["x"], ps0["y"])]
                latencies: list[float] = []
                door_shots = 0
                step = 0
                finished = False

                while not self._stop:
                    with self._lock:
                        if self._restart_req:
                            break
                        self.agent.deterministic = self.temp <= 0.0
                        self.agent.temperature = self.temp if self.temp > 0.0 else 0.40
                        paused = self.paused
                        step_once = self._step_once
                        if step_once:
                            self._step_once = False
                        fps = self.fps

                    if (paused and not step_once) or finished:
                        time.sleep(0.05)
                        continue

                    t_loop = time.perf_counter()
                    ft = obs["rays"]["hit_type"][FRONT_RAY_IDX]
                    fd = obs["rays"]["distance"][FRONT_RAY_IDX]
                    ents = obs["visible_entities"]
                    ev = ((ents["mask"] == 1) & np.isin(ents["entity_type"], [1, 2, 3])).any()

                    t0 = time.perf_counter()
                    act = self.agent.act(obs)
                    latencies.append((time.perf_counter() - t0) * 1000.0)
                    if act[6] == 1 and not ev and (((ft == RAY_DOOR_ID) | (ft == RAY_LOCKED_ID)) & (fd < 0.25)).any():
                        door_shots += 1

                    spatial_map = self.agent._state[1]
                    obs_next, _, term, trunc, _ = env.step(act)
                    ps = env.privileged_state()
                    traj.append((ps["x"], ps["y"]))
                    visited.add((int(ps["x"] // CELL_SIZE), int(ps["y"] // CELL_SIZE)))
                    if ps["sector_ceiling"] is not None and ps["door_lock"] is not None and prev_ceil is not None:
                        for sid in np.flatnonzero((ps["door_lock"] >= 0) & (ps["sector_ceiling"] > prev_ceil + 0.5)):
                            opened_doors.add(int(sid))
                        prev_ceil = ps["sector_ceiling"].copy()

                    title = f"{suite} (seed {seed})"
                    frame = render_frame(obs, ps, act, spatial_map, sectors, traj, bounds, step, title, len(opened_doors))
                    jpeg = self._encode_jpeg(frame)
                    with self._frame_cond:
                        self._latest_jpeg = jpeg
                        self._frame_seq += 1
                        self._frame_cond.notify_all()

                    status = "EXIT REACHED" if ps["exit_reached"] else ("DIED" if ps["dead"] else ("TICK CAP" if trunc else "RUNNING"))
                    with self._lock:
                        self.status_dict = {
                            "step": step,
                            "health": float(ps["health"]),
                            "kills": int(ps["kills"]),
                            "doors": len(opened_doors),
                            "keys": int(np.sum(ps["keys"])),
                            "cells": len(visited),
                            "door_shots": door_shots,
                            "p50_ms": float(np.median(latencies)) if latencies else 0.0,
                            "status": status,
                        }

                    obs = obs_next
                    step += 1
                    if term or trunc:
                        finished = True
                    dt = time.perf_counter() - t_loop
                    sleep_s = max(0.0, (1.0 / fps) - dt)
                    if sleep_s > 0:
                        time.sleep(sleep_s)
        finally:
            if env is not None:
                env.close()


def make_handler(runner: LiveEpisodeRunner):
    class DemoHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                body = HTML_PAGE.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/api/state":
                body = json.dumps(runner.get_state()).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/stream.mjpg":
                self.send_response(HTTPStatus.OK)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Pragma", "no-cache")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
                self.end_headers()
                last_seq = -1
                try:
                    while True:
                        seq, jpg = runner.wait_next_frame(last_seq, timeout=1.0)
                        if seq == last_seq or not jpg:
                            continue
                        last_seq = seq
                        hdr = (
                            f"--FRAME\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpg)}\r\n\r\n"
                        ).encode("ascii")
                        self.wfile.write(hdr + jpg + b"\r\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self):
            path = urlparse(self.path).path
            if path == "/api/control":
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                runner.control(json.loads(raw.decode("utf-8")))
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"ok":true}')
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

    return DemoHandler


def main() -> None:
    ap = argparse.ArgumentParser(description="Live web demo server for DeadLock DOOM agent")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    args = ap.parse_args()

    runner = LiveEpisodeRunner(device=args.device)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(runner))
    print(f"DeadLock live web demo running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        runner.stop()
        server.server_close()


if __name__ == "__main__":
    main()
