"""Hugging Face Spaces (Gradio + ZeroGPU) live web app for DeadLock."""

from __future__ import annotations

import os
import random
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if not (ROOT / "submission").is_dir():
    # Support running directly from repo checkout (deploy/hf_gradio/app.py)
    ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "env") not in sys.path:
    sys.path.insert(0, str(ROOT / "env"))

try:
    import spaces

    gpu_decorator = spaces.GPU
except ImportError:
    def gpu_decorator(fn=None, **_kwargs):
        if fn is None:
            return lambda f: f
        return fn

import gradio as gr
import numpy as np
import torch
import vizdoom as vd
from doom_env import make_env

from submission.agent import DoomAgent
from training.env_wrapper import FRONT_RAY_IDX, RAY_DOOR_ID, RAY_LOCKED_ID
from training.visualize import render_frame
from training.wad_gen import write_procedural_wad

FREEDOOM2_IWAD = os.path.join(os.path.dirname(vd.__file__), "freedoom2.wad")
CELL_SIZE = 96.0
TMP_DIR = Path(tempfile.mkdtemp(prefix="deadlock_gradio_"))

SUITE_CHOICES = [
    ("Tier 3 — Keycard + Locked Door", "tier3"),
    ("Tier 4 — Full Gauntlet", "tier4"),
    ("Tier 2 — Multi-Room Exploration", "tier2"),
    ("Tier 1 — Combat & Survival", "tier1"),
    ("Starter — two_rooms.wad", "two_rooms"),
    ("Freedoom2 — MAP01", "fd_map01"),
    ("Freedoom2 — MAP03", "fd_map03"),
    ("Freedoom2 — MAP06", "fd_map06"),
]

_CPU_AGENT: DoomAgent | None = None


def _get_cpu_agent() -> DoomAgent:
    global _CPU_AGENT
    if _CPU_AGENT is None:
        dummy = make_env(ROOT / "maps" / "two_rooms.wad", "MAP01", episode_ticks=700, render_buffers=False)
        _CPU_AGENT = DoomAgent(dummy.observation_space, dummy.action_space, device="cpu")
        dummy.close()
    return _CPU_AGENT


def resolve_wad(suite: str, seed: int) -> tuple[str | Path, str, int]:
    if suite == "two_rooms":
        return ROOT / "maps" / "two_rooms.wad", "MAP01", 2100
    if suite.startswith("fd_map") and os.path.isfile(FREEDOOM2_IWAD):
        return FREEDOOM2_IWAD, "MAP" + suite[-2:], 2800
    tier = {"tier1": 1, "tier2": 2, "tier3": 3, "tier4": 4}.get(suite, 3)
    ticks = {1: 1400, 2: 2100, 3: 2800, 4: 3500}[tier]
    wad_path = write_procedural_wad(TMP_DIR / f"gradio_t{tier}_{seed}.wad", tier=tier, seed=int(seed))
    return wad_path, "MAP01", ticks


def _format_stats(
    suite: str,
    seed: int,
    step: int,
    ps: dict,
    doors: int,
    cells: int,
    door_shots: int,
    latencies: list[float],
    status: str,
    device_label: str,
) -> str:
    p50 = float(np.median(latencies)) if latencies else 0.0
    keys = int(np.sum(ps["keys"]))
    return (
        f"### Status: **{status}** (`{device_label}`)\n"
        f"| Metric | Value |\n| --- | --- |\n"
        f"| **Map / Seed** | `{suite}` (seed `{seed}`) |\n"
        f"| **Step** | `{step}` |\n"
        f"| **Health** | `{int(ps['health'])}` |\n"
        f"| **Kills** | `{int(ps['kills'])}` |\n"
        f"| **Doors Opened** | `{doors}` |\n"
        f"| **Keycards Held** | `{keys}` |\n"
        f"| **Cells Visited** | `{cells}` |\n"
        f"| **Closed-Door Shots** | `{door_shots}` |\n"
        f"| **Decision p50 Latency** | `{p50:.2f} ms` |"
    )


def run_live_episode(suite: str, seed: int, fps: float, temp: float):
    """Stream live 4-panel DOOM gameplay on CPU (preserves ZeroGPU daily quota)."""
    agent = _get_cpu_agent()
    agent.deterministic = float(temp) <= 0.0
    agent.temperature = float(temp) if float(temp) > 0.0 else 0.40

    wad, map_name, ticks = resolve_wad(suite, int(seed))
    env = make_env(wad, map_name, episode_ticks=ticks, render_buffers=True)
    try:
        obs, _ = env.reset(seed=int(seed))
        agent.reset()
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
        stride = 2 if fps >= 16 else 1

        while True:
            t_loop = time.perf_counter()
            ft = obs["rays"]["hit_type"][FRONT_RAY_IDX]
            fd = obs["rays"]["distance"][FRONT_RAY_IDX]
            ents = obs["visible_entities"]
            ev = ((ents["mask"] == 1) & np.isin(ents["entity_type"], [1, 2, 3])).any()

            t0 = time.perf_counter()
            act = agent.act(obs)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            if act[6] == 1 and not ev and (((ft == RAY_DOOR_ID) | (ft == RAY_LOCKED_ID)) & (fd < 0.25)).any():
                door_shots += 1

            spatial_map = agent._state[1]
            obs_next, _, term, trunc, _ = env.step(act)
            ps = env.privileged_state()
            traj.append((ps["x"], ps["y"]))
            visited.add((int(ps["x"] // CELL_SIZE), int(ps["y"] // CELL_SIZE)))
            if ps["sector_ceiling"] is not None and ps["door_lock"] is not None and prev_ceil is not None:
                for sid in np.flatnonzero((ps["door_lock"] >= 0) & (ps["sector_ceiling"] > prev_ceil + 0.5)):
                    opened_doors.add(int(sid))
                prev_ceil = ps["sector_ceiling"].copy()

            done = term or trunc
            status = "EXIT REACHED" if ps["exit_reached"] else ("DIED" if ps["dead"] else ("TICK CAP" if trunc else "RUNNING"))
            if step % stride == 0 or done:
                frame = render_frame(obs, ps, act, spatial_map, sectors, traj, bounds, step, f"{suite} (seed {int(seed)})", len(opened_doors))
                yield frame, _format_stats(suite, int(seed), step, ps, len(opened_doors), len(visited), door_shots, latencies, status, "cpu")
                dt = time.perf_counter() - t_loop
                sleep_s = max(0.0, (1.0 / max(1.0, float(fps))) - dt)
                if sleep_s > 0 and not done:
                    time.sleep(sleep_s)

            obs = obs_next
            step += 1
            if done:
                break
    finally:
        env.close()


@gpu_decorator(duration=30)
def run_gpu_benchmark(suite: str, seed: int, temp: float):
    """Run one complete episode inside ZeroGPU (falls back to CPU if CUDA unavailable)."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    wad, map_name, ticks = resolve_wad(suite, int(seed))
    env = make_env(wad, map_name, episode_ticks=ticks, render_buffers=True)
    try:
        agent = DoomAgent(env.observation_space, env.action_space, device=dev)
        agent.deterministic = float(temp) <= 0.0
        agent.temperature = float(temp) if float(temp) > 0.0 else 0.40
        obs, _ = env.reset(seed=int(seed))
        agent.reset()

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
        last_frame = None

        while True:
            ft = obs["rays"]["hit_type"][FRONT_RAY_IDX]
            fd = obs["rays"]["distance"][FRONT_RAY_IDX]
            ents = obs["visible_entities"]
            ev = ((ents["mask"] == 1) & np.isin(ents["entity_type"], [1, 2, 3])).any()

            t0 = time.perf_counter()
            act = agent.act(obs)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            if act[6] == 1 and not ev and (((ft == RAY_DOOR_ID) | (ft == RAY_LOCKED_ID)) & (fd < 0.25)).any():
                door_shots += 1

            spatial_map = agent._state[1]
            obs_next, _, term, trunc, _ = env.step(act)
            ps = env.privileged_state()
            traj.append((ps["x"], ps["y"]))
            visited.add((int(ps["x"] // CELL_SIZE), int(ps["y"] // CELL_SIZE)))
            if ps["sector_ceiling"] is not None and ps["door_lock"] is not None and prev_ceil is not None:
                for sid in np.flatnonzero((ps["door_lock"] >= 0) & (ps["sector_ceiling"] > prev_ceil + 0.5)):
                    opened_doors.add(int(sid))
                prev_ceil = ps["sector_ceiling"].copy()

            if term or trunc:
                status = "EXIT REACHED" if ps["exit_reached"] else ("DIED" if ps["dead"] else "TICK CAP")
                last_frame = render_frame(obs, ps, act, spatial_map, sectors, traj, bounds, step, f"{suite} (seed {int(seed)})", len(opened_doors))
                return last_frame, _format_stats(suite, int(seed), step, ps, len(opened_doors), len(visited), door_shots, latencies, status, dev)

            obs = obs_next
            step += 1
    finally:
        env.close()


def randomize_seed() -> int:
    return random.randint(80000, 99999)


with gr.Blocks(title="DeadLock — Autonomous DOOM RL", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# DeadLock — Live Autonomous DOOM RL Agent\n"
        "Zero-shot PyTorch policy (**940,720 params · 3.6 MB**) fusing circular 360° rays, RGB+Depth bands, "
        "masked entity cross-attention, and dual Neural + Persistent Geometric Frontier Memory (`EgoSpatialMemory`)."
    )
    with gr.Row():
        with gr.Column(scale=3):
            frame_out = gr.Image(label="4-Panel Live Gameplay (RGB+HUD | 360° Rays | Spatial Memory | Top-Down Map)", type="pil")
        with gr.Column(scale=1):
            suite_in = gr.Dropdown(choices=SUITE_CHOICES, value="tier3", label="Map / Difficulty Tier")
            with gr.Row():
                seed_in = gr.Number(value=99305, precision=0, label="Map Seed")
                rand_btn = gr.Button("Random Seed")
            fps_in = gr.Slider(minimum=4, maximum=30, step=1, value=14, label="Playback FPS")
            temp_in = gr.Slider(minimum=0.0, maximum=0.70, step=0.05, value=0.40, label="Sampling Temperature")
            with gr.Row():
                play_btn = gr.Button("Play Live Episode", variant="primary")
                stop_btn = gr.Button("Stop")
            gpu_btn = gr.Button("Run ZeroGPU Quick Benchmark")
            stats_out = gr.Markdown("Click **Play Live Episode** to start streaming.")

    rand_btn.click(fn=randomize_seed, inputs=None, outputs=seed_in, api_name="randomize_seed")
    live_ev = play_btn.click(
        fn=run_live_episode,
        inputs=[suite_in, seed_in, fps_in, temp_in],
        outputs=[frame_out, stats_out],
        api_name="play_live",
    )
    stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[live_ev], api_name=False)
    gpu_btn.click(
        fn=run_gpu_benchmark,
        inputs=[suite_in, seed_in, temp_in],
        outputs=[frame_out, stats_out],
        api_name="gpu_benchmark",
    )


if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        ssr_mode=False,
    )
