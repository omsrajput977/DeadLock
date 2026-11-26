"""Zero-shot held-out evaluation suite across Tiers 1-4 + two_rooms.wad."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import vizdoom as vd
from doom_env import make_env

from submission.agent import DoomAgent
from training.wad_gen import write_procedural_wad

ROOT = Path(__file__).resolve().parents[1]
CELL_SIZE = 96.0


def eval_map(agent: DoomAgent, wad_path: str | Path, map_name: str, episode_ticks: int, seed: int) -> dict:
    env = make_env(wad_path, map_name, episode_ticks=episode_ticks, render_buffers=True)
    try:
        obs, _ = env.reset(seed=seed)
        agent.reset()
        visited = set()
        opened_doors = set()
        ps0 = env.privileged_state()
        prev_ceil = ps0["sector_ceiling"].copy() if ps0["sector_ceiling"] is not None else None
        visited.add((int(ps0["x"] // CELL_SIZE), int(ps0["y"] // CELL_SIZE)))

        latencies = []
        steps = 0
        while True:
            t0 = time.perf_counter()
            act = agent.act(obs)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            obs, _, term, trunc, info = env.step(act)
            steps += 1
            ps = env.privileged_state()
            visited.add((int(ps["x"] // CELL_SIZE), int(ps["y"] // CELL_SIZE)))
            if ps["sector_ceiling"] is not None and ps["door_lock"] is not None and prev_ceil is not None:
                rising = (ps["door_lock"] >= 0) & (ps["sector_ceiling"] > prev_ceil + 0.5)
                for sid in np.flatnonzero(rising):
                    opened_doors.add(int(sid))
                prev_ceil = ps["sector_ceiling"].copy()
            if term or trunc:
                return {
                    "exit": 1.0 if ps["exit_reached"] else 0.0,
                    "dead": 1.0 if ps["dead"] else 0.0,
                    "kills": float(ps["kills"]),
                    "keys": float(np.sum(ps["keys"])),
                    "doors": float(len(opened_doors)),
                    "cells": float(len(visited)),
                    "steps": float(steps),
                    "p50_ms": float(np.median(latencies)) if latencies else 0.0,
                }
    finally:
        env.close()


def run_eval(args: argparse.Namespace) -> None:
    dummy = make_env(ROOT / "maps" / "two_rooms.wad", "MAP01", episode_ticks=700, render_buffers=False)
    agent = DoomAgent(dummy.observation_space, dummy.action_space, device=args.device)
    if args.deterministic:
        agent.deterministic = True
    dummy.close()

    eval_dir = Path(args.eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # 1) Starter test map & Freedoom2 MAP01
    res_tr = eval_map(agent, ROOT / "maps" / "two_rooms.wad", "MAP01", episode_ticks=2100, seed=0)
    print(
        f"[two_rooms] exit={res_tr['exit']:.0f} kills={res_tr['kills']:.0f} doors={res_tr['doors']:.0f} "
        f"cells={res_tr['cells']:.0f} p50={res_tr['p50_ms']:.2f}ms"
    )
    iwad = os.path.join(os.path.dirname(vd.__file__), "freedoom2.wad")
    if os.path.isfile(iwad):
        res_fd = eval_map(agent, iwad, "MAP01", episode_ticks=2100, seed=0)
        print(
            f"[freedoom2:MAP01] exit={res_fd['exit']:.0f} kills={res_fd['kills']:.0f} doors={res_fd['doors']:.0f} "
            f"cells={res_fd['cells']:.0f} p50={res_fd['p50_ms']:.2f}ms"
        )

    # 2) Held-out procedural maps (seeds 99000+ never seen in training)
    for tier in (1, 2, 3, 4):
        ticks = {1: 1400, 2: 2100, 3: 2800, 4: 3500}[tier]
        rows = []
        for idx in range(args.maps_per_tier):
            wad = write_procedural_wad(eval_dir / f"eval_t{tier}_{idx}.wad", tier=tier, seed=99000 + tier * 100 + idx)
            rows.append(eval_map(agent, wad, "MAP01", episode_ticks=ticks, seed=idx))
        print(
            f"[Tier {tier} ({args.maps_per_tier} maps)] "
            f"exit={np.mean([r['exit'] for r in rows]):.2f} "
            f"kills={np.mean([r['kills'] for r in rows]):.2f} "
            f"keys={np.mean([r['keys'] for r in rows]):.2f} "
            f"doors={np.mean([r['doors'] for r in rows]):.2f} "
            f"cells={np.mean([r['cells'] for r in rows]):.1f} "
            f"p50={np.mean([r['p50_ms'] for r in rows]):.2f}ms"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate DoomAgent across held-out Tier 1-4 maps")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--maps-per-tier", type=int, default=4)
    p.add_argument("--eval-dir", type=str, default="/tmp/deadlock_eval_maps")
    p.add_argument("--deterministic", action="store_true", help="Force greedy argmax instead of agent default")
    return p.parse_args()


if __name__ == "__main__":
    run_eval(parse_args())
