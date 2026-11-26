"""End-to-end 4-stage curriculum trainer tuned for Apple Silicon (Mac M4).

Usage:
    PYTHONPATH=. .venv/bin/python -m training.run_curriculum --preset m4_fast
    PYTHONPATH=. .venv/bin/python -m training.run_curriculum --preset m4_full
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from training.eval import run_eval
from training.train import ROOT, parse_args as parse_train_args, train

PRESETS = {
    "m4_fast": {
        1: (24_000, 1400, 16),
        2: (36_000, 1800, 24),
        3: (64_000, 2400, 24),
        4: (56_000, 2800, 24),
    },
    "m4_medium": {
        1: (60_000, 1400, 24),
        2: (80_000, 2100, 32),
        3: (120_000, 2800, 32),
        4: (120_000, 3500, 32),
    },
    "m4_full": {
        1: (120_000, 1400, 32),
        2: (160_000, 2100, 48),
        3: (200_000, 2800, 48),
        4: (240_000, 3500, 48),
    },
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Run 4-stage curriculum training on Mac M4")
    ap.add_argument("--preset", default="m4_fast", choices=list(PRESETS))
    ap.add_argument("--num-envs", type=int, default=6)
    ap.add_argument("--rollout-len", type=int, default=64)
    ap.add_argument("--ppo-epochs", type=int, default=3)
    ap.add_argument("--cpu-threads", type=int, default=4)
    ap.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    args = ap.parse_args()

    out_path = ROOT / "submission" / "weights" / "model.pt"
    ckpt_dir = Path("/tmp/deadlock_ckpts")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    schedule = PRESETS[args.preset]
    resume_path = ""
    for stage in (1, 2, 3, 4):
        steps, ticks, maps_per_tier = schedule[stage]
        print(f"\n=== Curriculum Stage {stage}/4 ({steps} steps, {ticks} ticks) ===")
        t_args = parse_train_args([])
        t_args.stage = stage
        t_args.total_steps = steps
        t_args.episode_ticks = ticks
        t_args.maps_per_tier = maps_per_tier
        t_args.num_envs = args.num_envs
        t_args.rollout_len = args.rollout_len
        t_args.ppo_epochs = args.ppo_epochs
        t_args.num_minibatches = 2
        t_args.cpu_threads = args.cpu_threads
        t_args.device = args.device
        t_args.resume = resume_path
        t_args.out = str(out_path)
        t_args.log_every = 5
        train(t_args)
        stage_copy = ckpt_dir / f"stage{stage}.pt"
        shutil.copy2(out_path, stage_copy)
        resume_path = str(out_path)

    print("\n=== Held-Out Zero-Shot Evaluation ===")
    e_args = argparse.Namespace(device="cpu", maps_per_tier=4, eval_dir="/tmp/deadlock_eval_maps", deterministic=False)
    run_eval(e_args)


if __name__ == "__main__":
    main()
