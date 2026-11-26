# DeadLock — Hybrid Vision & Ego-Spatial Memory DOOM RL Agent

Submission for **Yugaantar 2026 · DevForge — Track 1: "Can It Play DOOM?"**

DeadLock is a custom PyTorch reinforcement learning agent trained from scratch to play unseen DOOM levels zero-shot under partial observability—navigating multi-room layouts, fighting enemies, collecting keycards, unlocking locked doors, and reaching level exits at **~1.0 ms** per decision on a single CPU thread.

---

## Architecture (`DoomSpatialPolicy`)

Implemented in [submission/model.py](submission/model.py) and [submission/agent.py](submission/agent.py) using pure `torch` + `numpy` (zero extra dependencies beyond the official base image):

1. **Direction-Preserving Circular Ray Encoder (`RayEncoder`):**
   - Processes all 64 360° ego-centric raycasts (`distance` + 6 one-hot `hit_type` classes + `sin/cos` angular embeddings) using a 3-layer **Circular 1D CNN** (`padding_mode="circular"`), preserving 16 directional angular sectors alongside an explicit 9-ray front-cone (`[-22.5°, +22.5°]`) and global pools.
2. **Masked Entity Set Cross-Attention (`EntityAttentionEncoder`):**
   - Permutation-invariant attention over variable numbers (`0–32`) of visible entities (`features` + learned `entity_type` embeddings), conditioned on ray + vitals context with NaN-safe empty-mask handling.
3. **Lightweight Vision Branch (`RgbEncoder`):**
   - 3-layer strided ConvNet on `60×80×3` downsampled `rgb` frames (`AdaptiveAvgPool2d((2, 3))`) to detect floor exit pads (`GATE3`), exit switches, and door trims invisible to 2D raycasts.
4. **Differentiable Egocentric 2D Spatial Memory (`EgoSpatialMemory`):**
   - Maintains a `(16, 25, 25)` egocentric spatial grid (`±1200` map units). Each step warps the previous memory grid via estimated ego-motion (`F.affine_grid` + `F.grid_sample` from `vel_forward`, `vel_right`, and yaw command), splats current ray hits and visible entities into the grid, updates via a gated `3×3` conv residual cell, and reads out via vitals/keycard-conditioned spatial attention.
5. **Control GRU & Conflict-Free Factorized Action Head:**
   - A `192`-dim `GRUCell` feeds actor, critic, and auxiliary next-vitals prediction heads.
   - Outputs `12` factorized logits (`Move(3) × Strafe(3) × Turn(3) × Attack × Use × SwitchWeapon`) mapped deterministically to `MultiBinary(9)`—preventing canceling simultaneous button presses (`FORWARD+BACKWARD`, `TURN_LEFT+TURN_RIGHT`) and masking `SELECT_NEXT_WEAPON` off while the active weapon holds ammo.

---

## Training & Exploration Pipeline

- **Procedural UDMF Map Generator ([training/wad_gen.py](training/wad_gen.py)):** Generates 3-lump ZDoom UDMF PWADs across Tiers 1–4 (combat arenas with cover pillars, multi-room door mazes, keycard → locked door → exit branching dungeons, and full gauntlets) without external BSP tools.
- **Reward Shaping & Backtracking ([training/env_wrapper.py](training/env_wrapper.py)):** Uses `env.privileged_state()` during training for `192×192` room-scale episodic exploration, monotonic keycard/door/locked-door approach shaping, keycard-pickup exploration reset (`_post_key_mode`) to incentivize backtracking out of key branches, ammo-discipline shaping, and a bounded 4-map LRU cache per worker (`--num-envs 6`) for 16 GB RAM efficiency.
- **Vectorized Recurrent PPO ([training/train.py](training/train.py), [training/run_curriculum.py](training/run_curriculum.py)):** Multi-process `SubprocVecEnv` rollouts with truncated BPTT, GAE, batched feed-forward encoders (`encode_inputs`), `target_kl=0.03` early stopping, entropy annealing, and auxiliary next-vitals supervision across synthetic maps + `freedoom2.wad` (`MAP01`, `MAP02`, `MAP04`, `MAP05`, `MAP06`).

---

## Zero-Shot Evaluation & Validator Results

Evaluated using [training/eval.py](training/eval.py) / [training/visualize.py](training/visualize.py) (`submission/weights/model.pt`, `2.6 MB` via Git LFS) across **8 unseen procedural maps per tier** and unmodified `freedoom2.wad:MAP01`:

| Benchmark | Exit Rate | Keycard Pickup | Doors Opened | Avg Kills | Cells Visited | p50 Latency |
| --- | --- | --- | --- | --- | --- | --- |
| **Tier 1 (8 unseen)** | **50.0%** | 0.0% | 0.25 | 2.25 | 19.6 | 0.96 ms |
| **Tier 2 (8 unseen)** | **25.0%** | 0.0% | 1.12 | 0.88 | 26.5 | 0.95 ms |
| **Tier 3 (8 unseen)** | **25.0%** | **50.0%** | 2.62 | 2.00 | 31.4 | 0.95 ms |
| **Tier 4 (8 unseen)** | **12.5%** | **37.5%** | 4.62 | 6.25 | 41.4 | 0.94 ms |
| **Freedoom2 MAP01** | 0% | — | 2 | 4 | 73 | 0.94 ms |

**Official Validator (`validate.py`):** **PASSED** across `maps/two_rooms.wad` and unseen Tier 1–4 maps (`2.8 MB` of `1.0 GB` repo size, `0.4s` init time, `p50 = 1.1 ms`, `p99 = 1.3–1.7 ms` vs `40.0 ms` budget).

---

## Repository Layout

```text
env/                     Official ViZDoom Gymnasium wrapper (doom_env)
maps/                    Starter test map (two_rooms.wad)
submission/
  agent.py               DoomAgent evaluation entry point (__init__, reset, act)
  model.py               DoomSpatialPolicy (RayEncoder, RgbEncoder, EntityAttention, EgoSpatialMemory)
  weights/model.pt       Trained weights (Git LFS)
  requirements.txt       No extra packages required beyond base image
training/
  wad_gen.py             Procedural Tier 1-4 UDMF WAD generator
  env_wrapper.py         ShapedDoomEnv, backtracking breadcrumbs, Freedoom2 Stage 4 pool
  train.py               Vectorized Recurrent PPO + BPTT trainer
  run_curriculum.py      4-stage curriculum launcher (m4_fast / m4_medium / m4_full)
  eval.py                Zero-shot Tier 1-4 + Freedoom2 evaluator
  visualize.py           4-panel replay GIFs, storyboards, live viewer & dashboard generator
validate.py              Official local submission validator
tests/                   Wrapper and validator test suite
docs/                    Contract, rules, evaluation, and submission docs
```

---

## Quick Usage

```bash
# Validate submission interface, size, and latency
python validate.py
# Or inside the official Docker sandbox:
docker build -t devforge-doom . && python validate.py --docker --image devforge-doom

# Run 4-stage curriculum training
PYTHONPATH=. python -m training.run_curriculum --preset m4_medium

# Evaluate zero-shot across unseen Tier 1-4 maps & Freedoom2 MAP01
PYTHONPATH=. python -m training.eval --maps-per-tier 8

# Generate replay GIFs, dashboard PNG, and HTML viewer (or pass --live --wad ...)
PYTHONPATH=. python -m training.visualize --out-dir /tmp/deadlock_viz
```

See [docs/CONTRACT.md](docs/CONTRACT.md), [docs/RULES.md](docs/RULES.md), [docs/EVALUATION.md](docs/EVALUATION.md), and [docs/SUBMISSION.md](docs/SUBMISSION.md) for the full competition specification.
