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
3. **Vision + Depth Branch (`RgbDepthEncoder`):**
   - 3-layer strided ConvNet over concatenated `60×80×4` `rgb` + aligned `depth` frames preserving a `(3×5)` spatial horizon/floor band grid (`24×3×5` -> `96` dims) to detect floor exit pads (`GATE3`), wall exit switches (`SW1BRCOM`), and colored keycard door trims invisible to 2D raycasts.
4. **Differentiable Egocentric 2D Spatial Memory & Persistent Frontier Map (`EgoSpatialMemory`):**
   - Maintains a `(16, 25, 25)` egocentric neural memory grid (`±1200` map units) alongside a **3-channel persistent geometric frontier grid `(3, 33, 33)`** (`±1920` units: visited trail with `0.25x` attenuation on new keycard pickup for backtracking, seen doors minus visited regions -> unexplored-door novelty, and persistent locked-door locations).
   - Each step ego-warps both maps via `F.affine_grid` + `F.grid_sample` using measured ViZDoom turn calibration (`0.245` rad/step) and velocity displacement, splats 64 rays + 32 entities, reads 16-angle × 3-distance radial frontier probes `[140, 320, 560]` units, and fuses with conditioned spatial attention.
5. **Observation Telemetry, Control GRU & Conflict-Free Action Head:**
   - Feeds `vitals` + `prev_action` + smooth observation-derived front-cone door/lock/wall/aim & `use_cooldown` state into a `256`-dim `GRUCell` driving actor, critic, and auxiliary next-vitals prediction heads.
   - Outputs `12` factorized logits (`Move(3) × Strafe(3) × Turn(3) × Attack × Use × SwitchWeapon`) mapped deterministically to `MultiBinary(9)`—preventing canceling simultaneous button presses (`FORWARD+BACKWARD`, `TURN_LEFT+TURN_RIGHT`) and masking `SELECT_NEXT_WEAPON` off while the active weapon holds ammo.

---

## Training & Exploration Pipeline

- **Procedural UDMF Map Generator ([training/wad_gen.py](training/wad_gen.py)):** Generates 3-lump ZDoom UDMF PWADs + training-time layout metadata across Tiers 1–4 (combat arenas with cover pillars, multi-room door mazes with varied elevation/lighting, keycard → locked door → exit branching dungeons with colored door trims, and full gauntlets + wall exit switches) without external BSP tools.
- **Exploration-First Privileged Teacher & Recurrent DAgger ([training/pretrain_bc.py](training/pretrain_bc.py)):** Uses `env.privileged_state()` + map topology strictly at training time (`Rule §2`) across `6` scheduled DAgger rounds (`80` eps/round, `β: 1.0 → 0.08`, `96`-step BPTT) combining shortest-path runs with partial-observability frontier exploration & locked-door bounce demonstrations (`0.00` door shots).
- **KL-Anchored Recurrent PPO & Anti-Loop Shaping ([training/env_wrapper.py](training/env_wrapper.py), [training/train.py](training/train.py)):** `270,000` steps of vectorized Recurrent PPO (`gamma=0.995`, truncated BPTT, GAE, PPO clipped value loss, critic-warmup `--freeze-actor-updates 25`, factorized KL anchor `--anchor-kl 0.035` toward the DAgger policy, `96×96` tile exploration + `_post_key_mode` backtracking breadcrumbs, stale-room penalty, and locked-door hug penalty) across `96` procedural maps/tier + `freedoom2.wad` (`MAP01–MAP06`).

---

## Zero-Shot Evaluation & Validator Results

Evaluated using [training/eval.py](training/eval.py) / [training/visualize.py](training/visualize.py) (`submission/weights/model.pt`, `temperature=0.40`, `3.6 MB` via Git LFS) across **8 unseen procedural maps per tier** and unmodified `freedoom2.wad:MAP01`:

| Benchmark | Exit Rate | Keycard Pickup | Doors Opened | Avg Kills | Cells Visited | p50 Latency |
| --- | --- | --- | --- | --- | --- | --- |
| **Tier 1 (8 unseen)** | **100.0%** | 0.0% | 0.25 | 1.75 | 15.2 | 1.18 ms |
| **Tier 2 (8 unseen)** | **50.0%** | 0.0% | 1.25 | 1.12 | 28.1 | 1.18 ms |
| **Tier 3 (8 unseen)** | **62.5%** | **87.5%** | 2.62 | 3.75 | 43.9 | 1.18 ms |
| **Tier 4 (8 unseen)** | **37.5%** | **50.0%** | 4.75 | 7.12 | 55.5 | 1.18 ms |
| **Freedoom2 MAP01** | 0% | — | 2 | 6 | 63 | 1.21 ms |

- **24-map/tier (96 unseen episodes):** **Tier 1 `58.3%`**, **Tier 2 `70.8%`**, **Tier 3 `70.8%` (`95.8%` keycards, `40.8` cells, `0%` deaths)**, **Tier 4 `25.0%` (`50.0%` keycards, `54.6` cells, `7.96` kills)**, **`56.2%` mean exit**, `0.00` door shots.
- **48-map/tier (192 unseen episodes):** **Tier 1 `52.1%`**, **Tier 2 `58.3%`**, **Tier 3 `50.0%` (`85%` keycards, `41.0` cells)**, **Tier 4 `18.8%` (`35%` keycards, `48.4` cells, `7.96` kills)**, **`44.8%` mean exit**, `0.00` door shots (`two_rooms.wad`: **Exit in 54 steps**).
- **Unseen Complex Map Stress Suite:** Fresh `88000` seeds (`12`/tier) reached **`50.0%` Tier 3 (`92%` keycards)** and `41.7%` mean exit; guaranteed **8-room key+lock mazes** averaged **`58%` keycards, `5.58` doors, `65.9` cells, `0%` deaths**; guaranteed **10-room heavy gauntlets** averaged **`11.75` kills, `8.08` doors, `60.2` cells** (`0.00` door shots everywhere).

**Official Validator (`validate.py`):** **PASSED** on `maps/two_rooms.wad` and unseen Tier 1–4 procedural maps (well below `1.0 GB` repo cap and `40.0 ms/step` compute budget).

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
  wad_gen.py             Procedural Tier 1-4 UDMF WAD generator + training layout export
  pretrain_bc.py         Privileged Teacher Recurrent DAgger warmstart trainer
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
