---
title: DeadLock — Autonomous DOOM RL
emoji: 🎮
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 5.12.0
python_version: "3.12"
app_file: app.py
pinned: false
---

# DeadLock — Live Autonomous DOOM Agent

Zero-shot PyTorch reinforcement learning agent (**940,720 parameters**, `p50 = 1.18–1.4 ms`, `0.00` closed-door shots) combining circular 1D ray convolutions, RGB+Depth spatial bands, entity cross-attention, and a dual **Neural + Persistent Geometric Frontier Memory** (`EgoSpatialMemory`).

Click **Play Live Episode** to watch the 4-panel telemetry stream across **Tiers 1–4**, `two_rooms.wad`, and **Freedoom2** (`MAP01`, `MAP03`, `MAP06`), or run the **ZeroGPU Quick Benchmark**.
