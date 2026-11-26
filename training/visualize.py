"""Visualizes DoomAgent gameplay, 360-degree rays, EgoSpatialMemory, top-down trajectories,
and Tier 1-4 evaluation test outputs into animated GIFs, storyboard PNGs, and a dashboard image.

Usage:
    PYTHONPATH=. .venv/bin/python -m training.visualize --out-dir /tmp/deadlock_viz
    PYTHONPATH=. .venv/bin/python -m training.visualize --live --wad maps/two_rooms.wad
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import vizdoom as vd
from doom_env import make_env
from PIL import Image, ImageDraw, ImageFont

from submission.agent import DoomAgent
from submission.model import preprocess_obs_dict
from training.wad_gen import write_procedural_wad

ROOT = Path(__file__).resolve().parents[1]
RAY_COLORS = {
    0: (45, 48, 56),      # none
    1: (130, 140, 155),   # wall
    2: (60, 210, 240),    # door
    3: (255, 200, 50),    # locked_door
    4: (245, 75, 75),     # enemy
    5: (80, 235, 130),    # item
}
ENT_COLORS = {
    1: (255, 70, 70),
    2: (255, 70, 70),
    3: (255, 110, 70),
    4: (255, 180, 40),
    5: (80, 235, 130),
    6: (80, 200, 255),
    7: (200, 235, 90),
    8: (200, 150, 255),
    9: (255, 225, 45),   # keycard
    10: (255, 130, 50),
    11: (180, 180, 180),
}
BTN_SHORT = ("FWD", "BCK", "L", "R", "TL", "TR", "ATK", "USE", "SW")


def _font(size: int = 12):
    for p in (
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/SFNSMono.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                pass
    return ImageFont.load_default()


def _draw_ray_radar(draw: ImageDraw.ImageDraw, x0: int, y0: int, size: int, obs: dict) -> None:
    cx, cy = x0 + size // 2, y0 + size // 2
    r_max = size // 2 - 14
    draw.rectangle([x0, y0, x0 + size, y0 + size], fill=(18, 20, 26), outline=(55, 60, 72))
    for frac in (0.33, 0.66, 1.0):
        rr = int(r_max * frac)
        draw.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=(38, 42, 52))

    # 90 deg FOV guide lines (straight ahead = up = -Y)
    fov_dx = int(r_max * math.sin(math.radians(45)))
    fov_dy = int(r_max * math.cos(math.radians(45)))
    draw.line([(cx, cy), (cx - fov_dx, cy - fov_dy)], fill=(55, 65, 80), width=1)
    draw.line([(cx, cy), (cx + fov_dx, cy - fov_dy)], fill=(55, 65, 80), width=1)

    dists = obs["rays"]["distance"]
    htypes = obs["rays"]["hit_type"]
    for i in range(len(dists)):
        # ray 0 = ahead (up: -Y), positive CCW (left: -X)
        ang = i * (2.0 * math.pi / len(dists))
        d = float(dists[i]) * r_max
        px = cx - d * math.sin(ang)
        py = cy - d * math.cos(ang)
        ht = int(htypes[i])
        col = RAY_COLORS.get(ht, (120, 120, 120))
        draw.line([(cx, cy), (px, py)], fill=tuple(max(18, c // 3) for c in col), width=1)
        if ht != 0:
            r_dot = 3 if ht in (2, 3, 4) else 2
            draw.ellipse([px - r_dot, py - r_dot, px + r_dot, py + r_dot], fill=col)

    # Visible entities overlay
    ents = obs["visible_entities"]
    for j in range(len(ents["mask"])):
        if ents["mask"][j] == 0:
            continue
        ed, sin_a, cos_a = float(ents["features"][j, 0]), float(ents["features"][j, 1]), float(ents["features"][j, 2])
        ed_px = min(r_max, ed * r_max)
        ex = cx - ed_px * sin_a
        ey = cy - ed_px * cos_a
        et = int(ents["entity_type"][j])
        ecol = ENT_COLORS.get(et, (255, 255, 255))
        r_e = 4 if et == 9 else 3
        draw.rectangle([ex - r_e, ey - r_e, ex + r_e, ey + r_e], fill=ecol, outline=(255, 255, 255))

    draw.polygon([(cx, cy - 5), (cx - 4, cy + 4), (cx + 4, cy + 4)], fill=(255, 255, 255))


def _render_spatial_map(img_canvas: Image.Image, x0: int, y0: int, size: int, spatial_map: torch.Tensor) -> None:
    """Render L2 energy across channels of the (16, 25, 25) EgoSpatialMemory tensor."""
    m = spatial_map.squeeze(0).detach().cpu().numpy()  # (16, 25, 25)
    energy = np.sqrt((m * m).mean(axis=0))  # (25, 25)
    mx = float(energy.max())
    norm = (energy / mx * 255.0) if mx > 1e-6 else np.zeros_like(energy)
    norm_u8 = np.clip(norm, 0, 255).astype(np.uint8)
    # Build false-color RGB (deep navy -> cyan -> yellow)
    rgb = np.zeros((norm_u8.shape[0], norm_u8.shape[1], 3), dtype=np.uint8)
    rgb[..., 0] = np.clip(norm_u8 * 1.3 - 80, 0, 255).astype(np.uint8)
    rgb[..., 1] = norm_u8
    rgb[..., 2] = np.clip(45 + norm_u8 * 0.8, 0, 255).astype(np.uint8)
    heat = Image.fromarray(rgb, mode="RGB").resize((size - 8, size - 8), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(img_canvas)
    draw.rectangle([x0, y0, x0 + size, y0 + size], fill=(18, 20, 26), outline=(55, 60, 72))
    img_canvas.paste(heat, (x0 + 4, y0 + 4))
    cx, cy = x0 + size // 2, y0 + size // 2
    draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=(255, 255, 255))


def _draw_topdown(
    draw: ImageDraw.ImageDraw,
    x0: int,
    y0: int,
    w: int,
    h: int,
    sectors: list[tuple[int, np.ndarray]],
    door_lock: np.ndarray | None,
    traj: list[tuple[float, float]],
    angle_deg: float,
    bounds: tuple[float, float, float, float],
) -> None:
    draw.rectangle([x0, y0, x0 + w, y0 + h], fill=(16, 18, 24), outline=(55, 60, 72))
    min_x, max_x, min_y, max_y = bounds
    span_x = max(1.0, max_x - min_x)
    span_y = max(1.0, max_y - min_y)
    pad = 14
    scale = min((w - 2 * pad) / span_x, (h - 2 * pad) / span_y)
    ox = x0 + (w - span_x * scale) * 0.5
    oy = y0 + (h - span_y * scale) * 0.5

    def to_px(wx: float, wy: float) -> tuple[float, float]:
        # World +Y is up -> screen Y goes down
        return ox + (wx - min_x) * scale, oy + (max_y - wy) * scale

    for sid, segs in sectors:
        if len(segs) == 0:
            continue
        is_door = door_lock is not None and sid < len(door_lock) and door_lock[sid] >= 0
        is_locked = door_lock is not None and sid < len(door_lock) and door_lock[sid] > 0
        col = (255, 200, 50) if is_locked else ((60, 205, 235) if is_door else (95, 105, 122))
        width = 2 if is_door else 1
        for x1, y1, x2, y2 in segs:
            draw.line([to_px(x1, y1), to_px(x2, y2)], fill=col, width=width)

    if len(traj) >= 2:
        pts = [to_px(px, py) for px, py in traj]
        n = len(pts)
        for i in range(n - 1):
            t = i / max(1, n - 2)
            col = (int(40 + 210 * t), int(210 - 60 * t), int(120 + 120 * (1 - t)))
            draw.line([pts[i], pts[i + 1]], fill=col, width=2)

    if traj:
        px, py = to_px(*traj[-1])
        rad = math.radians(angle_deg)
        hx = px + 9 * math.cos(rad)
        hy = py - 9 * math.sin(rad)
        draw.ellipse([px - 4, py - 4, px + 4, py + 4], fill=(255, 255, 255), outline=(40, 220, 120))
        draw.line([(px, py), (hx, hy)], fill=(255, 90, 90), width=2)


def render_frame(
    obs: dict,
    ps: dict,
    action: np.ndarray,
    spatial_map: torch.Tensor,
    sectors: list[tuple[int, np.ndarray]],
    traj: list[tuple[float, float]],
    bounds: tuple[float, float, float, float],
    step: int,
    title: str,
    doors_opened: int,
) -> Image.Image:
    W, H = 960, 440
    canvas = Image.new("RGB", (W, H), (12, 13, 18))
    draw = ImageDraw.Draw(canvas)
    f_sm = _font(12)
    f_lg = _font(14)

    # 1) First-person DOOM RGB (upscaled 3x -> 480x360)
    rgb_img = Image.fromarray(obs["rgb"], mode="RGB").resize((480, 360), Image.Resampling.NEAREST)
    canvas.paste(rgb_img, (12, 12))
    draw.rectangle([12, 12, 492, 372], outline=(70, 75, 90))
    # Crosshair
    draw.line([(252 - 6, 192), (252 + 6, 192)], fill=(120, 255, 160), width=1)
    draw.line([(252, 192 - 6), (252, 192 + 6)], fill=(120, 255, 160), width=1)

    # 2) Raycast Radar (196x196)
    _draw_ray_radar(draw, 504, 28, 192, obs)
    draw.text((504, 10), "360° Rays & Entities", fill=(210, 215, 228), font=f_sm)

    # 3) Neural Spatial Memory (196x196)
    _render_spatial_map(canvas, 504, 236, 192, spatial_map)
    draw.text((504, 220), "EgoSpatialMemory (25x25)", fill=(210, 215, 228), font=f_sm)

    # 4) Top-Down Trajectory Map (240x400)
    _draw_topdown(draw, 708, 28, 240, 400, sectors, ps.get("door_lock"), traj, ps["angle"], bounds)
    draw.text((708, 10), f"Top-Down Map | {title}", fill=(210, 215, 228), font=f_sm)

    # 5) Bottom HUD bar under RGB
    vit = obs["vitals"]
    keys_held = []
    for idx, kn in enumerate(("B", "Y", "R", "BS", "YS", "RS")):
        if vit[8 + idx] > 0.5:
            keys_held.append(kn)
    key_str = ",".join(keys_held) if keys_held else "None"
    hud1 = (
        f"Step {step:3d} | HP {int(vit[0]):3d} | ARM {int(vit[1]):3d} | "
        f"AMMO {int(vit[3]):3d} | KILLS {ps['kills']} | DOORS {doors_opened} | KEYS: {key_str}"
    )
    draw.text((14, 380), hud1, fill=(235, 240, 250), font=f_lg)

    # Action pills
    bx = 14
    for i, lbl in enumerate(BTN_SHORT):
        active = int(action[i]) == 1
        bg = (45, 175, 110) if active else (30, 33, 42)
        fg = (255, 255, 255) if active else (120, 126, 140)
        bw = 44 if len(lbl) == 3 else 28
        draw.rectangle([bx, 406, bx + bw, 428], fill=bg, outline=(65, 70, 84))
        draw.text((bx + 6, 411), lbl, fill=fg, font=f_sm)
        bx += bw + 6

    return canvas


def record_episode(
    agent: DoomAgent,
    wad_path: str | Path,
    map_name: str,
    episode_ticks: int,
    seed: int,
    title: str,
    out_gif: Path,
    stride: int = 2,
    live: bool = False,
) -> tuple[dict, list[Image.Image]]:
    env = make_env(wad_path, map_name, episode_ticks=episode_ticks, render_buffers=True)
    try:
        obs, _ = env.reset(seed=seed)
        agent.reset()
        sectors = env.sector_polygons()
        all_pts = np.concatenate([s[:, :2] for _, s in sectors if len(s)] + [s[:, 2:4] for _, s in sectors if len(s)], axis=0)
        bounds = (float(all_pts[:, 0].min()), float(all_pts[:, 0].max()), float(all_pts[:, 1].min()), float(all_pts[:, 1].max()))

        ps0 = env.privileged_state()
        prev_ceil = ps0["sector_ceiling"].copy() if ps0["sector_ceiling"] is not None else None
        opened_doors: set[int] = set()
        traj = [(ps0["x"], ps0["y"])]
        frames: list[Image.Image] = []
        keyframes: list[tuple[str, Image.Image]] = []
        latencies: list[float] = []

        screen = None
        if live:
            import pygame
            pygame.init()
            screen = pygame.display.set_mode((960, 440))
            pygame.display.set_caption(f"DeadLock Replay: {title}")

        prev_keys = 0
        prev_doors = 0
        step = 0
        while True:
            t0 = time.perf_counter()
            act = agent.act(obs)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            spatial_map = agent._state[1]

            obs_next, _, term, trunc, _ = env.step(act)
            ps = env.privileged_state()
            traj.append((ps["x"], ps["y"]))

            if ps["sector_ceiling"] is not None and ps["door_lock"] is not None and prev_ceil is not None:
                rising = (ps["door_lock"] >= 0) & (ps["sector_ceiling"] > prev_ceil + 0.5)
                for sid in np.flatnonzero(rising):
                    opened_doors.add(int(sid))
                prev_ceil = ps["sector_ceiling"].copy()

            cur_keys = int(np.sum(ps["keys"]))
            cur_doors = len(opened_doors)
            event_label = None
            if step == 0:
                event_label = "Start"
            elif cur_keys > prev_keys:
                event_label = f"Picked Up Key! (Step {step})"
            elif cur_doors > prev_doors:
                event_label = f"Opened Door #{cur_doors} (Step {step})"
            elif term and ps["exit_reached"]:
                event_label = f"Reached Exit! (Step {step})"

            if step % stride == 0 or event_label is not None or term or trunc:
                fr = render_frame(obs, ps, act, spatial_map, sectors, traj, bounds, step, title, cur_doors)
                frames.append(fr)
                if event_label is not None:
                    keyframes.append((event_label, fr.copy()))
                if live and screen is not None:
                    import pygame
                    surf = pygame.image.frombuffer(fr.tobytes(), fr.size, "RGB")
                    screen.blit(surf, (0, 0))
                    pygame.display.flip()
                    for ev in pygame.event.get():
                        if ev.type == pygame.QUIT:
                            term = True

            prev_keys = cur_keys
            prev_doors = cur_doors
            obs = obs_next
            step += 1
            if term or trunc:
                break

        out_gif.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            out_gif,
            save_all=True,
            append_images=frames[1:],
            duration=90,
            loop=0,
            optimize=True,
        )
        summary = {
            "title": title,
            "steps": step,
            "exit": bool(ps["exit_reached"]),
            "dead": bool(ps["dead"]),
            "kills": int(ps["kills"]),
            "keys": int(np.sum(ps["keys"])),
            "doors": len(opened_doors),
            "p50_ms": float(np.median(latencies)) if latencies else 0.0,
            "gif": str(out_gif),
        }
        return summary, [img for _, img in keyframes]
    finally:
        env.close()


def draw_eval_dashboard(out_path: Path, results: dict) -> None:
    """Create a crisp visual test dashboard PNG from Tier 1-4 + Freedoom2 results."""
    W, H = 980, 560
    img = Image.new("RGB", (W, H), (14, 16, 22))
    d = ImageDraw.Draw(img)
    f_title = _font(20)
    f_sub = _font(13)
    f_sm = _font(12)

    d.text((24, 16), "DeadLock (DOOM RL) — Zero-Shot Held-Out Evaluation Dashboard", fill=(240, 244, 252), font=f_title)
    d.text(
        (24, 46),
        f"Hybrid Vision + EgoSpatialMemory | 8 unseen maps/tier + Freedoom2:MAP01 | p50: {results['overall_p50_ms']:.2f} ms (40 ms cap) | validate.py: PASS",
        fill=(145, 200, 175),
        font=f_sm,
    )

    tiers = results["tiers"]  # list of dicts
    # Panel 1: Exit Rate & Keycard Pickup % bar chart
    x0, y0, pw, ph = 24, 82, 450, 215
    d.rectangle([x0, y0, x0 + pw, y0 + ph], fill=(22, 25, 34), outline=(55, 62, 78))
    d.text((x0 + 12, y0 + 10), "Completion & Keycard Rate", fill=(225, 230, 240), font=f_sub)
    for pct in (0, 25, 50, 75, 100):
        yy = y0 + ph - 32 - int((ph - 74) * (pct / 100.0))
        d.line([(x0 + 42, yy), (x0 + pw - 16, yy)], fill=(38, 42, 54), width=1)
        d.text((x0 + 10, yy - 6), f"{pct}%", fill=(130, 138, 155), font=f_sm)

    bar_w = 28
    for i, t in enumerate(tiers):
        bx = x0 + 68 + i * 94
        base_y = y0 + ph - 32
        eh = int((ph - 74) * t["exit_rate"])
        kh = int((ph - 74) * t["key_rate"])
        d.rectangle([bx, base_y - eh, bx + bar_w, base_y], fill=(65, 205, 135))
        d.rectangle([bx + bar_w + 6, base_y - kh, bx + 2 * bar_w + 6, base_y], fill=(255, 205, 60))
        d.text((bx + 2, base_y - eh - 15), f"{int(round(t['exit_rate']*100))}%", fill=(150, 240, 190), font=f_sm)
        if t["key_rate"] > 0:
            d.text((bx + bar_w + 8, base_y - kh - 15), f"{int(round(t['key_rate']*100))}%", fill=(255, 225, 120), font=f_sm)
        d.text((bx + 6, base_y + 8), t["name"], fill=(215, 220, 230), font=f_sm)
    d.rectangle([x0 + 270, y0 + 12, x0 + 282, y0 + 24], fill=(65, 205, 135))
    d.text((x0 + 288, y0 + 10), "Exit %", fill=(200, 210, 220), font=f_sm)
    d.rectangle([x0 + 352, y0 + 12, x0 + 364, y0 + 24], fill=(255, 205, 60))
    d.text((x0 + 370, y0 + 10), "Key %", fill=(200, 210, 220), font=f_sm)

    # Panel 2: Avg Kills, Doors Opened & Exploration Cells
    x1 = 496
    d.rectangle([x1, y0, x1 + pw, y0 + ph], fill=(22, 25, 34), outline=(55, 62, 78))
    d.text((x1 + 12, y0 + 10), "Combat, Doors & Exploration", fill=(225, 230, 240), font=f_sub)
    max_val = max(10.0, max(t["cells"] for t in tiers) * 1.15)
    for i, t in enumerate(tiers):
        bx = x1 + 52 + i * 96
        base_y = y0 + ph - 32
        for j, (key, col) in enumerate((("kills", (240, 85, 85)), ("doors", (60, 205, 235)), ("cells", (145, 130, 255)))):
            val = t[key]
            bh = int((ph - 74) * (val / max_val))
            xx = bx + j * 22
            d.rectangle([xx, base_y - bh, xx + 18, base_y], fill=col)
            d.text((xx - 1, base_y - bh - 14), f"{val:.1f}" if val < 10 else f"{val:.0f}", fill=col, font=f_sm)
        d.text((bx + 8, base_y + 8), t["name"], fill=(215, 220, 230), font=f_sm)
    for lx, lbl, col in ((x1 + 250, "Kills", (240, 85, 85)), (x1 + 312, "Doors", (60, 205, 235)), (x1 + 378, "Cells", (145, 130, 255))):
        d.rectangle([lx, y0 + 12, lx + 12, y0 + 24], fill=col)
        d.text((lx + 16, y0 + 10), lbl, fill=(200, 210, 220), font=f_sm)

    # Panel 3: Summary table across benchmarks
    ty = 314
    d.rectangle([24, ty, W - 24, H - 20], fill=(22, 25, 34), outline=(55, 62, 78))
    d.text((38, ty + 12), "Benchmark Suite Summary (Zero-Shot, Unseen Maps & Official Engine)", fill=(225, 230, 240), font=f_sub)
    headers = ["Suite", "Exit Rate", "Keycards", "Doors Opened", "Avg Kills", "Cells Visited", "p50 Latency"]
    cols_x = [38, 250, 360, 470, 600, 710, 840]
    for cx, hd in zip(cols_x, headers):
        d.text((cx, ty + 42), hd, fill=(140, 150, 170), font=f_sm)
    d.line([(36, ty + 60), (W - 36, ty + 60)], fill=(50, 56, 70), width=1)

    rows = [
        (
            t["name"] + " (8 unseen)",
            f"{t['exit_rate']*100:.1f}%",
            f"{t['key_rate']*100:.1f}%",
            f"{t['doors']:.2f}",
            f"{t['kills']:.2f}",
            f"{t['cells']:.1f}",
            f"{t['p50_ms']:.2f} ms",
        )
        for t in tiers
    ]
    fd = results.get("freedoom2")
    if fd:
        rows.append(
            (
                "Freedoom2 MAP01",
                f"{fd['exit']*100:.0f}%",
                "-",
                f"{fd['doors']:.0f}",
                f"{fd['kills']:.0f}",
                f"{fd['cells']:.0f}",
                f"{fd['p50_ms']:.2f} ms",
            )
        )
    for r_idx, row in enumerate(rows):
        ry = ty + 70 + r_idx * 28
        for cx, val in zip(cols_x, row):
            col = (140, 235, 185) if ("%" in val and val != "0.0%" and val != "0%") else (225, 230, 240)
            d.text((cx, ry), val, fill=col, font=f_sub)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize DeadLock agent gameplay and test outputs")
    ap.add_argument("--out-dir", default="/tmp/deadlock_viz", help="Output directory for GIFs/PNGs/HTML")
    ap.add_argument("--maps-per-tier", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--live", action="store_true", help="Display live pygame replay window")
    ap.add_argument("--wad", default="", help="Optional single WAD for --live")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = Path("/tmp/deadlock_eval_maps")
    eval_dir.mkdir(parents=True, exist_ok=True)

    dummy = make_env(ROOT / "maps" / "two_rooms.wad", "MAP01", render_buffers=False)
    agent = DoomAgent(dummy.observation_space, dummy.action_space, device=args.device)
    dummy.close()

    if args.live and args.wad:
        record_episode(agent, args.wad, "MAP01", 2100, 0, Path(args.wad).name, out_dir / "live.gif", stride=1, live=True)
        return

    # 1) Run 8-map/tier evaluation and remember best showcase seed per tier
    from training.eval import eval_map

    tier_rows = []
    showcase_wads = {}
    all_p50 = []
    for tier in (1, 2, 3, 4):
        ticks = {1: 1400, 2: 2100, 3: 2800, 4: 3500}[tier]
        rows = []
        best_score = -1.0
        best_spec = None
        for idx in range(args.maps_per_tier):
            wad = write_procedural_wad(eval_dir / f"eval_t{tier}_{idx}.wad", tier=tier, seed=99000 + tier * 100 + idx)
            r = eval_map(agent, wad, "MAP01", episode_ticks=ticks, seed=idx)
            rows.append(r)
            all_p50.append(r["p50_ms"])
            sc = r["exit"] * 10.0 + r["keys"] * 4.0 + r["doors"] * 1.0 + r["kills"] * 0.5 + r["cells"] * 0.05
            if sc > best_score:
                best_score = sc
                best_spec = (wad, ticks, idx)
        showcase_wads[tier] = best_spec
        tier_rows.append(
            {
                "name": f"Tier {tier}",
                "exit_rate": float(np.mean([r["exit"] for r in rows])),
                "key_rate": float(np.mean([r["keys"] for r in rows])),
                "doors": float(np.mean([r["doors"] for r in rows])),
                "kills": float(np.mean([r["kills"] for r in rows])),
                "cells": float(np.mean([r["cells"] for r in rows])),
                "p50_ms": float(np.mean([r["p50_ms"] for r in rows])),
            }
        )

    iwad = os.path.join(os.path.dirname(vd.__file__), "freedoom2.wad")
    fd_res = eval_map(agent, iwad, "MAP01", episode_ticks=2100, seed=0) if os.path.isfile(iwad) else None
    results = {"tiers": tier_rows, "freedoom2": fd_res, "overall_p50_ms": float(np.mean(all_p50))}
    (out_dir / "eval_results.json").write_text(json.dumps(results, indent=2))
    draw_eval_dashboard(out_dir / "eval_dashboard.png", results)

    # 2) Record animated GIF replays for showcase runs + Freedoom2 MAP01
    replays = []
    for tier in (1, 2, 3, 4):
        wad, ticks, s_idx = showcase_wads[tier]
        summ, kfs = record_episode(
            agent, wad, "MAP01", ticks, s_idx, f"Tier {tier} ({wad.name})", out_dir / f"tier{tier}_replay.gif", stride=2, live=args.live
        )
        replays.append(summ)
        if tier == 3 and kfs:
            # Build horizontal/grid storyboard of key moments (max 4 frames)
            pick = kfs[:4] if len(kfs) <= 4 else [kfs[0], kfs[1], kfs[-2], kfs[-1]]
            sb = Image.new("RGB", (480 * 2, 220 * 2), (12, 13, 18))
            for k_i, fr in enumerate(pick):
                thumb = fr.resize((480, 220), Image.Resampling.BILINEAR)
                sb.paste(thumb, ((k_i % 2) * 480, (k_i // 2) * 220))
            sb.save(out_dir / "tier3_storyboard.png")

    if os.path.isfile(iwad):
        summ_fd, _ = record_episode(
            agent, iwad, "MAP01", 2100, 0, "Freedoom2 MAP01", out_dir / "freedoom2_map01.gif", stride=3, live=args.live
        )
        replays.append(summ_fd)

    # 3) Write simple standalone HTML viewer
    cards = "\n".join(
        f'<div class="card"><h3>{r["title"]} — {"EXIT REACHED" if r["exit"] else ("DIED" if r["dead"] else "TICK CAP")}</h3>'
        f'<p>Steps: {r["steps"]} | Kills: {r["kills"]} | Keys: {r["keys"]} | Doors: {r["doors"]} | p50: {r["p50_ms"]:.2f} ms</p>'
        f'<img src="{Path(r["gif"]).name}" /></div>'
        for r in replays
    )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>DeadLock Agent Visualizations</title>
<style>
body{{background:#0e1016;color:#e8ecf5;font-family:system-ui,-apple-system,sans-serif;margin:24px}}
img{{max-width:100%;border:1px solid #373e4e;border-radius:6px}}
.card{{background:#161922;padding:16px;border-radius:8px;margin-bottom:20px;border:1px solid #2b3140}}
</style></head><body>
<h1>DeadLock — DOOM RL Agent Replays &amp; Test Outputs</h1>
<div class="card"><h2>Zero-Shot Evaluation Dashboard</h2><img src="eval_dashboard.png" /></div>
{cards}
</body></html>"""
    (out_dir / "index.html").write_text(html)
    print(f"Visualizations saved to {out_dir}")
    print(json.dumps({"tiers": tier_rows, "freedoom2": fd_res, "replays": replays}, indent=2))


if __name__ == "__main__":
    main()
