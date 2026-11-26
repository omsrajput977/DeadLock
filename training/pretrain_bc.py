"""Training-time Privileged Teacher & Recurrent DAgger Pretraining for DoomSpatialPolicy.

Strictly training-side scaffolding (Rule §2 allows privileged state & topology during
training). Collects rollouts under scheduled DAgger mixing (beta: teacher -> policy),
trains `DoomSpatialPolicy` via truncated BPTT on factor actions, critic returns, and
auxiliary vitals prediction, and calibrates both door USE timing and target-aligned aim.
"""

from __future__ import annotations

import argparse
import math
import random
import tempfile
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from doom_env import make_env
from doom_env.wad import Lump, write_wad

from submission.model import (
    DoomSpatialPolicy,
    ModelConfig,
    factor_to_multibinary,
    mask_switch_logit,
    normalize_vitals,
    preprocess_obs_dict,
    sample_factor_actions,
)
from training.env_wrapper import FRONT_RAY_IDX, RAY_DOOR_ID, RAY_LOCKED_ID, strip_obs
from training.wad_gen import generate_map_with_layout


def _wrap_angle(rad: float) -> float:
    return (rad + math.pi) % (2.0 * math.pi) - math.pi


def _has_lock_key(keys: np.ndarray, lock_num: int) -> bool:
    # LOCK_RED=1 -> RedCard(2)/RedSkull(5); LOCK_BLUE=2 -> Blue(0/3); LOCK_YELLOW=3 -> Yellow(1/4)
    if lock_num <= 0:
        return True
    if lock_num == 1:
        return bool(keys[2] > 0 or keys[5] > 0)
    if lock_num == 2:
        return bool(keys[0] > 0 or keys[3] > 0)
    if lock_num == 3:
        return bool(keys[1] > 0 or keys[4] > 0)
    return True


class PrivilegedDoomTeacher:
    """BFS + tactical combat + door-USE choreography teacher (training-only)."""

    def __init__(self, layout: dict):
        self.layout = layout
        self.tile_size: float = float(layout["tile_size"])
        self.grid: dict[tuple[int, int], int] = layout["grid"]
        self.door_lock_map: dict[int, int] = layout["door_lock_map"]
        self.door_sectors: set[int] = layout["door_sectors"]
        self.key_items: list[tuple[float, float, int]] = layout["key_items"]
        self.exit_xy: tuple[int, int] = layout["exit_xy"]
        self.exit_switch_only: bool = bool(layout["exit_switch_only"])
        self.exit_wall_dir: tuple[int, int] | None = layout["exit_wall_dir"]

        self._prev_ceil: np.ndarray | None = None
        self._use_cd: int = 0
        self._stuck_steps: int = 0
        self._last_xy: tuple[float, float] | None = None
        self._step_idx: int = 0

    def _tile_passable(self, tile: tuple[int, int], keys: np.ndarray) -> bool:
        sid = self.grid.get(tile)
        if sid is None:
            return False
        lock = self.door_lock_map.get(sid, 0)
        return _has_lock_key(keys, lock)

    def _nearest_valid_tile(self, x: float, y: float, keys: np.ndarray) -> tuple[int, int] | None:
        ix, iy = int(x // self.tile_size), int(y // self.tile_size)
        if self._tile_passable((ix, iy), keys):
            return (ix, iy)
        best = None
        best_d = 1e12
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                cand = (ix + dx, iy + dy)
                if self._tile_passable(cand, keys):
                    cx, cy = (cand[0] + 0.5) * self.tile_size, (cand[1] + 0.5) * self.tile_size
                    d = (cx - x) ** 2 + (cy - y) ** 2
                    if d < best_d:
                        best_d = d
                        best = cand
        return best

    def _bfs_path(self, start: tuple[int, int], goals: set[tuple[int, int]], keys: np.ndarray) -> list[tuple[int, int]]:
        if start in goals:
            return [start]
        q = deque([start])
        parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        found = None
        while q:
            cur = q.popleft()
            if cur in goals:
                found = cur
                break
            cx, cy = cur
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (cx + dx, cy + dy)
                if nxt not in parent and self._tile_passable(nxt, keys):
                    parent[nxt] = cur
                    q.append(nxt)
        if found is None:
            return [start]
        path = []
        node: tuple[int, int] | None = found
        while node is not None:
            path.append(node)
            node = parent[node]
        path.reverse()
        return path

    def act(self, obs: dict, ps: dict) -> tuple[np.ndarray, np.ndarray]:
        self._step_idx += 1
        if self._use_cd > 0:
            self._use_cd -= 1

        x, y = float(ps["x"]), float(ps["y"])
        heading = math.radians(float(ps["angle"]))
        keys = ps["keys"]
        vitals = obs["vitals"]
        ceil = ps["sector_ceiling"]
        floor = ps["sector_floor"]
        if self._prev_ceil is None and ceil is not None:
            self._prev_ceil = ceil.copy()

        moved = 999.0 if self._last_xy is None else math.hypot(x - self._last_xy[0], y - self._last_xy[1])
        self._last_xy = (x, y)

        start = self._nearest_valid_tile(x, y, keys)
        path: list[tuple[int, int]] = []
        if start is not None:
            # 1) Prefer uncollected keycard if still unheld and reachable
            key_goals: set[tuple[int, int]] = set()
            for kx, ky, lock_num in self.key_items:
                if not _has_lock_key(keys, lock_num):
                    kt = (int(kx // self.tile_size), int(ky // self.tile_size))
                    if self._tile_passable(kt, keys):
                        key_goals.add(kt)
            if key_goals:
                cand_path = self._bfs_path(start, key_goals, keys)
                if len(cand_path) >= 1 and cand_path[-1] in key_goals:
                    path = cand_path
            if not path:
                path = self._bfs_path(start, {self.exit_xy}, keys)
        else:
            path = [self.exit_xy]

        # Detect closed door immediately ahead along BFS path
        closed_door_xy: tuple[float, float] | None = None
        closed_door_rising = False
        if ceil is not None and floor is not None:
            for tile in path[1:3]:
                sid = self.grid.get(tile)
                if sid is not None and sid in self.door_sectors:
                    c_val = float(ceil[sid])
                    f_val = float(floor[sid])
                    prev_c = float(self._prev_ceil[sid]) if self._prev_ceil is not None else c_val
                    if c_val < f_val + 52.0:
                        closed_door_xy = ((tile[0] + 0.5) * self.tile_size, (tile[1] + 0.5) * self.tile_size)
                        closed_door_rising = c_val > prev_c + 0.5
                        break

        # Detect exit switch interaction at wall edge
        at_exit_switch = False
        switch_target_xy: tuple[float, float] | None = None
        if self.exit_switch_only and self.exit_wall_dir is not None and start == self.exit_xy and closed_door_xy is None:
            ex_wx = (self.exit_xy[0] + 0.5 + 0.45 * self.exit_wall_dir[0]) * self.tile_size
            ex_wy = (self.exit_xy[1] + 0.5 + 0.45 * self.exit_wall_dir[1]) * self.tile_size
            switch_target_xy = (ex_wx, ex_wy)
            at_exit_switch = True

        # Waypoint world target
        if at_exit_switch and switch_target_xy is not None:
            tx, ty = switch_target_xy
        elif closed_door_xy is not None:
            tx, ty = closed_door_xy
        else:
            wp_idx = 2 if len(path) >= 3 and _line_walkable(self.grid, self.door_lock_map, keys, path[0], path[2]) else min(1, len(path) - 1)
            wt = path[wp_idx]
            tx = (wt[0] + 0.5) * self.tile_size
            ty = (wt[1] + 0.5) * self.tile_size

            # Opportunistic pickup (medkit when injured, shotgun, or nearby ammo)
            ents = obs["visible_entities"]
            best_item = None
            best_idist = 180.0
            for k in range(len(ents["mask"])):
                if ents["mask"][k] < 0.5:
                    continue
                et = int(ents["entity_type"][k])
                edist = float(ents["features"][k, 0]) * 1024.0
                cos_a = float(ents["features"][k, 2])
                if cos_a < 0.4 or edist > best_idist:
                    continue
                want_pickup = (et == 5 and float(vitals[0]) < 78.0) or (et in (7, 8) and float(vitals[3]) < 16.0) or (et == 6 and float(vitals[1]) < 40.0)
                if want_pickup:
                    sin_a = float(ents["features"][k, 1])
                    item_ang = heading + math.atan2(sin_a, cos_a)
                    best_item = (x + edist * math.cos(item_ang), y + edist * math.sin(item_ang))
                    best_idist = edist
            if best_item is not None:
                tx, ty = best_item

        nav_ang = math.atan2(ty - y, tx - x)
        nav_err = _wrap_angle(nav_ang - heading)
        target_dist = math.hypot(tx - x, ty - y)

        # Front door proximity check from raw rays too (matches policy observation)
        ft = obs["rays"]["hit_type"][FRONT_RAY_IDX]
        fd = obs["rays"]["distance"][FRONT_RAY_IDX]
        door_rays = (ft == RAY_DOOR_ID) | (ft == RAY_LOCKED_ID)
        min_front_door_units = float(fd[door_rays].min()) * 1024.0 if door_rays.any() else 9999.0

        # Nearest visible enemy
        ents = obs["visible_entities"]
        enemy_dist = 9999.0
        enemy_sin = 0.0
        enemy_cos = 1.0
        has_enemy = False
        for k in range(len(ents["mask"])):
            if ents["mask"][k] < 0.5:
                continue
            et = int(ents["entity_type"][k])
            if et in (1, 2, 3):
                d_u = float(ents["features"][k, 0]) * 1024.0
                if d_u < enemy_dist:
                    enemy_dist = d_u
                    enemy_sin = float(ents["features"][k, 1])
                    enemy_cos = float(ents["features"][k, 2])
                    has_enemy = True

        move = 0    # 0 idle, 1 fwd, 2 back
        strafe = 0  # 0 idle, 1 left, 2 right
        turn = 0    # 0 idle, 1 left, 2 right
        atk = 0
        use = 0
        sw = 1 if (float(vitals[3]) <= 0.0 and float(np.sum(vitals[4:8])) > 0.0) else 0

        waiting_for_door = False
        if closed_door_xy is not None and target_dist < 84.0:
            waiting_for_door = True
            if nav_err > 0.20:
                turn = 1
            elif nav_err < -0.20:
                turn = 2
            if not closed_door_rising and self._use_cd == 0 and abs(nav_err) < 0.50:
                use = 1
                self._use_cd = 11
            elif target_dist > 56.0:
                move = 1
        elif at_exit_switch and target_dist < 56.0:
            waiting_for_door = True
            if nav_err > 0.16:
                turn = 1
            elif nav_err < -0.16:
                turn = 2
            if self._use_cd == 0 and abs(nav_err) < 0.40:
                use = 1
                self._use_cd = 10
            elif target_dist > 34.0:
                move = 1
        else:
            # Combat + navigation
            engage_enemy = has_enemy and enemy_dist < 560.0 and (enemy_dist < min_front_door_units - 16.0)
            if engage_enemy:
                rel_ang = math.atan2(enemy_sin, enemy_cos)
                if rel_ang > 0.09:
                    turn = 1
                elif rel_ang < -0.09:
                    turn = 2
                if abs(rel_ang) < 0.18 and enemy_cos > 0.0 and sw == 0:
                    atk = 1
                if enemy_dist < 130.0:
                    move = 2
                    strafe = 1 if (self._step_idx // 10) % 2 == 0 else 2
                elif abs(nav_err) < 0.85:
                    move = 1
                    if enemy_dist < 300.0:
                        strafe = 1 if (self._step_idx // 12) % 2 == 0 else 2
            else:
                if nav_err > 0.14:
                    turn = 1
                elif nav_err < -0.14:
                    turn = 2
                if abs(nav_err) < 0.90:
                    move = 1

        if not waiting_for_door and move == 1 and moved < 1.8:
            self._stuck_steps += 1
        else:
            self._stuck_steps = 0
        if self._stuck_steps >= 4:
            strafe = 1 if ((self._step_idx // 6) % 2 == 0) else 2
            if turn == 0:
                turn = 1 if nav_err >= 0 else 2

        if ceil is not None:
            self._prev_ceil = ceil.copy()

        factor_act = np.array([move, strafe, turn, atk, use, sw], dtype=np.int64)
        mb = factor_to_multibinary(torch.from_numpy(factor_act)).numpy().astype(np.int8)
        return factor_act, mb


def _line_walkable(grid: dict, door_lock_map: dict, keys: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> bool:
    for tx in range(min(a[0], b[0]), max(a[0], b[0]) + 1):
        for ty in range(min(a[1], b[1]), max(a[1], b[1]) + 1):
            sid = grid.get((tx, ty))
            if sid is None or not _has_lock_key(keys, door_lock_map.get(sid, 0)):
                return False
    return True


def _make_wad_and_teacher(tier: int, seed: int, wad_path: Path) -> tuple[Path, dict]:
    text, layout = generate_map_with_layout(tier=tier, seed=seed)
    write_wad(wad_path, [Lump("MAP01", b""), Lump("TEXTMAP", text.encode("utf-8")), Lump("ENDMAP", b"")])
    return wad_path, layout


def test_teacher(maps_per_tier: int = 5) -> None:
    tmp_dir = Path(tempfile.mkdtemp(prefix="deadlock_teacher_"))
    for tier in (1, 2, 3, 4):
        ticks = {1: 1400, 2: 2100, 3: 2800, 4: 3500}[tier]
        exits, kills, keys, doors, shot_at_door = [], [], [], [], []
        for idx in range(maps_per_tier):
            wad_path, layout = _make_wad_and_teacher(tier, seed=70000 + tier * 100 + idx, wad_path=tmp_dir / f"t{tier}_{idx}.wad")
            env = make_env(wad_path, "MAP01", episode_ticks=ticks, render_buffers=False)
            obs, _ = env.reset(seed=idx)
            teacher = PrivilegedDoomTeacher(layout)
            done = False
            sad = 0
            opened = set()
            prev_c = env.privileged_state()["sector_ceiling"]
            while not done:
                ps = env.privileged_state()
                _, act = teacher.act(obs, ps)
                ft = obs["rays"]["hit_type"][FRONT_RAY_IDX]
                fd = obs["rays"]["distance"][FRONT_RAY_IDX]
                ev = ((obs["visible_entities"]["mask"] == 1) & np.isin(obs["visible_entities"]["entity_type"], [1, 2, 3])).any()
                if act[6] == 1 and not ev and (((ft == RAY_DOOR_ID) | (ft == RAY_LOCKED_ID)) & (fd < 0.25)).any():
                    sad += 1
                obs, _, term, trunc, _ = env.step(act)
                ps2 = env.privileged_state()
                if prev_c is not None and ps2["sector_ceiling"] is not None and ps2["door_lock"] is not None:
                    rising = (ps2["door_lock"] >= 0) & (ps2["sector_ceiling"] > prev_c + 0.5)
                    for sid in np.flatnonzero(rising):
                        opened.add(int(sid))
                    prev_c = ps2["sector_ceiling"].copy()
                done = term or trunc
            ps = env.privileged_state()
            env.close()
            exits.append(1.0 if ps["exit_reached"] else 0.0)
            kills.append(ps["kills"])
            keys.append(int(np.sum(ps["keys"])))
            doors.append(len(opened))
            shot_at_door.append(sad)
        print(
            f"[Teacher T{tier}] exit={np.mean(exits):.2f} kills={np.mean(kills):.2f} "
            f"keys={np.mean(keys):.2f} doors={np.mean(doors):.2f} door_shots={np.mean(shot_at_door):.2f}"
        )


def _discounted_returns(rews: list[float], gamma: float = 0.995) -> np.ndarray:
    g = 0.0
    out = np.zeros(len(rews), dtype=np.float32)
    for t in reversed(range(len(rews))):
        g = rews[t] + gamma * g
        out[t] = g
    return out


def collect_dagger_episodes(
    policy: DoomSpatialPolicy,
    device: torch.device,
    beta: float,
    num_episodes: int,
    seed_offset: int,
    seq_len: int = 64,
) -> tuple[list[tuple], dict]:
    tmp_dir = Path(tempfile.mkdtemp(prefix="deadlock_bc_"))
    sequences: list[tuple] = []
    exits, kills, keys = [], [], []
    policy.eval()

    tier_weights = (1, 2, 3, 3, 4, 4)
    rng = random.Random(seed_offset)
    for ep_i in range(num_episodes):
        tier = tier_weights[ep_i % len(tier_weights)]
        ticks = {1: 1200, 2: 1800, 3: 2400, 4: 2800}[tier]
        wad_path, layout = _make_wad_and_teacher(tier, seed=seed_offset + ep_i, wad_path=tmp_dir / f"ep_{ep_i}.wad")
        env = make_env(wad_path, "MAP01", episode_ticks=ticks, render_buffers=True)
        obs, _ = env.reset(seed=ep_i)
        teacher = PrivilegedDoomTeacher(layout)
        state = policy.init_state(1, device)

        ep_obs, ep_prev_act, ep_teacher_f, ep_exec_act, ep_rews, ep_next_v, ep_dones = [], [], [], [], [], [], []
        prev_ps = env.privileged_state()
        done = False
        while not done:
            s_obs = strip_obs(obs)
            t_factor, t_mb = teacher.act(obs, prev_ps)
            obs_t = preprocess_obs_dict(s_obs, device)
            with torch.no_grad():
                logits, _, _, next_state = policy.forward_step(obs_t, state)
                _, p_act_t, _ = sample_factor_actions(logits, obs_t["vitals"], deterministic=False, temperature=0.55)
            p_mb = p_act_t.squeeze(0).cpu().numpy().astype(np.int8)

            exec_mb = t_mb if rng.random() < beta else p_mb
            ep_obs.append(s_obs)
            ep_prev_act.append(state[2].squeeze(0).cpu().clone())
            ep_teacher_f.append(torch.from_numpy(t_factor.copy()))
            exec_t = torch.from_numpy(exec_mb.astype(np.float32)).unsqueeze(0).to(device)
            ep_exec_act.append(exec_t.squeeze(0).cpu().clone())

            next_obs, base_r, term, trunc, _ = env.step(exec_mb)
            done = term or trunc
            ps2 = env.privileged_state()
            shaped_r = float(base_r) + 0.4 * float(ps2["kills"] - prev_ps["kills"]) + 2.0 * float(np.sum(ps2["keys"] - prev_ps["keys"]))
            if ps2["exit_reached"]:
                shaped_r += 8.0
            ep_rews.append(shaped_r)
            ep_next_v.append(torch.from_numpy(np.ascontiguousarray(next_obs["vitals"], dtype=np.float32)))
            ep_dones.append(1.0 if done else 0.0)

            state = policy.advance_state(next_state, exec_t, torch.tensor([1.0 if done else 0.0], device=device))
            obs = next_obs
            prev_ps = ps2

        exits.append(1.0 if prev_ps["exit_reached"] else 0.0)
        kills.append(prev_ps["kills"])
        keys.append(int(np.sum(prev_ps["keys"])))
        env.close()

        rets = _discounted_returns(ep_rews)
        n_steps = len(ep_obs)
        for start in range(0, n_steps, seq_len):
            end = min(n_steps, start + seq_len)
            if end - start < 8:
                continue
            sequences.append(
                (
                    ep_obs[start:end],
                    torch.stack(ep_prev_act[start:end]),
                    torch.stack(ep_teacher_f[start:end]),
                    torch.stack(ep_exec_act[start:end]),
                    torch.from_numpy(rets[start:end]),
                    torch.stack(ep_next_v[start:end]),
                    torch.tensor(ep_dones[start:end], dtype=torch.float32),
                )
            )
    stats = {"exit": float(np.mean(exits)), "kills": float(np.mean(kills)), "keys": float(np.mean(keys))}
    return sequences, stats


def train_bc(args: argparse.Namespace) -> None:
    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = ModelConfig()
    if args.resume and Path(args.resume).is_file():
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        if isinstance(ckpt, dict) and "config" in ckpt:
            cfg = ModelConfig(**ckpt["config"])
        policy = DoomSpatialPolicy(cfg).to(device)
        policy.load_state_dict(ckpt["model"] if "model" in ckpt else ck, strict=True)
        print(f"Resumed BC from {args.resume}")
    else:
        policy = DoomSpatialPolicy(cfg).to(device)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)
    replay_buffer: list[tuple] = []
    betas = [float(b.strip()) for b in args.betas.split(",") if b.strip()]
    total_steps = 0

    for rnd in range(args.rounds):
        beta = betas[min(rnd, len(betas) - 1)]
        cur_lr = args.lr * (0.35 + 0.65 * (1.0 - rnd / max(1, args.rounds - 1)))
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr
        t0 = time.perf_counter()
        seqs, stats = collect_dagger_episodes(
            policy, device=device, beta=beta, num_episodes=args.episodes_per_round, seed_offset=10000 + rnd * 1000, seq_len=args.seq_len
        )
        replay_buffer.extend(seqs)
        if len(replay_buffer) > args.max_buffer_seqs:
            replay_buffer = replay_buffer[-args.max_buffer_seqs :]
        total_steps += sum(len(s[0]) for s in seqs)

        policy.train()
        use_pos_w = torch.tensor([2.5], device=device)
        atk_pos_w = torch.tensor([1.5], device=device)
        losses = []
        for _epoch in range(args.bc_epochs):
            order = list(range(len(replay_buffer)))
            random.shuffle(order)
            for s_idx in range(0, len(order), args.batch_seqs):
                batch_ids = order[s_idx : s_idx + args.batch_seqs]
                if len(batch_ids) < 2:
                    continue
                min_t = min(len(replay_buffer[i][0]) for i in batch_ids)
                b_len = len(batch_ids)

                # Collate aligned sequences of length `min_t`
                obs_seq_list = []
                for t in range(min_t):
                    obs_seq_list.append({
                        "rgb": torch.stack([torch.from_numpy(replay_buffer[i][0][t]["rgb"]) for i in batch_ids]),
                        "depth": torch.stack([torch.from_numpy(replay_buffer[i][0][t]["depth"]) for i in batch_ids]),
                        "vitals": torch.stack([torch.from_numpy(replay_buffer[i][0][t]["vitals"]) for i in batch_ids]),
                        "ray_dist": torch.stack([torch.from_numpy(replay_buffer[i][0][t]["rays"]["distance"]) for i in batch_ids]),
                        "ray_type": torch.stack([torch.from_numpy(replay_buffer[i][0][t]["rays"]["hit_type"]) for i in batch_ids]),
                        "ent_feat": torch.stack([torch.from_numpy(replay_buffer[i][0][t]["visible_entities"]["features"]) for i in batch_ids]),
                        "ent_type": torch.stack([torch.from_numpy(replay_buffer[i][0][t]["visible_entities"]["entity_type"]) for i in batch_ids]),
                        "ent_mask": torch.stack([torch.from_numpy(replay_buffer[i][0][t]["visible_entities"]["mask"]) for i in batch_ids]),
                    })

                flat_obs = {k: torch.cat([step[k] for step in obs_seq_list], dim=0).to(device) for k in obs_seq_list[0]}
                prev_act_seq = torch.stack([replay_buffer[i][1][:min_t] for i in batch_ids], dim=1)  # (T, B, 9)
                flat_prev = prev_act_seq.reshape(min_t * b_len, -1).to(device)
                r_all, vraw_all, vis_all, s_all, ef_all, et_all, em_all = policy.encode_inputs(flat_obs, flat_prev)
                r_seq = r_all.view(min_t, b_len, -1)
                vraw_seq = vraw_all.view(min_t, b_len, -1)
                vis_seq = vis_all.view(min_t, b_len, -1)
                s_seq = s_all.view(min_t, b_len, *s_all.shape[1:])
                ef_seq = ef_all.view(min_t, b_len, *ef_all.shape[1:])
                et_seq = et_all.view(min_t, b_len, *et_all.shape[1:])
                em_seq = em_all.view(min_t, b_len, *em_all.shape[1:])
                vit_seq = torch.stack([step["vitals"] for step in obs_seq_list], dim=0).to(device)

                exec_seq = torch.stack([replay_buffer[i][3][:min_t] for i in batch_ids], dim=1).to(device)
                done_seq = torch.stack([replay_buffer[i][6][:min_t] for i in batch_ids], dim=1).to(device)
                state = policy.init_state(b_len, device)

                logits_seq, vals_seq, aux_seq = [], [], []
                for t in range(min_t):
                    lg, vl, ax, nst = policy.recurrent_step(
                        r_seq[t], vraw_seq[t], vis_seq[t], s_seq[t], ef_seq[t], et_seq[t], em_seq[t], vit_seq[t], state
                    )
                    logits_seq.append(lg)
                    vals_seq.append(vl)
                    aux_seq.append(ax)
                    state = policy.advance_state(nst, exec_seq[t], done_seq[t])

                lg_all = mask_switch_logit(torch.stack(logits_seq), vit_seq)  # (T, B, 12)
                vl_all = torch.stack(vals_seq)
                ax_all = torch.stack(aux_seq)

                tf_seq = torch.stack([replay_buffer[i][2][:min_t] for i in batch_ids], dim=1).to(device)
                ret_seq = torch.stack([replay_buffer[i][4][:min_t] for i in batch_ids], dim=1).to(device)
                nv_seq = torch.stack([replay_buffer[i][5][:min_t] for i in batch_ids], dim=1).to(device)

                loss_m = F.cross_entropy(lg_all[..., 0:3].reshape(-1, 3), tf_seq[..., 0].reshape(-1))
                loss_s = F.cross_entropy(lg_all[..., 3:6].reshape(-1, 3), tf_seq[..., 1].reshape(-1))
                loss_t = F.cross_entropy(lg_all[..., 6:9].reshape(-1, 3), tf_seq[..., 2].reshape(-1))
                loss_atk = F.binary_cross_entropy_with_logits(
                    lg_all[..., 9].reshape(-1), tf_seq[..., 3].float().reshape(-1), pos_weight=atk_pos_w
                )
                loss_use = F.binary_cross_entropy_with_logits(
                    lg_all[..., 10].reshape(-1), tf_seq[..., 4].float().reshape(-1), pos_weight=use_pos_w
                )
                loss_sw = F.binary_cross_entropy_with_logits(
                    lg_all[..., 11].reshape(-1), tf_seq[..., 5].float().reshape(-1)
                )
                vf_loss = F.smooth_l1_loss(vl_all, ret_seq)
                target_v = normalize_vitals(nv_seq)
                nonterm = (1.0 - done_seq).unsqueeze(-1)
                aux_loss = (F.mse_loss(ax_all, target_v, reduction="none") * nonterm).mean()

                loss = loss_m + loss_s + loss_t + 1.2 * loss_atk + 2.0 * loss_use + 0.5 * loss_sw + 0.25 * vf_loss + 0.1 * aux_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                losses.append(loss.item())

        dt = time.perf_counter() - t0
        print(
            f"[DAgger round {rnd + 1}/{args.rounds} | beta={beta:.2f} | steps={total_steps} | {dt:.1f}s] "
            f"rollout_exit={stats['exit']:.2f} kills={stats['kills']:.2f} keys={stats['keys']:.2f} loss={np.mean(losses):.3f}"
        )
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": policy.state_dict(), "config": asdict(policy.cfg), "step": total_steps}, out_p)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recurrent DAgger pretraining for DoomSpatialPolicy")
    p.add_argument("--test-teacher", action="store_true", help="Benchmark privileged teacher before training")
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--betas", type=str, default="1.0,0.75,0.50,0.30,0.15,0.05")
    p.add_argument("--episodes-per-round", type=int, default=80)
    p.add_argument("--bc-epochs", type=int, default=4)
    p.add_argument("--batch-seqs", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--max-buffer-seqs", type=int, default=1200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--out", type=str, default="/tmp/deadlock_ckpts/dagger_v2.pt")
    return p.parse_args(argv)


if __name__ == "__main__":
    a = parse_args()
    if a.test_teacher:
        test_teacher()
    else:
        train_bc(a)
