"""Training environment wrapper with dense progression & exploration reward shaping.

Uses `env.privileged_state()` (explicitly allowed during training) to compute:
- Episodic area coverage (96x96 unit cells visited, matching scorer coverage metric)
- Monotonic keycard-approach & locked-door-approach shaping (non-farmable)
- Door opening, keycard pickup, locked-door unlocking, and level exit bonuses
- Combat kills, item pickups, damage penalties, and anti-door-bounce shaping
Uses a bounded LRU cache (`max_cached=4` per worker) to stay comfortably within 16 GB RAM.
"""

from __future__ import annotations

import gc
import os
import random
from collections import OrderedDict
from pathlib import Path

import gymnasium as gym
import numpy as np
import vizdoom as vd
from doom_env import make_env
from doom_env import spaces as S

from training.wad_gen import build_map_pool

CELL_SIZE = 128.0  # Tile-scale exploration grid (rewards reaching corners, switches & exits)
KEY_ENTITY_ID = S.ENTITY_TYPES.index("key")
ENEMY_ENTITY_IDS = {
    S.ENTITY_TYPES.index("enemy_melee"),
    S.ENTITY_TYPES.index("enemy_hitscan"),
    S.ENTITY_TYPES.index("enemy_projectile"),
}
RAY_DOOR_ID = 2
RAY_LOCKED_ID = 3  # env/doom_env/sensors.py RAY_LOCKED
RAY_ENEMY_ID = 4
FRONT_RAY_IDX = np.array([60, 61, 62, 63, 0, 1, 2, 3, 4], dtype=np.int64)
FREEDOOM2_IWAD = os.path.join(os.path.dirname(vd.__file__), "freedoom2.wad")
FREEDOOM_STAGE4_MAPS = ("MAP01", "MAP02", "MAP03", "MAP04", "MAP05", "MAP06")


def stripped_obs_space() -> gym.spaces.Dict:
    full = S.observation_space()
    return gym.spaces.Dict(
        {
            "rgb": gym.spaces.Box(0, 255, (60, 80, 3), np.uint8),
            "vitals": full["vitals"],
            "rays": full["rays"],
            "visible_entities": full["visible_entities"],
        }
    )


def strip_obs(obs: dict) -> dict:
    return {
        "rgb": np.ascontiguousarray(obs["rgb"][::2, ::2, :]),
        "vitals": obs["vitals"],
        "rays": obs["rays"],
        "visible_entities": obs["visible_entities"],
    }


class ShapedDoomEnv(gym.Env):
    """Wraps a pool of WAD maps with curriculum tier sampling and dense reward shaping."""

    def __init__(
        self,
        map_entries: list[tuple[str, str]],
        episode_ticks: int = 2100,
        seed: int = 0,
        max_cached: int = 4,
    ):
        super().__init__()
        if not map_entries:
            raise ValueError("map_entries must be non-empty")
        self.map_entries = list(map_entries)
        self.episode_ticks = int(episode_ticks)
        self.max_cached = max(1, int(max_cached))
        self.rng = random.Random(seed)
        self.observation_space = stripped_obs_space()
        self.action_space = S.action_space()

        self._envs: OrderedDict[tuple[str, str], gym.Env] = OrderedDict()
        self._cur_env = None
        self._visited: set[tuple[int, int]] = set()
        self._opened_doors: set[int] = set()
        self._prev_ps: dict | None = None
        self._prev_use: int = 0
        self._prev_door_ceil: np.ndarray | None = None
        self._saw_key: bool = False
        self._best_key_dist: float = 10.0
        self._saw_locked_with_key: bool = False
        self._best_locked_dist: float = 2.0

    def _get_env(self, wad: str, map_name: str):
        key = (wad, map_name)
        env = self._envs.get(key)
        if env is not None:
            self._envs.move_to_end(key)
            return env
        while len(self._envs) >= self.max_cached:
            _, old_env = self._envs.popitem(last=False)
            old_env.close()
            gc.collect()
        env = make_env(wad, map_name, episode_ticks=self.episode_ticks, render_buffers=True)
        self._envs[key] = env
        return env

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng.seed(seed)
        wad, map_name = self.rng.choice(self.map_entries)
        self._cur_env = self._get_env(wad, map_name)
        ep_seed = self.rng.randrange(0, 2**31 - 1)
        obs, info = self._cur_env.reset(seed=ep_seed)
        ps = self._cur_env.privileged_state()

        self._visited = {(int(ps["x"] // CELL_SIZE), int(ps["y"] // CELL_SIZE))}
        self._all_visited = set(self._visited)
        self._post_key_mode = False
        self._opened_doors = set()
        self._prev_ps = ps
        self._prev_use = 0
        self._prev_door_ceil = ps["sector_ceiling"].copy() if ps["sector_ceiling"] is not None else None
        self._saw_key = False
        self._best_key_dist = 10.0
        self._saw_door = False
        self._best_door_dist = 2.0
        self._saw_locked_with_key = False
        self._best_locked_dist = 2.0
        return strip_obs(obs), info

    def step(self, action):
        act = np.asarray(action, dtype=np.int8)
        obs, _, terminated, truncated, info = self._cur_env.step(act)
        ps = self._cur_env.privileged_state()
        prev = self._prev_ps

        reward = -0.001  # small time-efficiency pressure
        # Urgency cost when out of combat & anti-wall-stall shaping (discourages circling/hugging walls)
        ents = obs["visible_entities"]
        enemy_visible = ((ents["mask"] == 1) & np.isin(ents["entity_type"], list(ENEMY_ENTITY_IDS))).any()
        enemy_in_front = (obs["rays"]["hit_type"][FRONT_RAY_IDX] == RAY_ENEMY_ID).any()
        if not enemy_visible and not enemy_in_front:
            reward -= 0.003
            if abs(float(obs["vitals"][14])) < 1.5 and abs(float(obs["vitals"][15])) < 1.5:
                reward -= 0.015
        if self._prev_use == 1 and int(act[7]) == 1:
            # USE is rising-edge button-down: holding it continuously is ignored by DOOM
            reward -= 0.015

        # 1) Tile-scale episodic exploration (128x128 cells): rewards sweeping rooms & corners
        #    and resets exploration breadcrumbs once a keycard is picked up to reward backtracking.
        cell = (int(ps["x"] // CELL_SIZE), int(ps["y"] // CELL_SIZE))
        self._all_visited.add(cell)
        if cell not in self._visited:
            self._visited.add(cell)
            reward += 0.05 if self._post_key_mode else 0.08

        # 2) Combat & ammo discipline (reward decisive kills & survival pickups without timidity)
        dkills = ps["kills"] - prev["kills"]
        if dkills > 0:
            reward += 0.75 * dkills
        ditems = ps["items"] - prev["items"]
        if ditems > 0:
            reward += 0.12 * ditems
        dheal = max(0.0, ps["health"] - prev["health"])
        if dheal > 0:
            reward += 0.25
        ddmg = max(0.0, ps["damage_taken"] - prev["damage_taken"])
        if ddmg > 0:
            reward -= 0.010 * ddmg

        if act[6] == 1 and not enemy_visible and not enemy_in_front:
            reward -= 0.008

        # 3) Keycards sighting, monotonic approach & pickup
        key_slots = (ents["mask"] == 1) & (ents["entity_type"] == KEY_ENTITY_ID)
        if key_slots.any():
            if not self._saw_key:
                self._saw_key = True
                reward += 0.30
            min_kdist = float(ents["features"][key_slots, 0].min())
            if min_kdist < self._best_key_dist:
                if self._best_key_dist < 10.0:
                    reward += 0.5 * (self._best_key_dist - min_kdist)
                self._best_key_dist = min_kdist

        dkeys = float(np.sum(ps["keys"] - prev["keys"]))
        if dkeys > 0:
            reward += 2.5 * dkeys
            self._saw_key = False
            self._best_key_dist = 10.0
            self._visited = {cell}
            self._post_key_mode = True

        # 4) Normal doors & locked doors: front-cone approach + opening
        front_types = obs["rays"]["hit_type"][FRONT_RAY_IDX]
        front_dists = obs["rays"]["distance"][FRONT_RAY_IDX]

        front_doors = front_types == RAY_DOOR_ID
        if front_doors.any():
            if not self._saw_door:
                self._saw_door = True
                reward += 0.10
            min_ddist = float(front_dists[front_doors].min())
            if min_ddist < self._best_door_dist:
                if self._best_door_dist < 2.0:
                    reward += 0.45 * (self._best_door_dist - min_ddist)
                self._best_door_dist = min_ddist

        if np.sum(ps["keys"]) > 0:
            locked_rays = obs["rays"]["hit_type"] == RAY_LOCKED_ID
            if locked_rays.any():
                if not self._saw_locked_with_key:
                    self._saw_locked_with_key = True
                    reward += 0.30
                min_ldist = float(obs["rays"]["distance"][locked_rays].min())
                if min_ldist < self._best_locked_dist:
                    if self._best_locked_dist < 2.0:
                        reward += 0.6 * (self._best_locked_dist - min_ldist)
                    self._best_locked_dist = min_ldist

        ceil = ps["sector_ceiling"]
        dlock = ps["door_lock"]
        if ceil is not None and dlock is not None and self._prev_door_ceil is not None:
            rising = (dlock >= 0) & (ceil > self._prev_door_ceil + 0.5)
            for sid in np.flatnonzero(rising):
                sid_int = int(sid)
                if sid_int not in self._opened_doors:
                    self._opened_doors.add(sid_int)
                    reward += 3.5 if dlock[sid_int] > 0 else 0.70
                    self._best_door_dist = 2.0
                    self._saw_door = False
            closing_door = (dlock >= 0) & (ceil < self._prev_door_ceil - 0.5) & (ceil > 8.0)
            if act[7] == 1 and closing_door.any():
                reward -= 0.08
            self._prev_door_ceil = ceil.copy()

        # 5) Terminal outcomes
        if ps["exit_reached"]:
            time_frac = max(0.0, 1.0 - ps["tic"] / max(1, ps["episode_ticks"]))
            health_frac = min(1.0, max(0.0, ps["health"] / 100.0))
            reward += 8.5 + 2.0 * time_frac + 1.0 * health_frac
        elif ps["dead"]:
            reward -= 2.0

        self._prev_ps = ps
        self._prev_use = int(act[7])

        info = {
            **info,
            "kills": ps["kills"],
            "keys_held": int(np.sum(ps["keys"])),
            "cells_visited": len(self._all_visited),
            "doors_opened": len(self._opened_doors),
            "dead": ps["dead"],
        }
        return strip_obs(obs), float(reward), bool(terminated), bool(truncated), info

    def close(self):
        for env in self._envs.values():
            env.close()
        self._envs.clear()
        self._cur_env = None


def curriculum_map_entries(
    stage: int,
    map_cache_dir: str | Path = "/tmp/deadlock_maps",
    maps_per_tier: int = 16,
) -> list[tuple[str, str]]:
    """Return list of (wad_path, map_name) for curriculum stage 1..4."""
    pool = build_map_pool(map_cache_dir, maps_per_tier=maps_per_tier)
    entries: list[tuple[str, str]] = []
    if stage == 1:
        tiers = [1]
    elif stage == 2:
        tiers = [1, 2, 2]
    elif stage == 3:
        tiers = [1, 2, 2, 3, 3, 3]
    else:
        tiers = [1, 1, 2, 2, 3, 3, 4, 4]
    for t in tiers:
        for p in pool[t]:
            entries.append((str(p), "MAP01"))
    if stage >= 2:
        for _ in range(max(1, maps_per_tier // 32)):
            entries.append((str(Path(__file__).resolve().parents[1] / "maps" / "two_rooms.wad"), "MAP01"))
    if stage == 4 and os.path.isfile(FREEDOOM2_IWAD):
        repeats = max(1, maps_per_tier // 48)
        for m in FREEDOOM_STAGE4_MAPS:
            # Keep Freedoom2 MAP01..06 present (~2-5% of episodes) without drowning procedural tiers
            for _ in range(repeats):
                entries.append((FREEDOOM2_IWAD, m))
    return entries
