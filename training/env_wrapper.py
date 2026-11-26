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

CELL_SIZE = 96.0
KEY_ENTITY_ID = S.ENTITY_TYPES.index("key")
RAY_LOCKED_ID = 3  # env/doom_env/sensors.py RAY_LOCKED
FREEDOOM2_IWAD = os.path.join(os.path.dirname(vd.__file__), "freedoom2.wad")
FREEDOOM_STAGE4_MAPS = ("MAP01", "MAP02", "MAP04", "MAP05", "MAP06")


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
        self._opened_doors = set()
        self._prev_ps = ps
        self._prev_use = 0
        self._prev_door_ceil = ps["sector_ceiling"].copy() if ps["sector_ceiling"] is not None else None
        self._saw_key = False
        self._best_key_dist = 10.0
        self._saw_locked_with_key = False
        self._best_locked_dist = 2.0
        return strip_obs(obs), info

    def step(self, action):
        act = np.asarray(action, dtype=np.int8)
        obs, _, terminated, truncated, info = self._cur_env.step(act)
        ps = self._cur_env.privileged_state()
        prev = self._prev_ps

        reward = -0.001  # small time-efficiency pressure

        # 1) Episodic exploration: reward new 96x96 cells visited
        cell = (int(ps["x"] // CELL_SIZE), int(ps["y"] // CELL_SIZE))
        if cell not in self._visited:
            self._visited.add(cell)
            reward += 0.05

        # 2) Combat & health/item management
        dkills = ps["kills"] - prev["kills"]
        if dkills > 0:
            reward += 0.45 * dkills
        ditems = ps["items"] - prev["items"]
        if ditems > 0:
            reward += 0.08 * ditems
        ddmg = max(0.0, ps["damage_taken"] - prev["damage_taken"])
        if ddmg > 0:
            reward -= 0.01 * ddmg

        # 3) Keycards sighting, monotonic approach & pickup
        ents = obs["visible_entities"]
        key_slots = (ents["mask"] == 1) & (ents["entity_type"] == KEY_ENTITY_ID)
        if key_slots.any():
            if not self._saw_key:
                self._saw_key = True
                reward += 0.25
            min_kdist = float(ents["features"][key_slots, 0].min())
            if min_kdist < self._best_key_dist:
                if self._best_key_dist < 10.0:
                    reward += 0.4 * (self._best_key_dist - min_kdist)
                self._best_key_dist = min_kdist

        dkeys = float(np.sum(ps["keys"] - prev["keys"]))
        if dkeys > 0:
            reward += 2.0 * dkeys
            self._saw_key = False
            self._best_key_dist = 10.0

        # 4) Doors & locked doors sighting (when holding a key), approach & opening
        if np.sum(ps["keys"]) > 0:
            locked_rays = obs["rays"]["hit_type"] == RAY_LOCKED_ID
            if locked_rays.any():
                if not self._saw_locked_with_key:
                    self._saw_locked_with_key = True
                    reward += 0.25
                min_ldist = float(obs["rays"]["distance"][locked_rays].min())
                if min_ldist < self._best_locked_dist:
                    if self._best_locked_dist < 2.0:
                        reward += 0.5 * (self._best_locked_dist - min_ldist)
                    self._best_locked_dist = min_ldist

        ceil = ps["sector_ceiling"]
        dlock = ps["door_lock"]
        if ceil is not None and dlock is not None and self._prev_door_ceil is not None:
            rising = (dlock >= 0) & (ceil > self._prev_door_ceil + 0.5)
            for sid in np.flatnonzero(rising):
                sid_int = int(sid)
                if sid_int not in self._opened_doors:
                    self._opened_doors.add(sid_int)
                    reward += 2.5 if dlock[sid_int] > 0 else 0.45
            if act[7] == 1 and self._prev_use == 1:
                reward -= 0.015
            closing_door = (dlock >= 0) & (ceil < self._prev_door_ceil - 0.5) & (ceil > 8.0)
            if act[7] == 1 and closing_door.any():
                reward -= 0.05
            self._prev_door_ceil = ceil.copy()

        # 5) Terminal outcomes
        if ps["exit_reached"]:
            time_frac = max(0.0, 1.0 - ps["tic"] / max(1, ps["episode_ticks"]))
            health_frac = min(1.0, max(0.0, ps["health"] / 100.0))
            reward += 6.0 + 1.0 * time_frac + 0.5 * health_frac
        elif ps["dead"]:
            reward -= 1.0

        self._prev_ps = ps
        self._prev_use = int(act[7])

        info = {
            **info,
            "kills": ps["kills"],
            "keys_held": int(np.sum(ps["keys"])),
            "cells_visited": len(self._visited),
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
        tiers = [2, 3, 3, 3]
    else:
        tiers = [1, 2, 3, 4, 4]
    for t in tiers:
        for p in pool[t]:
            entries.append((str(p), "MAP01"))
    if stage >= 2:
        entries.append((str(Path(__file__).resolve().parents[1] / "maps" / "two_rooms.wad"), "MAP01"))
    if stage == 4 and os.path.isfile(FREEDOOM2_IWAD):
        for m in FREEDOOM_STAGE4_MAPS:
            # Repeat twice so Freedoom2 levels make up ~10% of Stage 4 episodes
            entries.append((FREEDOOM2_IWAD, m))
            entries.append((FREEDOOM2_IWAD, m))
    return entries
