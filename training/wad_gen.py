"""Procedural UDMF WAD generator for Tier 1-4 DOOM training and validation maps.

Generates clean, guaranteed-connected multi-room maps in ZDoom UDMF format
(`TEXTMAP` inside a 3-lump PWAD) with zero external BSP builder required.
Uses `doom_env.wad.write_wad` so maps load directly in `doom_env.make_env`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from doom_env.wad import Lump, write_wad

TILE_SIZE = 96.0
SLOT_STRIDE = 8  # Grid stride between room centers (guarantees >= 3 corridor tiles between rooms)
CEIL_HEIGHT = 128

# Lock numbers matching env/doom_env/mapinfo.py (LOCK_RED=1, LOCK_BLUE=2, LOCK_YELLOW=3)
LOCK_KEYS = (
    (1, 13, "RedCard"),
    (2, 5, "BlueCard"),
    (3, 6, "YellowCard"),
)

# Verified textures & flats present in freedoom2.wad / two_rooms.wad
WALL_TEXTURES = ("STARTAN2", "STONE2", "GRAY1")
FLOOR_FLATS = ("FLOOR0_1", "FLOOR4_8", "FLAT5_4", "FLAT1", "FLAT10", "FLAT14", "FLAT18")
CEIL_FLAT = "CEIL1_1"
DOOR_TEX = "BIGDOOR2"
DOOR_TRACK = "DOORTRAK"
LOCK_TRIMS = {1: "DOORRED", 2: "DOORBLU", 3: "DOORYEL"}
EXIT_FLAT = "GATE3"
EXIT_SWITCH_TEX = "SW1BRCOM"

# DoomEdNum thing IDs
THING_PLAYER = 1
THING_ZOMBIEMAN = 3004
THING_SHOTGUNGUY = 9
THING_IMP = 3001
THING_DEMON = 3002
THING_STIMPACK = 2011
THING_MEDIKIT = 2012
THING_CLIP = 2007
THING_SHELLS = 2008
THING_SHOTGUN = 2001
THING_ARMOR = 2018


@dataclass
class RoomRect:
    room_id: int
    slot_x: int
    slot_y: int
    x0: int
    y0: int
    w: int
    h: int
    sector_id: int

    @property
    def cx(self) -> float:
        return (self.x0 + self.w * 0.5) * TILE_SIZE

    @property
    def cy(self) -> float:
        return (self.y0 + self.h * 0.5) * TILE_SIZE


@dataclass
class SectorSpec:
    floor_tex: str
    ceil_tex: str
    floor_h: int
    ceil_h: int
    light: int = 192
    is_door: bool = False
    door_lock: int = 0  # 0 = unlocked door, 1/2/3 = R/B/Y locked door
    is_exit: bool = False
    exit_switch_only: bool = False


def _fmt_thing(x: float, y: float, angle: int, doomed: int) -> str:
    return (
        f"thing {{ x = {x:.1f}; y = {y:.1f}; angle = {angle}; type = {doomed}; "
        f"skill1 = true; skill2 = true; skill3 = true; skill4 = true; skill5 = true; single = true; }}"
    )


def _grid_to_udmf(
    grid: dict[tuple[int, int], int],
    sectors: list[SectorSpec],
    things: list[tuple[float, float, int, int]],
    wall_tex_by_sec: list[str],
) -> str:
    """Convert a tile-to-sector grid into a valid ZDoom UDMF TEXTMAP string."""
    verts: dict[tuple[int, int], int] = {}

    def get_v(gx: int, gy: int) -> int:
        key = (gx, gy)
        idx = verts.get(key)
        if idx is None:
            idx = len(verts)
            verts[key] = idx
        return idx

    sidedefs: list[str] = []
    linedefs: list[str] = []

    def add_side(sec: int, top: str = "-", mid: str = "-", bot: str = "-") -> int:
        idx = len(sidedefs)
        sidedefs.append(
            f'sidedef {{ sector = {sec}; texturetop = "{top}"; texturemiddle = "{mid}"; texturebottom = "{bot}"; }}'
        )
        return idx

    def emit_edge(v1_xy: tuple[int, int], v2_xy: tuple[int, int], front_sec: int, back_sec: int) -> None:
        v1 = get_v(*v1_xy)
        v2 = get_v(*v2_xy)
        if back_sec < 0:
            s_front = sectors[front_sec]
            if s_front.is_door:
                tex = LOCK_TRIMS.get(s_front.door_lock, DOOR_TRACK)
            elif s_front.is_exit and s_front.exit_switch_only:
                tex = EXIT_SWITCH_TEX
            else:
                tex = wall_tex_by_sec[front_sec]
            sf = add_side(front_sec, mid=tex)
            if s_front.is_exit and s_front.exit_switch_only:
                linedefs.append(
                    f"linedef {{ v1 = {v1}; v2 = {v2}; sidefront = {sf}; blocking = true; dontpegbottom = true; "
                    f"special = 243; playeruse = true; }}"
                )
            else:
                linedefs.append(
                    f"linedef {{ v1 = {v1}; v2 = {v2}; sidefront = {sf}; blocking = true; dontpegbottom = true; }}"
                )
            return

        # Two-sided edge between front_sec and back_sec.
        # Ensure any door or exit sector is on the BACK side so front->back activation/trigger works.
        if sectors[front_sec].is_door or (sectors[front_sec].is_exit and not sectors[back_sec].is_door):
            v1, v2 = v2, v1
            front_sec, back_sec = back_sec, front_sec

        b_spec = sectors[back_sec]
        if b_spec.is_door:
            sf = add_side(front_sec, top=DOOR_TEX)
            sb = add_side(back_sec)
            if b_spec.door_lock > 0:
                linedefs.append(
                    f"linedef {{ v1 = {v1}; v2 = {v2}; sidefront = {sf}; sideback = {sb}; twosided = true; "
                    f"special = 13; arg0 = 0; arg1 = 16; arg2 = 150; arg3 = {b_spec.door_lock}; "
                    f"playeruse = true; repeatspecial = true; }}"
                )
            else:
                linedefs.append(
                    f"linedef {{ v1 = {v1}; v2 = {v2}; sidefront = {sf}; sideback = {sb}; twosided = true; "
                    f"special = 12; arg0 = 0; arg1 = 16; arg2 = 150; "
                    f"playeruse = true; monsteruse = true; repeatspecial = true; }}"
                )
        elif b_spec.is_exit and not b_spec.exit_switch_only:
            sf = add_side(front_sec)
            sb = add_side(back_sec)
            linedefs.append(
                f"linedef {{ v1 = {v1}; v2 = {v2}; sidefront = {sf}; sideback = {sb}; twosided = true; "
                f"special = 243; playercross = true; }}"
            )
        else:
            sf = add_side(front_sec)
            sb = add_side(back_sec)
            linedefs.append(
                f"linedef {{ v1 = {v1}; v2 = {v2}; sidefront = {sf}; sideback = {sb}; twosided = true; }}"
            )

    # Iterate over all active tiles; emit outer walls and unique internal sector boundaries.
    for (ix, iy), sec in sorted(grid.items()):
        # Left edge: (ix, iy) -> (ix, iy+1) has (ix, iy) on its right
        left = grid.get((ix - 1, iy), -1)
        if left < 0:
            emit_edge((ix, iy), (ix, iy + 1), sec, -1)
        elif left != sec and (ix - 1, iy) < (ix, iy):
            emit_edge((ix, iy), (ix, iy + 1), sec, left)

        # Right edge: (ix+1, iy+1) -> (ix+1, iy) has (ix, iy) on its right
        right = grid.get((ix + 1, iy), -1)
        if right < 0:
            emit_edge((ix + 1, iy + 1), (ix + 1, iy), sec, -1)
        elif right != sec and (ix + 1, iy) < (ix, iy):
            emit_edge((ix + 1, iy + 1), (ix + 1, iy), sec, right)

        # Bottom edge: (ix+1, iy) -> (ix, iy) has (ix, iy) on its right
        bot = grid.get((ix, iy - 1), -1)
        if bot < 0:
            emit_edge((ix + 1, iy), (ix, iy), sec, -1)
        elif bot != sec and (ix, iy - 1) < (ix, iy):
            emit_edge((ix + 1, iy), (ix, iy), sec, bot)

        # Top edge: (ix, iy+1) -> (ix+1, iy+1) has (ix, iy) on its right
        top = grid.get((ix, iy + 1), -1)
        if top < 0:
            emit_edge((ix, iy + 1), (ix + 1, iy + 1), sec, -1)
        elif top != sec and (ix, iy + 1) < (ix, iy):
            emit_edge((ix, iy + 1), (ix + 1, iy + 1), sec, top)

    lines = ['namespace = "zdoom";']
    for (gx, gy) in verts:
        lines.append(f"vertex {{ x = {gx * TILE_SIZE:.1f}; y = {gy * TILE_SIZE:.1f}; }}")
    for s in sectors:
        lines.append(
            f'sector {{ texturefloor = "{s.floor_tex}"; textureceiling = "{s.ceil_tex}"; '
            f"heightfloor = {s.floor_h}; heightceiling = {s.ceil_h}; lightlevel = {s.light}; }}"
        )
    lines.extend(sidedefs)
    lines.extend(linedefs)
    for x, y, ang, tid in things:
        lines.append(_fmt_thing(x, y, ang, tid))
    return "\n".join(lines) + "\n"


def _carve_corridor(
    grid: dict[tuple[int, int], int],
    r_a: RoomRect,
    r_b: RoomRect,
    corridor_sec: int,
    door_sec: int | None,
) -> list[tuple[int, int]]:
    """Carve a 1-tile-wide straight corridor between adjacent room slots, placing door_sec in the middle."""
    tiles: list[tuple[int, int]] = []
    if r_a.slot_y == r_b.slot_y:
        if r_a.slot_x > r_b.slot_x:
            r_a, r_b = r_b, r_a
        y_lo = max(r_a.y0, r_b.y0)
        y_hi = min(r_a.y0 + r_a.h, r_b.y0 + r_b.h) - 1
        cy = (y_lo + y_hi) // 2
        x_start = r_a.x0 + r_a.w
        x_end = r_b.x0 - 1
        mid = (x_start + x_end) // 2
        for x in range(x_start, x_end + 1):
            grid[(x, cy)] = door_sec if (door_sec is not None and x == mid) else corridor_sec
            tiles.append((x, cy))
    else:
        if r_a.slot_y > r_b.slot_y:
            r_a, r_b = r_b, r_a
        x_lo = max(r_a.x0, r_b.x0)
        x_hi = min(r_a.x0 + r_a.w, r_b.x0 + r_b.w) - 1
        cx = (x_lo + x_hi) // 2
        y_start = r_a.y0 + r_a.h
        y_end = r_b.y0 - 1
        mid = (y_start + y_end) // 2
        for y in range(y_start, y_end + 1):
            grid[(cx, y)] = door_sec if (door_sec is not None and y == mid) else corridor_sec
            tiles.append((cx, y))
    return tiles


def generate_map_text(tier: int, seed: int) -> str:
    """Generate a procedural UDMF map string for tier in {1, 2, 3, 4}."""
    rng = random.Random(seed)
    grid: dict[tuple[int, int], int] = {}
    sectors: list[SectorSpec] = []
    wall_tex: list[str] = []
    things: list[tuple[float, float, int, int]] = []

    def new_sector(spec: SectorSpec, wtex: str) -> int:
        sid = len(sectors)
        sectors.append(spec)
        wall_tex.append(wtex)
        return sid

    if tier == 1 and rng.random() < 0.6:
        # Tier 1a: Combat & Survival arena (1 room, cover pillars, enemies, pickups, exit)
        w = rng.randint(5, 7)
        h = rng.randint(5, 7)
        wtex = rng.choice(WALL_TEXTURES)
        ftex = rng.choice(FLOOR_FLATS)
        main_sec = new_sector(SectorSpec(ftex, CEIL_FLAT, 0, CEIL_HEIGHT), wtex)
        for ix in range(w):
            for iy in range(h):
                grid[(ix, iy)] = main_sec

        # Optional 1-2 cover pillars away from center & exit
        if w >= 6 and h >= 6 and rng.random() < 0.7:
            grid.pop((2, 2), None)
            if rng.random() < 0.6:
                grid.pop((w - 3, h - 3), None)

        # Exit tile in a corner opposite player
        exit_xy = (w - 1, h - 1)
        exit_sec = new_sector(SectorSpec(EXIT_FLAT, CEIL_FLAT, 0, CEIL_HEIGHT, is_exit=True), wtex)
        grid[exit_xy] = exit_sec

        px, py = 1.2 * TILE_SIZE, 1.2 * TILE_SIZE
        things.append((px, py, rng.choice((0, 45, 90)), THING_PLAYER))

        candidates = [
            (ix, iy)
            for ix, iy in grid
            if (ix, iy) != exit_xy and abs(ix - 1) + abs(iy - 1) >= 2
        ]
        rng.shuffle(candidates)
        n_enemies = rng.randint(3, 6)
        enemy_pool = [THING_ZOMBIEMAN, THING_SHOTGUNGUY, THING_IMP, THING_DEMON]
        for ix, iy in candidates[:n_enemies]:
            things.append(((ix + 0.5) * TILE_SIZE, (iy + 0.5) * TILE_SIZE, rng.randrange(0, 360, 45), rng.choice(enemy_pool)))
        # Pickups (shotgun, shells, stimpack, armor)
        pickup_pool = [THING_SHOTGUN, THING_STIMPACK, THING_SHELLS, THING_CLIP, THING_ARMOR]
        for ix, iy in candidates[n_enemies : n_enemies + rng.randint(2, 4)]:
            things.append(((ix + 0.5) * TILE_SIZE, (iy + 0.5) * TILE_SIZE, 0, rng.choice(pickup_pool)))

        return _grid_to_udmf(grid, sectors, things, wall_tex)

    # Multi-room connected layout on a slot grid (includes 2-room combat+door for 40% of Tier 1)
    if tier == 1:
        grid_cols, grid_rows, target_rooms = 2, 1, 2
    elif tier == 3:
        r = rng.random()
        if r < 0.30:
            # 2-room ultra-compact keycard->locked-door warm-up
            grid_cols, grid_rows, target_rooms = 2, 1, 2
        elif r < 0.65:
            # Compact 3-4 room Tier 3 variant
            grid_cols, grid_rows, target_rooms = 2, 2, rng.randint(3, 4)
        else:
            grid_cols, grid_rows, target_rooms = 3, 3, rng.randint(5, 7)
    elif tier == 2 and rng.random() < 0.35:
        grid_cols, grid_rows, target_rooms = 2, 2, 3
    else:
        grid_cols = 3 if tier >= 3 else rng.choice((2, 3))
        grid_rows = 3 if tier == 4 else rng.choice((2, 3))
        target_rooms = {2: rng.randint(4, 6), 4: rng.randint(6, 9)}[tier]
        target_rooms = min(target_rooms, grid_cols * grid_rows)

    # Grow a connected set of slots starting from (0, 0)
    active_slots: list[tuple[int, int]] = [(0, 0)]
    seen = {(0, 0)}
    tree_edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    while len(active_slots) < target_rooms:
        frontier = []
        for sx, sy in active_slots:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = sx + dx, sy + dy
                if 0 <= nx < grid_cols and 0 <= ny < grid_rows and (nx, ny) not in seen:
                    frontier.append(((sx, sy), (nx, ny)))
        if not frontier:
            break
        u, v = rng.choice(frontier)
        seen.add(v)
        active_slots.append(v)
        tree_edges.append((u, v))

    # Build rooms inside each active slot (centered so adjacent rooms always share overlapping bands)
    rooms_by_slot: dict[tuple[int, int], RoomRect] = {}
    room_heights: dict[tuple[int, int], tuple[int, int]] = {}
    for rid, (sx, sy) in enumerate(active_slots):
        rw = rng.randint(3, 5)
        rh = rng.randint(3, 5)
        x0 = sx * SLOT_STRIDE + (5 - rw) // 2
        y0 = sy * SLOT_STRIDE + (5 - rh) // 2
        wtex = rng.choice(WALL_TEXTURES)
        ftex = rng.choice(FLOOR_FLATS)
        fh = 0 if (sx, sy) == (0, 0) else rng.choice((-16, 0, 0, 16))
        ch = fh + rng.choice((112, 128, 160))
        light = rng.choice((160, 192, 224))
        sec_id = new_sector(SectorSpec(ftex, CEIL_FLAT, fh, ch, light=light), wtex)
        room_heights[(sx, sy)] = (fh, ch)
        rect = RoomRect(rid, sx, sy, x0, y0, rw, rh, sec_id)
        rooms_by_slot[(sx, sy)] = rect
        for ix in range(x0, x0 + rw):
            for iy in range(y0, y0 + rh):
                grid[(ix, iy)] = sec_id

    # Adjacency in tree
    adj: dict[tuple[int, int], list[tuple[int, int]]] = {s: [] for s in active_slots}
    for u, v in tree_edges:
        adj[u].append(v)
        adj[v].append(u)

    # Choose exit room (prefer farthest leaf from start (0, 0))
    dist = {(0, 0): 0}
    queue = [(0, 0)]
    for cur in queue:
        for nxt in adj[cur]:
            if nxt not in dist:
                dist[nxt] = dist[cur] + 1
                queue.append(nxt)
    leaves = [s for s in active_slots if s != (0, 0) and len(adj[s]) == 1]
    exit_slot = max(leaves or [s for s in active_slots if s != (0, 0)], key=lambda s: dist[s])

    # For Tier 3 & 4: lock the single edge entering exit_slot, place keycard in a non-exit room
    locked_edges: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}
    key_placements: list[tuple[tuple[int, int], int]] = []
    if tier >= 3:
        parent_of_exit = adj[exit_slot][0]
        lock_num, key_tid, _ = rng.choice(LOCK_KEYS)
        edge_key = (min(parent_of_exit, exit_slot), max(parent_of_exit, exit_slot))
        locked_edges[edge_key] = lock_num
        # Key room: pick another room (prefer far from start and not exit_slot)
        key_candidates = [s for s in active_slots if s not in ((0, 0), exit_slot)]
        key_slot = max(key_candidates, key=lambda s: dist[s]) if key_candidates else (0, 0)
        key_placements.append((key_slot, key_tid))

    # Optional extra loop edge between non-exit rooms for Tier 2 & 4
    extra_edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    if tier in (2, 4) and rng.random() < 0.65:
        edge_set = {(min(u, v), max(u, v)) for u, v in tree_edges}
        for sx, sy in active_slots:
            for dx, dy in ((1, 0), (0, 1)):
                nb = (sx + dx, sy + dy)
                if nb in rooms_by_slot and exit_slot not in ((sx, sy), nb):
                    ek = (min((sx, sy), nb), max((sx, sy), nb))
                    if ek not in edge_set:
                        extra_edges.append(ek)
                        edge_set.add(ek)
                        break
            if extra_edges:
                break

    # Carve corridors and doors along all edges
    for u, v in tree_edges + extra_edges:
        ek = (min(u, v), max(u, v))
        r_u, r_v = rooms_by_slot[u], rooms_by_slot[v]
        corr_sec = new_sector(SectorSpec(rng.choice(FLOOR_FLATS), CEIL_FLAT, 0, CEIL_HEIGHT), rng.choice(WALL_TEXTURES))
        if ek in locked_edges:
            door_sec = new_sector(
                SectorSpec("FLAT5_4", CEIL_FLAT, 0, 0, is_door=True, door_lock=locked_edges[ek]),
                DOOR_TRACK,
            )
        elif rng.random() < (0.75 if tier >= 2 else 0.5):
            door_sec = new_sector(
                SectorSpec("FLAT5_4", CEIL_FLAT, 0, 0, is_door=True, door_lock=0),
                DOOR_TRACK,
            )
        else:
            door_sec = None
        _carve_corridor(grid, r_u, r_v, corr_sec, door_sec)

    # Carve exit tile inside exit room (85% walk-over GATE3 pad, 15% wall exit switch on edge)
    exit_rect = rooms_by_slot[exit_slot]
    ex_fh, ex_ch = room_heights[exit_slot]
    use_switch = tier >= 2 and rng.random() < 0.15
    if use_switch:
        # Pick a corner of exit_rect so at least two walls border solid void (-1) for the switch
        ex_ix = exit_rect.x0 + exit_rect.w - 1
        ex_iy = exit_rect.y0 + exit_rect.h - 1
        exit_sec = new_sector(
            SectorSpec(EXIT_FLAT, CEIL_FLAT, ex_fh, ex_ch, is_exit=True, exit_switch_only=True), WALL_TEXTURES[0]
        )
    else:
        ex_ix = exit_rect.x0 + exit_rect.w // 2
        ex_iy = exit_rect.y0 + exit_rect.h // 2
        exit_sec = new_sector(SectorSpec(EXIT_FLAT, CEIL_FLAT, ex_fh, ex_ch, is_exit=True), WALL_TEXTURES[0])
    grid[(ex_ix, ex_iy)] = exit_sec

    # Spawn player in room (0, 0)
    start_rect = rooms_by_slot[(0, 0)]
    things.append((start_rect.cx, start_rect.cy, rng.choice((0, 90, 180, 270)), THING_PLAYER))

    # Place keycards
    for kslot, ktid in key_placements:
        kr = rooms_by_slot[kslot]
        things.append((kr.cx, kr.cy, 0, ktid))

    # Populate enemies and pickups in non-start rooms
    enemy_pool = {
        1: [THING_ZOMBIEMAN, THING_SHOTGUNGUY, THING_IMP],
        2: [THING_ZOMBIEMAN],
        3: [THING_ZOMBIEMAN, THING_SHOTGUNGUY, THING_IMP],
        4: [THING_ZOMBIEMAN, THING_SHOTGUNGUY, THING_IMP, THING_DEMON],
    }[tier]
    enemy_chance = {1: 0.9, 2: 0.45, 3: 0.65, 4: 0.9}[tier]
    for slot, r in rooms_by_slot.items():
        if slot == (0, 0):
            # Starter supplies in start room
            if tier >= 3:
                things.append(((r.x0 + 0.6) * TILE_SIZE, (r.y0 + 0.6) * TILE_SIZE, 0, THING_SHOTGUN))
                things.append(((r.x0 + r.w - 0.6) * TILE_SIZE, (r.y0 + 0.6) * TILE_SIZE, 0, THING_SHELLS))
            continue
        if rng.random() < enemy_chance:
            n_e = rng.randint(1, 2 if tier < 4 else 3)
            for _ in range(n_e):
                tx = (rng.randint(r.x0, r.x0 + r.w - 1) + 0.5) * TILE_SIZE
                ty = (rng.randint(r.y0, r.y0 + r.h - 1) + 0.5) * TILE_SIZE
                if slot == exit_slot and abs(tx - (ex_ix + 0.5) * TILE_SIZE) < 32 and abs(ty - (ex_iy + 0.5) * TILE_SIZE) < 32:
                    continue
                things.append((tx, ty, rng.randrange(0, 360, 45), rng.choice(enemy_pool)))
        if rng.random() < 0.5:
            px = (rng.randint(r.x0, r.x0 + r.w - 1) + 0.5) * TILE_SIZE
            py = (rng.randint(r.y0, r.y0 + r.h - 1) + 0.5) * TILE_SIZE
            if slot == exit_slot and abs(px - (ex_ix + 0.5) * TILE_SIZE) < 32 and abs(py - (ex_iy + 0.5) * TILE_SIZE) < 32:
                continue
            things.append((px, py, 0, rng.choice((THING_STIMPACK, THING_MEDIKIT, THING_CLIP, THING_SHELLS, THING_ARMOR))))

    return _grid_to_udmf(grid, sectors, things, wall_tex)


def write_procedural_wad(out_path: str | Path, tier: int, seed: int, map_name: str = "MAP01") -> Path:
    """Generate and write a procedural UDMF PWAD file."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = generate_map_text(tier=tier, seed=seed)
    write_wad(p, [Lump(map_name.upper(), b""), Lump("TEXTMAP", text.encode("utf-8")), Lump("ENDMAP", b"")])
    return p


def build_map_pool(out_dir: str | Path, maps_per_tier: int = 16, base_seed: int = 1000) -> dict[int, list[Path]]:
    """Pre-generate a pool of procedural WADs across Tiers 1-4."""
    root = Path(out_dir)
    pool: dict[int, list[Path]] = {}
    for tier in (1, 2, 3, 4):
        tier_dir = root / f"tier{tier}"
        tier_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for idx in range(maps_per_tier):
            seed = base_seed + tier * 10000 + idx
            wad_path = tier_dir / f"t{tier}_{idx:03d}.wad"
            if not wad_path.exists():
                write_procedural_wad(wad_path, tier=tier, seed=seed)
            paths.append(wad_path)
        pool[tier] = paths
    return pool
