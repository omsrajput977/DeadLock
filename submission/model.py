"""Custom Ego-Centric Spatial-Memory Policy for DOOM (PyTorch).

Architecture highlights:
1. Circular 1D ConvNet over the 64 360-degree ego-centric raycasts (preserves angular wrap-around).
2. Masked Multi-Head Cross-Attention over variable numbers of visible entities (0..32).
3. Differentiable Egocentric 2D Spatial Memory Map (16 x 25 x 25) warped every step via
   estimated ego-motion (velocity + yaw delta) and updated with ray/entity splatting + gated fusion.
4. Spatial Attention readout conditioned on held keycards/vitals + GRU control memory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_RAYS = 64
NUM_RAY_TYPES = 6
MAX_ENTITIES = 32
NUM_ENTITY_TYPES = 12
VITALS_DIM = 17
NUM_ACTIONS = 9
ACTOR_LOGITS = 12  # move(3) + strafe(3) + turn(3) + attack(1) + use(1) + switch_weapon(1)


def mask_switch_logit(logits: torch.Tensor, vitals: torch.Tensor | None) -> torch.Tensor:
    """Mask SELECT_NEXT_WEAPON off when current weapon still has ammo or no backup ammo exists."""
    if vitals is None:
        return logits
    can_switch = (vitals[..., 3] <= 0.0) & (vitals[..., 4:8].sum(dim=-1) > 0.0)
    out = logits.clone()
    out[..., 11] = torch.where(can_switch, out[..., 11], torch.full_like(out[..., 11], -20.0))
    return out


def factor_to_multibinary(factor_act: torch.Tensor) -> torch.Tensor:
    """Convert (..., 6) discrete factor indices [move, strafe, turn, atk, use, sw] -> (..., 9) MultiBinary."""
    m, s, t, atk, use, sw = factor_act.unbind(dim=-1)
    mb = torch.stack(
        [
            (m == 1).float(),
            (m == 2).float(),
            (s == 1).float(),
            (s == 2).float(),
            (t == 1).float(),
            (t == 2).float(),
            atk.float(),
            use.float(),
            sw.float(),
        ],
        dim=-1,
    )
    return mb


def sample_factor_actions(
    logits: torch.Tensor,
    vitals: torch.Tensor | None = None,
    deterministic: bool = False,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (factor_act (..., 6), multibinary (..., 9), log_prob (...))."""
    lg = mask_switch_logit(logits, vitals)
    if not deterministic and temperature != 1.0:
        lg = lg / temperature
    dm = torch.distributions.Categorical(logits=lg[..., 0:3])
    ds = torch.distributions.Categorical(logits=lg[..., 3:6])
    dt = torch.distributions.Categorical(logits=lg[..., 6:9])
    db = torch.distributions.Bernoulli(logits=lg[..., 9:12])
    if deterministic:
        am = lg[..., 0:3].argmax(dim=-1)
        as_ = lg[..., 3:6].argmax(dim=-1)
        at = lg[..., 6:9].argmax(dim=-1)
        ab = (lg[..., 9:12] > 0.0).long()
    else:
        am = dm.sample()
        as_ = ds.sample()
        at = dt.sample()
        ab = db.sample().long()
    lp = dm.log_prob(am) + ds.log_prob(as_) + dt.log_prob(at) + db.log_prob(ab.float()).sum(dim=-1)
    factor_act = torch.cat([am.unsqueeze(-1), as_.unsqueeze(-1), at.unsqueeze(-1), ab], dim=-1)
    return factor_act, factor_to_multibinary(factor_act), lp


def eval_factor_logprob_entropy(
    logits: torch.Tensor, factor_act: torch.Tensor, vitals: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate exact log_prob and entropy for stored (T, M, 6) factor actions."""
    lg = mask_switch_logit(logits, vitals)
    dm = torch.distributions.Categorical(logits=lg[..., 0:3])
    ds = torch.distributions.Categorical(logits=lg[..., 3:6])
    dt = torch.distributions.Categorical(logits=lg[..., 6:9])
    db = torch.distributions.Bernoulli(logits=lg[..., 9:12])
    lp = (
        dm.log_prob(factor_act[..., 0])
        + ds.log_prob(factor_act[..., 1])
        + dt.log_prob(factor_act[..., 2])
        + db.log_prob(factor_act[..., 3:6].float()).sum(dim=-1)
    )
    ent = dm.entropy() + ds.entropy() + dt.entropy() + db.entropy().sum(dim=-1)
    return lp, ent


@dataclass(frozen=True)
class ModelConfig:
    map_channels: int = 16
    map_size: int = 25
    map_range: float = 1200.0  # +/- 1200 map units around player (~96 units/cell)
    gru_hidden: int = 192
    frame_skip: int = 4


def preprocess_obs_dict(obs: dict, device: torch.device) -> dict[str, torch.Tensor]:
    """Convert a single gym observation (or batched numpy dict) into batched torch tensors on `device`."""
    dist = np.asarray(obs["rays"]["distance"], dtype=np.float32)
    single = dist.ndim == 1
    if single:
        dist = dist[None, :]
        hit_type = np.asarray(obs["rays"]["hit_type"], dtype=np.int64)[None, :]
        ent_feat = np.asarray(obs["visible_entities"]["features"], dtype=np.float32)[None, :, :]
        ent_type = np.asarray(obs["visible_entities"]["entity_type"], dtype=np.int64)[None, :]
        ent_mask = np.asarray(obs["visible_entities"]["mask"], dtype=np.float32)[None, :]
        vitals = np.asarray(obs["vitals"], dtype=np.float32)[None, :]
        rgb = np.asarray(obs["rgb"], dtype=np.uint8)[None, ...]
    else:
        hit_type = np.asarray(obs["rays"]["hit_type"], dtype=np.int64)
        ent_feat = np.asarray(obs["visible_entities"]["features"], dtype=np.float32)
        ent_type = np.asarray(obs["visible_entities"]["entity_type"], dtype=np.int64)
        ent_mask = np.asarray(obs["visible_entities"]["mask"], dtype=np.float32)
        vitals = np.asarray(obs["vitals"], dtype=np.float32)
        rgb = np.asarray(obs["rgb"], dtype=np.uint8)

    if rgb.shape[1:3] == (120, 160):
        rgb = rgb[:, ::2, ::2, :]

    return {
        "ray_dist": torch.from_numpy(np.ascontiguousarray(dist)).to(device),
        "ray_type": torch.from_numpy(np.ascontiguousarray(hit_type)).to(device),
        "ent_feat": torch.from_numpy(np.ascontiguousarray(ent_feat)).to(device),
        "ent_type": torch.from_numpy(np.ascontiguousarray(ent_type)).to(device),
        "ent_mask": torch.from_numpy(np.ascontiguousarray(ent_mask)).to(device),
        "vitals": torch.from_numpy(np.ascontiguousarray(vitals)).to(device),
        "rgb": torch.from_numpy(np.ascontiguousarray(rgb)).to(device),
    }


def normalize_vitals(vitals: torch.Tensor) -> torch.Tensor:
    """Scale raw vitals (B, 17) into well-behaved neural input ranges."""
    scale = vitals.new_tensor(
        [
            100.0, 100.0, 7.0, 50.0,
            100.0, 50.0, 20.0, 100.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            15.0, 15.0, 15.0,
        ]
    )
    x = vitals / scale
    # Compress large ammo/health outliers while preserving key flags (indices 8..13)
    cont = torch.tanh(x[..., :8] * 0.5)
    keys = x[..., 8:14]
    vel = torch.tanh(x[..., 14:17])
    return torch.cat([cont, keys, vel], dim=-1)


class RayEncoder(nn.Module):
    """Direction-preserving Circular 1D CNN over 64 ego-centric rays + front-cone features."""

    def __init__(self, out_dim: int = 128):
        super().__init__()
        angles = torch.arange(NUM_RAYS, dtype=torch.float32) * (2.0 * math.pi / NUM_RAYS)
        # Store (1, 2, 64) sin/cos positional channels
        self.register_buffer("sincos", torch.stack([torch.sin(angles), torch.cos(angles)], dim=0).unsqueeze(0))
        self.register_buffer("front_idx", torch.tensor([60, 61, 62, 63, 0, 1, 2, 3, 4], dtype=torch.long))
        in_ch = 1 + NUM_RAY_TYPES + 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 48, kernel_size=5, padding=2, padding_mode="circular"),
            nn.SiLU(),
            nn.Conv1d(48, 64, kernel_size=5, stride=2, padding=2, padding_mode="circular"),
            nn.SiLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1, padding_mode="circular"),
            nn.SiLU(),
        )
        self.dir_reduce = nn.Sequential(nn.Conv1d(64, 16, kernel_size=1), nn.SiLU())
        self.proj = nn.Linear(16 * 16 + 128 + 9 * (1 + NUM_RAY_TYPES), out_dim)

    def forward(self, ray_dist: torch.Tensor, ray_type: torch.Tensor) -> torch.Tensor:
        bsz = ray_dist.shape[0]
        one_hot = F.one_hot(ray_type.clamp(0, NUM_RAY_TYPES - 1), num_classes=NUM_RAY_TYPES).float()  # (B, 64, 6)
        x = torch.cat([ray_dist.unsqueeze(1), one_hot.permute(0, 2, 1), self.sincos.expand(bsz, -1, -1)], dim=1)
        h = self.net(x)  # (B, 64, 16)
        dir_bins = self.dir_reduce(h).flatten(1)  # (B, 256) preserves 16 angular sectors
        front_cone = x[:, : 1 + NUM_RAY_TYPES, self.front_idx].flatten(1)  # (B, 63) explicit ahead rays
        pooled = torch.cat([dir_bins, h.mean(dim=-1), h.amax(dim=-1), front_cone], dim=-1)
        return F.silu(self.proj(pooled))


class EntityAttentionEncoder(nn.Module):
    """Permutation-invariant masked cross-attention over up to 32 visible entities."""

    def __init__(self, query_ctx_dim: int, d_model: int = 64, n_queries: int = 3, out_dim: int = 96):
        super().__init__()
        self.type_emb = nn.Embedding(NUM_ENTITY_TYPES, 16)
        self.ent_mlp = nn.Sequential(
            nn.Linear(5 + 16, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
        )
        self.static_queries = nn.Parameter(torch.randn(1, n_queries - 1, d_model) * 0.02)
        self.ctx_query = nn.Linear(query_ctx_dim, d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
        self.out_proj = nn.Sequential(nn.Linear(n_queries * d_model, out_dim), nn.SiLU())

    def forward(
        self, ent_feat: torch.Tensor, ent_type: torch.Tensor, ent_mask: torch.Tensor, ctx: torch.Tensor
    ) -> torch.Tensor:
        bsz = ent_feat.shape[0]
        # Scale distance/velocities cleanly
        f = ent_feat.clone()
        f[..., 0] = torch.log1p(f[..., 0].clamp(min=0.0))
        f[..., 3:5] = torch.tanh(f[..., 3:5] / 15.0)
        te = self.type_emb(ent_type.clamp(0, NUM_ENTITY_TYPES - 1))
        tokens = self.ent_mlp(torch.cat([f, te], dim=-1))  # (B, 32, d_model)

        has_any = (ent_mask.sum(dim=-1, keepdim=True) > 0).float()  # (B, 1)
        # Guard against all-masked rows (softmax over all -inf -> NaN)
        safe_mask = ent_mask.clone()
        safe_mask[:, 0] = torch.where(has_any.squeeze(-1) > 0, safe_mask[:, 0], torch.ones_like(safe_mask[:, 0]))
        key_pad = safe_mask <= 0.5  # True = ignore

        q_ctx = self.ctx_query(ctx).unsqueeze(1)
        queries = torch.cat([self.static_queries.expand(bsz, -1, -1), q_ctx], dim=1)  # (B, Q, d_model)
        attn_out, _ = self.attn(queries, tokens, tokens, key_padding_mask=key_pad, need_weights=False)
        attn_out = attn_out * has_any.unsqueeze(-1)
        return self.out_proj(attn_out.reshape(bsz, -1))


class EgoSpatialMemory(nn.Module):
    """Differentiable 2D Egocentric Spatial Memory Map updated via ego-motion warp + ray/entity splatting."""

    def __init__(self, cfg: ModelConfig, cond_dim: int, out_dim: int = 96):
        super().__init__()
        self.cfg = cfg
        c = cfg.map_channels
        s = cfg.map_size

        # Precompute ray unit directions in ego frame: (fwd, right)
        angles = torch.arange(NUM_RAYS, dtype=torch.float32) * (2.0 * math.pi / NUM_RAYS)
        # In Doom ego frame: angle 0 = straight ahead, positive = counter-clockwise (left = -right)
        self.register_buffer("ray_fwd_dir", torch.cos(angles))
        self.register_buffer("ray_right_dir", -torch.sin(angles))

        # Project ray hit types + entity hits into C-dim spatial tokens
        self.ray_write_emb = nn.Embedding(NUM_RAY_TYPES, c)
        self.ent_write_emb = nn.Embedding(NUM_ENTITY_TYPES, c)

        # Learned turn rate per step (radians) when TURN_LEFT / TURN_RIGHT is pressed (~11 deg/step initial)
        self.turn_rate = nn.Parameter(torch.tensor(0.20))

        # Gated map update (2xC -> C)
        self.update_conv = nn.Conv2d(2 * c, 2 * c, kernel_size=3, padding=1)

        # Spatial readout: downsample 25x25 -> 12x12 -> 6x6 and query via conditioned attention
        self.read_conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )
        # Readout grid is 7x7 for map_size=25 with two stride-2 convs
        read_hw = ((s + 1) // 2 + 1) // 2
        self.register_buffer("pos_grid", self._make_pos_grid(read_hw))
        self.cell_proj = nn.Linear(32 + 2, 48)
        self.read_query = nn.Linear(cond_dim, 48)
        self.out_proj = nn.Sequential(nn.Linear(48 + 32, out_dim), nn.SiLU())

    @staticmethod
    def _make_pos_grid(hw: int) -> torch.Tensor:
        coords = torch.linspace(-1.0, 1.0, hw)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        return torch.stack([xx, yy], dim=0).unsqueeze(0)  # (1, 2, hw, hw)

    def _warp_map(self, prev_map: torch.Tensor, vitals: torch.Tensor, prev_action: torch.Tensor) -> torch.Tensor:
        """Rotate and translate `prev_map` (B, C, H, W) into the current ego frame."""
        bsz = prev_map.shape[0]
        # Ego displacement over frame_skip tics, normalized to [-1, 1] grid coordinates
        # Grid coordinates: x = right / map_range, y = -forward / map_range (so top of map = ahead)
        df = vitals[:, 14] * self.cfg.frame_skip / self.cfg.map_range
        dr = vitals[:, 15] * self.cfg.frame_skip / self.cfg.map_range

        # Estimated yaw delta: TURN_LEFT (idx 4) - TURN_RIGHT (idx 5)
        turn_cmd = prev_action[:, 4] - prev_action[:, 5]
        dtheta = turn_cmd * self.turn_rate.clamp(0.02, 0.6)
        c = torch.cos(dtheta)
        s = torch.sin(dtheta)

        # Affine matrix mapping target normalized (x_new, y_new) -> source (x_old, y_old):
        # Rotate by +dtheta and shift by (+dr, -df)
        theta = torch.zeros(bsz, 2, 3, device=prev_map.device, dtype=prev_map.dtype)
        theta[:, 0, 0] = c
        theta[:, 0, 1] = -s
        theta[:, 0, 2] = dr
        theta[:, 1, 0] = s
        theta[:, 1, 1] = c
        theta[:, 1, 2] = -df

        grid = F.affine_grid(theta, prev_map.size(), align_corners=False)
        return F.grid_sample(prev_map, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    def _splat_observations(
        self,
        ray_dist: torch.Tensor,
        ray_type: torch.Tensor,
        ent_feat: torch.Tensor,
        ent_type: torch.Tensor,
        ent_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Project current ray hits and visible entities into an ego-centric 2D observation grid."""
        bsz = ray_dist.shape[0]
        c = self.cfg.map_channels
        s = self.cfg.map_size
        obs_grid = ray_dist.new_zeros(bsz, c, s * s)

        # 1) Splat ray hits (ignore RAY_NONE == 0)
        r_dist_units = ray_dist * 1024.0
        rf = r_dist_units * self.ray_fwd_dir.unsqueeze(0)
        rr = r_dist_units * self.ray_right_dir.unsqueeze(0)
        gx = ((rr / self.cfg.map_range + 1.0) * 0.5 * (s - 1)).round().long().clamp(0, s - 1)
        gy = ((-rf / self.cfg.map_range + 1.0) * 0.5 * (s - 1)).round().long().clamp(0, s - 1)
        r_idx = (gy * s + gx).unsqueeze(1).expand(-1, c, -1)  # (B, C, 64)
        r_valid = (ray_type > 0).unsqueeze(1).float()  # (B, 1, 64)
        r_emb = self.ray_write_emb(ray_type.clamp(0, NUM_RAY_TYPES - 1)).permute(0, 2, 1) * r_valid
        obs_grid.scatter_add_(2, r_idx, r_emb)

        # 2) Splat visible entities (keys, enemies, items)
        edist = ent_feat[..., 0] * 1024.0
        sin_a = ent_feat[..., 1]
        cos_a = ent_feat[..., 2]
        ef = edist * cos_a
        er = -edist * sin_a
        egx = ((er / self.cfg.map_range + 1.0) * 0.5 * (s - 1)).round().long().clamp(0, s - 1)
        egy = ((-ef / self.cfg.map_range + 1.0) * 0.5 * (s - 1)).round().long().clamp(0, s - 1)
        e_idx = (egy * s + egx).unsqueeze(1).expand(-1, c, -1)  # (B, C, 32)
        e_emb = self.ent_write_emb(ent_type.clamp(0, NUM_ENTITY_TYPES - 1)).permute(0, 2, 1) * ent_mask.unsqueeze(1)
        obs_grid.scatter_add_(2, e_idx, e_emb)

        return torch.tanh(obs_grid.view(bsz, c, s, s))

    def warp_and_update(
        self,
        prev_map: torch.Tensor,
        splat: torch.Tensor,
        vitals: torch.Tensor,
        prev_action: torch.Tensor,
        cond: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Warp `prev_map` into current ego frame, fuse precomputed `splat`, and read out."""
        warped = self._warp_map(prev_map, vitals, prev_action)
        gate_cand = self.update_conv(torch.cat([warped, splat], dim=1))
        gate, cand = torch.chunk(gate_cand, 2, dim=1)
        z = torch.sigmoid(gate)
        new_map = (1.0 - z) * warped + z * torch.tanh(cand)

        feat_map = self.read_conv(new_map)  # (B, 32, hw, hw)
        bsz, _ch, _hw, _ = feat_map.shape
        cells = torch.cat([feat_map, self.pos_grid.expand(bsz, -1, -1, -1)], dim=1)  # (B, 34, hw, hw)
        cells_flat = cells.flatten(2).transpose(1, 2)  # (B, hw*hw, 34)
        keys = torch.tanh(self.cell_proj(cells_flat))  # (B, hw*hw, 48)
        q = self.read_query(cond).unsqueeze(1)  # (B, 1, 48)
        attn_w = F.softmax((q * keys).sum(dim=-1) / math.sqrt(48.0), dim=-1)  # (B, hw*hw)
        attended = (attn_w.unsqueeze(-1) * keys).sum(dim=1)  # (B, 48)
        pooled = feat_map.mean(dim=(2, 3))  # (B, 32)
        return self.out_proj(torch.cat([attended, pooled], dim=-1)), new_map

    def forward(
        self,
        prev_map: torch.Tensor,
        obs_t: dict[str, torch.Tensor],
        prev_action: torch.Tensor,
        cond: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        splat = self._splat_observations(
            obs_t["ray_dist"], obs_t["ray_type"], obs_t["ent_feat"], obs_t["ent_type"], obs_t["ent_mask"]
        )
        return self.warp_and_update(prev_map, splat, obs_t["vitals"], prev_action, cond)


class RgbEncoder(nn.Module):
    """Ultra-lightweight 60x80 RGB ConvNet for detecting floor exit pads, signs, and door trims."""

    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((2, 3)),
        )
        self.proj = nn.Sequential(nn.Linear(32 * 2 * 3, out_dim), nn.SiLU())

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        x = rgb.permute(0, 3, 1, 2).float() / 255.0
        return self.proj(self.conv(x).flatten(1))


class DoomSpatialPolicy(nn.Module):
    """Hybrid Vision + Ego-Spatial-Memory Actor-Critic policy for DOOM."""

    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.ray_enc = RayEncoder(out_dim=128)
        self.rgb_enc = RgbEncoder(out_dim=64)
        self.vitals_mlp = nn.Sequential(
            nn.Linear(VITALS_DIM + NUM_ACTIONS, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
        )
        self.ent_enc = EntityAttentionEncoder(query_ctx_dim=128 + 64, d_model=64, n_queries=3, out_dim=96)
        self.spatial_mem = EgoSpatialMemory(self.cfg, cond_dim=64 + 128, out_dim=96)

        fused_dim = 128 + 64 + 96 + 96 + 64
        self.gru = nn.GRUCell(fused_dim, self.cfg.gru_hidden)

        h_dim = self.cfg.gru_hidden
        self.actor = nn.Sequential(
            nn.Linear(h_dim, 128),
            nn.SiLU(),
            nn.Linear(128, ACTOR_LOGITS),
        )
        with torch.no_grad():
            # Prefer looking forward where FOV/RGB points rather than blind sideways strafing
            self.actor[-1].bias.zero_()
            self.actor[-1].bias[1] = 0.6   # MOVE_FORWARD
            self.actor[-1].bias[2] = -0.3  # MOVE_BACKWARD
            self.actor[-1].bias[3] = 0.4   # STRAFE_IDLE
            self.actor[-1].bias[9] = -0.4  # ATTACK (avoid blind initial spam)
            self.actor[-1].bias[10] = -0.2  # USE
        self.critic = nn.Sequential(
            nn.Linear(h_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )
        # Auxiliary head predicting next normalized vitals (training-only representation regularizer)
        self.aux_vitals = nn.Linear(h_dim, VITALS_DIM)

    def init_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return zeroed (gru_h, spatial_map, prev_action) recurrent state."""
        gru_h = torch.zeros(batch_size, self.cfg.gru_hidden, device=device)
        spatial_map = torch.zeros(
            batch_size, self.cfg.map_channels, self.cfg.map_size, self.cfg.map_size, device=device
        )
        prev_action = torch.zeros(batch_size, NUM_ACTIONS, device=device)
        return gru_h, spatial_map, prev_action

    def encode_inputs(
        self, obs_t: dict[str, torch.Tensor], prev_action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Feed-forward observation encoders (batchable across T*B during PPO BPTT)."""
        v_norm = normalize_vitals(obs_t["vitals"])
        v_feat = self.vitals_mlp(torch.cat([v_norm, prev_action], dim=-1))
        r_feat = self.ray_enc(obs_t["ray_dist"], obs_t["ray_type"])
        vis_feat = self.rgb_enc(obs_t["rgb"])
        e_feat = self.ent_enc(
            obs_t["ent_feat"], obs_t["ent_type"], obs_t["ent_mask"], torch.cat([r_feat, v_feat], dim=-1)
        )
        splat = self.spatial_mem._splat_observations(
            obs_t["ray_dist"], obs_t["ray_type"], obs_t["ent_feat"], obs_t["ent_type"], obs_t["ent_mask"]
        )
        return r_feat, v_feat, e_feat, vis_feat, splat

    def recurrent_step(
        self,
        r_feat: torch.Tensor,
        v_feat: torch.Tensor,
        e_feat: torch.Tensor,
        vis_feat: torch.Tensor,
        splat: torch.Tensor,
        vitals: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        gru_h, spatial_map, prev_action = state
        m_feat, new_map = self.spatial_mem.warp_and_update(
            spatial_map, splat, vitals, prev_action, torch.cat([v_feat, r_feat], dim=-1)
        )
        fused = torch.cat([r_feat, v_feat, e_feat, m_feat, vis_feat], dim=-1)
        new_h = self.gru(fused, gru_h)
        logits = self.actor(new_h)
        value = self.critic(new_h).squeeze(-1)
        aux_pred = self.aux_vitals(new_h)
        return logits, value, aux_pred, (new_h, new_map, prev_action)

    def forward_step(
        self,
        obs_t: dict[str, torch.Tensor],
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        r_feat, v_feat, e_feat, vis_feat, splat = self.encode_inputs(obs_t, state[2])
        return self.recurrent_step(r_feat, v_feat, e_feat, vis_feat, splat, obs_t["vitals"], state)
