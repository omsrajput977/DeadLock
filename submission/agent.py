"""Evaluation entry point. The scorer instantiates DoomAgent from this file.

Contract: docs/CONTRACT.md §6.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

try:
    from submission.model import DoomSpatialPolicy, ModelConfig, preprocess_obs_dict, sample_factor_actions
except ImportError:
    from model import DoomSpatialPolicy, ModelConfig, preprocess_obs_dict, sample_factor_actions

SUBMISSION_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = SUBMISSION_DIR / "weights" / "model.pt"


class DoomAgent:
    def __init__(self, observation_space, action_space, device: str):
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = torch.device(device)
        self.deterministic = False
        self.temperature = 0.40

        ckpt = torch.load(WEIGHTS_PATH, map_location=self.device, weights_only=True) if WEIGHTS_PATH.is_file() else None
        cfg = ModelConfig(**ckpt["config"]) if (ckpt and "config" in ckpt) else ModelConfig()
        self.policy = DoomSpatialPolicy(cfg).to(self.device)
        if ckpt is not None:
            state_dict = ckpt["model"] if "model" in ckpt else ckpt
            self.policy.load_state_dict(state_dict, strict=True)
        self.policy.eval()

        self._state = None
        self.reset()

    def reset(self) -> None:
        self._state = self.policy.init_state(batch_size=1, device=self.device)

    @torch.inference_mode()
    def act(self, observation: dict) -> np.ndarray:
        obs_t = preprocess_obs_dict(observation, self.device)
        logits, _, _, next_state = self.policy.forward_step(obs_t, self._state)
        _, action_t, _ = sample_factor_actions(
            logits, obs_t["vitals"], deterministic=self.deterministic, temperature=self.temperature
        )

        self._state = self.policy.advance_state(next_state, action_t)
        return action_t.squeeze(0).cpu().numpy().astype(np.int8)
