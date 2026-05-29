"""Termination terms for AIR2 pick-place tasks."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv

from .constants import BASKET_POS_LOCAL


def target_reached_basket(
    env: ManagerBasedRLEnv,
    target_key: str,
    radius: float = 0.30,
) -> torch.Tensor:
    """Terminate when target object is inside the basket. (num_envs,) bool."""
    basket = BASKET_POS_LOCAL.to(env.device)
    asset = env.scene[target_key]
    obj_pos = asset.data.root_pos_w - env.scene.env_origins
    dist = torch.linalg.norm(obj_pos - basket, dim=-1)
    return dist < radius
