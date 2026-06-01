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
    # DEBUG (one-shot): print actual brush pos + dist so we can see why
    # mimic annotate is rejecting demos despite the demo HDF5 ending near
    # the basket. Remove after the pipeline is verified.
    import os
    if os.environ.get("AIR2_DEBUG_BASKET_CHECK") == "1":
        print(f"[target_reached_basket DEBUG] target={target_key} obj_pos={obj_pos.tolist()} basket={basket.tolist()} dist={dist.tolist()} pass={(dist < radius).tolist()}", flush=True)
    return dist < radius


def target_dropped_in_basket(
    env: ManagerBasedRLEnv,
    target_key: str,
    xy_radius: float = 0.18,
    rim_z_offset: float = 0.15,
    finger_open_thresh: float = 0.03,
) -> torch.Tensor:
    """Strict drop-in success: object inside basket footprint AND below rim AND gripper released.

    Three conditions must all hold:
      1. XY distance to basket center  < xy_radius          (inside footprint)
      2. Z below basket center + rim_z_offset               (object has dropped past rim)
      3. Sum of two Franka finger joint positions > thresh  (gripper has opened)

    Hovering with the object still gripped inside the basket bounds does NOT count.
    """
    basket = BASKET_POS_LOCAL.to(env.device)
    asset = env.scene[target_key]
    obj_pos = asset.data.root_pos_w - env.scene.env_origins

    xy_dist = torch.linalg.norm(obj_pos[..., :2] - basket[:2], dim=-1)
    inside_xy = xy_dist < xy_radius
    below_rim = obj_pos[..., 2] < basket[2] + rim_z_offset

    robot = env.scene["robot"]
    finger_sum = robot.data.joint_pos[:, -2:].sum(dim=-1)
    gripper_open = finger_sum > finger_open_thresh

    return inside_xy & below_rim & gripper_open
