"""AIR2-scene reward terms.

The default `lift_env_cfg` rewards are designed for a 30cm tabletop pick-place:

  - `reaching_object` uses std=0.1 (10cm) — at AIR2's ~3m EE-to-object distance
    this Gaussian is ~0, so it gives no gradient.
  - `lifting_object(minimal_height=0.04)` — fires whenever the object is above
    floor level. AIR2 objects spawn on hooks at z=1.3-1.7m, so this is always
    True regardless of policy. Reward is a constant, not a signal.
  - `object_goal_tracking` uses a goal pose in robot-root frame at small ranges
    that map to a world position ~7m from the actual object. Impossible to
    satisfy and gives the `position_error: ~7m` we see in the PPO logs.

These functions replace the broken ones with rewards that reflect the AIR2
task: pick objects off the pegboard hooks and drop them in the basket.
"""

from __future__ import annotations

import torch

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from ..objects import OBJECT_NAMES


# Basket world-position in env-local frame. Matches BASKET_POS_LOCAL in
# scripted_controller.py so demo trajectories and reward signal agree.
BASKET_POS_LOCAL = torch.tensor([-3.560, -5.370, 1.040])

# y-coordinate of the hook line. Objects start at y=-5.9; "off hook" means
# the object has been pulled at least HOOK_CLEAR cm toward +Y (toward robot).
HOOK_LINE_Y = -5.9
HOOK_CLEAR = 0.20  # 20 cm

_OBJECT_NAMES = tuple(OBJECT_NAMES)


def _object_local_pos(env: ManagerBasedRLEnv, name: str) -> torch.Tensor:
    """Get an object's position in env-local frame (num_envs, 3)."""
    asset: RigidObject = env.scene[name]
    return asset.data.root_pos_w - env.scene.env_origins


def all_objects_to_basket(env: ManagerBasedRLEnv, std: float = 0.5) -> torch.Tensor:
    """Sum of Gaussian rewards over distance(each object → basket).

    Returns (num_envs,). Maximum value = len(_OBJECT_NAMES) when every object
    sits on the basket. The gradient pulls objects toward the basket.
    """
    basket = BASKET_POS_LOCAL.to(env.device)
    total = torch.zeros(env.num_envs, device=env.device)
    for name in _OBJECT_NAMES:
        obj_pos = _object_local_pos(env, name)
        dist = torch.linalg.norm(obj_pos - basket, dim=-1)
        total = total + torch.exp(-(dist / std) ** 2)
    return total


def all_objects_in_basket(env: ManagerBasedRLEnv, radius: float = 0.30) -> torch.Tensor:
    """Number of objects within `radius` m of the basket center. (num_envs,)."""
    basket = BASKET_POS_LOCAL.to(env.device)
    count = torch.zeros(env.num_envs, device=env.device)
    for name in _OBJECT_NAMES:
        obj_pos = _object_local_pos(env, name)
        dist = torch.linalg.norm(obj_pos - basket, dim=-1)
        count = count + (dist < radius).float()
    return count


def object_in_basket(
    env: ManagerBasedRLEnv,
    radius: float = 0.30,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Bool: the specified object's center is within `radius` m of the basket.

    Used as the success termination for teleop demo recording — the recorder
    only writes an episode to HDF5 once this fires for num_success_steps
    consecutive frames.
    """
    basket = BASKET_POS_LOCAL.to(env.device)
    asset: RigidObject = env.scene[asset_cfg.name]
    obj_pos = asset.data.root_pos_w - env.scene.env_origins
    dist = torch.linalg.norm(obj_pos - basket, dim=-1)
    return dist < radius


def all_objects_off_hook(env: ManagerBasedRLEnv, clearance: float = HOOK_CLEAR) -> torch.Tensor:
    """Number of objects pulled at least `clearance` m off the hook line.

    Pegboard is at y=-5.9; a successful pick moves the object toward +Y.
    This is a coarse "have we made progress" signal — fires earlier than
    `all_objects_in_basket` so PPO has gradient before the object reaches
    the basket.
    """
    threshold = HOOK_LINE_Y + clearance
    count = torch.zeros(env.num_envs, device=env.device)
    for name in _OBJECT_NAMES:
        obj_pos = _object_local_pos(env, name)
        count = count + (obj_pos[..., 1] > threshold).float()
    return count


def ee_to_nearest_object(env: ManagerBasedRLEnv, std: float = 0.5) -> torch.Tensor:
    """Gaussian reward on distance from EE to the NEAREST remaining-on-hook object.

    std=0.5 means the reward saturates at ~50cm — appropriate for AIR2's
    scene scale. Drives the arm toward whichever object is still on the
    pegboard. Once all objects are off, falls back to nearest object regardless.
    """
    ee_frame = env.scene["ee_frame"]
    ee_pos = ee_frame.data.target_pos_w[..., 0, :] - env.scene.env_origins  # (num_envs, 3)
    threshold_y = HOOK_LINE_Y + HOOK_CLEAR
    best = None
    for name in _OBJECT_NAMES:
        obj_pos = _object_local_pos(env, name)
        on_hook = obj_pos[..., 1] <= threshold_y           # still on pegboard
        d = torch.linalg.norm(obj_pos - ee_pos, dim=-1)
        # If the object is off-hook, pretend it's far away so reward focuses
        # on objects still needing pick.
        d = torch.where(on_hook, d, torch.full_like(d, 999.0))
        best = d if best is None else torch.minimum(best, d)
    if best is None:
        return torch.zeros(env.num_envs, device=env.device)
    # If every object is off-hook (best is 999 for all envs), reward = 0.
    all_done = (best > 100.0)
    reward = torch.exp(-(best / std) ** 2)
    return torch.where(all_done, torch.zeros_like(reward), reward)
