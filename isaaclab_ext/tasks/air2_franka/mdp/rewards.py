"""Target-specific reward terms for AIR2 pick-place tasks.

Each function takes a `target_key` parameter — the scene key of the object
this task is trying to pick (e.g. "object" for brush, "tool_pliers" for pliers).
All rewards are in env-local frame (world position minus env origin).

The base LiftEnvCfg rewards are broken for this scene:
  - reaching_object: std=0.1 gives ~0 gradient at AIR2's 3m scale
  - lifting_object: always fires — objects spawn on hooks above floor level
  - object_goal_tracking: goal pose in wrong frame, position error ~7m
These are all disabled in joint_pos_env_cfg.py.
"""

from __future__ import annotations

import torch

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv

from .constants import BASKET_POS_LOCAL, SLOT_LINE_Y, SLOT_CLEAR
from ..objects import OBJECT_NAMES


def _obj_local_pos(env: ManagerBasedRLEnv, key: str) -> torch.Tensor:
    asset: RigidObject = env.scene[key]
    return asset.data.root_pos_w - env.scene.env_origins


def _ee_local_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.scene["ee_frame"].data.target_pos_w[..., 0, :] - env.scene.env_origins


# ---------------------------------------------------------------------------
# Positive reward terms
# ---------------------------------------------------------------------------

def ee_to_target(env: ManagerBasedRLEnv, target_key: str, std: float = 0.5) -> torch.Tensor:
    """Gaussian reward pulling EE toward the target object."""
    dist = torch.linalg.norm(_obj_local_pos(env, target_key) - _ee_local_pos(env), dim=-1)
    return torch.exp(-(dist / std) ** 2)


def target_off_slot(
    env: ManagerBasedRLEnv,
    target_key: str,
    slot_line_y: float = SLOT_LINE_Y,
    clearance: float = SLOT_CLEAR,
) -> torch.Tensor:
    """Binary: target object has been pulled off its slot."""
    obj_pos = _obj_local_pos(env, target_key)
    return (obj_pos[..., 1] > slot_line_y + clearance).float()


def target_in_hand(
    env: ManagerBasedRLEnv,
    target_key: str,
    grasp_radius: float = 0.08,
) -> torch.Tensor:
    """Dense reward for maintaining grasp on target during carry.

    Approximated with physics: object within grasp_radius of EE AND
    gripper fingers are nearly closed. No CNN required.
    """
    dist = torch.linalg.norm(_obj_local_pos(env, target_key) - _ee_local_pos(env), dim=-1)
    robot = env.scene["robot"]
    # Franka finger joints are the last 2. Closed = sum < 0.04 m.
    finger_sum = robot.data.joint_pos[:, -2:].sum(dim=-1)
    gripper_closed = finger_sum < 0.04
    return ((dist < grasp_radius) & gripper_closed).float()


def target_to_basket(
    env: ManagerBasedRLEnv,
    target_key: str,
    std: float = 0.5,
) -> torch.Tensor:
    """Gaussian reward pulling target object toward the basket."""
    basket = BASKET_POS_LOCAL.to(env.device)
    dist = torch.linalg.norm(_obj_local_pos(env, target_key) - basket, dim=-1)
    return torch.exp(-(dist / std) ** 2)


def target_in_basket(
    env: ManagerBasedRLEnv,
    target_key: str,
    radius: float = 0.30,
) -> torch.Tensor:
    """Binary: target object is inside the basket. Primary success signal."""
    basket = BASKET_POS_LOCAL.to(env.device)
    dist = torch.linalg.norm(_obj_local_pos(env, target_key) - basket, dim=-1)
    return (dist < radius).float()


# ---------------------------------------------------------------------------
# Penalty terms
# ---------------------------------------------------------------------------

def wrong_object_moved(
    env: ManagerBasedRLEnv,
    target_key: str,
    slot_line_y: float = SLOT_LINE_Y,
    clearance: float = SLOT_CLEAR,
) -> torch.Tensor:
    """Penalty: any non-target object has been displaced from its slot.

    Blocks shortcuts where PPO knocks everything off the board.
    Returns count of displaced non-target objects (num_envs,).
    """
    total = torch.zeros(env.num_envs, device=env.device)
    for name in OBJECT_NAMES:
        if name == target_key:
            continue
        obj_pos = _obj_local_pos(env, name)
        moved = (obj_pos[..., 1] > slot_line_y + clearance).float()
        total = total + moved
    return total


# Module-level state for step-to-step tracking. Keyed by id(env).
_prev_progress: dict[int, torch.Tensor] = {}
_was_off_slot: dict[int, torch.Tensor] = {}
_was_in_hand: dict[int, torch.Tensor] = {}


def object_slipped(
    env: ManagerBasedRLEnv,
    target_key: str,
    slot_line_y: float = SLOT_LINE_Y,
    clearance: float = SLOT_CLEAR,
) -> torch.Tensor:
    """Penalty: target fell back onto its slot after previously being off it.

    Fires once per slip event (edge-triggered, not level). Does not fire
    when the object moves from off-slot directly into the basket.
    """
    obj_pos = _obj_local_pos(env, target_key)
    currently_off = obj_pos[..., 1] > slot_line_y + clearance

    env_id = id(env)
    is_reset = env.episode_length_buf <= 1

    if env_id not in _was_off_slot:
        _was_off_slot[env_id] = currently_off.clone()
        return torch.zeros(env.num_envs, device=env.device)

    was_off = _was_off_slot[env_id].to(env.device)
    was_off = torch.where(is_reset, currently_off, was_off)
    slipped = (was_off & ~currently_off).float()
    _was_off_slot[env_id] = currently_off.clone()
    return slipped


def grasp_lost(
    env: ManagerBasedRLEnv,
    target_key: str,
    grasp_radius: float = 0.08,
    basket_radius: float = 0.30,
) -> torch.Tensor:
    """Penalty: target was in gripper last step and dropped this step.

    Does not fire when the object lands in the basket (intentional release).
    Edge-triggered — one penalty per drop event.
    """
    dist_to_ee = torch.linalg.norm(_obj_local_pos(env, target_key) - _ee_local_pos(env), dim=-1)
    robot = env.scene["robot"]
    finger_sum = robot.data.joint_pos[:, -2:].sum(dim=-1)
    gripper_closed = finger_sum < 0.04
    currently_in_hand = (dist_to_ee < grasp_radius) & gripper_closed

    basket = BASKET_POS_LOCAL.to(env.device)
    in_basket = torch.linalg.norm(_obj_local_pos(env, target_key) - basket, dim=-1) < basket_radius

    env_id = id(env)
    is_reset = env.episode_length_buf <= 1

    if env_id not in _was_in_hand:
        _was_in_hand[env_id] = currently_in_hand.clone()
        return torch.zeros(env.num_envs, device=env.device)

    was_in_hand = _was_in_hand[env_id].to(env.device)
    was_in_hand = torch.where(is_reset, currently_in_hand, was_in_hand)
    lost = (was_in_hand & ~currently_in_hand & ~in_basket).float()
    _was_in_hand[env_id] = currently_in_hand.clone()
    return lost


def progress_stall(env: ManagerBasedRLEnv, target_key: str) -> torch.Tensor:
    """Penalty: task-relevant distance did not decrease this step.

    Tracks min(ee→target, target→basket) per env. Returns 1.0 for envs
    where this metric did not improve, 0.0 where it did.
    Penalises dithering without penalising slow careful motion.
    """
    basket = BASKET_POS_LOCAL.to(env.device)
    obj_pos = _obj_local_pos(env, target_key)
    ee_to_obj = torch.linalg.norm(obj_pos - _ee_local_pos(env), dim=-1)
    obj_to_basket = torch.linalg.norm(obj_pos - basket, dim=-1)
    current = torch.minimum(ee_to_obj, obj_to_basket)

    env_id = id(env)
    is_reset = env.episode_length_buf <= 1

    if env_id not in _prev_progress:
        _prev_progress[env_id] = current.clone()
        return torch.zeros(env.num_envs, device=env.device)

    prev = _prev_progress[env_id].to(env.device)
    prev = torch.where(is_reset, current, prev)
    no_progress = (current >= prev - 0.005).float()
    _prev_progress[env_id] = current.clone()
    return no_progress
