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


def grasp_shaping(
    env: ManagerBasedRLEnv,
    target_key: str,
    near_radius: float = 0.15,
) -> torch.Tensor:
    """Dense per-step reward for closing the gripper while near the target.

    Bridges the gap between `ee_to_target` (always-on Gaussian near object) and
    `target_in_hand` (binary AND-of-two-conditions). Without this, PPO finds
    the local optimum of "hover near target, do not move" — `target_in_hand`
    never fires because closing the gripper temporarily reduces `ee_to_target`
    (the body moves a tiny bit) and incurs `action_rate` penalty.

    Returns (num_envs,) in [0, 1]:
        near    = 1 if EE within near_radius of target, smoothly decays outside
        closure = 1 when fingers fully closed (sum=0), 0 when fully open (sum=0.08)
        score   = near * closure
    """
    dist = torch.linalg.norm(_obj_local_pos(env, target_key) - _ee_local_pos(env), dim=-1)
    near = torch.exp(-(dist / near_radius) ** 2)              # Gaussian, std=near_radius
    robot = env.scene["robot"]
    finger_sum = robot.data.joint_pos[:, -2:].sum(dim=-1)     # 0 closed, 0.08 open
    closure = torch.clamp((0.08 - finger_sum) / 0.08, 0.0, 1.0)
    return near * closure


def lift_progress(
    env: ManagerBasedRLEnv,
    target_key: str,
    base_z: float = 1.61,
    max_lift: float = 0.30,
) -> torch.Tensor:
    """Continuous reward for raising the target object above its spawn z.

    Slot z is ~1.61 m (see _SLOTS in air2_robotis_franka cfg). Reward climbs
    linearly from 0 (at spawn height) to 1.0 (when target is 30 cm above).

    Implicitly conditions on grasp: only a truly grasped object stays in the
    air; bumped/swatted objects fall back down within a few steps. This is the
    missing gradient between "grasp briefly" and "carry to basket" — without
    it, PPO has no incentive to LIFT after grasping.
    """
    target_pos = _obj_local_pos(env, target_key)
    height_above = torch.clamp(target_pos[..., 2] - base_z, 0.0, max_lift)
    return height_above / max_lift


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


def release_above_basket(
    env: ManagerBasedRLEnv,
    target_key: str,
    xy_radius: float = 0.18,
    height_above_rim: float = 0.10,
    finger_open_thresh: float = 0.03,
) -> torch.Tensor:
    """Reward for opening the gripper while object is hovering over the basket.

    Bridges the gap between the loose `target_to_basket` Gaussian (always-on
    pull) and the strict `target_dropped_in_basket` termination (gripper must
    be open AND object inside basket). Without this term PPO has no incentive
    to release — it just hovers gripped above the basket forever.

    Returns 1.0 when:
        - object's XY position is inside the basket footprint
        - object's Z is at least height_above_rim above basket center
        - gripper fingers are open (sum > finger_open_thresh)
    Otherwise 0.0.
    """
    basket = BASKET_POS_LOCAL.to(env.device)
    obj_pos = _obj_local_pos(env, target_key)
    xy_dist = torch.linalg.norm(obj_pos[..., :2] - basket[:2], dim=-1)
    inside_xy = xy_dist < xy_radius
    above_rim = obj_pos[..., 2] > basket[2] + height_above_rim

    robot = env.scene["robot"]
    finger_sum = robot.data.joint_pos[:, -2:].sum(dim=-1)
    gripper_open = finger_sum > finger_open_thresh

    return (inside_xy & above_rim & gripper_open).float()


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


def tool_fell_to_floor(
    env: ManagerBasedRLEnv,
    target_key: str,
    floor_z: float = 1.0,
) -> torch.Tensor:
    """Penalty: count of ANY tools (target + distractors) whose Z fell below floor_z.

    Slot height is ~1.61 m and basket rim ~1.19 m; anything below floor_z = 1.0 m
    is clearly knocked off and dropped. Counts target too — keeps the policy from
    just batting the brush onto the floor rather than carrying it.
    """
    total = torch.zeros(env.num_envs, device=env.device)
    for name in OBJECT_NAMES:
        obj_pos = _obj_local_pos(env, name)
        total = total + (obj_pos[..., 2] < floor_z).float()
    return total


def joint_near_limit(
    env: ManagerBasedRLEnv,
    margin_frac: float = 0.05,
) -> torch.Tensor:
    """Penalty: count of arm joints within margin_frac of their hard limit.

    Proxy for near-singularity / wrist lock. The Franka panda hits gimbal
    lock most often via joint 4 / 6 reaching their lower bound; once any
    joint sits at its limit the differential-IK controller can't follow
    further pose deltas in that direction.
    """
    robot = env.scene["robot"]
    joint_pos = robot.data.joint_pos[:, :7]  # arm only (panda_joint1..7); skip the 2 finger joints
    limits = robot.data.joint_pos_limits[:, :7]  # (num_envs, 7, 2)
    lower = limits[..., 0]
    upper = limits[..., 1]
    pos_range = upper - lower
    margin = pos_range * margin_frac
    near = ((joint_pos < lower + margin) | (joint_pos > upper - margin)).float()
    return near.sum(dim=-1)


def arm_stuck(
    env: ManagerBasedRLEnv,
    jvel_threshold: float = 0.01,
    action_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalty: arm commanded to move but joints aren't moving.

    Fires when the policy outputs a non-trivial arm action but the actual
    joint velocities are ~zero. Catches singular configurations, contacts
    against the environment, and joint-limit lockup.
    """
    robot = env.scene["robot"]
    joint_vel_max = robot.data.joint_vel[:, :7].abs().max(dim=-1).values
    last_action = env.action_manager.action  # (num_envs, action_dim)
    action_mag = last_action[:, :6].abs().max(dim=-1).values  # arm action only (first 6 dims)
    stuck = (joint_vel_max < jvel_threshold) & (action_mag > action_threshold)
    return stuck.float()


def ee_out_of_workspace(
    env: ManagerBasedRLEnv,
    x_min: float = -5.5, x_max: float = -3.5,
    y_min: float = -6.0, y_max: float = -5.0,
    z_min: float = 0.95, z_max: float = 2.2,
) -> torch.Tensor:
    """Penalty proportional to how far the EE has left the workspace box.

    Returns the Euclidean distance from the EE position to the nearest
    point inside the box (0 when EE is inside). Box edges chosen to keep
    the EE between the pegboard (y < -5.95 would collide) and the open
    right side (x > -3.5), above the table (z < 0.95), and below ceiling
    (z > 2.2).
    """
    ee_pos = _ee_local_pos(env)
    dx = torch.clamp(x_min - ee_pos[..., 0], min=0.0) + torch.clamp(ee_pos[..., 0] - x_max, min=0.0)
    dy = torch.clamp(y_min - ee_pos[..., 1], min=0.0) + torch.clamp(ee_pos[..., 1] - y_max, min=0.0)
    dz = torch.clamp(z_min - ee_pos[..., 2], min=0.0) + torch.clamp(ee_pos[..., 2] - z_max, min=0.0)
    return torch.sqrt(dx * dx + dy * dy + dz * dz)


def gripper_sky_pointing(
    env: ManagerBasedRLEnv,
    sky_thresh: float = 0.3,
) -> torch.Tensor:
    """Penalty: gripper forward axis pointing upward (sky-pointing pose).

    panda_hand local +Z is the gripper-forward direction (the way the
    fingers point). For a sensible grasp pose this should point
    horizontally or downward, NOT toward the sky. Returns the surplus
    above sky_thresh of the world-Z component of the rotated forward
    axis — soft proportional penalty.

    Quaternion identity: rotating local [0,0,1] by q=(w,x,y,z) gives a
    world vector whose z-component is `1 - 2*(x^2 + y^2)`.
    """
    # Use the existing ee_frame FrameTransformer's TCP quat (= panda_hand quat).
    ee_quat = env.scene["ee_frame"].data.target_quat_w[..., 0, :]  # (num_envs, 4) wxyz
    x = ee_quat[..., 1]
    y = ee_quat[..., 2]
    forward_world_z = 1.0 - 2.0 * (x * x + y * y)
    return torch.clamp(forward_world_z - sky_thresh, min=0.0)


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
