# Reset-time randomization events for Isaac Lab manager-based RL tasks.
#
# Extracted from robotis_lab's FFW-SG2 pick-place task. The notable piece is
# `randomize_table_with_objects_on_slots`: the table is jittered first, then each
# object is placed at a fixed offset *in the table frame*, so the whole scene
# moves rigidly. This preserves relative geometry between objects and any anchor
# (e.g. a basket), which is what makes recorded demos still valid after
# augmentation.
#
# Wiring example (inside an EventCfg):
#
#   from .randomization_events import (
#       set_default_joint_pose, randomize_joint_by_gaussian_offset,
#       randomize_table_with_objects_on_slots, randomize_camera_pose,
#       randomize_scene_lighting_domelight, randomize_background_color,
#   )
#
#   @configclass
#   class EventCfg:
#       set_robot_pose = EventTerm(
#           func=set_default_joint_pose, mode="reset",
#           params={"joint_positions": {"joint1": 0.0, ...},
#                   "asset_cfg": SceneEntityCfg("robot")},
#       )
#       jitter_robot = EventTerm(
#           func=randomize_joint_by_gaussian_offset, mode="reset",
#           params={"mean": 0.0, "std": 0.05,
#                   "joint_names": ["joint1", "joint2", ...],
#                   "asset_cfg": SceneEntityCfg("robot")},
#       )
#       randomize_scene = EventTerm(
#           func=randomize_table_with_objects_on_slots, mode="reset",
#           params={
#               "table_cfg": SceneEntityCfg("table"),
#               "anchor_cfg": SceneEntityCfg("basket"),
#               "anchor_relative_pose": {"x": 0.41, "y": 0.0, "z": 0.72},
#               "target_asset_cfg": SceneEntityCfg("brush"),
#               "other_asset_cfgs": [SceneEntityCfg(n) for n in
#                                    ("scissors", "pliers", "tooth_brush", ...)],
#               "slots": LEFT_SLOTS + RIGHT_SLOTS,
#               "target_slots": LEFT_SLOTS,        # which subset target may use
#               "table_pose_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02),
#                                    "yaw": (-0.05, 0.05)},
#           },
#       )

from __future__ import annotations

import random
import torch
from typing import Literal, TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, AssetBase
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.camera import Camera

from pxr import Gf, UsdShade

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# -----------------------------------------------------------------------------
# Slot layouts — example values from the FFW-SG2 task. Replace with your own
# slot positions in the table's local frame (x forward, y left, z up).
# -----------------------------------------------------------------------------

LEFT_SLOTS = [
    (0.555, 0.260, 0.95),
    (0.555, 0.087, 0.95),
    (0.555, 0.260, 1.235),
    (0.555, 0.087, 1.235),
]

RIGHT_SLOTS = [
    (0.555, -0.087, 0.95),
    (0.555, -0.260, 0.95),
    (0.555, -0.087, 1.235),
    (0.555, -0.260, 1.235),
]


# -----------------------------------------------------------------------------
# Robot joint randomization
# -----------------------------------------------------------------------------

def _joint_position_mapping(joint_names: list[str], desired: dict[str, float]) -> torch.Tensor:
    return torch.tensor(
        [desired.get(n, 0.0) for n in joint_names], dtype=torch.float32
    )


def set_default_joint_pose(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    joint_positions: dict[str, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Write a fixed start pose to the robot, addressing joints by name."""
    asset: Articulation = env.scene[asset_cfg.name]
    pose = _joint_position_mapping(asset.joint_names, joint_positions).to(env.device)
    if pose.dim() == 1:
        pose = pose.unsqueeze(0).repeat(len(env_ids), 1)
    asset.set_joint_position_target(pose, env_ids=env_ids)
    asset.write_joint_state_to_sim(pose, torch.zeros_like(pose), env_ids=env_ids)


def randomize_joint_by_gaussian_offset(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    mean: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_names: list[str] | None = None,
):
    """Add Gaussian noise to current joint positions, clamped to soft limits."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[env_ids].clone()
    joint_vel = asset.data.joint_vel[env_ids].clone()

    if joint_names is not None:
        for name in joint_names:
            if name in asset.joint_names:
                idx = asset.joint_names.index(name)
                noise = math_utils.sample_gaussian(mean, std, (len(env_ids), 1), joint_pos.device)
                joint_pos[:, idx:idx + 1] += noise
    else:
        joint_pos += math_utils.sample_gaussian(mean, std, joint_pos.shape, joint_pos.device)

    limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(limits[..., 0], limits[..., 1])

    asset.set_joint_position_target(joint_pos, env_ids=env_ids)
    asset.set_joint_velocity_target(joint_vel, env_ids=env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


# -----------------------------------------------------------------------------
# Slot-based scene placement (table + objects move as a rigid frame)
# -----------------------------------------------------------------------------

def randomize_table_with_objects_on_slots(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    table_cfg: SceneEntityCfg,
    target_asset_cfg: SceneEntityCfg,
    other_asset_cfgs: list[SceneEntityCfg],
    slots: list[tuple[float, float, float]],
    target_slots: list[tuple[float, float, float]] | None = None,
    table_pose_range: dict[str, tuple[float, float]] | None = None,
    anchor_cfg: SceneEntityCfg | None = None,
    anchor_relative_pose: dict[str, float] | None = None,
):
    """Jitter the table, then place a target + N other objects on table-frame slots.

    The table pose is sampled once; every object (and the optional anchor, e.g.
    a basket) is positioned at a fixed offset in the table's local frame and
    rotated/translated into world space. This keeps the relative scene geometry
    intact across resets, so recorded demos remain valid after augmentation.

    target_slots: subset of `slots` that the target object may occupy. Defaults
    to all slots. The remaining slots are shuffled and assigned to others.
    """
    if env_ids is None:
        return

    if table_pose_range is None:
        table_pose_range = {}
    if target_slots is None:
        target_slots = slots

    table_asset = env.scene[table_cfg.name]
    target_asset = env.scene[target_asset_cfg.name]
    anchor_asset = env.scene[anchor_cfg.name] if anchor_cfg is not None else None

    for cur_env in env_ids.tolist():
        env_id_t = torch.tensor([cur_env], device=env.device)

        # 1. Sample table pose
        ranges = [table_pose_range.get(k, (0.0, 0.0)) for k in
                  ("x", "y", "z", "roll", "pitch", "yaw")]
        s = [random.uniform(lo, hi) for (lo, hi) in ranges]
        table_pos = torch.tensor([s[0:3]], device=env.device) + env.scene.env_origins[cur_env, 0:3]
        table_quat = math_utils.quat_from_euler_xyz(
            torch.tensor([s[3]], device=env.device),
            torch.tensor([s[4]], device=env.device),
            torch.tensor([s[5]], device=env.device),
        )

        # 2. Write table
        table_asset.write_root_pose_to_sim(
            torch.cat([table_pos, table_quat], dim=-1), env_ids=env_id_t,
        )
        table_asset.write_root_velocity_to_sim(
            torch.zeros(1, 6, device=env.device), env_ids=env_id_t,
        )

        # 3. Optional anchor (e.g. basket) at fixed offset in table frame
        if anchor_asset is not None and anchor_relative_pose is not None:
            rel_pos = torch.tensor([[
                anchor_relative_pose.get("x", 0.0),
                anchor_relative_pose.get("y", 0.0),
                anchor_relative_pose.get("z", 0.0),
            ]], device=env.device)
            rel_quat = math_utils.quat_from_euler_xyz(
                torch.tensor([anchor_relative_pose.get("roll", 0.0)], device=env.device),
                torch.tensor([anchor_relative_pose.get("pitch", 0.0)], device=env.device),
                torch.tensor([anchor_relative_pose.get("yaw", 0.0)], device=env.device),
            )
            world_pos = table_pos + math_utils.quat_apply(table_quat, rel_pos)
            world_quat = math_utils.quat_mul(table_quat, rel_quat)
            anchor_asset.write_root_pose_to_sim(
                torch.cat([world_pos, world_quat], dim=-1), env_ids=env_id_t,
            )
            anchor_asset.write_root_velocity_to_sim(
                torch.zeros(1, 6, device=env.device), env_ids=env_id_t,
            )

        # 4. Place target on a random allowed slot
        remaining = list(slots)
        target_slot = random.choice(target_slots)
        if target_slot in remaining:
            remaining.remove(target_slot)

        _place_on_slot(env, target_asset, target_slot, table_pos, table_quat, env_id_t)

        # 5. Place others on shuffled remaining slots
        random.shuffle(remaining)
        for cfg, slot in zip(other_asset_cfgs, remaining):
            asset = env.scene[cfg.name]
            _place_on_slot(env, asset, slot, table_pos, table_quat, env_id_t)


def _place_on_slot(env, asset, slot, table_pos, table_quat, env_id_t):
    rel = torch.tensor([list(slot)], device=env.device)
    pos = table_pos + math_utils.quat_apply(table_quat, rel)
    quat = table_quat.clone()
    asset.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_id_t)
    asset.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device), env_ids=env_id_t)


# -----------------------------------------------------------------------------
# Visual domain randomization
# -----------------------------------------------------------------------------

def randomize_camera_pose(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose_range: dict[str, tuple[float, float]] | None = None,
    convention: Literal["opengl", "ros", "world"] = "ros",
):
    """Jitter a camera around its initial pose. Initial pose is captured on first call."""
    if pose_range is None:
        pose_range = {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.02, 0.02),
                      "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.1, 0.1)}

    asset: Camera = env.scene[asset_cfg.name]

    if not hasattr(asset, "_initial_pos_w"):
        asset._initial_pos_w = asset.data.pos_w.clone()
        asset._initial_quat_w_ros = asset.data.quat_w_ros.clone()
        asset._initial_quat_w_opengl = asset.data.quat_w_opengl.clone()
        asset._initial_quat_w_world = asset.data.quat_w_world.clone()

    ori_pos_w = asset._initial_pos_w
    ori_quat_w = {
        "ros": asset._initial_quat_w_ros,
        "opengl": asset._initial_quat_w_opengl,
        "world": asset._initial_quat_w_world,
    }[convention]

    ranges = torch.tensor(
        [pose_range.get(k, (0.0, 0.0)) for k in ("x", "y", "z", "roll", "pitch", "yaw")],
        device=asset.device,
    )
    samples = math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device,
    )

    for i, env_id in enumerate(env_ids.tolist()):
        pos = ori_pos_w[env_id, 0:3] + samples[i, 0:3]
        delta = math_utils.quat_from_euler_xyz(samples[i, 3], samples[i, 4], samples[i, 5])
        quat = math_utils.quat_mul(ori_quat_w[env_id], delta)
        asset.set_world_poses(
            pos.unsqueeze(0), quat.unsqueeze(0),
            env_ids=torch.tensor([env_id], device=asset.device),
            convention=convention,
        )


def randomize_scene_lighting_domelight(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    intensity_range: tuple[float, float] = (1000.0, 3000.0),
    color_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = ((0.5, 1.0), (0.5, 1.0), (0.5, 1.0)),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("light"),
):
    """Randomize a DomeLight's intensity and tint each reset."""
    asset: AssetBase = env.scene[asset_cfg.name]
    light_prim = asset.prims[0]

    light_prim.GetAttribute("inputs:intensity").Set(
        random.uniform(*intensity_range)
    )
    light_prim.GetAttribute("inputs:color").Set(Gf.Vec3f(
        random.uniform(*color_range[0]),
        random.uniform(*color_range[1]),
        random.uniform(*color_range[2]),
    ))


def randomize_background_color(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    color_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
    prim_name: str = "BackgroundCube",
):
    """Randomize the diffuse color of a background prim across all envs."""
    if env_ids is None:
        return

    import omni.usd
    stage = omni.usd.get_context().get_stage()

    for cur_env in env_ids.tolist():
        prim = stage.GetPrimAtPath(f"/World/envs/env_{cur_env}/{prim_name}")
        if not prim.IsValid():
            continue
        color = Gf.Vec3f(
            random.uniform(*color_range[0]),
            random.uniform(*color_range[1]),
            random.uniform(*color_range[2]),
        )
        _set_material_color(stage, prim, color)


def _set_material_color(stage, prim, color: Gf.Vec3f) -> bool:
    prim_path = prim.GetPath().pathString
    for mat_suffix in ("/Looks/PreviewSurface", "/Looks/material_0", "/Material", "/Looks"):
        mat = stage.GetPrimAtPath(f"{prim_path}{mat_suffix}")
        if mat.IsValid():
            shader = stage.GetPrimAtPath(f"{mat.GetPath().pathString}/Shader")
            if not shader.IsValid():
                shader = mat
            for attr in ("inputs:diffuseColor", "inputs:diffuse_color_constant", "inputs:baseColor"):
                a = shader.GetAttribute(attr)
                if a.IsValid():
                    a.Set(color)
                    return True
    return _walk_geometry(prim, color)


def _walk_geometry(node, color: Gf.Vec3f) -> bool:
    for child in node.GetChildren():
        if child.GetTypeName() in ("Mesh", "Cube") or "Geom" in child.GetTypeName():
            mat, _ = UsdShade.MaterialBindingAPI(child).ComputeBoundMaterial()
            if mat:
                for shader in mat.GetPrim().GetChildren():
                    if shader.GetTypeName() == "Shader":
                        for attr in ("inputs:diffuseColor", "inputs:diffuse_color_constant", "inputs:baseColor"):
                            a = shader.GetAttribute(attr)
                            if a.IsValid():
                                a.Set(color)
                                return True
        if _walk_geometry(child, color):
            return True
    return False
