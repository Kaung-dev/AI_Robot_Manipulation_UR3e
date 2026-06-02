"""Grasp-assist env wrapper v2 — B1 (scripted gripper) + B2 (KINEMATIC ATTACH).

Extends Stephen's GraspAssistWrapper. The v1 wrapper scripted the gripper open/
close timing but relied on PhysX to physically hold the object after closing — and
on thin/awkward grasps it SLIPS (the same failure we saw at eval), so PPO never got
a reliable `target_in_hand` signal and never learned the carry.

v2 adds a **kinematic attach**: the instant the grip closes, we capture the object's
pose relative to the end-effector and then, every step while held, rigidly snap the
object to follow the EE (zeroing its velocity). It physically *cannot* slip, so the
carry reward gradient is clean. When the policy opens the gripper during CARRY (over
the basket), we detach and let physics drop it in.

Phases (per env):
    0 APPROACH  policy arm; gripper FORCED OPEN; count steps within NEAR_RADIUS
    1 GRIP      arm FROZEN; gripper FORCED CLOSED; hold GRIP_HOLD steps; on entry,
                capture grasp offset and mark held
    2 CARRY     passthrough (policy arm + gripper). While held, object follows EE.
                If the policy opens the gripper -> detach (release).
"""
from __future__ import annotations

import gymnasium as gym
import torch

from isaaclab.utils.math import subtract_frame_transforms, combine_frame_transforms


class GraspAssistWrapperV2(gym.Wrapper):
    NEAR_THRESH = 75      # steps within radius before the grip fires (~1.5s)
    GRIP_HOLD   = 50      # steps to hold the closed grip before CARRY (~1s)
    NEAR_RADIUS = 0.15    # m — basin for the auto-grip
    OPEN_VAL    = 1.0     # BinaryJointPositionActionCfg: +1 open
    CLOSE_VAL   = -1.0    # -1 close
    RELEASE_OPEN_THRESH = 0.0   # in CARRY, gripper action > this => policy wants to release

    def __init__(self, env: gym.Env, target_key: str = "object"):
        super().__init__(env)
        self.target_key = target_key
        self.num_envs = env.unwrapped.num_envs
        self.device = env.unwrapped.device
        self._init_state()

    def _init_state(self) -> None:
        n, dev = self.num_envs, self.device
        self.phase = torch.zeros(n, dtype=torch.long, device=dev)
        self.near_counter = torch.zeros(n, dtype=torch.long, device=dev)
        self.grip_steps = torch.zeros(n, dtype=torch.long, device=dev)
        # B2 attach state
        self.held = torch.zeros(n, dtype=torch.bool, device=dev)
        self.off_pos = torch.zeros(n, 3, device=dev)
        self.off_quat = torch.zeros(n, 4, device=dev); self.off_quat[:, 0] = 1.0

    def _ee_pose(self):
        ee = self.env.unwrapped.scene["ee_frame"]
        return ee.data.target_pos_w[:, 0, :], ee.data.target_quat_w[:, 0, :]

    def _obj(self):
        return self.env.unwrapped.scene[self.target_key]

    def step(self, action: torch.Tensor):
        ee_pos, ee_quat = self._ee_pose()
        obj = self._obj()
        obj_pos, obj_quat = obj.data.root_pos_w, obj.data.root_quat_w
        ee_obj_dist = torch.linalg.norm(ee_pos - obj_pos, dim=-1)

        # --- phase 0 -> 1: count steps near, then grip ---
        self.near_counter = torch.where(
            (self.phase == 0) & (ee_obj_dist < self.NEAR_RADIUS),
            self.near_counter + 1, self.near_counter)
        self.phase = torch.where(
            (self.phase == 0) & (self.near_counter >= self.NEAR_THRESH),
            torch.ones_like(self.phase), self.phase)

        # --- phase 1 -> 2: hold grip, then hand control to policy ---
        self.grip_steps = torch.where(self.phase == 1, self.grip_steps + 1, self.grip_steps)
        entering_carry = (self.phase == 1) & (self.grip_steps >= self.GRIP_HOLD)
        # On entry to CARRY, capture the grasp offset (object pose in EE frame) + mark held.
        if entering_carry.any():
            off_p, off_q = subtract_frame_transforms(ee_pos, ee_quat, obj_pos, obj_quat)
            self.off_pos = torch.where(entering_carry.unsqueeze(-1), off_p, self.off_pos)
            self.off_quat = torch.where(entering_carry.unsqueeze(-1), off_q, self.off_quat)
            self.held = self.held | entering_carry
        self.phase = torch.where(entering_carry, torch.full_like(self.phase, 2), self.phase)

        # --- action override by phase (B1) ---
        assist = action.clone()
        p0, p1 = self.phase == 0, self.phase == 1
        assist[..., -1] = torch.where(p0, torch.full_like(assist[..., -1], self.OPEN_VAL), assist[..., -1])
        p1_arm = p1.unsqueeze(-1).expand_as(assist[..., :-1])
        assist[..., :-1] = torch.where(p1_arm, torch.zeros_like(assist[..., :-1]), assist[..., :-1])
        assist[..., -1] = torch.where(p1, torch.full_like(assist[..., -1], self.CLOSE_VAL), assist[..., -1])

        # Release: in CARRY, if the policy opens the gripper -> detach.
        releasing = self.held & (self.phase == 2) & (assist[..., -1] > self.RELEASE_OPEN_THRESH)
        self.held = self.held & ~releasing

        obs, reward, terminated, truncated, info = self.env.step(assist)

        # --- B2 kinematic attach: snap held objects to the EE (after physics step) ---
        if self.held.any():
            ee_pos2, ee_quat2 = self._ee_pose()
            tgt_pos, tgt_quat = combine_frame_transforms(ee_pos2, ee_quat2, self.off_pos, self.off_quat)
            obj = self._obj()
            root_pose = torch.cat([obj.data.root_pos_w, obj.data.root_quat_w], dim=-1).clone()
            root_pose[self.held, :3] = tgt_pos[self.held]
            root_pose[self.held, 3:7] = tgt_quat[self.held]
            obj.write_root_pose_to_sim(root_pose)
            vel = obj.data.root_vel_w.clone()
            vel[self.held] = 0.0
            obj.write_root_velocity_to_sim(vel)

        # --- reset state for done envs ---
        done = terminated | truncated
        if done.any():
            self.phase[done] = 0
            self.near_counter[done] = 0
            self.grip_steps[done] = 0
            self.held[done] = False

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._init_state()
        return self.env.reset(**kwargs)
