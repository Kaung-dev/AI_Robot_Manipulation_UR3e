"""Grasp+Drop-assist env wrapper v3 — V2 (auto-grip + B2 kinematic attach) PLUS a
policy-gated, reward-shaped RELEASE over the basket.

Why this exists
---------------
With V2 (B2 kinematic attach) the policy learned to carry the object into the basket
and then JUST HOLD IT THERE: `target_in_basket` fires every step for the held object,
so opening the gripper is pure downside (risk of bounce-out, loss of the guaranteed
attach). At eval this showed up as `release_above_basket ~= 0` and only ~30% landings —
the object reaches the basket but is never actually dropped.

V3 is the symmetric counterpart of the grasp helper. The grasp helper removed the
"learn the precise close-timing" problem (auto-close when near). V3 removes the
"learn to open over the basket" problem the same way, but keeps the RELEASE as the
policy's decision (policy-gated) and only *assists* it:

  * ANTI-HOVER  : while the object is HELD and sitting in the drop zone, apply a small
                  per-step penalty -> "sit in the basket holding it" no longer pays.
  * RELEASE BONUS: when the policy opens the gripper INSIDE the drop zone, add a large
                  one-time bonus -> opening here strictly beats holding.
  * CLEAN DROP  : on release, zero the object's velocity once so it falls straight into
                  the basket instead of inheriting the kinematic carry velocity.

All shaping lives in the wrapper (it modifies the scalar reward returned by env.step),
so there are NO env-reward edits and the SAME wrapper is used in training and eval ->
contract parity, exactly like V2.

Phases (per env) — identical to V2:
    0 APPROACH  policy arm; gripper FORCED OPEN; count steps within NEAR_RADIUS
    1 GRIP      arm FROZEN; gripper FORCED CLOSED; hold GRIP_HOLD steps; capture offset
    2 CARRY     passthrough. While held, object follows EE. Policy opens -> release.

Drop zone uses the SAME frame as the eval landing metric:
    obj_local = obj.root_pos_w - env_origins ; basket_local = (-3.941, -5.785, 1.140)
so "in the zone" lines up with what eval_ppo scores as LANDED.
"""
from __future__ import annotations

import gymnasium as gym
import torch

from isaaclab.utils.math import subtract_frame_transforms, combine_frame_transforms


class GraspAssistWrapperV3(gym.Wrapper):
    # --- grasp (identical to V2) ---
    NEAR_THRESH = 75      # steps within radius before the grip fires (~1.5s)
    GRIP_HOLD   = 50      # steps to hold the closed grip before CARRY (~1s)
    NEAR_RADIUS = 0.15    # m — basin for the auto-grip
    OPEN_VAL    = 1.0     # BinaryJointPositionActionCfg: +1 open
    CLOSE_VAL   = -1.0    # -1 close
    RELEASE_OPEN_THRESH = 0.0   # in CARRY, gripper action > this => policy wants to release

    # --- drop assist (new in V3) ---
    BASKET_LOCAL = (-3.941, -5.785, 1.140)  # env-local basket center (matches eval metric)
    R_DROP   = 0.12       # m — XY radius of the drop zone (tighter than eval's 0.25 so it lands centered)
    Z_MIN    = 1.14       # m — object must be at/above the basket rim to count as "over the basket"
    Z_MAX    = 1.55       # m — and below this (a sane carry height) so it's genuinely above the basket
    HOLD_PENALTY  = 0.5   # per-step penalty while HELD and hovering in the drop zone (anti-gaming)
    RELEASE_BONUS = 40.0  # one-time reward when the policy opens INSIDE the drop zone
    OUT_PENALTY   = 5.0   # one-time penalty for releasing OUTSIDE the zone (throwing the object away)

    def __init__(self, env: gym.Env, target_key: str = "object"):
        super().__init__(env)
        self.target_key = target_key
        self.num_envs = env.unwrapped.num_envs
        self.device = env.unwrapped.device
        self.basket_local = torch.tensor(self.BASKET_LOCAL, device=self.device)
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
        # latch so a released env stays released (no re-attach) until episode reset
        self.released = torch.zeros(n, dtype=torch.bool, device=dev)

    def _ee_pose(self):
        ee = self.env.unwrapped.scene["ee_frame"]
        return ee.data.target_pos_w[:, 0, :], ee.data.target_quat_w[:, 0, :]

    def _obj(self):
        return self.env.unwrapped.scene[self.target_key]

    def _in_drop_zone(self, obj_pos_w: torch.Tensor) -> torch.Tensor:
        """True where the object is over the basket (env-local frame, == eval metric)."""
        obj_local = obj_pos_w - self.env.unwrapped.scene.env_origins
        d_xy = torch.linalg.norm(obj_local[:, :2] - self.basket_local[:2], dim=-1)
        return (d_xy < self.R_DROP) & (obj_local[:, 2] >= self.Z_MIN) & (obj_local[:, 2] <= self.Z_MAX)

    def step(self, action: torch.Tensor):
        ee_pos, ee_quat = self._ee_pose()
        obj = self._obj()
        obj_pos, obj_quat = obj.data.root_pos_w, obj.data.root_quat_w
        ee_obj_dist = torch.linalg.norm(ee_pos - obj_pos, dim=-1)
        in_zone = self._in_drop_zone(obj_pos)   # evaluated on pre-step pose (held obj tracks EE)

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

        # --- policy-gated RELEASE: in CARRY, if the policy opens the gripper -> detach ---
        opening = assist[..., -1] > self.RELEASE_OPEN_THRESH
        releasing = self.held & (self.phase == 2) & opening
        release_in_zone  = releasing & in_zone
        release_out_zone = releasing & ~in_zone
        # hovering = held in the zone but NOT opening this step (the gaming behavior)
        hovering = self.held & (self.phase == 2) & in_zone & ~opening

        self.held = self.held & ~releasing
        self.released = self.released | releasing

        obs, reward, terminated, truncated, info = self.env.step(assist)

        # --- B2 kinematic attach: snap still-held objects to the EE (after physics step) ---
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

        # --- CLEAN DROP: zero velocity once on the release step so it falls straight in ---
        if releasing.any():
            obj = self._obj()
            vel = obj.data.root_vel_w.clone()
            vel[releasing] = 0.0
            obj.write_root_velocity_to_sim(vel)

        # --- drop-assist reward shaping (all in the wrapper; no env-reward edits) ---
        reward = reward.clone()
        reward = reward - self.HOLD_PENALTY  * hovering.float()          # anti-hover
        reward = reward + self.RELEASE_BONUS * release_in_zone.float()   # drop here -> big payoff
        reward = reward - self.OUT_PENALTY   * release_out_zone.float()  # don't throw it away early

        # --- reset state for done envs ---
        done = terminated | truncated
        if done.any():
            self.phase[done] = 0
            self.near_counter[done] = 0
            self.grip_steps[done] = 0
            self.held[done] = False
            self.released[done] = False

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._init_state()
        return self.env.reset(**kwargs)
