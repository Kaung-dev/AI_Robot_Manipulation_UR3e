"""Scripted waypoint controller for AIR2 pick-place.

Pure library code (no argparse, no AppLauncher) so multiple scripts can import
PickPlaceController without each one's argparse colliding with the importer.

Per-object sequence (real pick-and-place):

  1. APPROACH    — hover 20cm in front of object on +Y side, gripper OPEN
  2. PRE_GRASP   — close to object (5cm offset), gripper OPEN
  3. AT_OBJECT   — go to object position exactly, gripper STILL OPEN (let fingers wrap)
  4. CLOSE       — same position, gripper CLOSE, dwell to let physics settle
  5. LIFT        — +Z 20cm (clear the hook), gripper CLOSED
  6. ABOVE_BASKET— move to (basket_xy, +30cm Z), gripper CLOSED
  7. RELEASE     — same position, gripper OPEN, dwell to let object drop

Then move to the next object.

This is the version that *actually* picks objects up. The previous version
('visual-demo') only navigated and never grasped — fine for generating CNN
training images, useless for training a policy that solves the env reward.
"""

from __future__ import annotations

import torch


# Match collect_air2_demos.py expectations.
OBJECT_NAMES = ["object", "tool_pliers", "tool_scissors", "tool_silicone"]
BASKET_POS_LOCAL = torch.tensor([-3.560, -5.370, 1.040])

# Approach geometry — hooks are at y=-5.9 so objects sit on the -Y side of the
# pegboard. Approaching from +Y (toward the robot) makes sense.
APPROACH_OFFSET_Y = 0.20    # 20 cm in +Y, hover in front of object
PRE_GRASP_OFFSET_Y = 0.05   # 5 cm in +Y, just before contact
LIFT_OFFSET_Z = 0.20        # lift 20 cm above the object to clear the hook
BASKET_HOVER = 0.30         # 30 cm above the basket
GRIPPER_OPEN = 1.0          # +1 = open (matches BinaryJointPositionActionCfg)
GRIPPER_CLOSE = -1.0        # -1 = close
POSITION_TOLERANCE = 0.04   # 4 cm tolerance

ACTION_DIM = 7              # 6-DOF IK-Rel pose + 1 gripper


def make_object_waypoints(obj_xyz: torch.Tensor) -> list[tuple[torch.Tensor, float, int]]:
    """Real pick-and-place sequence per object.

    Each waypoint is (target_xyz_local, gripper_cmd, min_dwell_steps).
    The min_dwell ensures gripper actions and physics have time to settle —
    too short and the object falls before the gripper closes.
    """
    approach    = obj_xyz + torch.tensor([0.0, APPROACH_OFFSET_Y, 0.0])
    pre_grasp   = obj_xyz + torch.tensor([0.0, PRE_GRASP_OFFSET_Y, 0.0])
    at_object   = obj_xyz.clone()
    after_lift  = obj_xyz + torch.tensor([0.0, 0.0, LIFT_OFFSET_Z])
    above_basket = BASKET_POS_LOCAL + torch.tensor([0.0, 0.0, BASKET_HOVER])
    return [
        (approach,     GRIPPER_OPEN,  5),   # 1. approach from front
        (pre_grasp,    GRIPPER_OPEN,  5),   # 2. close in
        (at_object,    GRIPPER_OPEN,  5),   # 3. fingers around object
        (at_object,    GRIPPER_CLOSE, 15),  # 4. CLOSE — long dwell for physics
        (after_lift,   GRIPPER_CLOSE, 5),   # 5. lift up off hook
        (above_basket, GRIPPER_CLOSE, 5),   # 6. transport to basket
        (above_basket, GRIPPER_OPEN,  10),  # 7. RELEASE — long dwell for drop
    ]


class PickPlaceController:
    """Per-env scripted controller. Stores a waypoint queue per env."""

    def __init__(self, num_envs: int, device: torch.device):
        self.num_envs = num_envs
        self.device = device
        self.waypoints: list[list] = [[] for _ in range(num_envs)]
        self.wp_idx = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.dwell = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.done = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset(self, env_objects: list[list[torch.Tensor]]):
        """Build waypoints for each env from current object positions."""
        for i, objs in enumerate(env_objects):
            wps = []
            for obj_xyz in objs:
                wps.extend(make_object_waypoints(obj_xyz))
            self.waypoints[i] = wps
        self.wp_idx.zero_()
        self.dwell.zero_()
        self.done.zero_()

    def step(self, ee_pos_local: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute (delta_pos[num_envs, 3], gripper[num_envs]) for the env step."""
        delta_pos = torch.zeros(self.num_envs, 3, device=self.device)
        gripper = torch.full((self.num_envs,), GRIPPER_OPEN, device=self.device)

        for i in range(self.num_envs):
            if self.done[i]:
                gripper[i] = GRIPPER_OPEN
                continue
            idx = int(self.wp_idx[i])
            if idx >= len(self.waypoints[i]):
                self.done[i] = True
                continue
            target_xyz, grip_cmd, min_dwell = self.waypoints[i][idx]
            target_xyz = target_xyz.to(self.device)
            d = target_xyz - ee_pos_local[i]
            dist = torch.linalg.norm(d)
            delta_pos[i] = d
            gripper[i] = grip_cmd

            self.dwell[i] += 1
            # For gripper-action waypoints (CLOSE / RELEASE), we don't need to
            # move further — only dwell for min_dwell steps to let physics settle.
            # For position waypoints, also require reaching the target.
            position_reached = dist < POSITION_TOLERANCE
            dwell_done = self.dwell[i] >= min_dwell
            if position_reached and dwell_done:
                self.wp_idx[i] += 1
                self.dwell[i] = 0

        return delta_pos, gripper

    def all_done(self) -> bool:
        return bool(self.done.all().item())
