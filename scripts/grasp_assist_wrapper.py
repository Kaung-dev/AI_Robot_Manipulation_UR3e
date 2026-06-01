"""Grasp-assist env wrapper for PPO training and inference.

Automates the moment of grasp so the policy only needs to learn approach,
carry, and release. Ported from Stephen's eval_state_bc.py phase machine.

Per-env phase state machine:
    Phase 0  APPROACH  policy controls arm; gripper FORCED OPEN (+1)
                       count steps within NEAR_RADIUS of target
    Phase 1  GRIP      arm FROZEN (action zeroed); gripper FORCED CLOSED (-1)
                       hold for GRIP_HOLD steps
    Phase 2  CARRY     policy controls everything (passthrough)

The wrapped policy SEES its own raw action and the resulting state — it has
no idea the wrapper intervened. PPO therefore learns a policy that assumes
the gripper will auto-close at the right moment. Same wrapper is used at
inference to reproduce that contract.
"""
from __future__ import annotations

import gymnasium as gym
import torch


class GraspAssistWrapper(gym.Wrapper):
    NEAR_THRESH = 250    # steps within radius before phase 0 → 1 trigger (~5s at 50Hz)
    GRIP_HOLD = 50       # steps to hold the grip before phase 1 → 2 (~1s)
    NEAR_RADIUS = 0.08   # meters; EE-to-object proximity for "near"
    OPEN_VAL = 1.0       # BinaryJointPositionActionCfg: +1 → open_command
    CLOSE_VAL = -1.0     # -1 → close_command

    def __init__(self, env: gym.Env, target_key: str = "object"):
        super().__init__(env)
        self.target_key = target_key
        self.num_envs = env.unwrapped.num_envs
        self.device = env.unwrapped.device
        self._init_state()

    def _init_state(self) -> None:
        self.phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.near_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.grip_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _ee_obj_dist(self) -> torch.Tensor:
        scene = self.env.unwrapped.scene
        ee_pos = scene["ee_frame"].data.target_pos_w[..., 0, :]
        obj_pos = scene[self.target_key].data.root_pos_w
        return torch.linalg.norm(ee_pos - obj_pos, dim=-1)

    def step(self, action: torch.Tensor):
        ee_obj_dist = self._ee_obj_dist()

        # Phase 0 → 1: count steps near; trigger at NEAR_THRESH
        self.near_counter = torch.where(
            (self.phase == 0) & (ee_obj_dist < self.NEAR_RADIUS),
            self.near_counter + 1,
            self.near_counter,
        )
        self.phase = torch.where(
            (self.phase == 0) & (self.near_counter >= self.NEAR_THRESH),
            torch.ones_like(self.phase),
            self.phase,
        )

        # Phase 1 → 2: hold grip for GRIP_HOLD steps, then release control to policy
        self.grip_steps = torch.where(
            self.phase == 1,
            self.grip_steps + 1,
            self.grip_steps,
        )
        self.phase = torch.where(
            (self.phase == 1) & (self.grip_steps >= self.GRIP_HOLD),
            torch.full_like(self.phase, 2),
            self.phase,
        )

        # Override the action based on phase
        assist = action.clone()
        p0 = self.phase == 0
        p1 = self.phase == 1
        # Phase 0: gripper open
        assist[..., -1] = torch.where(p0, torch.full_like(assist[..., -1], self.OPEN_VAL), assist[..., -1])
        # Phase 1: arm zero + gripper closed
        p1_arm = p1.unsqueeze(-1).expand_as(assist[..., :-1])
        assist[..., :-1] = torch.where(p1_arm, torch.zeros_like(assist[..., :-1]), assist[..., :-1])
        assist[..., -1] = torch.where(p1, torch.full_like(assist[..., -1], self.CLOSE_VAL), assist[..., -1])
        # Phase 2: passthrough — leave assist unchanged

        obs, reward, terminated, truncated, info = self.env.step(assist)

        # Reset phase state for done envs (vectorized)
        done = terminated | truncated
        if done.any():
            self.phase[done] = 0
            self.near_counter[done] = 0
            self.grip_steps[done] = 0

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._init_state()
        return self.env.reset(**kwargs)
