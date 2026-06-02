"""Shared scene constants for AIR2 reward and termination functions."""

import torch

# Basket position in env-local frame. Must match BASKET_POS_LOCAL in
# scripted_controller.py so demo trajectories and reward signal agree.
BASKET_POS_LOCAL = torch.tensor([-3.941, -5.785, 1.140])

# Robotis table: slot cylinders sit at y = -5.960 in env-local frame.
SLOT_LINE_Y = -5.960
SLOT_CLEAR = 0.15   # 15 cm off slot = object has been moved

# AIR2 pegboard hooks sit at y = -5.9 in env-local frame.
HOOK_LINE_Y = -5.9
HOOK_CLEAR = 0.20   # 20 cm off hook = object has been moved
