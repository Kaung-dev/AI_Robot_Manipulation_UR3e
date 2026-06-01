"""Shared scene constants for AIR2 reward and termination functions."""

import torch

# Basket position in env-local frame. Must match BASKET_POS_LOCAL in
# scripted_controller.py so demo trajectories and reward signal agree.
#
# 2026-06-01: corrected via _probe_basket_pos.py. The Outer Xform prim
# `/Env_0/Environment/SM_BoxPortableD` sits at (-3.560, -5.370, 1.040), but
# the actual basket mesh (`SM_BoxPortableD/SM_BoxPortableD`) has a 36 cm / 42 cm
# local offset, so its world origin is at (-3.918, -5.787, 1.042). Reward and
# success checks must target the mesh, not the wrapper.
BASKET_POS_LOCAL = torch.tensor([-3.918, -5.787, 1.042])

# Robotis table: slot cylinders sit at y = -5.960 in env-local frame.
SLOT_LINE_Y = -5.960
SLOT_CLEAR = 0.15   # 15 cm off slot = object has been moved

# AIR2 pegboard hooks sit at y = -5.9 in env-local frame.
HOOK_LINE_Y = -5.9
HOOK_CLEAR = 0.20   # 20 cm off hook = object has been moved
