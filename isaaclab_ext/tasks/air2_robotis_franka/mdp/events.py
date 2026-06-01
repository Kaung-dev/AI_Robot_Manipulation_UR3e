"""Reset-time event: place 4 objects on the 4 right-side slots (R0–R3).

Slot layout (right side only):
    R0: idx 4  [-4.272, -5.960, 1.611]   upper-left of right panel
    R1: idx 5  [-4.445, -5.960, 1.611]   upper-right of right panel
    R2: idx 6  [-4.272, -5.960, 1.326]   lower-left of right panel
    R3: idx 7  [-4.445, -5.960, 1.326]   lower-right of right panel

Forbidden placements (from physical reachability testing):
    brush (col 0)       — cannot reach R3
    screwdriver (col 3) — cannot reach R3
R3 is therefore always assigned to pliers or scissors.
"""
import torch

from isaaclab_ext.tasks.air2_franka.objects import OBJECT_NAMES

SLOT_POSITIONS = [
    [-3.925, -5.960, 1.611],  # L0  idx 0
    [-4.098, -5.960, 1.611],  # L1  idx 1
    [-3.925, -5.960, 1.326],  # L2  idx 2
    [-4.098, -5.960, 1.326],  # L3  idx 3
    [-4.272, -5.960, 1.611],  # R0  idx 4
    [-4.445, -5.960, 1.611],  # R1  idx 5
    [-4.272, -5.960, 1.326],  # R2  idx 6
    [-4.445, -5.960, 1.326],  # R3  idx 7
]

# Only right-side slots are used.
RIGHT_SLOT_INDICES = [4, 5, 6, 7]  # R0, R1, R2, R3

_OBJECT_NAMES = list(OBJECT_NAMES)  # [brush, pliers, scissors, screwdriver]
_TOOL_QUAT = [0.7071, 0.0, 0.0, -0.7071]


def reset_objects_on_slots(env, env_ids: torch.Tensor):
    """Place all 4 objects on the 4 right-side slots, one per slot.

    Each reset picks a random permutation of R0-R3 and assigns one object per
    slot. Brush and screwdriver cannot reach R3 so any draw of R3 for those
    objects is swapped with the adjacent column (pliers or scissors).
    """
    slots = torch.tensor(SLOT_POSITIONS, device=env.device)     # [8, 3]
    right = torch.tensor(RIGHT_SLOT_INDICES, device=env.device) # [4]
    n = len(env_ids)
    quat = torch.tensor(_TOOL_QUAT, device=env.device).expand(n, -1)

    # Random permutation of the 4 right-slot indices per env → shape (n, 4)
    perm_within = torch.argsort(torch.rand(n, 4, device=env.device), dim=1)
    perms = right[perm_within]  # (n, 4) — global slot indices

    # Fix brush (col 0): swap with pliers (col 1) if brush drew R3.
    mask = perms[:, 0] == 7
    perms[mask, 0], perms[mask, 1] = perms[mask, 1].clone(), perms[mask, 0].clone()

    # Fix screwdriver (col 3): swap with scissors (col 2) if screwdriver drew R3.
    mask = perms[:, 3] == 7
    perms[mask, 3], perms[mask, 2] = perms[mask, 2].clone(), perms[mask, 3].clone()

    for i, name in enumerate(_OBJECT_NAMES):
        asset = env.scene[name]
        positions = env.scene.env_origins[env_ids] + slots[perms[:, i]]
        asset.write_root_pose_to_sim(
            torch.cat([positions, quat], dim=-1), env_ids=env_ids
        )
        asset.write_root_velocity_to_sim(
            torch.zeros(n, 6, device=env.device), env_ids=env_ids
        )
