"""Reset-time event: place 4 objects on 4 randomly chosen slots out of 8."""
import torch

SLOT_POSITIONS = [
    [-3.925, -5.960, 1.611],  # L0
    [-4.098, -5.960, 1.611],  # L1
    [-3.925, -5.960, 1.326],  # L2
    [-4.098, -5.960, 1.326],  # L3
    [-4.272, -5.960, 1.611],  # R0
    [-4.445, -5.960, 1.611],  # R1
    [-4.272, -5.960, 1.326],  # R2
    [-4.445, -5.960, 1.326],  # R3
]

_OBJECT_NAMES = ["object", "tool_pliers", "tool_scissors", "tool_silicone"]

_TOOL_QUAT = [0.7071, 0.0, 0.0, -0.7071]


def reset_objects_on_slots(env, env_ids: torch.Tensor):
    """Pick 4 random slots from 8 and place one object on each."""
    slots = torch.tensor(SLOT_POSITIONS, device=env.device)  # [8, 3]
    n = len(env_ids)
    quat = torch.tensor(_TOOL_QUAT, device=env.device).expand(n, -1)

    perms = torch.argsort(torch.rand(n, 8, device=env.device), dim=1)  # [n, 8]

    # object (paintbrush) col 0: avoid R3 (idx 7)
    # tool_silicone (screwdriver) col 3: avoid L1 (idx 1) and R3 (idx 7)
    for col, forbidden_slots in [(0, (7,)), (3, (1, 7))]:
        for forbidden in forbidden_slots:
            mask = perms[:, col] == forbidden
            tmp = perms[mask, col].clone()
            perms[mask, col] = perms[mask, col + 1]
            perms[mask, col + 1] = tmp

    for i, name in enumerate(_OBJECT_NAMES):
        asset = env.scene[name]
        positions = env.scene.env_origins[env_ids] + slots[perms[:, i]]
        asset.write_root_pose_to_sim(
            torch.cat([positions, quat], dim=-1), env_ids=env_ids
        )
        asset.write_root_velocity_to_sim(
            torch.zeros(n, 6, device=env.device), env_ids=env_ids
        )
