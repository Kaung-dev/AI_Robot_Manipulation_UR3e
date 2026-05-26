"""Reset-time event: each object picks randomly between its 2 allowed slots."""
import torch

# Each object has exactly 2 candidate slots [top, bottom].
_OBJECT_SLOTS = {
    "object":       [[-4.098, -5.960, 1.611],  # L1
                     [-4.098, -5.960, 1.326]], # L3
    "tool_pliers":  [[-3.925, -5.960, 1.611],  # L0
                     [-3.925, -5.960, 1.326]], # L2
    "tool_scissors":[[-4.272, -5.960, 1.611],  # R0
                     [-4.272, -5.960, 1.326]], # R2
    "tool_silicone":[[-4.445, -5.960, 1.611],  # R1
                     [-4.445, -5.960, 1.327]], # R3
}

_TOOL_QUAT = [0.7071, 0.0, 0.0, -0.7071]


def reset_objects_on_slots(env, env_ids: torch.Tensor):
    """Randomly pick top or bottom slot for each object independently."""
    n = len(env_ids)
    quat = torch.tensor(_TOOL_QUAT, device=env.device).expand(n, -1)

    for name, slot_pair in _OBJECT_SLOTS.items():
        slots = torch.tensor(slot_pair, device=env.device)  # [2, 3]
        # Random binary choice per env: 0 or 1.
        choice = torch.randint(0, 2, (n,), device=env.device)
        positions = env.scene.env_origins[env_ids] + slots[choice]
        asset = env.scene[name]
        asset.write_root_pose_to_sim(
            torch.cat([positions, quat], dim=-1), env_ids=env_ids
        )
        asset.write_root_velocity_to_sim(
            torch.zeros(n, 6, device=env.device), env_ids=env_ids
        )
