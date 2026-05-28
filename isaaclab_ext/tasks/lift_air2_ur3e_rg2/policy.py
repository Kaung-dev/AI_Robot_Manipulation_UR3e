"""BC / PPO policy network for the AIR2 pick-and-place task.

Architecture (matches CNN_PERCEPTION_AND_POLICY.md):

    wrist_rgb (224x224x3) ──► U-Net encoder (frozen) ──► pooled features (256)
    board_rgb (640x360x3) ──► U-Net encoder (frozen) ──► pooled features (256)
                                                              │
              + joint_pos (9) + joint_vel (9) + last_action (7) + target object (4)
                                                              │
                                                              ▼
                                                       MLP (state + vision)
                                                              │
                                                              ▼
                                                  7-D action: dx, dy, dz, dRx, dRy, dRz, gripper

The CNN encoder uses the teammate's `AIR2UNet` and is frozen during BC training.
Only the state encoder + policy head have gradients. Same `BCPolicy` instance is
reused as the PPO actor (with a fresh critic head trained from scratch).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from .cnn.model import AIR2UNet
from .objects import TARGET_DIM


# ----- vision encoder -------------------------------------------------------


class FrozenUNetEncoder(nn.Module):
    """Wraps an AIR2UNet and exposes a pooled feature vector from its bottleneck.

    We skip the decoder + head and only use encoder1..4. The deepest layer
    (encoder4) has 256 channels at base_channels=32 — global-average-pool that
    spatial map to get a 256-D scene feature per image.
    """

    def __init__(self, unet: AIR2UNet, freeze: bool = True):
        super().__init__()
        self.encoder1 = unet.encoder1
        self.encoder2 = unet.encoder2
        self.encoder3 = unet.encoder3
        self.encoder4 = unet.encoder4
        # bottleneck channel count (encoder4 doubles 4× from base)
        # at base_channels=32 this is 32 * 8 = 256
        self.feature_dim = self.encoder4.net[-1].net[-2].num_features
        if freeze:
            for p in self.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        x = self.encoder4(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)  # (B, 256)
        return x


def load_frozen_encoder(unet_ckpt: str | Path | None, num_classes: int = 9) -> FrozenUNetEncoder:
    """Build a FrozenUNetEncoder from a U-Net checkpoint (or random weights if None)."""
    unet = AIR2UNet(num_classes=num_classes)
    if unet_ckpt is not None:
        state = torch.load(str(unet_ckpt), map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        # Tolerate num_classes mismatch on the head: head weights are not used
        # by the encoder anyway.
        missing, unexpected = unet.load_state_dict(state, strict=False)
        print(f"[encoder] loaded U-Net ckpt, missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print("[encoder] WARNING: no U-Net checkpoint — using random encoder weights")
    unet.eval()
    return FrozenUNetEncoder(unet, freeze=True)


# ----- policy ---------------------------------------------------------------


JOINT_DIM = 9        # Franka: 7 arm + 2 fingers
ACTION_DIM = 7       # IK-Rel pose (6) + binary gripper (1)
CHUNK_SIZE = 16      # ACT-inspired: predict the next CHUNK_SIZE actions in one forward pass
STATE_DIM = JOINT_DIM * 2 + ACTION_DIM   # joint_pos + joint_vel + last_action
COMMAND_DIM = TARGET_DIM                  # one-hot target object command
DEFAULT_VISION_DIM = 256                  # FrozenUNetEncoder feature_dim


class BCPolicy(nn.Module):
    """Action-chunked actor: (wrist_img, board_img, state) → (chunk, 7).

    Predicts the next `chunk_size` actions per forward pass. At inference,
    execute only the first action of the chunk (or use temporal ensembling
    by averaging overlapping chunks — see eval_bc.py).

    Inspired by ACT (Action Chunking Transformer; Zhao et al. 2023) but uses
    a frozen U-Net encoder + MLP fusion instead of ResNet + Transformer.
    Reasons: simpler to train, matches our staged CNN/IL/RL design doc, and
    the AIR2 task is short-horizon so we don't need long-range temporal
    context from a Transformer.

    The PPO actor (bc_to_ppo.py) does NOT use chunking — it predicts one
    action per step. The action-head weights are still copied over for the
    last linear layer; chunked outputs are flattened and only the first
    action_dim slice is transferred.
    """

    def __init__(
        self,
        encoder: FrozenUNetEncoder,
        vision_dim: int = DEFAULT_VISION_DIM,
        state_dim: int = STATE_DIM,
        command_dim: int = COMMAND_DIM,
        action_dim: int = ACTION_DIM,
        chunk_size: int = CHUNK_SIZE,
        hidden: tuple[int, int] = (256, 128),
    ):
        super().__init__()
        self.encoder = encoder
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.command_dim = command_dim
        # one encoder pass per camera (shared weights) — input dim = 2 * vision_dim
        self.fusion = nn.Sequential(
            nn.Linear(2 * vision_dim + state_dim + command_dim, hidden[0]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(inplace=True),
        )
        # Output the entire chunk in one shot.
        self.action_head = nn.Linear(hidden[1], chunk_size * action_dim)

    def encode(self, wrist_rgb: torch.Tensor, board_rgb: torch.Tensor) -> torch.Tensor:
        """wrist_rgb / board_rgb: (B, 3, H, W), uint8 or normalized float in [0,1]."""
        if wrist_rgb.dtype == torch.uint8:
            wrist_rgb = wrist_rgb.float() / 255.0
        if board_rgb.dtype == torch.uint8:
            board_rgb = board_rgb.float() / 255.0
        with torch.no_grad():
            w_feat = self.encoder(wrist_rgb)
            b_feat = self.encoder(board_rgb)
        return torch.cat([w_feat, b_feat], dim=1)  # (B, 2*vision_dim)

    def forward(
        self,
        wrist_rgb: torch.Tensor,
        board_rgb: torch.Tensor,
        state: torch.Tensor,
        target_one_hot: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns predicted action chunk: (B, chunk_size, action_dim).

        state: (B, STATE_DIM) — joint_pos + joint_vel + last_action.
        """
        if target_one_hot is None:
            target_one_hot = torch.zeros(
                state.shape[0], self.command_dim, dtype=state.dtype, device=state.device
            )
        vision = self.encode(wrist_rgb, board_rgb)
        x = torch.cat([vision, state, target_one_hot], dim=1)
        x = self.fusion(x)
        flat = self.action_head(x)                                   # (B, chunk * action_dim)
        return flat.view(-1, self.chunk_size, self.action_dim)       # (B, chunk, action_dim)

    @torch.no_grad()
    def act_first(self, wrist_rgb, board_rgb, state, target_one_hot=None) -> torch.Tensor:
        """Convenience: predict chunk, return first action only. (B, action_dim)."""
        chunk = self.forward(wrist_rgb, board_rgb, state, target_one_hot)
        return chunk[:, 0, :]
