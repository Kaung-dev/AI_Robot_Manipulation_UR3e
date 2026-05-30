"""Diffusion policy for AIR2 pick-and-place.

Architecture:
    wrist_rgb + board_rgb  ──► frozen encoder ──► visual features
    joint_pos + joint_vel + last_action + target_one_hot ──► state features
                                                    │
                                         concat → obs_cond
                                                    │
    noisy_actions + timestep_emb + obs_cond  ──► MLPNoisePred ──► predicted_noise

Training: standard DDPM (cosine schedule, T=100, MSE noise loss).
Inference: DDIM deterministic sampling, configurable steps (default 10).

Action space: (CHUNK_SIZE=16, ACTION_DIM=7) flattened to (112,).
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .policy import (
    FrozenUNetEncoder, FrozenResNetEncoder,
    load_frozen_encoder, load_frozen_resnet_encoder,
    ACTION_DIM, STATE_DIM, CHUNK_SIZE, COMMAND_DIM,
    DEFAULT_VISION_DIM, RESNET_VISION_DIM,
)

ACTION_HORIZON = CHUNK_SIZE   # steps predicted per forward pass
FLAT_ACTION_DIM = ACTION_HORIZON * ACTION_DIM  # 16 × 7 = 112


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal embedding for DDPM timestep t ∈ [0, T-1]."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResidualBlock(nn.Module):
    """FC block with LayerNorm, SiLU, and residual + FiLM conditioning."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        # FiLM: scale + shift from conditioning vector
        self.film = nn.Linear(cond_dim, 2 * dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.film(cond).chunk(2, dim=-1)
        h = self.norm(x) * (1 + scale) + shift
        h = F.silu(self.linear1(h))
        h = self.linear2(h)
        return x + h


class MLPNoisePred(nn.Module):
    """MLP noise predictor with FiLM conditioning.

    Input:  noisy_actions (B, flat_action_dim)
    Cond:   obs_cond (B, obs_dim)  + timestep_emb (B, t_dim)
    Output: predicted_noise (B, flat_action_dim)
    """

    def __init__(
        self,
        flat_action_dim: int = FLAT_ACTION_DIM,
        obs_dim: int = 541,
        hidden_dim: int = 512,
        n_layers: int = 4,
        t_emb_dim: int = 64,
    ):
        super().__init__()
        self.t_emb = nn.Sequential(
            SinusoidalPosEmb(t_emb_dim),
            nn.Linear(t_emb_dim, t_emb_dim),
            nn.SiLU(),
        )
        cond_dim = obs_dim + t_emb_dim
        self.input_proj = nn.Linear(flat_action_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, cond_dim) for _ in range(n_layers)
        ])
        self.out = nn.Linear(hidden_dim, flat_action_dim)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        obs_cond: torch.Tensor,
    ) -> torch.Tensor:
        t_emb = self.t_emb(t)
        cond = torch.cat([obs_cond, t_emb], dim=-1)
        h = F.silu(self.input_proj(x))
        for block in self.blocks:
            h = block(h, cond)
        return self.out(h)


# ---------------------------------------------------------------------------
# Noise schedule helpers
# ---------------------------------------------------------------------------

def _cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """Cosine beta schedule (Nichol & Dhariwal 2021)."""
    steps = torch.arange(T + 1, dtype=torch.float64) / T
    alphas_cumprod = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return betas.clamp(0, 0.999).float()


# ---------------------------------------------------------------------------
# Diffusion policy
# ---------------------------------------------------------------------------

class DiffusionPolicy(nn.Module):
    """DDPM-based imitation learning policy for AIR2.

    Training: add noise to demo actions at random timestep, predict noise.
    Inference: start from Gaussian noise, denoise via DDIM in n_inf_steps.
    """

    def __init__(
        self,
        encoder: FrozenUNetEncoder | FrozenResNetEncoder,
        T: int = 100,
        n_inf_steps: int = 10,
        hidden_dim: int = 512,
        n_layers: int = 4,
        t_emb_dim: int = 64,
    ):
        super().__init__()
        self.encoder = encoder
        self.T = T
        self.n_inf_steps = n_inf_steps

        vision_dim = getattr(encoder, "feature_dim", DEFAULT_VISION_DIM)
        obs_dim = 2 * vision_dim + STATE_DIM + COMMAND_DIM

        self.noise_pred = MLPNoisePred(
            flat_action_dim=FLAT_ACTION_DIM,
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            t_emb_dim=t_emb_dim,
        )

        betas = _cosine_beta_schedule(T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1 - alphas_cumprod).sqrt())

    # ------------------------------------------------------------------
    # Observation encoding
    # ------------------------------------------------------------------

    def encode_obs(
        self,
        wrist_rgb: torch.Tensor,
        board_rgb: torch.Tensor,
        state: torch.Tensor,
        target_one_hot: torch.Tensor,
    ) -> torch.Tensor:
        """Encode observation into a flat conditioning vector.

        wrist_rgb / board_rgb: (B, 3, H, W) uint8 or float [0,1].
        state: (B, STATE_DIM)
        target_one_hot: (B, COMMAND_DIM)
        Returns: (B, obs_dim)
        """
        if wrist_rgb.dtype == torch.uint8:
            wrist_rgb = wrist_rgb.float() / 255.0
        if board_rgb.dtype == torch.uint8:
            board_rgb = board_rgb.float() / 255.0
        with torch.no_grad():
            w_feat = self.encoder(wrist_rgb)
            b_feat = self.encoder(board_rgb)
        return torch.cat([w_feat, b_feat, state, target_one_hot], dim=-1)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_loss(
        self,
        obs_cond: torch.Tensor,
        actions: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """DDPM training loss (MSE on noise prediction).

        obs_cond: (B, obs_dim) — from encode_obs()
        actions:  (B, CHUNK_SIZE, ACTION_DIM)
        mask:     (B, CHUNK_SIZE) — 1.0 valid, 0.0 padded
        """
        B = actions.shape[0]
        x0 = actions.reshape(B, -1)  # (B, flat_action_dim)

        t = torch.randint(0, self.T, (B,), device=x0.device)
        eps = torch.randn_like(x0)

        sqrt_a = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_1ma = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        x_t = sqrt_a * x0 + sqrt_1ma * eps

        eps_pred = self.noise_pred(x_t, t, obs_cond)

        # Expand mask to per-element weight: (B, chunk) → (B, flat)
        # Each of ACTION_DIM elements in a chunk step shares the same mask value.
        mask_flat = mask.unsqueeze(-1).expand(-1, -1, ACTION_DIM).reshape(B, -1)
        loss = (F.mse_loss(eps_pred, eps, reduction="none") * mask_flat).sum()
        loss = loss / mask_flat.sum().clamp(min=1.0)
        return loss

    # ------------------------------------------------------------------
    # Inference (DDIM deterministic)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        obs_cond: torch.Tensor,
        n_steps: int | None = None,
    ) -> torch.Tensor:
        """Denoise from Gaussian noise to actions via DDIM.

        obs_cond: (B, obs_dim)
        Returns:  (B, CHUNK_SIZE, ACTION_DIM)
        """
        n_steps = n_steps or self.n_inf_steps
        B = obs_cond.shape[0]
        device = obs_cond.device

        # Evenly spaced timesteps from T-1 down to 0
        step_indices = torch.linspace(self.T - 1, 0, n_steps, device=device).long()

        x = torch.randn(B, FLAT_ACTION_DIM, device=device)

        for i, t_now in enumerate(step_indices):
            t_batch = t_now.expand(B)
            alpha_now = self.alphas_cumprod[t_now]

            eps_pred = self.noise_pred(x, t_batch, obs_cond)

            # Predicted x0
            x0_pred = (x - (1 - alpha_now).sqrt() * eps_pred) / alpha_now.sqrt()
            x0_pred = x0_pred.clamp(-1.0, 1.0)

            if i == n_steps - 1:
                x = x0_pred
            else:
                t_prev = step_indices[i + 1]
                alpha_prev = self.alphas_cumprod[t_prev]
                # DDIM step (no stochastic noise)
                x = alpha_prev.sqrt() * x0_pred + (1 - alpha_prev).sqrt() * eps_pred

        return x.reshape(B, ACTION_HORIZON, ACTION_DIM)

    @torch.no_grad()
    def act(
        self,
        wrist_rgb: torch.Tensor,
        board_rgb: torch.Tensor,
        state: torch.Tensor,
        target_one_hot: torch.Tensor | None = None,
        n_steps: int | None = None,
    ) -> torch.Tensor:
        """Convenience: encode obs + sample. Returns (B, CHUNK_SIZE, ACTION_DIM)."""
        if target_one_hot is None:
            target_one_hot = torch.zeros(
                state.shape[0], COMMAND_DIM, dtype=state.dtype, device=state.device
            )
        obs_cond = self.encode_obs(wrist_rgb, board_rgb, state, target_one_hot)
        return self.sample(obs_cond, n_steps=n_steps)


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------

def build_diffusion_policy(
    backbone: str = "unet",
    ckpt_path: str | Path | None = None,
    num_classes: int = 9,
    T: int = 100,
    n_inf_steps: int = 10,
    hidden_dim: int = 512,
    n_layers: int = 4,
) -> DiffusionPolicy:
    if backbone == "resnet18":
        encoder = load_frozen_resnet_encoder(ckpt_path, num_classes=num_classes)
    else:
        encoder = load_frozen_encoder(ckpt_path, num_classes=num_classes)
    return DiffusionPolicy(
        encoder=encoder,
        T=T,
        n_inf_steps=n_inf_steps,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
    )
