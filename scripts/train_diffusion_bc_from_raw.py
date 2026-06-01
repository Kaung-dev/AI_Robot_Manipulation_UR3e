"""Train a Diffusion-Policy state-BC from raw manual-demo HDF5s.

Why diffusion over vanilla MLP-BC:
- Vanilla BC predicts a single mean action per obs; tiny eval-time obs drift makes
  the prediction increasingly wrong (compounding covariate shift). With our 40-demo
  brush case that meant the EE walked AWAY from the brush at eval despite val=0.0007.
- Diffusion models the full action *distribution*. At inference it samples from that
  distribution (denoising from random noise), so small obs perturbations produce
  diverse but task-consistent actions — much more robust to drift.
- Action chunking (predicting next K actions instead of just 1) smooths over per-step
  noise: the policy commits to a multi-step plan, executes it, then re-plans.

Architecture (state-only, no vision):
- Conditioning encoder: state(35) -> embed(256) via 2-layer MLP
- Time embedding: sinusoidal pos-encoding -> 256-D
- Denoising MLP: cat[noisy_chunk(K*7), state_embed(256), time_embed(256)] ->
                 hidden(512) x 4 -> noise_pred(K*7)
- DDPM with 100 diffusion timesteps, cosine beta schedule

Inputs match the env emission order exactly so the resulting policy can be loaded
into eval_diffusion_bc.py against `Isaac-AIR2-Robotis-Franka-<Tool>-Play-v0` envs.

Usage:
    python scripts/train_diffusion_bc_from_raw.py \\
        --hdf5 datasets/air2_manual_demos_brush/air2_mimic_source.hdf5 \\
        --out  checkpoints/policy_diffusion_bc_brush.pth
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# NOTE: h5py is imported lazily inside load_chunked_dataset() so that
# eval_diffusion_bc.py can `from train_diffusion_bc_from_raw import DiffusionPolicy`
# without triggering h5py's DLL load — which conflicts with Isaac Sim's bundled
# HDF5 plugins and crashes the eval before the policy even loads.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
# Parser construction is in a function so that importing this module from
# eval_diffusion_bc.py (which has its OWN argparse) doesn't fire parse_args()
# at module load and crash on missing --hdf5 / --out.

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--chunk_size", type=int, default=16,
                   help="Action horizon: number of future actions predicted per obs.")
    p.add_argument("--num_diffusion_steps", type=int, default=100)
    p.add_argument("--state_embed_dim", type=int, default=256)
    p.add_argument("--time_embed_dim", type=int, default=256)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--num_hidden_layers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


# `args` is assigned only when the module runs as __main__ (see bottom of file).
args = None  # type: ignore


# ---------------------------------------------------------------------------
# Data: same 35-D obs layout as train_state_bc_from_raw.py
# ---------------------------------------------------------------------------

HDF5_OBS_HEAD = ["joint_pos", "joint_vel", "object_position"]   # 21 dims
HDF5_OBS_TAIL = ["eef_pos", "eef_quat"]                          # 7 dims
# 7 dims in between come from last_action (derived from demo['actions'][t-1]).


def load_chunked_dataset(hdf5_path: str, chunk_size: int) -> tuple[np.ndarray, np.ndarray, int]:
    import h5py  # lazy: kept out of module scope so eval scripts can import DiffusionPolicy
    """Return (obs[N, 35], action_chunks[N, K, 7], n_demos).

    Each sample is (obs at time t, next K actions starting from t). Demos shorter
    than K get right-padded with the last recorded action (gripper held at final
    state, arm zeroed).
    """
    obs_list, chunk_list = [], []
    with h5py.File(hdf5_path, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1]))
        print(f"[diff-bc] {hdf5_path}: {len(demos)} demos, chunk_size={chunk_size}")
        for demo_key in demos:
            d = f["data"][demo_key]
            try:
                head_parts = [d["obs"][k][:] for k in HDF5_OBS_HEAD]
                tail_parts = [d["obs"][k][:] for k in HDF5_OBS_TAIL]
            except KeyError as e:
                print(f"[diff-bc] {demo_key}: missing obs key {e}, skipping")
                continue
            T = head_parts[0].shape[0]
            act_arr = d["actions"][:].astype(np.float32)  # (T, 7)
            if act_arr.shape[0] != T:
                continue
            last_action = np.zeros_like(act_arr)
            last_action[1:] = act_arr[:-1]
            obs = np.concatenate(head_parts + [last_action] + tail_parts, axis=-1).astype(np.float32)

            # Build (T, chunk_size, 7) chunks. Right-pad short tails with the
            # final-frame action (arm zeroed + gripper held at terminal value).
            padded_act = np.zeros((T + chunk_size, 7), dtype=np.float32)
            padded_act[:T] = act_arr
            # Pad rows = "stay still + keep gripper at last commanded value"
            final_grip = act_arr[-1, -1] if T > 0 else 0.0
            padded_act[T:, 6] = final_grip
            chunks = np.stack([padded_act[i:i + chunk_size] for i in range(T)], axis=0)

            obs_list.append(obs)
            chunk_list.append(chunks)

    obs_all = np.concatenate(obs_list, axis=0)
    chunks_all = np.concatenate(chunk_list, axis=0)
    n_demos = len(obs_list)
    print(f"[diff-bc] loaded {n_demos} demos -> {obs_all.shape[0]} samples, "
          f"obs_dim={obs_all.shape[1]}, chunk_shape={chunks_all.shape[1:]}")
    return obs_all, chunks_all, n_demos


# ---------------------------------------------------------------------------
# Diffusion components
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """Standard transformer-style sinusoidal embedding of diffusion step index."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        emb = t[:, None].float() * freqs[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class StateEncoder(nn.Module):
    def __init__(self, state_dim: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, embed_dim), nn.SiLU(),
            nn.Linear(embed_dim, embed_dim), nn.SiLU(),
        )
    def forward(self, x): return self.net(x)


class DenoiserMLP(nn.Module):
    """Predicts the noise that was added to the action chunk.

    Inputs concatenated: noisy_chunk (K*A) + state_embed (S) + time_embed (T)
    Output:              noise_pred  (K*A)
    """
    def __init__(self, chunk_dim: int, state_embed_dim: int, time_embed_dim: int,
                 hidden_dim: int, n_layers: int):
        super().__init__()
        in_dim = chunk_dim + state_embed_dim + time_embed_dim
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers += [nn.Linear(hidden_dim, chunk_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, noisy_chunk_flat: torch.Tensor, state_embed: torch.Tensor,
                time_embed: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([noisy_chunk_flat, state_embed, time_embed], dim=-1))


def cosine_beta_schedule(num_timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Improved DDPM cosine schedule (Nichol & Dhariwal 2021).

    Smoother than the linear schedule near t=0, T which improves sample quality.
    """
    steps = num_timesteps + 1
    x = torch.linspace(0, num_timesteps, steps)
    alpha_bar = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return torch.clamp(betas, 0.0001, 0.999)


class DiffusionPolicy(nn.Module):
    """State -> action-chunk diffusion policy.

    Trains a noise-prediction denoiser; samples actions via DDPM (or DDIM) at
    inference. Action chunks are normalised to mean=0/std=1 from training stats.
    """
    def __init__(self, state_dim: int, action_dim: int, chunk_size: int,
                 num_timesteps: int, state_embed_dim: int, time_embed_dim: int,
                 hidden_dim: int, num_hidden_layers: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.chunk_dim = chunk_size * action_dim
        self.num_timesteps = num_timesteps

        self.state_encoder = StateEncoder(state_dim, state_embed_dim)
        self.time_embedder = SinusoidalTimeEmbedding(time_embed_dim)
        self.denoiser = DenoiserMLP(self.chunk_dim, state_embed_dim, time_embed_dim,
                                    hidden_dim, num_hidden_layers)

        betas = cosine_beta_schedule(num_timesteps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        # Register as buffers so they move with .to(device) and serialise.
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        # Stats for action normalisation, filled at fit-time.
        self.register_buffer("action_mean", torch.zeros(action_dim))
        self.register_buffer("action_std", torch.ones(action_dim))

    def normalize(self, actions: torch.Tensor) -> torch.Tensor:
        return (actions - self.action_mean) / self.action_std.clamp(min=1e-6)

    def denormalize(self, actions: torch.Tensor) -> torch.Tensor:
        return actions * self.action_std + self.action_mean

    def add_noise(self, x0_flat: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(x0_flat)
        ab = self.alpha_bar[t][:, None]
        xt = ab.sqrt() * x0_flat + (1 - ab).sqrt() * noise
        return xt, noise

    def predict_noise(self, xt_flat: torch.Tensor, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        s_emb = self.state_encoder(state)
        t_emb = self.time_embedder(t)
        return self.denoiser(xt_flat, s_emb, t_emb)

    @torch.no_grad()
    def sample(self, state: torch.Tensor, num_inference_steps: int | None = None) -> torch.Tensor:
        """DDIM-style deterministic sampling. Returns (B, K, A) un-normalised action chunk."""
        B = state.shape[0]
        device = state.device
        K = self.num_timesteps if num_inference_steps is None else num_inference_steps
        # Subsample timesteps uniformly when K < num_timesteps (DDIM accelerates)
        step_indices = torch.linspace(self.num_timesteps - 1, 0, K, device=device).long()

        xt = torch.randn(B, self.chunk_dim, device=device)
        s_emb = self.state_encoder(state)
        for i, t in enumerate(step_indices):
            t_batch = t.repeat(B)
            t_emb = self.time_embedder(t_batch)
            eps = self.denoiser(xt, s_emb, t_emb)
            ab_t = self.alpha_bar[t]
            x0_pred = (xt - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
            x0_pred = x0_pred.clamp(-4.0, 4.0)  # normalised range
            if i < K - 1:
                t_prev = step_indices[i + 1]
                ab_prev = self.alpha_bar[t_prev]
                xt = ab_prev.sqrt() * x0_pred + (1 - ab_prev).sqrt() * eps
            else:
                xt = x0_pred
        chunk = xt.reshape(B, self.chunk_size, self.action_dim)
        return self.denormalize(chunk)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train():
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    obs_np, chunks_np, n_demos = load_chunked_dataset(args.hdf5, args.chunk_size)
    n = obs_np.shape[0]
    n_val = max(1, int(n * args.val_ratio))
    perm = np.random.permutation(n)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    dev = args.device
    obs_t = torch.from_numpy(obs_np).to(dev)
    chunks_t = torch.from_numpy(chunks_np).to(dev)
    obs_train, obs_val = obs_t[train_idx], obs_t[val_idx]
    ch_train, ch_val = chunks_t[train_idx], chunks_t[val_idx]
    n_train = len(train_idx)
    print(f"[diff-bc] device={dev}  train_samples={n_train}  val_samples={n_val}")

    state_dim = obs_np.shape[1]
    action_dim = chunks_np.shape[-1]
    model = DiffusionPolicy(
        state_dim=state_dim, action_dim=action_dim, chunk_size=args.chunk_size,
        num_timesteps=args.num_diffusion_steps,
        state_embed_dim=args.state_embed_dim, time_embed_dim=args.time_embed_dim,
        hidden_dim=args.hidden_dim, num_hidden_layers=args.num_hidden_layers,
    ).to(dev)

    # Fit normalisation stats on training actions only.
    train_flat = ch_train.reshape(-1, action_dim)
    model.action_mean.copy_(train_flat.mean(dim=0))
    model.action_std.copy_(train_flat.std(dim=0).clamp(min=1e-6))
    print(f"[diff-bc] action_mean={model.action_mean.tolist()}")
    print(f"[diff-bc] action_std={model.action_std.tolist()}")

    print(f"[diff-bc] model params: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    log = []
    best_val = float("inf")
    best_state: dict | None = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm_t = torch.randperm(n_train, device=dev)
        obs_shuf = obs_train[perm_t]
        ch_shuf = ch_train[perm_t]
        tr_loss = 0.0
        for i in range(0, n_train, args.batch_size):
            xb_obs = obs_shuf[i:i + args.batch_size]
            xb_ch = ch_shuf[i:i + args.batch_size]
            B = xb_obs.shape[0]
            x0 = model.normalize(xb_ch).reshape(B, model.chunk_dim)
            t = torch.randint(0, model.num_timesteps, (B,), device=dev)
            xt, noise = model.add_noise(x0, t)
            noise_pred = model.predict_noise(xt, t, xb_obs)
            loss = F.mse_loss(noise_pred, noise)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * B
        tr_loss /= n_train

        model.eval()
        with torch.no_grad():
            B = obs_val.shape[0]
            x0 = model.normalize(ch_val).reshape(B, model.chunk_dim)
            t = torch.randint(0, model.num_timesteps, (B,), device=dev)
            xt, noise = model.add_noise(x0, t)
            noise_pred = model.predict_noise(xt, t, obs_val)
            vl_loss = F.mse_loss(noise_pred, noise).item()
        scheduler.step()

        log.append({"epoch": epoch, "train": tr_loss, "val": vl_loss})
        if epoch % 25 == 0 or epoch == 1:
            print(f"  ep {epoch:>3}/{args.epochs}  train={tr_loss:.5f}  val={vl_loss:.5f}")

        if vl_loss < best_val:
            best_val = vl_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    print(f"[diff-bc] best val loss: {best_val:.5f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "chunk_size": args.chunk_size,
        "num_diffusion_steps": args.num_diffusion_steps,
        "state_embed_dim": args.state_embed_dim,
        "time_embed_dim": args.time_embed_dim,
        "hidden_dim": args.hidden_dim,
        "num_hidden_layers": args.num_hidden_layers,
        "obs_layout": "joint_pos(9)+joint_vel(9)+object_position(3)+last_action(7)+eef_pos(3)+eef_quat(4) = 35",
        "num_samples": int(obs_np.shape[0]),
        "num_demos": n_demos,
        "source_hdf5": str(args.hdf5),
    }, out_path)
    log_path = out_path.with_suffix(".log.json")
    log_path.write_text(json.dumps(log, indent=2))
    print(f"[diff-bc] saved -> {out_path}")
    print(f"[diff-bc] log   -> {log_path}")


if __name__ == "__main__":
    args = _build_parser().parse_args()
    train()
