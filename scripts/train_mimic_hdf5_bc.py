"""Train a state-only BC policy from Isaac Mimic HDF5 demonstrations.

This is intentionally separate from train_bc.py. The normal AIR2 BC trainer
learns from camera PNGs plus states.npz object labels. Isaac Mimic HDF5 files
usually store only state/action arrays, so this script trains a small MLP:

    flat state observation -> 7-D action

The output checkpoint uses the same format as train_state_bc.py, so it can be
loaded by scripts/eval_state_bc.py for a quick brush-only trial.

Windows example:
    D:\\IsaacLab\\isaaclab.bat -p scripts\\train_mimic_hdf5_bc.py `
        --hdf5 datasets\\air2_mimic\\air2_mimic_demos_annotated.hdf5 `
        --epochs 100 `
        --batch_size 256 `
        --out checkpoints\\brush_mimic_state_bc.pth

Linux example:
    ~/IsaacLab/isaaclab.sh -p scripts/train_mimic_hdf5_bc.py \\
        --hdf5 datasets/air2_mimic/air2_mimic_demos_annotated.hdf5 \\
        --epochs 100 \\
        --batch_size 256 \\
        --out checkpoints/brush_mimic_state_bc.pth
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from rsl_rl.networks import MLP  # type: ignore
except ModuleNotFoundError:
    class MLP(nn.Module):
        """Small rsl_rl-compatible actor MLP fallback.

        Some Isaac Lab installs expose rsl_rl differently on Windows. Keeping
        this fallback local lets the HDF5 trial train without changing the
        normal PPO/BC code paths.
        """

        def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int], activation: str):
            super().__init__()
            act = _activation(activation)
            layers: list[nn.Module] = []
            last_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(last_dim, hidden_dim))
                layers.append(act())
                last_dim = hidden_dim
            layers.append(nn.Linear(last_dim, output_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


def _activation(name: str) -> type[nn.Module]:
    table = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "leaky_relu": nn.LeakyReLU,
    }
    key = name.lower()
    if key not in table:
        raise ValueError(f"Unsupported activation '{name}'. Choose from {sorted(table)}.")
    return table[key]


DEFAULT_OBS_KEYS = (
    "joint_pos",
    "joint_vel",
    "object_position",
    "target_object_position",
    "actions",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a state-only AIR2 brush BC policy from Isaac Mimic HDF5 demos."
    )
    parser.add_argument(
        "--hdf5",
        nargs="+",
        required=True,
        help="One or more HDF5 demo files, for example air2_mimic_demos_annotated.hdf5.",
    )
    parser.add_argument(
        "--obs_keys",
        nargs="+",
        default=list(DEFAULT_OBS_KEYS),
        help="Datasets under demo_i/obs to concatenate into the policy input.",
    )
    parser.add_argument(
        "--action_key",
        default="actions",
        choices=("actions", "processed_actions"),
        help="Action dataset to train against. Use actions for the current eval_state_bc.py path.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument(
        "--success_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use only demo groups with success=True when that attribute is present.",
    )
    parser.add_argument(
        "--limit_demos",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 128, 64])
    parser.add_argument("--activation", default="elu")
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Training device.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="checkpoints/brush_mimic_state_bc.pth")
    return parser.parse_args()


def _read_array(group: h5py.Group, key: str, n: int) -> np.ndarray:
    if key not in group:
        raise KeyError(f"Missing obs/{key}. Available keys: {list(group.keys())}")
    arr = np.asarray(group[key][:n], dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return arr


def _demo_sort_key(name: str) -> tuple[int, str]:
    if name.startswith("demo_"):
        try:
            return int(name.split("_", 1)[1]), name
        except ValueError:
            pass
    return 10**9, name


def load_hdf5_demos(
    paths: list[Path],
    obs_keys: list[str],
    action_key: str,
    frame_stride: int,
    success_only: bool,
    limit_demos: int | None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    obs_chunks: list[np.ndarray] = []
    act_chunks: list[np.ndarray] = []
    summary = {
        "files": [],
        "obs_keys": obs_keys,
        "action_key": action_key,
        "success_only": success_only,
        "frame_stride": frame_stride,
    }
    demos_seen = 0
    demos_used = 0
    demos_skipped = 0

    for path in paths:
        path = path.expanduser().resolve()
        if not path.exists():
            nearby = sorted(path.parent.glob("*.hdf5")) if path.parent.exists() else []
            hint = ""
            if nearby:
                hint = "\nNearby HDF5 files:\n" + "\n".join(f"  - {p}" for p in nearby)
            raise FileNotFoundError(
                f"HDF5 file not found: {path}\n"
                "Check the extracted folder name and pass the exact --hdf5 path."
                f"{hint}"
            )
        file_summary = {"path": str(path), "used": 0, "skipped": 0, "samples": 0}
        with h5py.File(path, "r") as h5:
            if "data" not in h5:
                raise KeyError(f"{path} does not contain a top-level 'data' group.")
            data_group = h5["data"]
            env_args = data_group.attrs.get("env_args")
            if isinstance(env_args, bytes):
                env_args = env_args.decode("utf-8", errors="replace")
            file_summary["env_args"] = env_args

            demo_names = sorted(data_group.keys(), key=_demo_sort_key)
            for demo_name in demo_names:
                if limit_demos is not None and demos_used >= limit_demos:
                    break
                demo = data_group[demo_name]
                demos_seen += 1
                success = bool(demo.attrs.get("success", True))
                if success_only and not success:
                    demos_skipped += 1
                    file_summary["skipped"] += 1
                    continue
                if action_key not in demo:
                    raise KeyError(f"{path}:{demo_name} missing '{action_key}'.")

                actions = np.asarray(demo[action_key], dtype=np.float32)
                if actions.ndim == 1:
                    actions = actions[:, None]
                if actions.ndim > 2:
                    actions = actions.reshape(actions.shape[0], -1)
                if action_key == "processed_actions" and actions.shape[1] > 7:
                    print(
                        "[hdf5-bc] processed_actions has more than 7 dims; "
                        "using the first 7 so eval_state_bc.py can execute it."
                    )
                    actions = actions[:, :7]

                obs_group = demo["obs"]
                n = min(actions.shape[0], *(obs_group[k].shape[0] for k in obs_keys))
                if n <= 1:
                    demos_skipped += 1
                    file_summary["skipped"] += 1
                    continue

                idx = np.arange(0, n, max(1, frame_stride), dtype=np.int64)
                obs = np.concatenate([_read_array(obs_group, key, n)[idx] for key in obs_keys], axis=1)
                act = actions[:n][idx]

                obs_chunks.append(obs)
                act_chunks.append(act)
                demos_used += 1
                file_summary["used"] += 1
                file_summary["samples"] += int(obs.shape[0])

            summary["files"].append(file_summary)

    if not obs_chunks:
        raise RuntimeError("No usable samples were loaded from the provided HDF5 files.")

    obs_np = np.concatenate(obs_chunks, axis=0).astype(np.float32)
    act_np = np.concatenate(act_chunks, axis=0).astype(np.float32)
    summary.update(
        {
            "demos_seen": demos_seen,
            "demos_used": demos_used,
            "demos_skipped": demos_skipped,
            "num_samples": int(obs_np.shape[0]),
            "input_dim": int(obs_np.shape[1]),
            "action_dim": int(act_np.shape[1]),
        }
    )
    return obs_np, act_np, summary


def train_mlp(
    obs_np: np.ndarray,
    act_np: np.ndarray,
    hidden_dims: list[int],
    activation: str,
    epochs: int,
    batch_size: int,
    lr: float,
    val_ratio: float,
    device: str,
) -> tuple[MLP, list[dict]]:
    obs = torch.from_numpy(obs_np).float()
    act = torch.from_numpy(act_np).float()
    n = obs.shape[0]
    n_val = max(1, int(n * val_ratio))
    perm = torch.randperm(n)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_dl = DataLoader(TensorDataset(obs[train_idx], act[train_idx]), batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(TensorDataset(obs[val_idx], act[val_idx]), batch_size=batch_size, shuffle=False)

    model = MLP(obs.shape[1], act.shape[1], hidden_dims, activation).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    log: list[dict] = []
    print(
        f"[hdf5-bc] MLP input={obs.shape[1]} hidden={hidden_dims} "
        f"output={act.shape[1]} activation={activation}"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for xb, yb in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = nn.functional.smooth_l1_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss_sum += float(loss.item()) * xb.shape[0]
            train_count += xb.shape[0]

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = nn.functional.smooth_l1_loss(pred, yb)
                val_loss_sum += float(loss.item()) * xb.shape[0]
                val_count += xb.shape[0]

        train_loss = train_loss_sum / max(1, train_count)
        val_loss = val_loss_sum / max(1, val_count)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        log.append(row)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"[hdf5-bc] epoch {epoch:04d}/{epochs} "
                f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}",
                flush=True,
            )

    return model, log


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    hdf5_paths = [Path(p) for p in args.hdf5]
    obs_np, act_np, summary = load_hdf5_demos(
        paths=hdf5_paths,
        obs_keys=list(args.obs_keys),
        action_key=args.action_key,
        frame_stride=args.frame_stride,
        success_only=args.success_only,
        limit_demos=args.limit_demos,
    )
    print(
        f"[hdf5-bc] loaded {summary['num_samples']} samples from "
        f"{summary['demos_used']} demos; input_dim={summary['input_dim']} "
        f"action_dim={summary['action_dim']}"
    )
    for file_summary in summary["files"]:
        print(
            f"[hdf5-bc] source {file_summary['path']} "
            f"used={file_summary['used']} skipped={file_summary['skipped']} "
            f"samples={file_summary['samples']}"
        )

    model, log = train_mlp(
        obs_np=obs_np,
        act_np=act_np,
        hidden_dims=args.hidden_dims,
        activation=args.activation,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_ratio=args.val_ratio,
        device=args.device,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": int(obs_np.shape[1]),
            "action_dim": int(act_np.shape[1]),
            "hidden_dims": list(args.hidden_dims),
            "activation": args.activation,
            "num_pairs": int(obs_np.shape[0]),
            "source": "mimic_hdf5",
            "hdf5_summary": summary,
        },
        out_path,
    )
    log_path = out_path.with_suffix(".log.json")
    log_path.write_text(
        json.dumps(
            {
                "epochs": log,
                "hdf5_summary": summary,
                "hidden_dims": list(args.hidden_dims),
                "activation": args.activation,
                "lr": args.lr,
                "batch_size": args.batch_size,
            },
            indent=2,
        )
    )
    print(f"[hdf5-bc] saved checkpoint -> {out_path}")
    print(f"[hdf5-bc] saved log        -> {log_path}")


if __name__ == "__main__":
    main()
