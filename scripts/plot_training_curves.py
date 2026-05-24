"""Generate training-curve plots from the U-Net and BC log JSONs.

Reads:
  checkpoints/air2_segmentation_metrics.json        (U-Net per-epoch metrics)
  checkpoints/policy_bc_mvp.log.json                (BC per-epoch metrics — or whichever is current)
  eval_results/bc_mvp.json                          (BC rollout summary)

Writes:
  eval_results/plots/unet_curves.png
  eval_results/plots/bc_curves.png
  eval_results/plots/bc_eval_summary.png

Usage:
  python scripts/plot_training_curves.py
  python scripts/plot_training_curves.py --bc_log checkpoints/policy_bc_v2.log.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt


def plot_unet(metrics_path: Path, out_path: Path) -> bool:
    if not metrics_path.exists():
        print(f"[plot] skip U-Net: {metrics_path} not found")
        return False
    data = json.loads(metrics_path.read_text())
    # The U-Net trainer writes a list of per-epoch dicts.
    epochs = [d.get("epoch", i + 1) for i, d in enumerate(data)]
    train_loss = [d["train_loss"] for d in data]
    val_loss = [d["val_loss"] for d in data]
    pixel_acc = [d["pixel_accuracy"] for d in data]
    miou = [d["mean_iou"] for d in data]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, train_loss, "o-", label="train")
    axes[0].plot(epochs, val_loss, "s-", label="val")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].set_title("U-Net loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, pixel_acc, "o-", color="tab:green")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("pixel accuracy")
    axes[1].set_title("U-Net pixel accuracy (val)")
    axes[1].set_ylim(0, 1.0); axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, miou, "o-", color="tab:red")
    axes[2].set_xlabel("epoch"); axes[2].set_ylabel("mean IoU")
    axes[2].set_title("U-Net mean IoU (val)")
    axes[2].set_ylim(0, 1.0); axes[2].grid(alpha=0.3)

    fig.suptitle(f"U-Net training — {len(epochs)} epochs", fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"[plot] wrote {out_path}  (final: train={train_loss[-1]:.4f} val={val_loss[-1]:.4f} mIoU={miou[-1]:.3f})")
    return True


def plot_bc(log_path: Path, out_path: Path) -> bool:
    if not log_path.exists():
        print(f"[plot] skip BC: {log_path} not found")
        return False
    data = json.loads(log_path.read_text())
    epochs = [d["epoch"] for d in data]
    tr_pose = [d["train"]["pose_l1"] for d in data]
    tr_grip = [d["train"]["grip_bce"] for d in data]
    val_pose = [d["val"]["pose_l1"] for d in data]
    val_grip = [d["val"]["grip_bce"] for d in data]
    lrs = [d.get("lr", 0) for d in data]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, tr_pose, "o-", label="train")
    axes[0].plot(epochs, val_pose, "s-", label="val")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("smooth-L1 loss")
    axes[0].set_title("BC pose loss (chunked actions)")
    axes[0].legend(); axes[0].grid(alpha=0.3); axes[0].set_yscale("log")

    axes[1].plot(epochs, tr_grip, "o-", label="train")
    axes[1].plot(epochs, val_grip, "s-", label="val")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("BCE loss")
    axes[1].set_title("BC gripper loss")
    axes[1].legend(); axes[1].grid(alpha=0.3); axes[1].set_yscale("log")

    axes[2].plot(epochs, lrs, "o-", color="tab:purple")
    axes[2].set_xlabel("epoch"); axes[2].set_ylabel("learning rate")
    axes[2].set_title("LR schedule (cosine)")
    axes[2].grid(alpha=0.3)

    fig.suptitle(f"BC training — {len(epochs)} epochs", fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"[plot] wrote {out_path}  (final val: pose={val_pose[-1]:.4f} grip={val_grip[-1]:.4f})")
    return True


def plot_bc_eval(eval_path: Path, out_path: Path) -> bool:
    if not eval_path.exists():
        print(f"[plot] skip BC eval: {eval_path} not found")
        return False
    data = json.loads(eval_path.read_text())
    episodes = data.get("episodes", [])
    if not episodes:
        return False
    ep_idx = [e["ep_idx"] for e in episodes]
    rewards = [e["cumulative_reward"] for e in episodes]
    basket_dists = [e["min_basket_dist"] for e in episodes]
    reached = [e["reached_basket"] for e in episodes]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    bars = axes[0].bar(ep_idx, rewards,
                       color=["tab:green" if r else "tab:red" for r in reached])
    axes[0].axhline(data["mean_reward"], color="black", linestyle="--",
                    label=f"mean={data['mean_reward']:.1f}")
    axes[0].set_xlabel("episode"); axes[0].set_ylabel("cumulative reward")
    axes[0].set_title("Per-episode reward (green=reached basket)")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].bar(ep_idx, basket_dists,
                color=["tab:green" if r else "tab:red" for r in reached])
    axes[1].axhline(0.40, color="black", linestyle="--", label="40cm threshold")
    axes[1].set_xlabel("episode"); axes[1].set_ylabel("min EE→basket distance (m)")
    axes[1].set_title("Closest approach to basket")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    n = len(episodes)
    n_success = sum(reached)
    axes[2].pie([n_success, n - n_success],
                labels=[f"reached\n({n_success}/{n})", f"missed\n({n - n_success}/{n})"],
                colors=["tab:green", "tab:red"], autopct="%1.0f%%",
                wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[2].set_title(f"Basket-reach rate: {data['basket_reach_rate']*100:.0f}%")

    fig.suptitle(f"BC rollout eval — {n} episodes", fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"[plot] wrote {out_path}  (reach rate: {data['basket_reach_rate']*100:.0f}%)")
    return True


def plot_ppo_training(tb_dir: Path, out_path: Path) -> bool:
    """Read rsl_rl's TensorBoard event files and plot reward + losses.

    rsl_rl/OnPolicyRunner writes scalars like 'Train/mean_reward',
    'Loss/value_function', 'Loss/surrogate' to TensorBoard. We pull them out
    and turn into a 3-panel matplotlib figure — no TensorBoard needed at
    view time.
    """
    if not tb_dir.exists():
        print(f"[plot] skip PPO: {tb_dir} not found")
        return False
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("[plot] skip PPO: tensorboard package not available")
        return False

    # Find the event file (rsl_rl writes one per run, usually under tb_dir directly).
    event_files = list(tb_dir.glob("events.out.tfevents.*")) or list(tb_dir.rglob("events.out.tfevents.*"))
    if not event_files:
        print(f"[plot] skip PPO: no events.out.tfevents.* under {tb_dir}")
        return False

    acc = EventAccumulator(str(event_files[0].parent), size_guidance={"scalars": 0})
    acc.Reload()
    tags = set(acc.Tags().get("scalars", []))

    def get_series(name):
        if name not in tags:
            return None, None
        evs = acc.Scalars(name)
        return [e.step for e in evs], [e.value for e in evs]

    # rsl_rl tag names can vary by version. Try a few common ones.
    reward_steps, reward_vals = (get_series("Train/mean_reward")
                                 or get_series("rewards/mean")
                                 or get_series("Reward/mean"))
    value_steps, value_vals = (get_series("Loss/value_function")
                               or get_series("loss/value_function"))
    surr_steps, surr_vals = (get_series("Loss/surrogate")
                             or get_series("loss/surrogate"))

    if reward_steps is None:
        print(f"[plot] skip PPO: no recognizable reward tag in {tags}")
        return False

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(reward_steps, reward_vals, color="tab:blue")
    axes[0].set_xlabel("iteration"); axes[0].set_ylabel("mean reward")
    axes[0].set_title("PPO mean episode reward"); axes[0].grid(alpha=0.3)

    if value_vals is not None:
        axes[1].plot(value_steps, value_vals, color="tab:orange")
    axes[1].set_xlabel("iteration"); axes[1].set_ylabel("value loss")
    axes[1].set_title("PPO critic loss"); axes[1].grid(alpha=0.3); axes[1].set_yscale("log")

    if surr_vals is not None:
        axes[2].plot(surr_steps, surr_vals, color="tab:green")
    axes[2].set_xlabel("iteration"); axes[2].set_ylabel("surrogate loss")
    axes[2].set_title("PPO actor (PPO-clip) loss"); axes[2].grid(alpha=0.3)

    fig.suptitle(f"PPO training — {len(reward_steps)} iterations", fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"[plot] wrote {out_path}  (final reward: {reward_vals[-1]:.2f})")
    return True


def plot_bc_vs_ppo(bc_eval_path: Path, ppo_eval_path: Path, out_path: Path) -> bool:
    """Side-by-side comparison of BC vs PPO rollout metrics."""
    if not bc_eval_path.exists() or not ppo_eval_path.exists():
        print(f"[plot] skip BC-vs-PPO: need both {bc_eval_path.name} and {ppo_eval_path.name}")
        return False

    bc = json.loads(bc_eval_path.read_text())
    ppo = json.loads(ppo_eval_path.read_text())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Pane 1: basket-reach rate
    rates = [bc["basket_reach_rate"] * 100, ppo["basket_reach_rate"] * 100]
    bars = axes[0].bar(["BC", "PPO"], rates, color=["tab:blue", "tab:orange"])
    for b, r in zip(bars, rates):
        axes[0].text(b.get_x() + b.get_width() / 2, r + 1, f"{r:.0f}%", ha="center", fontweight="bold")
    axes[0].set_ylabel("basket-reach rate (%)"); axes[0].set_ylim(0, 105)
    axes[0].set_title("Success rate: BC vs PPO"); axes[0].grid(alpha=0.3, axis="y")

    # Pane 2: reward distribution
    axes[1].boxplot(
        [[e["cumulative_reward"] for e in bc["episodes"]],
         [e["cumulative_reward"] for e in ppo["episodes"]]],
        labels=["BC", "PPO"], patch_artist=True,
        boxprops={"facecolor": "tab:blue", "alpha": 0.6})
    axes[1].set_ylabel("cumulative reward per episode")
    axes[1].set_title("Reward distribution"); axes[1].grid(alpha=0.3, axis="y")

    # Pane 3: min basket distance distribution
    axes[2].boxplot(
        [[e["min_basket_dist"] for e in bc["episodes"]],
         [e["min_basket_dist"] for e in ppo["episodes"]]],
        labels=["BC", "PPO"], patch_artist=True,
        boxprops={"facecolor": "tab:orange", "alpha": 0.6})
    axes[2].axhline(0.40, color="black", linestyle="--", alpha=0.5, label="40cm threshold")
    axes[2].set_ylabel("min EE→basket distance (m)")
    axes[2].set_title("Closest approach distribution"); axes[2].legend(); axes[2].grid(alpha=0.3, axis="y")

    delta = (ppo["basket_reach_rate"] - bc["basket_reach_rate"]) * 100
    fig.suptitle(f"BC ({len(bc['episodes'])} eps) vs PPO ({len(ppo['episodes'])} eps) — "
                 f"Δsuccess = {delta:+.0f} pp", fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"[plot] wrote {out_path}  (BC={bc['basket_reach_rate']*100:.0f}%, PPO={ppo['basket_reach_rate']*100:.0f}%)")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unet_metrics", default="checkpoints/air2_segmentation_metrics.json")
    parser.add_argument("--bc_log", default="checkpoints/policy_bc_mvp.log.json")
    parser.add_argument("--bc_eval", default="eval_results/bc_mvp.json")
    parser.add_argument("--ppo_tb_dir", default=None,
                        help="Path to rsl_rl run dir (contains events.out.tfevents.*).")
    parser.add_argument("--ppo_eval", default=None,
                        help="JSON from eval_ppo.py for the BC-vs-PPO comparison.")
    parser.add_argument("--out_dir", default="eval_results/plots")
    args = parser.parse_args()

    out = Path(args.out_dir)
    plot_unet(Path(args.unet_metrics), out / "unet_curves.png")
    plot_bc(Path(args.bc_log), out / "bc_curves.png")
    plot_bc_eval(Path(args.bc_eval), out / "bc_eval_summary.png")
    if args.ppo_tb_dir:
        plot_ppo_training(Path(args.ppo_tb_dir), out / "ppo_curves.png")
    if args.ppo_eval:
        plot_bc_vs_ppo(Path(args.bc_eval), Path(args.ppo_eval), out / "bc_vs_ppo.png")


if __name__ == "__main__":
    main()
