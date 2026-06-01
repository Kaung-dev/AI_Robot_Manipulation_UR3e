"""Mixture-of-Experts orchestrator for the AIR2 pick-and-place pipeline.

At each episode reset:
    1. Read wrist + board cameras
    2. Run the CNN tool classifier → predict which tool is the target
    3. Load the matching per-tool PPO checkpoint
    4. Roll out PPO (state-only, no rendering → real-time on T4)

Repeats forever (or until --num_episodes). Cameras are only used at reset.

Required artifacts (build per docs/CNN_ROUTER_PLAN.md):
    checkpoints/cnn_tool_classifier.pth                                 ← train_tool_classifier.py
    logs/rsl_rl/air2_ppo/<tool>_ppo_v6/model_final.pt   for each tool   ← bc_to_ppo.py per tool

Usage:
    isaaclab.bat -p scripts/_play_router.py \\
        --task Isaac-AIR2-Robotis-Franka-Segmentation-Play-v0 \\
        --classifier_ckpt checkpoints/cnn_tool_classifier.pth \\
        --ppo_dir logs/rsl_rl/air2_ppo \\
        --num_envs 1 --num_episodes 5 --enable_cameras
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-AIR2-Robotis-Franka-Segmentation-Play-v0",
                    help="Task with cameras enabled (so the classifier can read images at reset).")
parser.add_argument("--classifier_ckpt", default="checkpoints/cnn_tool_classifier.pth")
parser.add_argument("--ppo_dir", default="logs/rsl_rl/air2_ppo",
                    help="Directory with per-tool subfolders like brush_ppo_v6/model_final.pt")
parser.add_argument("--ppo_pattern", default="{tool}_ppo_v6/model_final.pt",
                    help="Path inside --ppo_dir for each tool's PPO checkpoint")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_episodes", type=int, default=5)
parser.add_argument("--max_steps", type=int, default=1000)
parser.add_argument("--episode_length_s", type=float, default=20.0)
parser.add_argument("--confidence_threshold", type=float, default=0.5,
                    help="If max classifier prob < threshold, skip the episode (refuse to act).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import gymnasium as gym

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import isaaclab_ext.tasks.air2_franka  # noqa: F401
import isaaclab_ext.tasks.air2_robotis_franka  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper


class ToolClassifier(nn.Module):
    """Mirror of train_tool_classifier.py — frozen ResNet-18 + 4-way head."""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        backbone = torchvision.models.resnet18(weights=None)
        backbone.fc = nn.Identity()
        for p in backbone.parameters():
            p.requires_grad = False
        backbone.eval()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(512 * 2, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    @torch.no_grad()
    def forward(self, wrist: torch.Tensor, board: torch.Tensor) -> torch.Tensor:
        fw = self.backbone(wrist)
        fb = self.backbone(board)
        return self.head(torch.cat([fw, fb], dim=-1))


def load_classifier(ckpt_path: str, device: str):
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    labels = blob["labels"]
    model = ToolClassifier(num_classes=len(labels)).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, labels


def read_cameras(env, device):
    """Pull a single wrist + board frame from the env's camera sensors."""
    wrist = env.unwrapped.scene["wrist_camera"].data.output["rgb"][0]  # (H, W, 3) uint8
    board = env.unwrapped.scene["board_camera"].data.output["rgb"][0]
    tfm = T.Compose([
        T.Resize((224, 224)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    wrist_t = tfm((wrist.float() / 255.0).permute(2, 0, 1)).unsqueeze(0).to(device)
    board_t = tfm((board.float() / 255.0).permute(2, 0, 1)).unsqueeze(0).to(device)
    return wrist_t, board_t


def load_ppo_for_tool(tool: str, task_id: str, env_cfg, num_envs: int):
    """Build a fresh OnPolicyRunner + load the tool-specific checkpoint.

    Returns (runner, policy) — policy is the deterministic inference fn.
    """
    agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(task_id, "rsl_rl_cfg_entry_point")
    agent_cfg.policy.noise_std_type = "log"            # match training-time config
    agent_cfg.empirical_normalization = False

    env = gym.make(task_id, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    ckpt_path = Path(args_cli.ppo_dir) / args_cli.ppo_pattern.format(tool=tool)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"PPO checkpoint missing for tool '{tool}': {ckpt_path}")
    runner.load(str(ckpt_path))
    policy = runner.get_inference_policy(device=agent_cfg.device)
    return env, runner, policy


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier, labels = load_classifier(args_cli.classifier_ckpt, device)
    print(f"[router] classifier loaded — labels: {labels}", flush=True)

    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs)
    env_cfg.episode_length_s = args_cli.episode_length_s
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    print(f"[router] env launched ({args_cli.task}, num_envs={args_cli.num_envs})", flush=True)

    ppo_cache: dict[str, tuple] = {}  # tool → (env, runner, policy) — cached so we don't re-init Isaac each episode

    episodes_done = 0
    while episodes_done < args_cli.num_episodes and simulation_app.is_running():
        env.reset()
        # Step a few times so cameras render their first frame
        zero_act = torch.zeros(args_cli.num_envs, 7, device=device)
        for _ in range(5):
            env.unwrapped.step(zero_act)

        wrist, board = read_cameras(env, device)
        logits = classifier(wrist, board)
        probs = torch.softmax(logits, dim=-1)[0]
        pred_idx = int(probs.argmax().item())
        pred_conf = float(probs.max().item())
        tool = labels[pred_idx]
        print(f"[router] ep {episodes_done+1}: classifier → {tool} (conf={pred_conf:.2f})", flush=True)

        if pred_conf < args_cli.confidence_threshold:
            print(f"[router] confidence below {args_cli.confidence_threshold} — skipping", flush=True)
            episodes_done += 1
            continue

        task_id = f"Isaac-AIR2-Robotis-Franka-{tool.capitalize()}-v0"
        if tool not in ppo_cache:
            tool_env_cfg = parse_env_cfg(task_id, device="cuda:0", num_envs=args_cli.num_envs)
            ppo_cache[tool] = load_ppo_for_tool(tool, task_id, tool_env_cfg, args_cli.num_envs)
        tool_env, runner, policy = ppo_cache[tool]

        obs = tool_env.get_observations()[0]
        step = 0
        while step < args_cli.max_steps and simulation_app.is_running():
            with torch.inference_mode():
                action = policy(obs)
            step_out = tool_env.step(action)
            obs = step_out[0]
            done = step_out[2] if len(step_out) == 4 else (step_out[2] | step_out[3])
            if done.any():
                break
            step += 1
        print(f"[router] ep {episodes_done+1} done — steps={step}", flush=True)
        episodes_done += 1

    print("[router] all episodes complete")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
