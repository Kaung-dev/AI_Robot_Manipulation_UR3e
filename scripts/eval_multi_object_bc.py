"""Multi-object pick-and-place orchestrator using per-object state-BC policies.

At each round (episode):
    1. Read ground-truth world positions of all four objects + basket
    2. Sort objects by distance to basket (closest first)
    3. For each object in order:
       a. Load its per-object state-BC checkpoint
       b. Run phase-based rollout (APPROACH → GRIP → CARRY → RELEASE)
       c. On success or timeout, move to the next object
    4. After all objects attempted, reset the env (re-randomizes positions)

The phase machine is the same as eval_state_bc.py:
    Phase 0 — APPROACH: BC arm, gripper open, count steps near object
    Phase 1 — GRIP: hold arm still, gripper closed
    Phase 2 — CARRY: BC arm, gripper closed, wait for basket proximity
    Phase 3 — RELEASE: object placed, gripper open, done

Usage (GUI — watch the robot):
    cd /mnt/extra/IsaacLab && ./isaaclab.sh -p \\
        /mnt/extra/ai_ws/AI_Robot_Manipulation_UR3e/scripts/eval_multi_object_bc.py \\
        --task Isaac-AIR2-Robotis-Franka-Segmentation-Play-v0 \\
        --ckpt_dir /mnt/extra/ai_ws/AI_Robot_Manipulation_UR3e/checkpoints \\
        --num_envs 1 --num_rounds 3 --enable_cameras

Headless:
    Add --headless to the command above.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Multi-object state-BC orchestrator.")
parser.add_argument("--task", default="Isaac-AIR2-Robotis-Franka-Segmentation-Play-v0")
parser.add_argument("--ckpt_dir", default=str(REPO_ROOT / "checkpoints"),
                    help="Directory containing policy_state_bc_<object>.pth files.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_rounds", type=int, default=1,
                    help="Number of full rounds (env resets between rounds, re-randomizing objects).")
parser.add_argument("--max_steps_per_object", type=int, default=2000)
parser.add_argument("--reset_delay_steps", type=int, default=150,
                    help="Steps to pause after placing before moving to next object (~3s at 50Hz).")
parser.add_argument("--episode_length_s", type=float, default=120.0)
parser.add_argument("--out", default="eval_results/multi_object_bc.json")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -- post-sim-init imports --------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

import isaaclab_tasks  # noqa: F401
import isaaclab_ext.tasks.air2_franka          # noqa: F401
import isaaclab_ext.tasks.air2_robotis_franka  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from isaaclab_ext.tasks.air2_franka.objects import OBJECT_SPECS


# ---------------------------------------------------------------------------
# Model helpers (same MLP as train_state_bc_from_hdf5.py / eval_state_bc.py)
# ---------------------------------------------------------------------------

def _make_mlp(input_dim, output_dim, hidden_dims, activation="elu"):
    act = nn.ELU if activation == "elu" else nn.ReLU
    layers = []
    in_d = input_dim
    for h in hidden_dims:
        layers += [nn.Linear(in_d, h), act()]
        in_d = h
    layers.append(nn.Linear(in_d, output_dim))
    return nn.Sequential(*layers)


def load_state_bc_policy(ckpt_path: str, device: str):
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = _make_mlp(
        blob["input_dim"], blob["action_dim"],
        blob["hidden_dims"], blob.get("activation", "elu"),
    ).to(device)
    model.load_state_dict(blob["state_dict"], strict=True)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASKET_POS_LOCAL = torch.tensor([-3.941, -5.785, 1.140])

# Phase machine thresholds (same as eval_state_bc.py)
NEAR_THRESH      = 250   # steps within 0.08m before closing (5s)
GRIP_HOLD        = 50    # steps holding gripper closed (1s)
MIN_CARRY_STEPS  = 200   # min carry before XY check (~4s)
BASKET_XY_RADIUS = 0.35
BASKET_REACH_RADIUS = 0.40


# ---------------------------------------------------------------------------
# Single-object rollout
# ---------------------------------------------------------------------------

def run_single_object(env, policy, target_scene_key: str, target_label: str,
                      device, origins, basket_dev, max_steps: int,
                      reset_delay: int) -> dict:
    """Run phase-based rollout for one object. Returns result dict."""
    phase = torch.zeros(1, dtype=torch.long, device=device)
    near_counter = torch.zeros(1, dtype=torch.long, device=device)
    grip_steps = torch.zeros(1, dtype=torch.long, device=device)
    carry_steps = torch.zeros(1, dtype=torch.long, device=device)
    ep_step = 0
    ep_reward = 0.0
    min_basket_dist = float("inf")
    success = False

    while ep_step < max_steps and simulation_app.is_running():
        with torch.inference_mode():
            ee = env.unwrapped.scene["ee_frame"]
            target_obj = env.unwrapped.scene[target_scene_key]

            ee_pos = ee.data.target_pos_w[:, 0, :] - origins
            obj_pos = target_obj.data.root_pos_w
            obj_local = obj_pos - origins
            ee_obj_dist = torch.linalg.norm(ee_pos - obj_local, dim=-1)

            # Build observation
            obs_dict = env.unwrapped.observation_manager.compute()
            obs_policy = obs_dict["policy"]
            if isinstance(obs_policy, dict):
                obs_policy = torch.cat(list(obs_policy.values()), dim=-1)

            # Patch target_object_position (dims 21:28) with current target
            obs_policy = obs_policy.clone()
            obs_policy[:, 21:24] = obj_pos  # world pos
            id_quat = torch.zeros(1, 4, device=device)
            id_quat[:, 0] = 1.0
            obs_policy[:, 24:28] = id_quat

            phase_bit = (phase >= 2).float().unsqueeze(-1)
            obs_input = torch.cat([obs_policy, phase_bit], dim=-1)

            action = policy(obs_input)

            # Phase 0 — APPROACH
            near_counter = torch.where(
                (phase == 0) & (ee_obj_dist < 0.08),
                near_counter + 1, near_counter)
            phase = torch.where(
                (phase == 0) & (near_counter >= NEAR_THRESH),
                torch.ones_like(phase), phase)

            # Phase 1 — GRIP
            grip_steps = torch.where(phase == 1, grip_steps + 1, grip_steps)
            phase = torch.where(
                (phase == 1) & (grip_steps >= GRIP_HOLD),
                torch.full_like(phase, 2), phase)

            # Phase 2 — CARRY
            carry_steps = torch.where(phase == 2, carry_steps + 1, carry_steps)
            xy_dist = torch.linalg.norm(obj_local[:, :2] - basket_dev[:2], dim=-1)
            phase = torch.where(
                (phase == 2) & (carry_steps >= MIN_CARRY_STEPS)
                & (xy_dist < BASKET_XY_RADIUS) & (obj_local[:, 2] <= 1.4),
                torch.full_like(phase, 3), phase)

            # Build action from phase
            arm_action = action[:, :-1].clamp(-0.15, 0.15)
            arm_action = torch.where(
                (phase == 1).unsqueeze(-1).expand_as(arm_action),
                torch.zeros_like(arm_action), arm_action)
            grip_val = torch.where(
                (phase == 0) | (phase == 3),
                torch.ones(1, device=device),
                -torch.ones(1, device=device))
            action_out = torch.cat([arm_action, grip_val.unsqueeze(-1)], dim=-1)

            if ep_step % 100 == 0:
                print(f"  [{target_label}] step={ep_step:4d}  phase={phase[0].item()}  "
                      f"ee_dist={ee_obj_dist[0].item():.3f}m  "
                      f"carry={carry_steps[0].item()}  xy_basket={xy_dist[0].item():.3f}m",
                      flush=True)

            _, rew, terminated, truncated, _ = env.step(action_out)
            ep_step += 1
            ep_reward += rew[0].item()

            dist_to_basket = torch.linalg.norm(obj_local - basket_dev, dim=1)
            min_basket_dist = min(min_basket_dist, dist_to_basket[0].item())

            # Phase 3 reached — object placed, hold briefly then move on
            if phase[0].item() == 3:
                success = True
                print(f"  [{target_label}] PLACED at step {ep_step}! "
                      f"Holding {reset_delay} steps...", flush=True)
                for _ in range(reset_delay):
                    if not simulation_app.is_running():
                        break
                    hold_action = torch.zeros_like(action_out)
                    hold_action[:, -1] = 1.0  # open gripper
                    env.step(hold_action)
                break

    return {
        "object": target_label,
        "scene_key": target_scene_key,
        "steps": ep_step,
        "reward": ep_reward,
        "min_basket_dist": min_basket_dist,
        "success": success,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_path = Path(args_cli.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- environment -------------------------------------------------------
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs)
    env_cfg.episode_length_s = args_cli.episode_length_s
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    device = env.unwrapped.device
    origins = env.unwrapped.scene.env_origins  # (N, 3)
    basket_dev = BASKET_POS_LOCAL.to(device)

    # ---- discover per-object checkpoints -----------------------------------
    ckpt_dir = Path(args_cli.ckpt_dir)
    object_info = {}
    for spec in OBJECT_SPECS:
        for pattern in [
            f"policy_state_bc_{spec.label}.pth",
            f"policy_state_bc_{spec.label}_mimic.pth",
        ]:
            p = ckpt_dir / pattern
            if p.exists():
                object_info[spec.label] = {
                    "scene_key": spec.scene_key,
                    "ckpt": str(p),
                    "class_id": spec.class_id,
                }
                break

    if not object_info:
        print("[FATAL] No per-object checkpoints found in", ckpt_dir, flush=True)
        print(f"  expected: policy_state_bc_<tool>.pth for tools: "
              f"{[s.label for s in OBJECT_SPECS]}", flush=True)
        env.close()
        return

    available = list(object_info.keys())
    print(f"[orchestrator] found {len(available)} policies: {available}", flush=True)

    # ---- preload all policies into GPU -------------------------------------
    policies = {}
    for label, info in object_info.items():
        policies[label] = load_state_bc_policy(info["ckpt"], device)
        print(f"  loaded {label}: {info['ckpt']}", flush=True)

    # ---- helper: sort objects by distance to basket ------------------------
    def get_object_distances():
        dists = []
        for label, info in object_info.items():
            obj = env.unwrapped.scene[info["scene_key"]]
            obj_pos_local = obj.data.root_pos_w[0] - origins[0]
            dist = torch.linalg.norm(obj_pos_local - basket_dev).item()
            dists.append((label, info["scene_key"], dist))
        dists.sort(key=lambda x: x[2])
        return dists

    # ---- run rounds --------------------------------------------------------
    all_rounds = []

    for round_idx in range(args_cli.num_rounds):
        if not simulation_app.is_running():
            break

        # Re-sort every round since randomization changes object positions
        sorted_objects = get_object_distances()

        print(f"\n{'#'*60}", flush=True)
        print(f"[orchestrator] ROUND {round_idx+1}/{args_cli.num_rounds}", flush=True)
        print(f"[orchestrator] pick order (closest to basket first):", flush=True)
        for i, (label, skey, dist) in enumerate(sorted_objects):
            print(f"  {i+1}. {label} (dist={dist:.3f}m)", flush=True)
        print(f"{'#'*60}", flush=True)

        round_results = []
        for obj_idx, (target_label, target_scene_key, _) in enumerate(sorted_objects):
            if not simulation_app.is_running():
                break
            if target_label not in policies:
                print(f"[orchestrator] skipping {target_label} — no policy", flush=True)
                continue

            print(f"\n[orchestrator] object {obj_idx+1}/{len(sorted_objects)}: "
                  f"{target_label}", flush=True)

            result = run_single_object(
                env, policies[target_label], target_scene_key, target_label,
                device, origins, basket_dev,
                args_cli.max_steps_per_object, args_cli.reset_delay_steps,
            )
            round_results.append(result)

            status = "SUCCESS" if result["success"] else "TIMEOUT"
            print(f"[orchestrator] {target_label}: {status} "
                  f"(steps={result['steps']}, min_dist={result['min_basket_dist']:.3f}m)",
                  flush=True)

        n_success = sum(1 for r in round_results if r["success"])
        print(f"\n[orchestrator] round {round_idx+1} complete: "
              f"{n_success}/{len(round_results)} placed", flush=True)

        all_rounds.append({
            "round": round_idx + 1,
            "pick_order": [r["object"] for r in round_results],
            "successful": n_success,
            "total": len(round_results),
            "objects": round_results,
        })

        # Reset env for next round (re-randomizes object positions)
        if round_idx < args_cli.num_rounds - 1:
            env.reset()

    # ---- summary -----------------------------------------------------------
    total_attempts = sum(r["total"] for r in all_rounds)
    total_success = sum(r["successful"] for r in all_rounds)
    summary = {
        "num_rounds": len(all_rounds),
        "total_objects_attempted": total_attempts,
        "total_successful_placements": total_success,
        "overall_success_rate": total_success / total_attempts if total_attempts else 0.0,
        "rounds": all_rounds,
    }
    out_path.write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*60}", flush=True)
    print(f"[orchestrator] DONE: {total_success}/{total_attempts} objects placed "
          f"across {len(all_rounds)} rounds "
          f"({summary['overall_success_rate']*100:.0f}%)", flush=True)
    print(f"[orchestrator] results -> {out_path}", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
