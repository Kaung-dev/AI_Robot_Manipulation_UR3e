"""Roll out a trained STATE-only BC policy (from train_state_bc.py) and report metrics.

This is the smaller MLP that mirrors rsl_rl's actor architecture — the one we
use to warm-start PPO ([bc_to_ppo.py](bc_to_ppo.py)). Same metrics + JSON
schema as [eval_bc.py](eval_bc.py) so [plot_training_curves.py](plot_training_curves.py)
can plot a state-BC line on the same axes.

Use in GUI (drop --headless to watch):
    C:\\isaac\\IsaacLab\\isaaclab.bat -p scripts/eval_state_bc.py ^
        --state_bc_ckpt checkpoints/policy_state_bc.pth ^
        --task Isaac-AIR2-Franka-Play-v0 ^
        --enable_cameras ^
        --num_envs 1 --num_episodes 5

Headless eval (for metrics + JSON):
    C:\\isaac\\IsaacLab\\isaaclab.bat -p scripts/eval_state_bc.py ^
        --state_bc_ckpt checkpoints/policy_state_bc.pth ^
        --task Isaac-AIR2-Franka-v0 ^
        --headless --enable_cameras ^
        --num_envs 4 --num_episodes 20 ^
        --out eval_results/state_bc.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a trained STATE-only BC policy.")
parser.add_argument("--state_bc_ckpt", required=True,
                    help="Path to checkpoints/policy_state_bc.pth from train_state_bc.py.")
parser.add_argument("--task", default="Isaac-AIR2-Franka-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_episodes", type=int, default=5)
parser.add_argument("--max_steps", type=int, default=2000)
parser.add_argument("--reset_delay", type=int, default=250,
                    help="Steps to hold at release before resetting (lets you see the result). 250=~5s at 50Hz.")
parser.add_argument("--episode_length_s", type=float, default=40.0)
parser.add_argument("--exclude_slots", type=str, default="",
                    help="Comma-separated slots to SKIP at spawn (e.g. 'R1'). On reset, if the object "
                         "lands on an excluded slot, the env is re-rolled until it doesn't. Use for "
                         "objects that can't be solved at a given slot (e.g. pliers R1).")
parser.add_argument("--servo_dist", type=float, default=0.18,
                    help="v2: when EE is within this dist of the object during APPROACH, "
                         "servo the EE straight onto the object (robot-root frame) instead "
                         "of trusting the under/over-shooting BC. Fixes the R2 downward "
                         "overshoot. Set 0 to disable.")
parser.add_argument("--out", default="eval_results/state_bc.json")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -- post-sim-init imports --------------------------------------------------

import numpy as np
import torch
import gymnasium as gym

import isaaclab_tasks  # noqa: F401
import isaaclab_ext.tasks.air2_franka          # noqa: F401
import isaaclab_ext.tasks.air2_robotis_franka  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import subtract_frame_transforms  # v2 servo: world->robot-root


def _make_mlp(input_dim, output_dim, hidden_dims, activation="elu"):
    import torch.nn as nn
    act = nn.ELU if activation == "elu" else nn.ReLU
    layers = []
    in_d = input_dim
    for h in hidden_dims:
        layers += [nn.Linear(in_d, h), act()]
        in_d = h
    layers.append(nn.Linear(in_d, output_dim))
    return nn.Sequential(*layers)


# Same proxy success metric used by eval_bc.py / eval_ppo.py.
BASKET_POS_LOCAL = torch.tensor([-3.941, -5.785, 1.140])
BASKET_REACH_RADIUS = 0.40

# Slot positions in ROBOT-ROOT frame; (x, z) distinguish them (y≈-0.67 constant).
_SLOTS_ROOT = {"R0": (-0.03, 0.57), "R1": (-0.20, 0.57), "R2": (-0.03, 0.29), "R3": (-0.20, 0.29)}


def _current_slot(env, tol=0.08):
    """Which slot the target object spawned on (robot-root frame), or None."""
    obj = env.unwrapped.scene["object"]
    p = (obj.data.root_pos_w - env.unwrapped.scene.env_origins)[0]
    x, z = float(p[0]), float(p[2])
    best, bestd = None, 1e9
    for name, (sx, sz) in _SLOTS_ROOT.items():
        d = ((x - sx) ** 2 + (z - sz) ** 2) ** 0.5
        if d < bestd:
            best, bestd = name, d
    return best if bestd < tol else None


def _reset_avoiding(env, excluded, tries=40):
    """Reset; if the object lands on an excluded slot (e.g. pliers R1), re-roll."""
    env.reset()
    if not excluded:
        return
    for _ in range(tries):
        if _current_slot(env) not in excluded:
            return
        env.reset()


def load_state_bc_policy(ckpt_path: str, device: str):
    """Reconstruct the MLP from the checkpoint's metadata + state_dict."""
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" not in blob:
        raise RuntimeError(f"Checkpoint {ckpt_path} missing 'state_dict' key")
    input_dim  = blob["input_dim"]
    action_dim = blob["action_dim"]
    hidden_dims = blob["hidden_dims"]
    activation  = blob.get("activation", "elu")
    model = _make_mlp(input_dim, action_dim, hidden_dims, activation).to(device)
    model.load_state_dict(blob["state_dict"], strict=True)
    model.eval()
    # v2: load normalization stats + obs-alignment flag (None if a legacy ckpt).
    obs_mean = obs_std = None
    if blob.get("obs_mean") is not None and blob.get("obs_std") is not None:
        obs_mean = torch.as_tensor(blob["obs_mean"], dtype=torch.float32, device=device)
        obs_std  = torch.as_tensor(blob["obs_std"],  dtype=torch.float32, device=device)
    obs_aligned = bool(blob.get("obs_aligned", False))
    print(f"[eval-state-bc] loaded MLP: input={input_dim} hidden={hidden_dims} "
          f"output={action_dim} act={activation} | normalized={obs_mean is not None} aligned={obs_aligned}")
    return model, obs_mean, obs_std, obs_aligned


def main():
    out_path = Path(args_cli.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs)
    env_cfg.episode_length_s = args_cli.episode_length_s

    # v2 fix: DISABLE the env's early terminations during eval. Otherwise the
    # task_success termination fires ~10 steps after release (as the brush enters
    # the basket region) and the env AUTO-RESETS mid-fall — so the brush never
    # physically lands and we score it False even though it would land. With these
    # off, the episode runs through our release-and-settle window so the brush
    # actually drops + settles, and we measure the true landed state. (We keep
    # time_out so episodes can't run forever; our loop also caps at --max_steps.)
    _disabled = []
    for _tname in list(vars(env_cfg.terminations).keys()):
        if _tname.startswith("_") or _tname in ("time_out", "time_outs"):
            continue
        if getattr(env_cfg.terminations, _tname) is not None:
            setattr(env_cfg.terminations, _tname, None)
            _disabled.append(_tname)
    print(f"[eval-state-bc] disabled env terminations (let object settle): {_disabled}", flush=True)

    # v2 fix: strip the cosmetic debug-vis marker FrameTransformers. They target
    # BRUSH-task prim paths (e.g. "Object/brush_ring"), which don't exist in the
    # pliers/scissors/screwdriver scenes -> target_pos_w is None and the debug-vis
    # callback CRASHES the app at startup. They're visualization-only; the eval
    # uses scene["ee_frame"] + scene["object"], not these. Safe to remove for all.
    _stripped = []
    for _mk in ("basket_frame", "basket_beacon_frame", "brush_frame", "pliers_frame",
                "scissors_frame", "screwdriver_frame", "ee_tcp_marker"):
        if hasattr(env_cfg.scene, _mk) and getattr(env_cfg.scene, _mk) is not None:
            setattr(env_cfg.scene, _mk, None)
            _stripped.append(_mk)
    print(f"[eval-state-bc] stripped cosmetic markers: {_stripped}", flush=True)

    excluded = {s.strip().upper() for s in args_cli.exclude_slots.split(",") if s.strip()}
    if excluded:
        print(f"[eval-state-bc] excluding spawn slots: {sorted(excluded)} (re-roll reset until allowed)", flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    _reset_avoiding(env, excluded)
    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs

    policy, obs_mean, obs_std, obs_aligned = load_state_bc_policy(args_cli.state_bc_ckpt, device=device)
    print(f"[eval-state-bc] task={args_cli.task} envs={num_envs} target_episodes={args_cli.num_episodes} episode_length_s={env_cfg.episode_length_s}",
          flush=True)

    basket_pos_dev = BASKET_POS_LOCAL.to(device)

    ep_step      = torch.zeros(num_envs, dtype=torch.long, device=device)
    ep_reward    = torch.zeros(num_envs, device=device)
    ep_min_basket_dist = torch.full((num_envs,), float("inf"), device=device)
    # Phase state: 0=APPROACH, 1=GRIP, 2=CARRY, 3=RELEASE
    phase        = torch.zeros(num_envs, dtype=torch.long, device=device)
    near_counter = torch.zeros(num_envs, dtype=torch.long, device=device)
    grip_steps   = torch.zeros(num_envs, dtype=torch.long, device=device)
    carry_steps  = torch.zeros(num_envs, dtype=torch.long, device=device)
    release_steps = torch.zeros(num_envs, dtype=torch.long, device=device)  # v2: settle window
    released      = torch.zeros(num_envs, dtype=torch.bool, device=device)  # v2: reached RELEASE
    NEAR_THRESH      = 250   # steps within 0.08m before closing (5s)
    GRIP_HOLD        = 50    # steps holding gripper closed while gripping (1s)
    MIN_CARRY_STEPS  = 200   # min carry steps before XY check activates (~5s)
    BASKET_XY_RADIUS = 0.35  # XY radius around basket to trigger release
    # v2 release+landed scoring:
    RELEASE_HOLD     = 120   # steps to hold (gripper open, arm still) after release so the object drops + settles (~2.4s)
    SUCCESS_RADIUS   = 0.25  # 3D object->basket dist that counts as "in the basket" AFTER the drop
    SUCCESS_Z_MAX    = 1.27  # object must settle at/below this height (basket z=1.14; >this = still hovering on rim)
    CARRY_ARM_CLAMP  = 0.15  # carry arm-action cap (== normal clamp = effectively off). 0.06 was too weak:
                             # it stalled R2's carry before the basket and didn't fix R1 (R1's issue is the
                             # GRASP not holding, not carry speed). Left tunable but default = no extra limit.
    EXTRACT_STEPS    = 0     # off (was an attempt to fix pliers R1 by pulling +y off the board first; it
                             # didn't reliably help R1, so we exclude R1 from spawning instead — see
                             # --exclude_slots. Set >0 to re-enable the off-board pull at the start of carry).
    EXTRACT_CMD      = 0.10  # +y in robot-root frame = toward the robot / away from the board
    saved_episodes: list[dict] = []

    while len(saved_episodes) < args_cli.num_episodes and simulation_app.is_running():
        with torch.inference_mode():
            ee    = env.unwrapped.scene["ee_frame"]
            brush = env.unwrapped.scene["object"]

            ee_pos  = ee.data.target_pos_w[:, 0, :] - env.unwrapped.scene.env_origins
            ee_quat = ee.data.target_quat_w[:, 0, :]
            obj_pos = brush.data.root_pos_w
            ee_obj_dist = torch.linalg.norm(ee_pos - (obj_pos - env.unwrapped.scene.env_origins), dim=-1)

            obs_dict   = env.unwrapped.observation_manager.compute()
            obs_policy = obs_dict["policy"]
            if isinstance(obs_policy, dict):
                obs_policy = torch.cat(list(obs_policy.values()), dim=-1)
            # v2 obs-alignment: match training — dims 21:28 (target_object_position)
            # = object position (robot-root, == dims 18:21) + identity quat, NOT the
            # resampling command / world pose. Only when the ckpt was trained aligned.
            obs_policy = obs_policy.clone()
            if obs_aligned:
                id_quat = torch.zeros(num_envs, 4, device=device)
                id_quat[:, 0] = 1.0
                obs_policy[:, 21:24] = obs_policy[:, 18:21]  # robot-root object pos
                obs_policy[:, 24:28] = id_quat
            else:
                # legacy ckpt: keep the original world-pos patch
                id_quat = torch.zeros(num_envs, 4, device=device)
                id_quat[:, 0] = 1.0
                obs_policy[:, 21:24] = obj_pos
                obs_policy[:, 24:28] = id_quat
            phase_bit  = (phase >= 2).float().unsqueeze(-1)
            obs_input  = torch.cat([obs_policy, phase_bit], dim=-1)  # (N,43)

            # v2 normalization: apply the SAME standardization used in training.
            if obs_mean is not None:
                obs_input = (obs_input - obs_mean) / obs_std

            action = policy(obs_input)

            # Phase 0 — APPROACH: BC arm, gripper open, count time near object
            near_counter = torch.where(
                (phase == 0) & (ee_obj_dist < 0.08),
                near_counter + 1, near_counter)
            phase = torch.where((phase == 0) & (near_counter >= NEAR_THRESH),
                                torch.ones_like(phase), phase)

            # Phase 1 — GRIP: hold arm still, gripper closed, wait GRIP_HOLD steps
            grip_steps = torch.where(phase == 1, grip_steps + 1, grip_steps)
            phase = torch.where((phase == 1) & (grip_steps >= GRIP_HOLD),
                                torch.full_like(phase, 2), phase)

            # Phase 2 — CARRY: BC arm, gripper closed
            # After MIN_CARRY_STEPS, trigger release when object enters basket XY radius
            carry_steps = torch.where(phase == 2, carry_steps + 1, carry_steps)
            obj_local = obj_pos - env.unwrapped.scene.env_origins
            xy_dist_to_basket = torch.linalg.norm(
                obj_local[:, :2] - basket_pos_dev[:2], dim=-1)
            # v2 release fix: trigger RELEASE once the brush is CENTERED over the
            # basket (XY) after carrying — NO height gate. The old `obj_z <= 1.4`
            # gate never passed because the gripper carries the brush ABOVE the rim
            # (>1.4 m), so it hovered forever and never dropped. We just need it
            # well-centred in XY; the drop is what gets it into the basket.
            RELEASE_XY = 0.12   # tighter than BASKET_XY_RADIUS so we drop on-centre
            phase = torch.where(
                (phase == 2) & (carry_steps >= MIN_CARRY_STEPS)
                & (xy_dist_to_basket < RELEASE_XY),
                torch.full_like(phase, 3), phase)

            # Phase 3 — RELEASE: gripper open, arm held still; let the object DROP
            # and SETTLE for RELEASE_HOLD steps before ending the episode. (v2 fix:
            # the original ended the episode the instant phase==3, so the gripper
            # never actually released and the object was never dropped.)
            released = released | (phase == 3)
            release_steps = torch.where(phase == 3, release_steps + 1, release_steps)

            # Build action based on phase — clip arm to prevent OOD large outputs
            arm_action = action[:, :-1].clamp(-0.15, 0.15)

            # v2 SERVO: closed-loop final approach. When the EE is within servo_dist
            # of the brush during APPROACH, override the BC *position* command with a
            # straight servo toward the object's true pose (robot-root frame). This
            # stops the BC's under/over-shoot — specifically the R2 downward overshoot
            # — because we drive to the object's actual height. BC's wrist *rotation*
            # (dims 3:6) is kept, since the grasp orientation already works at R0/R1.
            if args_cli.servo_dist > 0:
                robot = env.unwrapped.scene["robot"]
                root_pos_w, root_quat_w = robot.data.root_pos_w, robot.data.root_quat_w
                ee_root,  _ = subtract_frame_transforms(root_pos_w, root_quat_w, ee.data.target_pos_w[:, 0, :])
                obj_root, _ = subtract_frame_transforms(root_pos_w, root_quat_w, brush.data.root_pos_w)
                servo_cmd = ((obj_root - ee_root) * 2.0).clamp(-0.12, 0.12)  # *2 undoes IK scale=0.5
                servo_active = (phase == 0) & (ee_obj_dist < args_cli.servo_dist)
                arm_action[:, 0:3] = torch.where(
                    servo_active.unsqueeze(-1).expand(-1, 3), servo_cmd, arm_action[:, 0:3])

            # EXTRACT after grasp: for the first EXTRACT_STEPS of CARRY, pull the object
            # straight OFF the pegboard (+y in robot-root = toward the robot) instead of
            # carrying laterally. The slots are on a vertical board; sliding sideways while
            # the object is still seated drags it out of the gripper (R1's failure). Pull
            # it clear first, THEN let the BC carry to the basket. Gripper stays closed.
            extracting = (phase == 2) & (carry_steps < EXTRACT_STEPS)
            extract_cmd = torch.zeros_like(arm_action)
            extract_cmd[:, 1] = EXTRACT_CMD   # +y root = off the board, toward the robot
            arm_action = torch.where(extracting.unsqueeze(-1).expand_as(arm_action), extract_cmd, arm_action)

            # Gentle CARRY clamp (currently a no-op at 0.15; kept tunable).
            carry_mask = (phase == 2).unsqueeze(-1).expand_as(arm_action)
            arm_action = torch.where(
                carry_mask, arm_action.clamp(-CARRY_ARM_CLAMP, CARRY_ARM_CLAMP), arm_action)

            # Hold the arm still during GRIP (1) and RELEASE (3) so the grasp/drop is clean.
            hold_arm = (phase == 1) | (phase == 3)
            arm_action = torch.where(hold_arm.unsqueeze(-1).expand_as(arm_action),
                                     torch.zeros_like(arm_action), arm_action)
            grip_val = torch.where(phase == 0,
                                   torch.ones(num_envs, device=device),
                                   torch.where(phase == 3,
                                               torch.ones(num_envs, device=device),
                                               -torch.ones(num_envs, device=device)))
            action = torch.cat([arm_action, grip_val.unsqueeze(-1)], dim=-1)

            if ep_step[0] % 100 == 0:
                robot = env.unwrapped.scene["robot"]
                jvel = robot.data.joint_vel[0].abs().max().item()
                raw_arm_mag = action[0, :-1].abs().max().item()
                print(f"[phase] step={ep_step[0].item():4d}  phase={phase[0].item()}  "
                      f"near={near_counter[0].item()}  ee_dist={ee_obj_dist[0].item():.3f}m  "
                      f"carry={carry_steps[0].item()}  xy_basket={xy_dist_to_basket[0].item():.3f}m  "
                      f"max_jvel={jvel:.3f}  raw_arm={raw_arm_mag:.3f}", flush=True)

            _, rew, terminated, truncated, _ = env.step(action)
            ep_step += 1
            ep_reward += rew

            brush_local = brush.data.root_pos_w - env.unwrapped.scene.env_origins
            dist = torch.linalg.norm(brush_local - basket_pos_dev, dim=1)
            ep_min_basket_dist = torch.minimum(ep_min_basket_dist, dist)

            # v2: end the episode only AFTER the release-and-settle window, so the
            # object has actually been dropped + landed (not at the release *decision*).
            done = terminated | truncated | (ep_step >= args_cli.max_steps) \
                   | ((phase == 3) & (release_steps >= RELEASE_HOLD))
            # v2 success = released AND object settled inside the basket (post-drop),
            # NOT min-distance-while-gripped.
            landed = released & (dist < SUCCESS_RADIUS) & (brush_local[:, 2] <= SUCCESS_Z_MAX)
            if done.any():
                # Brief visual pause so you can see the final landed state.
                if released[0].item() and args_cli.reset_delay > 0:
                    import time; time.sleep(args_cli.reset_delay / 50.0)
                for i in done.nonzero(as_tuple=False).squeeze(-1).cpu().tolist():
                    if len(saved_episodes) >= args_cli.num_episodes:
                        break
                    success = bool(landed[i].item())
                    saved_episodes.append({
                        "ep_idx": len(saved_episodes),
                        "env_idx": int(i),
                        "steps": int(ep_step[i].item()),
                        "cumulative_reward": float(ep_reward[i].item()),
                        "min_basket_dist": float(ep_min_basket_dist[i].item()),
                        "final_basket_dist": float(dist[i].item()),
                        "released": bool(released[i].item()),
                        "landed_in_basket": success,
                        "reached_basket": success,  # success = released + landed (v2 definition)
                        "terminated": bool(terminated[i].item()),
                        "truncated": bool(truncated[i].item()),
                    })
                    print(f"[eval-state-bc] ep {len(saved_episodes)}/{args_cli.num_episodes}  "
                          f"steps={int(ep_step[i].item())}  reward={float(ep_reward[i].item()):.2f}  "
                          f"released={bool(released[i].item())}  final_dist={float(dist[i].item()):.3f}m  "
                          f"LANDED={success}", flush=True)
                # Terminations are disabled, so the env won't auto-reset — do it
                # explicitly so the next episode spawns a fresh (randomized) slot,
                # re-rolling past any excluded slot (e.g. pliers R1).
                _reset_avoiding(env, excluded)
                ep_step[done] = 0
                ep_reward[done] = 0.0
                ep_min_basket_dist[done] = float("inf")
                phase[done] = 0
                near_counter[done] = 0
                grip_steps[done] = 0
                carry_steps[done] = 0
                release_steps[done] = 0
                released[done] = False

    rewards = [e["cumulative_reward"] for e in saved_episodes]
    reached = sum(1 for e in saved_episodes if e["reached_basket"])
    n = len(saved_episodes)
    summary = {
        "policy": "state_bc",
        "checkpoint": str(args_cli.state_bc_ckpt),
        "num_episodes": n,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "std_reward": float(np.std(rewards)) if rewards else 0.0,
        "basket_reach_rate": reached / n if n else 0.0,
        "mean_steps": float(np.mean([e["steps"] for e in saved_episodes])) if n else 0.0,
        "episodes": saved_episodes,
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[eval-state-bc] basket-reach: {summary['basket_reach_rate']*100:.0f}% "
          f"({reached}/{n}), mean reward: {summary['mean_reward']:.2f}, results -> {out_path}",
          flush=True)

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
