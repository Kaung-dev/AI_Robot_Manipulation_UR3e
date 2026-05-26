"""Auto-pick the toothbrush from the pegboard with the UR3e + RG2.

Drives env `Isaac-Lift-Pegboard-UR3e-RG2-IK-Abs-v0` with a warp-based finite-state
machine: rest -> hover above cube -> descend -> close gripper -> lift to goal.

The state machine kernel + driver class are mirrored from Isaac Lab's
scripts/environments/state_machine/lift_cube_sm.py (the same one the Franka
demo uses) so any upstream fix flows in by re-syncing. Differences from the
Franka version are only in:
  - task ID (Isaac-Lift-Pegboard-UR3e-RG2-IK-Abs-v0)
  - approach offset and timing constants are unchanged (the RG2 reaches the
    cube with the same TCP frame the env defines).

The env's BinaryJointPositionActionCfg maps +1 -> open_command (jaws at 0 rad)
and -1 -> close_command (jaws at 0.60 rad), matching the state machine's
GripperState.OPEN / CLOSE constants without any remapping.

Usage:
    /home/user/IsaacLab/isaaclab.sh -p scripts/pick_pegboard_auto.py --num_envs 1
    /home/user/IsaacLab/isaaclab.sh -p scripts/pick_pegboard_auto.py --num_envs 32 --headless
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Auto-pick the toothbrush from the pegboard with the UR3e + RG2.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of envs to run in parallel.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False,
    help="Disable Fabric (slower, debug-only).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------

from collections.abc import Sequence

import gymnasium as gym
import torch
import warp as wp

from isaaclab.assets.rigid_object.rigid_object_data import RigidObjectData

import isaaclab_tasks  # noqa: F401 — registers gym envs
import isaaclab_ext.tasks.lift_pegboard_ur3e_rg2  # noqa: F401 — registers local pegboard envs
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

wp.init()

TASK = "Isaac-Lift-Pegboard-UR3e-RG2-IK-Abs-v0"


# ---------------------------------------------------------------------------
# State machine — copied from Isaac Lab's lift_cube_sm.py (BSD-3, NVIDIA).
# ---------------------------------------------------------------------------


class GripperState:
    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


class PickSmState:
    REST = wp.constant(0)
    APPROACH_ABOVE_OBJECT = wp.constant(1)
    APPROACH_OBJECT = wp.constant(2)
    GRASP_OBJECT = wp.constant(3)
    LIFT_OBJECT = wp.constant(4)
    RELEASE = wp.constant(5)    # open gripper at the place pose
    RETREAT = wp.constant(6)    # back away upward after release


class PickSmWaitTime:
    REST = wp.constant(0.2)
    APPROACH_ABOVE_OBJECT = wp.constant(0.5)
    APPROACH_OBJECT = wp.constant(0.6)
    GRASP_OBJECT = wp.constant(0.3)
    LIFT_OBJECT = wp.constant(1.0)
    RELEASE = wp.constant(0.5)
    RETREAT = wp.constant(0.5)


@wp.func
def distance_below_threshold(current_pos: wp.vec3, desired_pos: wp.vec3, threshold: float) -> bool:
    return wp.length(current_pos - desired_pos) < threshold


@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    object_pose: wp.array(dtype=wp.transform),
    des_object_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    offset: wp.array(dtype=wp.transform),
    position_threshold: float,
):
    tid = wp.tid()
    state = sm_state[tid]

    if state == PickSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if sm_wait_time[tid] >= PickSmWaitTime.REST:
            sm_state[tid] = PickSmState.APPROACH_ABOVE_OBJECT
            sm_wait_time[tid] = 0.0

    elif state == PickSmState.APPROACH_ABOVE_OBJECT:
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], object_pose[tid])
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
                sm_state[tid] = PickSmState.APPROACH_OBJECT
                sm_wait_time[tid] = 0.0

    elif state == PickSmState.APPROACH_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
                sm_state[tid] = PickSmState.GRASP_OBJECT
                sm_wait_time[tid] = 0.0

    elif state == PickSmState.GRASP_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        if sm_wait_time[tid] >= PickSmWaitTime.GRASP_OBJECT:
            sm_state[tid] = PickSmState.LIFT_OBJECT
            sm_wait_time[tid] = 0.0

    elif state == PickSmState.LIFT_OBJECT:
        des_ee_pose[tid] = des_object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            if sm_wait_time[tid] >= PickSmWaitTime.LIFT_OBJECT:
                sm_state[tid] = PickSmState.RELEASE
                sm_wait_time[tid] = 0.0

    elif state == PickSmState.RELEASE:
        # Hold at the place pose, open the gripper, wait for the cube to
        # settle on the table before backing away.
        des_ee_pose[tid] = des_object_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if sm_wait_time[tid] >= PickSmWaitTime.RELEASE:
            sm_state[tid] = PickSmState.RETREAT
            sm_wait_time[tid] = 0.0

    elif state == PickSmState.RETREAT:
        # Reuse the +Z `offset` (same constant the approach state uses) to
        # rise above the place pose. Terminal state — stays here.
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], des_object_pose[tid])
        gripper_state[tid] = GripperState.OPEN
        if sm_wait_time[tid] >= PickSmWaitTime.RETREAT:
            sm_state[tid] = PickSmState.RETREAT
            sm_wait_time[tid] = 0.0

    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


class PickAndLiftSm:
    """Warp-based pick-and-lift state machine running in parallel across envs."""

    def __init__(self, dt: float, num_envs: int, device: torch.device | str = "cpu",
                 position_threshold: float = 0.01):
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        self.position_threshold = position_threshold

        self.sm_dt = torch.full((self.num_envs,), self.dt, device=self.device)
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)

        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.des_gripper_state = torch.full((self.num_envs,), 0.0, device=self.device)

        # 10 cm above the object; identity orientation (x,y,z,w convention for warp)
        self.offset = torch.zeros((self.num_envs, 7), device=self.device)
        self.offset[:, 2] = 0.1
        self.offset[:, -1] = 1.0

        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
        self.offset_wp = wp.from_torch(self.offset, wp.transform)

    def reset_idx(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0

    def compute(self, ee_pose: torch.Tensor, object_pose: torch.Tensor,
                des_object_pose: torch.Tensor) -> torch.Tensor:
        # Isaac Lab returns quaternions as (w,x,y,z); warp wants (x,y,z,w).
        ee_pose = ee_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        object_pose = object_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        des_object_pose = des_object_pose[:, [0, 1, 2, 4, 5, 6, 3]]

        ee_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        obj_wp = wp.from_torch(object_pose.contiguous(), wp.transform)
        des_obj_wp = wp.from_torch(des_object_pose.contiguous(), wp.transform)

        wp.launch(
            kernel=infer_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp, self.sm_state_wp, self.sm_wait_time_wp,
                ee_wp, obj_wp, des_obj_wp,
                self.des_ee_pose_wp, self.des_gripper_state_wp, self.offset_wp,
                self.position_threshold,
            ],
            device=self.device,
        )

        # Convert back to (w,x,y,z) for Isaac Lab's IK action.
        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        return torch.cat([des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)


# ---------------------------------------------------------------------------


def main() -> None:
    env_cfg: LiftEnvCfg = parse_env_cfg(
        TASK,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # Keep the time_out termination enabled so each episode (~5 s) ends
    # cleanly, the cube re-randomises, and the FSM (reset by reset_idx
    # below) starts a new pick cycle. This is what gives the looping
    # demo behaviour.
    print("[pick_pegboard_auto] before gym.make ...", flush=True)
    env = gym.make(TASK, cfg=env_cfg)
    print("[pick_pegboard_auto] before env.reset ...", flush=True)
    env.reset()
    print("[pick_pegboard_auto] env.reset done", flush=True)

    # Read the home-pose orientation, then rotate it so the gripper points
    # straight down. With the current init joints, the TCP's local +Z axis
    # points along world +Y (gripper sideways). The fixed rotation below
    # (−90° about world +X) sends world +Y → world −Z, so the rotated frame
    # has local +Z pointing straight down — i.e. the fingers approach from
    # above. We rotate in code rather than in the joint config so we don't
    # depend on the URDF's wrist-joint axis conventions (which varied
    # between commercial UR3e and this Inria URDF).
    ee_init = env.unwrapped.scene["ee_frame"]
    q_home = ee_init.data.target_quat_w[..., 0, :].clone()  # (num_envs, 4) wxyz

    def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Hamilton product of two (w,x,y,z) quaternions."""
        aw, ax, ay, az = a.unbind(-1)
        bw, bx, by, bz = b.unbind(-1)
        return torch.stack([
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ], dim=-1)

    # −90° about world +X axis, i.e. q = (cos(-45°), sin(-45°), 0, 0).
    q_rot = torch.tensor(
        [0.7071067811865476, -0.7071067811865476, 0.0, 0.0],
        device=env.unwrapped.device, dtype=q_home.dtype,
    ).expand_as(q_home)
    desired_orientation = quat_mul(q_rot, q_home)  # pre-multiply = world-frame rotation

    def axes_in_world(quat: torch.Tensor) -> tuple[tuple[float, float, float], ...]:
        qw, qx, qy, qz = quat[0].tolist()
        return (
            (1 - 2*(qy*qy + qz*qz), 2*(qx*qy + qz*qw),       2*(qx*qz - qy*qw)),
            (2*(qx*qy - qz*qw),       1 - 2*(qx*qx + qz*qz), 2*(qy*qz + qx*qw)),
            (2*(qx*qz + qy*qw),       2*(qy*qz - qx*qw),     1 - 2*(qx*qx + qy*qy)),
        )

    h_x, h_y, h_z = axes_in_world(q_home)
    d_x, d_y, d_z = axes_in_world(desired_orientation)
    qhw, qhx, qhy, qhz = q_home[0].tolist()
    qdw, qdx, qdy, qdz = desired_orientation[0].tolist()
    # Write to a dedicated file — Omniverse's logger hijacks stdout after
    # env.reset(), eating Python prints, but our own file handle is fine.
    diag_path = "/tmp/tcp_orientation.txt"
    with open(diag_path, "w") as fh:
        fh.write("home-pose TCP orientation (gripper sideways):\n")
        fh.write(f"  quat (w,x,y,z) = ({qhw:+.3f}, {qhx:+.3f}, {qhy:+.3f}, {qhz:+.3f})\n")
        fh.write(f"  local +X axis in world = ({h_x[0]:+.3f}, {h_x[1]:+.3f}, {h_x[2]:+.3f})\n")
        fh.write(f"  local +Y axis in world = ({h_y[0]:+.3f}, {h_y[1]:+.3f}, {h_y[2]:+.3f})\n")
        fh.write(f"  local +Z axis in world = ({h_z[0]:+.3f}, {h_z[1]:+.3f}, {h_z[2]:+.3f})\n\n")
        fh.write("desired TCP orientation (gripper down):\n")
        fh.write(f"  quat (w,x,y,z) = ({qdw:+.3f}, {qdx:+.3f}, {qdy:+.3f}, {qdz:+.3f})\n")
        fh.write(f"  local +X axis in world = ({d_x[0]:+.3f}, {d_x[1]:+.3f}, {d_x[2]:+.3f})\n")
        fh.write(f"  local +Y axis in world = ({d_y[0]:+.3f}, {d_y[1]:+.3f}, {d_y[2]:+.3f})\n")
        fh.write(f"  local +Z axis in world = ({d_z[0]:+.3f}, {d_z[1]:+.3f}, {d_z[2]:+.3f})  <- should be (0, 0, -1)\n")
    print(f"[pick_pegboard_auto] orientation diagnostics written to {diag_path}", flush=True)

    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    # Seed the action's orientation with the *home* quaternion so the very
    # first physics step is an orientation no-op; the FSM (REST → APPROACH_
    # ABOVE_OBJECT) will then gradually rotate to gripper-down as it starts
    # commanding `desired_orientation`.
    actions[:, 3:7] = q_home

    # Fixed place target — every env drops the cube at the same spot, off to
    # one side of where it was picked. Pose is in robot base frame (= world,
    # since the robot is at the origin):
    #   x = 0.30 m forward (same as cube spawn — no fore-aft motion)
    #   y = -0.25 m  (25 cm to the robot's right; cube spawn is around y=0)
    #   z = 0.08 m  (cube center ~5 cm above the table top — when the
    #                gripper opens here the cube falls only a few cm)
    place_pos = torch.tensor(
        [[0.30, -0.25, 0.08]], device=env.unwrapped.device,
    ).expand(env.unwrapped.num_envs, 3).clone()

    pick_sm = PickAndLiftSm(
        dt=env_cfg.sim.dt * env_cfg.decimation,
        num_envs=env.unwrapped.num_envs,
        device=env.unwrapped.device,
        position_threshold=0.01,
    )

    while simulation_app.is_running():
        with torch.inference_mode():
            dones = env.step(actions)[-2]

            ee = env.unwrapped.scene["ee_frame"]
            tcp_pos = ee.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
            tcp_quat = ee.data.target_quat_w[..., 0, :].clone()

            obj: RigidObjectData = env.unwrapped.scene["object"].data
            obj_pos = obj.root_pos_w - env.unwrapped.scene.env_origins

            actions = pick_sm.compute(
                torch.cat([tcp_pos, tcp_quat], dim=-1),
                torch.cat([obj_pos, desired_orientation], dim=-1),
                torch.cat([place_pos, desired_orientation], dim=-1),
            )

            if dones.any():
                pick_sm.reset_idx(dones.nonzero(as_tuple=False).squeeze(-1))

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
