"""Read the live board_camera transform from the running Isaac Sim and append
it to _mvp_logs/board_camera_pose.log so the assistant can read it back.

How to use:
  1. In Isaac Sim, open Window > Script Editor
  2. Either paste this whole file, or click "File > Open" in the editor and
     pick this path:
       C:\\Users\\Administrator\\Desktop\\AI_Robot_Manipulation_UR3e\\scripts\\inspect_board_camera.py
  3. Move the board_camera in the Stage panel however you like
  4. Click Run. The current pose is appended to the log file.
  5. Repeat after each move — log accumulates a history.

The output gives both USD-convention values (matching the Property panel) and
ROS-convention values (matching CameraCfg.OffsetCfg(convention="ros")).
"""

import json
from pathlib import Path
from pxr import UsdGeom, Gf, Usd
import omni.usd

LOG = Path(r"C:\Users\Administrator\Desktop\AI_Robot_Manipulation_UR3e\_mvp_logs\board_camera_pose.log")
PRIM_PATH = "/World/envs/env_0/board_camera"

stage = omni.usd.get_context().get_stage()
prim = stage.GetPrimAtPath(PRIM_PATH)
if not prim or not prim.IsValid():
    raise RuntimeError(f"prim not found at {PRIM_PATH} — is the segmentation task loaded?")

xf = UsdGeom.Xformable(prim)
tc = Usd.TimeCode.Default()
local = xf.GetLocalTransformation(tc)
world = xf.ComputeLocalToWorldTransform(tc)


def _quat_wxyz(mat):
    q = mat.ExtractRotationQuat()
    img = q.GetImaginary()
    return (float(q.GetReal()), float(img[0]), float(img[1]), float(img[2]))


def _quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


# USD camera looks down -Z; ROS camera looks down +Z. The two conventions differ
# by a 180° rotation around the camera-frame X axis. In the parent frame:
#     q_ros_in_parent = q_usd_in_parent * q_swap_x180
SWAP_X180_WXYZ = (0.0, 1.0, 0.0, 0.0)

local_t = local.ExtractTranslation()
world_t = world.ExtractTranslation()
q_local_usd = _quat_wxyz(local)
q_world_usd = _quat_wxyz(world)
q_local_ros = _quat_mul(q_local_usd, SWAP_X180_WXYZ)
q_world_ros = _quat_mul(q_world_usd, SWAP_X180_WXYZ)

entry = {
    "prim_path": PRIM_PATH,
    "local_translate_xyz":   [float(local_t[0]), float(local_t[1]), float(local_t[2])],
    "local_quat_wxyz_usd":   [round(v, 4) for v in q_local_usd],
    "local_quat_wxyz_ros":   [round(v, 4) for v in q_local_ros],
    "world_translate_xyz":   [float(world_t[0]), float(world_t[1]), float(world_t[2])],
    "world_quat_wxyz_usd":   [round(v, 4) for v in q_world_usd],
    "world_quat_wxyz_ros":   [round(v, 4) for v in q_world_ros],
    "cfg_paste_pos":  "pos=({:.4f}, {:.4f}, {:.4f}),".format(*local_t),
    "cfg_paste_rot":  "rot=({:.4f}, {:.4f}, {:.4f}, {:.4f}),  # ROS w,x,y,z".format(*q_local_ros),
}

LOG.parent.mkdir(parents=True, exist_ok=True)
with LOG.open("a", encoding="utf-8") as f:
    f.write(json.dumps(entry, indent=2))
    f.write("\n---\n")

print(f"[inspect_board_camera] wrote pose to {LOG}")
print(json.dumps(entry, indent=2))
