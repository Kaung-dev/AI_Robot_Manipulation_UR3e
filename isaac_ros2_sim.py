from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False, "renderer": "RayTracedLighting"})

from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

import omni.kit.commands
import omni.graph.core as og
import omni.usd
from isaacsim.core.api import World
from isaacsim.asset.importer.urdf import _urdf
from pxr import UsdPhysics

URDF_PATH = "/home/user/Desktop/ur_pick/ur3e_rg2.urdf"
USD_PATH = "/home/user/Desktop/ur_pick/ur3e_rg2.usd"

cfg = _urdf.ImportConfig()
cfg.merge_fixed_joints = False
cfg.fix_base = True
cfg.import_inertia_tensor = True
cfg.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
cfg.default_drive_strength = 1e7
cfg.distance_scale = 1.0
cfg.make_default_prim = True

print("[bridge] re-importing URDF (rebuilds USD from current URDF)…")
omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path=URDF_PATH,
    import_config=cfg,
    dest_path=USD_PATH,
)
simulation_app.update()

omni.usd.get_context().open_stage(USD_PATH)
simulation_app.update()

stage = omni.usd.get_context().get_stage()
default_prim = stage.GetDefaultPrim()
print(f"[bridge] default prim: {default_prim.GetPath() if default_prim else 'NONE'}")

drive_set = 0
for prim in stage.Traverse():
    tn = prim.GetTypeName()
    if tn == "PhysicsRevoluteJoint":
        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.CreateStiffnessAttr(1e6)
        drive.CreateDampingAttr(1e5)
        drive.CreateMaxForceAttr(1e7)
        drive.CreateTargetPositionAttr(0.0)
        drive_set += 1
    elif tn == "PhysicsPrismaticJoint":
        drive = UsdPhysics.DriveAPI.Apply(prim, "linear")
        drive.CreateStiffnessAttr(1e6)
        drive.CreateDampingAttr(1e5)
        drive.CreateMaxForceAttr(1e6)
        drive.CreateTargetPositionAttr(0.0)
        drive_set += 1
print(f"[bridge] applied PD drives to {drive_set} joints (stiffness=1e6, damping=1e5, target=0)")

api_holders = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
print(f"[bridge] prims with ArticulationRootAPI before fix: {api_holders}")

# The URDF importer puts ArticulationRootAPI on /ur3e_rg2/root_joint (the fixed joint).
# That prim has 0 DOFs from PhysX's POV, so PublishJointState publishes empty messages.
# Move the API to the parent xform so PhysX sees the real articulation chain.
for path in api_holders:
    prim = stage.GetPrimAtPath(path)
    if prim and prim.GetTypeName() in ("PhysicsFixedJoint", "PhysicsRevoluteJoint", "PhysicsPrismaticJoint"):
        prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        print(f"[bridge] removed ArticulationRootAPI from joint prim: {path}")

robot_prim_path = str(default_prim.GetPath()) if default_prim else None
if robot_prim_path is None:
    raise RuntimeError("[bridge] no default prim")
target_prim = stage.GetPrimAtPath(robot_prim_path)
UsdPhysics.ArticulationRootAPI.Apply(target_prim)
print(f"[bridge] applied ArticulationRootAPI to: {robot_prim_path}")
print(f"[bridge] articulation target: {robot_prim_path}")

keys = og.Controller.Keys
og.Controller.edit(
    {"graph_path": "/ROS2_Bridge_Graph", "evaluator_name": "execution"},
    {
        keys.CREATE_NODES: [
            ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
            ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
            ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
            ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
            ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
        ],
        keys.CONNECT: [
            ("OnPlaybackTick.outputs:tick", "PublishJointState.inputs:execIn"),
            ("OnPlaybackTick.outputs:tick", "SubscribeJointState.inputs:execIn"),
            ("OnPlaybackTick.outputs:tick", "ArticulationController.inputs:execIn"),
            ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
            ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
            ("SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
            ("SubscribeJointState.outputs:velocityCommand", "ArticulationController.inputs:velocityCommand"),
            ("SubscribeJointState.outputs:effortCommand", "ArticulationController.inputs:effortCommand"),
        ],
        keys.SET_VALUES: [
            ("PublishJointState.inputs:topicName", "isaac_joint_states"),
            ("PublishJointState.inputs:targetPrim", [robot_prim_path]),
            ("SubscribeJointState.inputs:topicName", "joint_command"),
            ("ArticulationController.inputs:targetPrim", [robot_prim_path]),
        ],
    },
)

simulation_app.update()

world = World(stage_units_in_meters=1.0)
world.reset()
world.play()

print("[bridge] sim playing — publishing /joint_states, listening on /joint_command")
while simulation_app.is_running():
    world.step(render=True)

world.stop()
simulation_app.close()
