from isaacsim import SimulationApp

sim_app = SimulationApp({"headless": False})

import omni.kit.commands
from isaacsim.asset.importer.urdf import _urdf

cfg = _urdf.ImportConfig()
cfg.merge_fixed_joints = False
cfg.fix_base = True
cfg.import_inertia_tensor = True
cfg.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
cfg.default_drive_strength = 1e7
cfg.distance_scale = 1.0
cfg.make_default_prim = True

omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path="/home/user/Desktop/ur_pick/ur3e_rg2.urdf",
    import_config=cfg,
    dest_path="/home/user/Desktop/ur_pick/ur3e_rg2.usd",
)

while sim_app.is_running():
    sim_app.update()

sim_app.close()
