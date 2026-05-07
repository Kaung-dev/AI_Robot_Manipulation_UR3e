"""UR3e + OnRobot RG2 articulation config for Isaac Lab.

The USD path is resolved relative to this file's *real* location
(this file is symlinked into IsaacLab from <repo>/isaaclab_ext/robots/ur3e_rg2.py),
so the same code works on any teammate's machine without edits. Override with the
UR3E_RG2_USD environment variable if needed.

The scene has been pre-processed (by scripts/fix_scene_for_isaaclab.py) to have a
single articulation root at /World/ur3e/base_link.
"""

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# Resolve to <repo>/scene/scene_isaaclab.usd by walking up from this file.
# Path(__file__).resolve() follows the IsaacLab symlink to the real file in the repo.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
UR3E_RG2_USD = os.environ.get(
    "UR3E_RG2_USD",
    str(_REPO_ROOT / "scene" / "scene_isaaclab.usd"),
)

UR3E_RG2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=UR3E_RG2_USD,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.712,
            "elbow_joint": 1.712,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,
            "rg2_gripper_joint": 0.0,
            "rg2_gripper_mirror_joint": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_.*", "elbow_joint", "wrist_.*"],
            effort_limit_sim=150.0,
            stiffness=2000.0,
            damping=100.0,
            # Joint armature = effective gearbox rotor inertia reflected to the
            # joint side. Real UR3e harmonic drives are stiff and high-inertia;
            # without this PhysX sees the arm as nearly massless and lets the
            # gripper kickback shove the wrist around.
            armature=0.1,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["rg2_gripper_joint", "rg2_gripper_mirror_joint"],
            effort_limit_sim=80.0,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""UR3e arm + OnRobot RG2 parallel-jaw gripper, single articulation."""


UR3E_RG2_HIGH_PD_CFG = UR3E_RG2_CFG.copy()
UR3E_RG2_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
UR3E_RG2_HIGH_PD_CFG.actuators["arm"].stiffness = 8000.0
UR3E_RG2_HIGH_PD_CFG.actuators["arm"].damping = 400.0
UR3E_RG2_HIGH_PD_CFG.actuators["arm"].armature = 0.2
"""Stiffer-PD variant for differential-IK / teleop tracking. Gravity is disabled
on the arm bodies so the IK target is held precisely without sag."""
