# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg

from . import pre_grasping_env_cfg
from . import mdp


@configclass
class PoseTrackingEnvCfg(pre_grasping_env_cfg.PoseTrackingEnvCfg):
    """Environment configuration for UR5 pose tracking using IK-based control."""
    
    def __post_init__(self) -> None:
        """Post initialization."""
        # Call parent's post_init first
        super().__post_init__()
        # Override actions
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            body_name="ee_link",
            controller=DifferentialIKControllerCfg(
                command_type="pose", 
                use_relative_mode=True, 
                ik_method="dls",  # Damped least squares method
            ),
            scale=0.3,  # Reduced scale for smoother control
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
        )
        # Adjust for IK-specific settings
        self.decimation = 2  # Higher control frequency for IK
        
        # Add higher PD gains for more precise tracking
        self.scene.robot.actuators["arm_actuator"].stiffness = 150000000.0
        self.scene.robot.actuators["arm_actuator"].damping = 1000.0
