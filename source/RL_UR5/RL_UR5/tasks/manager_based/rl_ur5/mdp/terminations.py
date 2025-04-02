from __future__ import annotations

import torch
import math
from typing import TYPE_CHECKING
import numpy as np
import random
from .observations import *



import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# tasks/manager_based/rl_ur5/mdp/terminations.py

def pose_tracking_success(
    env: ManagerBasedRLEnv,
    position_threshold: float = 0.05,
    orientation_threshold: float = 0.1,
    velocity_threshold: float = 0.05,
    torque_threshold: float = 1.0,
    command_name: str = "tracking_pose",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Determine if the pose tracking task has been successfully completed.
    
    Args:
        env: The RL environment instance
        position_threshold: Maximum position error to consider successful
        orientation_threshold: Maximum orientation error to consider successful
        velocity_threshold: Maximum joint velocity magnitude to consider stable
        torque_threshold: Maximum joint torque magnitude to consider stable
        command_name: Name of the command containing the target pose
        asset_cfg: Configuration for the robot asset
        ee_frame_cfg: Configuration for the end-effector frame
        
    Returns:
        torch.Tensor: Boolean tensor indicating task success
    """
    # Get end-effector position and orientation using ee_frame
    asset = env.scene[asset_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    ee_position = ee_frame.data.target_pos_w[..., 0, :]
    ee_quat = ee_frame.data.target_quat_w[..., 0, :]
    
    # Get the desired position and orientation from the command
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_quat_b = command[:, 3:7]
    
    # Transform to world frame
    des_pos_w, _ = math_utils.combine_frame_transforms(
        asset.data.root_state_w[:, :3], 
        asset.data.root_state_w[:, 3:7], 
        des_pos_b
    )
    des_quat_w = math_utils.quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    
    # Calculate position error
    position_error = torch.norm(ee_position - des_pos_w, p=2, dim=-1)
    
    # Calculate orientation error
    orientation_error = math_utils.quat_error_magnitude(ee_quat, des_quat_w)
    
    # Get joint velocities and torques
    joint_velocities = torch.norm(asset.data.joint_vel, p=2, dim=-1)
    
    # Check if joint torques are available
    has_torques = hasattr(asset.data, 'joint_effort') and asset.data.joint_effort is not None
    if has_torques:
        joint_torques = torch.norm(asset.data.joint_effort, p=2, dim=-1)
    else:
        # If torques aren't available, just use a tensor of zeros
        joint_torques = torch.zeros_like(joint_velocities)
    
    # Check if all conditions are met
    position_success = position_error < position_threshold
    orientation_success = orientation_error < orientation_threshold
    velocity_success = joint_velocities < velocity_threshold
    torque_success = joint_torques < torque_threshold
    
    # Success is when all criteria are met (position, orientation, velocity, and torque)
    success = torch.logical_and(
        torch.logical_and(position_success, orientation_success),
        torch.logical_and(velocity_success, torque_success)
    )
    
    return success