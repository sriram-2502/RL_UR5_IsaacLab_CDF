from __future__ import annotations

import torch
import math
from typing import TYPE_CHECKING
import numpy as np
import random
from .observations import *
from .thresholds import *  # Import all constants from thresholds


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


    
    # Check if all conditions are met
    position_success = position_error < position_threshold
    orientation_success = orientation_error < orientation_threshold
    velocity_success = joint_velocities < velocity_threshold

    
    # Success is when all criteria are met (position, orientation, velocity, and torque)
    success = torch.logical_and(
        torch.logical_and(position_success, orientation_success),velocity_success
    )
    
    return success


def alignment_success(
    env: ManagerBasedRLEnv,
    position_threshold: float = 0.05,
    orientation_threshold: float = 0.9,
    velocity_threshold: float = 0.05,
    torque_threshold: float = 1.0,
    height_offset: float = 0.3,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Determine if the end-effector is successfully aligned above the target cube.
    
    Args:
        env: The RL environment instance
        position_threshold: Maximum position error to consider successful
        orientation_threshold: Minimum orientation alignment to consider successful (0-1)
        velocity_threshold: Maximum joint velocity magnitude to consider stable
        torque_threshold: Maximum joint torque magnitude to consider stable
        height_offset: Target height above the cube
        ee_frame_cfg: Configuration for the end-effector frame
        asset_cfg: Configuration for the robot asset
        
    Returns:
        torch.Tensor: Boolean tensor indicating alignment success
    """
    # Get end-effector position and orientation
    ee_frame = env.scene[ee_frame_cfg.name]
    ee_position = ee_frame.data.target_pos_w[..., 0, :]
    ee_quat = ee_frame.data.target_quat_w[..., 0, :]
    
    # Get robot asset for joint velocities and torques
    asset = env.scene[asset_cfg.name]
    joint_velocities = torch.norm(asset.data.joint_vel, p=2, dim=-1)
    
    # Check if joint torques are available
    has_torques = hasattr(asset.data, 'joint_effort') and asset.data.joint_effort is not None
    if has_torques:
        joint_torques = torch.norm(asset.data.joint_effort, p=2, dim=-1)
    else:
        # If torques aren't available, just use a tensor of zeros
        joint_torques = torch.zeros_like(joint_velocities)
    
    # Initialize success tensor
    success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    for i in range(env.num_envs):
        if hasattr(env, "task_info") and i in env.task_info:
            target_info = env.task_info[i]
            target_cube_name = target_info["target_cube"]
            cube = env.scene[target_cube_name]
            cube_position = cube.data.root_pos_w[i, :3]
            cube_quat = cube.data.root_quat_w[i, :]
            
            # Calculate target position (above cube)
            target_position = cube_position.clone()
            target_position[2] += height_offset
            
            # Check position
            position_error = torch.norm(ee_position[i] - target_position, p=2)
            position_success = position_error < position_threshold
            
            # Check orientation alignment
            ee_rot_mat = math_utils.matrix_from_quat(ee_quat[i].unsqueeze(0)).squeeze(0)
            cube_rot_mat = math_utils.matrix_from_quat(cube_quat.unsqueeze(0)).squeeze(0)
            
            # Extract axes
            ee_x_axis = ee_rot_mat[:, 0]
            ee_y_axis = ee_rot_mat[:, 1]
            cube_y_axis = cube_rot_mat[:, 1]
            cube_z_axis = cube_rot_mat[:, 2]
            
            # Calculate alignment
            x_z_alignment = torch.abs(torch.dot(-ee_x_axis, cube_z_axis))
            y_y_alignment = torch.abs(torch.dot(-ee_y_axis, cube_y_axis))
            combined_alignment = torch.sqrt(x_z_alignment * y_y_alignment)
            
            orientation_success = combined_alignment > orientation_threshold
            
            # Check velocity and torque stability
            velocity_success = joint_velocities[i] < velocity_threshold
            torque_success = joint_torques[i] < torque_threshold
            
            # Set success if all criteria are met
            if position_success and orientation_success and velocity_success and torque_success:
                success[i] = True
    
    return success


def task_success(
    env: ManagerBasedRLEnv,
    gripper_threshold: float = 5.0,
    distance_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Determine if the pick and place task has been successfully completed.
    
    The task is considered successful when:
    1. The target cube is at the desired placement position
    2. The gripper is open (cube has been released)
    
    Args:
        env: The RL environment instance
        gripper_threshold: Maximum gripper position to consider "open"
        distance_threshold: Maximum distance to consider "at target"
        asset_cfg: Configuration for the robot asset
        
    Returns:
        torch.Tensor: Boolean tensor indicating task success
    """
    # Initialize success tensor
    success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    # Get robot
    robot = env.scene[asset_cfg.name]
    
    # Find joint index by name
    joint_names = robot.joint_names
    gripper_joint_name = "robotiq_85_left_knuckle_joint"
    gripper_joint_idx = joint_names.index(gripper_joint_name) if gripper_joint_name in joint_names else -1
    
    if gripper_joint_idx == -1:
        return success
    
    # Get gripper position
    gripper_position = robot.data.joint_pos[:, gripper_joint_idx]
    
    for i in range(env.num_envs):
        # Skip if no task info
        if not hasattr(env, "task_info") or i not in env.task_info:
            continue
            
        target_info = env.task_info[i]
        target_cube_name = target_info["target_cube"]
        placement_position = target_info.get("placement_position", None)
        
        # Skip if no placement position is defined
        if placement_position is None:
            continue
            
        # Get cube position
        cube = env.scene[target_cube_name]
        cube_position = cube.data.root_pos_w[i, :3]
        
        # Calculate distance between cube and target
        distance = torch.norm(cube_position - placement_position, p=2)
        
        # Check if cube is at target and gripper is open
        is_at_target = distance < distance_threshold
        is_gripper_open = gripper_position[i] < gripper_threshold
        
        if is_at_target and is_gripper_open:
            success[i] = True
    
    return success

# Add to terminations.py
def robot_instability(
    env: ManagerBasedRLEnv,
    velocity_threshold: float = 5.0,
    torque_threshold: float = 50.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate episodes where the robot exhibits unstable behavior.
    
    Args:
        env: The RL environment instance
        velocity_threshold: Maximum joint velocity magnitude
        torque_threshold: Maximum joint torque magnitude
        asset_cfg: Configuration for the robot asset
        
    Returns:
        torch.Tensor: Boolean tensor indicating instability
    """
    # Get robot
    robot = env.scene[asset_cfg.name]
    
    # Check joint velocities
    joint_velocities = torch.norm(robot.data.joint_vel, p=2, dim=-1)
    velocity_unstable = joint_velocities > velocity_threshold
    
    # Check joint torques if available
    has_torques = hasattr(robot.data, 'joint_effort') and robot.data.joint_effort is not None
    if has_torques:
        joint_torques = torch.norm(robot.data.joint_effort, p=2, dim=-1)
        torque_unstable = joint_torques > torque_threshold
        
        # Terminate if either velocity or torque is unstable
        unstable = torch.logical_or(velocity_unstable, torque_unstable)
    else:
        # Just use velocity if torques not available
        unstable = velocity_unstable
    
    return unstable


# NEW TERMINATION FUNCTIONS FOR END-EFFECTOR HEIGHT CONSTRAINTS

def ee_frame_height_below_minimum(
    env: ManagerBasedRLEnv,
    minimum_height: float = -0.1,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Determine if the end-effector frame has gone below a minimum height threshold.
    
    This termination condition helps prevent the robot from hitting the table or 
    going below safe operating limits.
    
    Args:
        env: The RL environment instance
        minimum_height: Minimum z-coordinate threshold (relative to world frame)
        ee_frame_cfg: Configuration for the end-effector frame
        
    Returns:
        torch.Tensor: Boolean tensor indicating if end-effector is below minimum height
    """
    # Get end-effector position from the frame transformer
    ee_frame = env.scene[ee_frame_cfg.name]
    ee_position = ee_frame.data.target_pos_w[..., 0, :]  # Shape: (num_envs, 3)
    
    # Get z-coordinate (height) of end-effector
    ee_height = ee_position[:, 2]
    
    # Check if height is below minimum threshold
    below_minimum = ee_height < minimum_height
    
    return below_minimum


def ee_frame_table_collision(
    env: ManagerBasedRLEnv,
    table_height: float = TABLE_HEIGHT,
    safety_margin: float = 0.05,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Determine if the end-effector frame is too close to or below the table surface.
    
    Args:
        env: The RL environment instance
        table_height: Height of the table surface
        safety_margin: Safety margin above table surface
        ee_frame_cfg: Configuration for the end-effector frame
        
    Returns:
        torch.Tensor: Boolean tensor indicating if end-effector is too close to table
    """
    # Get end-effector position from the frame transformer
    ee_frame = env.scene[ee_frame_cfg.name]
    ee_position = ee_frame.data.target_pos_w[..., 0, :]
    
    # Get z-coordinate (height) of end-effector
    ee_height = ee_position[:, 2]
    
    # Check if height is below table + safety margin
    too_close_to_table = ee_height < (table_height + safety_margin)
    
    return too_close_to_table
