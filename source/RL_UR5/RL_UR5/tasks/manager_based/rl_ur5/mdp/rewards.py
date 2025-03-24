# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import math
from typing import TYPE_CHECKING
import isaaclab.utils.math as math_utils


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# tasks/mdp.py

import numpy as np
import random
from observations import *

from isaaclab.managers import SceneEntityCfg


# @configclass
# class JointPositionActionCfg:
#     """Configuration for joint position control."""
    
#     # Type annotations with explicit MISSING markers to indicate required fields
#     asset_name: str = dataclasses.MISSING
#     joint_names: List[str] = dataclasses.MISSING
#     scale: List[float] = dataclasses.MISSING






"""
Reward functions
"""

def distance_to_target_cube(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward based on distance between end-effector and target cube.
    
    Args:
        env: The RL environment instance
        asset_cfg: Configuration for the robot asset
        
    Returns:
        torch.Tensor: Negative distance between end-effector and target cube
    """
    # Get end-effector pose
    ee_pose = end_effector_pose(env, asset_cfg)
    ee_position = ee_pose[:, :3]
    
    # Get target cube positions for each environment
    cube_positions_tensor = torch.zeros((env.num_envs, 3), device=env.device)
    for i in range(env.num_envs):
        if hasattr(env, "task_info") and i in env.task_info:
            target_info = env.task_info[i]
            target_cube_name = target_info["target_cube"]
            cube = env.scene[target_cube_name]
            cube_positions_tensor[i] = cube.data.root_pos_w[i, :3]
    
    # Calculate distance
    distance = torch.norm(ee_position - cube_positions_tensor, p=2, dim=-1)
    
    return -distance  # Negative because smaller distance is better


def approach_reward(
    env: ManagerBasedRLEnv,
    distance_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_link"),
) -> torch.Tensor:
    """Reward for being close to the target cube.
    
    Args:
        env: The RL environment instance
        distance_threshold: Maximum distance to consider "close"
        asset_cfg: Configuration for the robot asset
        ee_frame_cfg: Configuration for the end-effector frame
        
    Returns:
        torch.Tensor: Binary reward (1.0 if close, 0.0 otherwise)
    """
    # Get end-effector position
    ee_frame = env.scene[ee_frame_cfg.name]
    ee_position = ee_frame.data.target_pos_w[..., 0, :]
    
    # Get target cube positions for each environment
    cube_positions_tensor = torch.zeros((env.num_envs, 3), device=env.device)
    for i in range(env.num_envs):
        if hasattr(env, "task_info") and i in env.task_info:
            target_info = env.task_info[i]
            target_cube_name = target_info["target_cube"]
            cube = env.scene[target_cube_name]
            cube_positions_tensor[i] = cube.data.root_pos_w[i, :3]
    
    # Calculate distance
    distance = torch.norm(ee_position - cube_positions_tensor, p=2, dim=-1)
    
    # Calculate reward (1.0 if close, 0.0 otherwise)
    reward = torch.where(distance < distance_threshold, 
                        torch.ones_like(distance), 
                        torch.zeros_like(distance))
    
    return reward


def grasp_reward(
    env: ManagerBasedRLEnv,
    gripper_threshold: float = 0.4,
    distance_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_link"),
) -> torch.Tensor:
    """Reward for grasping the cube.
    
    Args:
        env: The RL environment instance
        gripper_threshold: Minimum gripper position to consider "closed"
        distance_threshold: Maximum distance to consider "close"
        asset_cfg: Configuration for the robot asset
        ee_frame_cfg: Configuration for the end-effector frame
        
    Returns:
        torch.Tensor: Binary reward (2.0 if grasping, 0.0 otherwise)
    """
    # Get end-effector position
    ee_frame = env.scene[ee_frame_cfg.name]
    ee_position = ee_frame.data.target_pos_w[..., 0, :]
    
    # Get target cube positions for each environment
    cube_positions_tensor = torch.zeros((env.num_envs, 3), device=env.device)
    for i in range(env.num_envs):
        if hasattr(env, "task_info") and i in env.task_info:
            target_info = env.task_info[i]
            target_cube_name = target_info["target_cube"]
            cube = env.scene[target_cube_name]
            cube_positions_tensor[i] = cube.data.root_pos_w[i, :3]
    
    # Calculate distance
    distance = torch.norm(ee_position - cube_positions_tensor, p=2, dim=-1)
    is_close = distance < distance_threshold
    
    # Get gripper position
    robot = env.scene[asset_cfg.name]
    
    # Find joint index by name
    joint_names = robot.joint_names
    gripper_joint_name = "robotiq_85_left_knuckle_joint"
    gripper_joint_idx = joint_names.index(gripper_joint_name) if gripper_joint_name in joint_names else -1
    
    if gripper_joint_idx == -1:
        raise ValueError(f"Gripper joint '{gripper_joint_name}' not found in joint names")
    
    gripper_position = robot.data.joint_pos[:, gripper_joint_idx]
    
    # Check if gripper is closed enough
    is_closed = gripper_position > gripper_threshold
    
    # Calculate reward (2.0 if close and gripper closed, 0.0 otherwise)
    reward = torch.where(torch.logical_and(is_close, is_closed), 
                        2.0 * torch.ones_like(distance), 
                        torch.zeros_like(distance))
    
    return reward


def placement_reward(
    env: ManagerBasedRLEnv,
    distance_threshold: float = 0.05,
    gripper_threshold: float = 0.4,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for placing the cube at the target position.
    
    Args:
        env: The RL environment instance
        distance_threshold: Maximum distance to consider "at target"
        gripper_threshold: Minimum gripper position to consider "grasping"
        asset_cfg: Configuration for the robot asset
        
    Returns:
        torch.Tensor: Reward for placing the cube at the target (5.0 if successful)
    """
    rewards = torch.zeros(env.num_envs, device=env.device)
    
    for i in range(env.num_envs):
        if hasattr(env, "task_info") and i in env.task_info:
            target_info = env.task_info[i]
            target_cube_name = target_info["target_cube"]
            # Use placement_position from task_info
            target_position = target_info.get("placement_position", None)
            
            # Skip if no placement position is defined
            if target_position is None:
                continue
            
            # Get cube position
            cube = env.scene[target_cube_name]
            cube_position = cube.data.root_pos_w[i, :3]
            
            # Calculate distance between cube and target
            distance = torch.norm(cube_position - target_position, p=2)
            
            # Check if cube is at target
            is_at_target = distance < distance_threshold
            
            # Get gripper position
            robot = env.scene[asset_cfg.name]
            
            # Find joint index by name
            joint_names = robot.joint_names
            gripper_joint_name = "robotiq_85_left_knuckle_joint"
            gripper_joint_idx = joint_names.index(gripper_joint_name) if gripper_joint_name in joint_names else -1
            
            if gripper_joint_idx == -1:
                raise ValueError(f"Gripper joint '{gripper_joint_name}' not found in joint names")
            
            gripper_position = robot.data.joint_pos[i, gripper_joint_idx]
            is_grasping = gripper_position > gripper_threshold
            
            # Calculate reward (5.0 if cube at target and grasping, 0.0 otherwise)
            if is_at_target and is_grasping:
                rewards[i] = 5.0
    
    return rewards


def success_reward(
    env: ManagerBasedRLEnv,
    gripper_threshold: float = 0.2,
    distance_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for successfully completing the task.
    
    Args:
        env: The RL environment instance
        gripper_threshold: Maximum gripper position to consider "open"
        distance_threshold: Maximum distance to consider "at target"
        asset_cfg: Configuration for the robot asset
        
    Returns:
        torch.Tensor: Reward for successfully completing the task (10.0 if successful)
    """
    rewards = torch.zeros(env.num_envs, device=env.device)
    
    for i in range(env.num_envs):
        if hasattr(env, "task_info") and i in env.task_info:
            target_info = env.task_info[i]
            target_cube_name = target_info["target_cube"]
            # Use placement_position from task_info
            target_position = target_info.get("placement_position", None)
            
            # Skip if no placement position is defined
            if target_position is None:
                continue
            
            # Get cube position
            cube = env.scene[target_cube_name]
            cube_position = cube.data.root_pos_w[i, :3]
            
            # Calculate distance between cube and target
            distance = torch.norm(cube_position - target_position, p=2)
            
            # Check if cube is at target
            is_at_target = distance < distance_threshold
            
            # Get gripper position
            robot = env.scene[asset_cfg.name]
            
            # Find joint index by name
            joint_names = robot.joint_names
            gripper_joint_name = "robotiq_85_left_knuckle_joint"
            gripper_joint_idx = joint_names.index(gripper_joint_name) if gripper_joint_name in joint_names else -1
            
            if gripper_joint_idx == -1:
                raise ValueError(f"Gripper joint '{gripper_joint_name}' not found in joint names")
            
            gripper_position = robot.data.joint_pos[i, gripper_joint_idx]
            is_open = gripper_position < gripper_threshold
            
            # Calculate reward (10.0 if cube at target and gripper open, 0.0 otherwise)
            if is_at_target and is_open:
                rewards[i] = 10.0
    
    return rewards


def movement_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for excessive movement.
    
    Args:
        env: The RL environment instance
        asset_cfg: Configuration for the robot asset
        
    Returns:
        torch.Tensor: Penalty proportional to joint velocities
    """
    # Get joint velocities
    robot = env.scene[asset_cfg.name]
    joint_vels = robot.data.joint_vel
    
    # Calculate penalty (sum of absolute velocities)
    penalty = -0.01 * torch.sum(torch.abs(joint_vels), dim=-1)
    
    return penalty


