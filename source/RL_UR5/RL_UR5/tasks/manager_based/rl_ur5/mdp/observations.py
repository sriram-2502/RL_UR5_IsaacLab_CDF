from __future__ import annotations

import torch
import math
from typing import TYPE_CHECKING
import isaaclab.utils.math as math_utils
import random



from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedEnv


"""
Observation functions
"""

def joint_positions(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Get joint positions for the robot."""
    # Extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos

def joint_velocities(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Get joint velocities for the robot."""
    # Extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel

def ee_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Get end-effector position.
    
    Args:
        env: The RL environment instance
        asset_cfg: Configuration for the robot asset
        
    Returns:
        torch.Tensor: End-effector position [x, y, z]
    """
    # Extract the robot asset
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get end-effector body index
    ee_link_name = "ee_link"  # Using ee_link as the end-effector
    
    # Find the body index by name (compatible with IsaacLab 4.5)
    body_names = asset.body_names
    body_idx = body_names.index(ee_link_name) if ee_link_name in body_names else -1
    
    if body_idx == -1:
        raise ValueError(f"End-effector link '{ee_link_name}' not found in articulation body names")
    
    # Get body position
    ee_pos = asset.data.body_pos_w[:, body_idx, :]
    
    return ee_pos


def ee_quat(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Get end-effector orientation quaternion.
    
    Args:
        env: The RL environment instance
        asset_cfg: Configuration for the robot asset
        
    Returns:
        torch.Tensor: End-effector quaternion [qx, qy, qz, qw]
    """
    # Extract the robot asset
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get end-effector body index
    ee_link_name = "ee_link"  # Using ee_link as the end-effector
    
    # Find the body index by name (compatible with IsaacLab 4.5)
    body_names = asset.body_names
    body_idx = body_names.index(ee_link_name) if ee_link_name in body_names else -1
    
    if body_idx == -1:
        raise ValueError(f"End-effector link '{ee_link_name}' not found in articulation body names")
    
    # Get body orientation
    ee_quat = asset.data.body_quat_w[:, body_idx, :]
    
    return ee_quat


def end_effector_pose(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Get combined end-effector pose (position and orientation).
    
    Args:
        env: The RL environment instance
        asset_cfg: Configuration for the robot asset
        
    Returns:
        torch.Tensor: End-effector pose [x, y, z, qx, qy, qz, qw]
    """
    position = ee_pos(env, asset_cfg)
    quaternion = ee_quat(env, asset_cfg)
    
    # Combine into a single pose vector [x, y, z, qx, qy, qz, qw]
    pose = torch.cat([position, quaternion], dim=-1)
    
    return pose

"""
Updated functions for IsaacLab 4.5 compatibility
"""

def cube_positions(
    env: ManagerBasedRLEnv, 
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("red_cube"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("green_cube"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("blue_cube"),
) -> torch.Tensor:
    """Get positions of the cubes relative to environment origins.
    
    Args:
        env: The RL environment instance
        cube_1_cfg: Configuration for the first cube
        cube_2_cfg: Configuration for the second cube
        cube_3_cfg: Configuration for the third cube
        
    Returns:
        torch.Tensor: Concatenated positions of all cubes, shape (batch_size, 9)
    """
    # Get cubes from the scene
    cube_1 = env.scene[cube_1_cfg.name]
    cube_2 = env.scene[cube_2_cfg.name]
    cube_3 = env.scene[cube_3_cfg.name]
    
    # Get positions (world frame)
    cube_1_pos_w = cube_1.data.root_pos_w
    cube_2_pos_w = cube_2.data.root_pos_w
    cube_3_pos_w = cube_3.data.root_pos_w
    
    # Adjust positions relative to environment origins
    cube_1_pos = cube_1_pos_w - env.scene.env_origins[:, 0:3]
    cube_2_pos = cube_2_pos_w - env.scene.env_origins[:, 0:3]
    cube_3_pos = cube_3_pos_w - env.scene.env_origins[:, 0:3]
    
    # Concatenate all positions
    all_positions = torch.cat((cube_1_pos, cube_2_pos, cube_3_pos), dim=1)
    
    return all_positions


def target_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get the target position for placing the cube.
    
    Args:
        env: The RL environment instance
        
    Returns:
        torch.Tensor: Target positions for all environments
    """
    # Initialize tensor to hold target positions for all environments
    target_positions = torch.zeros((env.num_envs, 3), device=env.device)
    
    # Fill in target positions from task_info for each environment
    for i in range(env.num_envs):
        if hasattr(env, "task_info") and i in env.task_info:
            target_positions[i] = env.task_info[i]["placement_position"]
    
    return target_positions


def task_id(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get the one-hot encoded task ID (which cube to pick).
    
    Args:
        env: The RL environment instance
        
    Returns:
        torch.Tensor: One-hot encoded task IDs for all environments
    """
    # Initialize tensor for task IDs
    batch_size = env.num_envs
    num_cubes = 3  # Assuming 3 cubes (red, green, blue)
    task_ids = torch.zeros((batch_size, num_cubes), device=env.device)
    
    # Fill in one-hot encoded task IDs
    for i in range(batch_size):
        if hasattr(env, "task_info") and i in env.task_info:
            cube_idx = env.task_info[i]["target_cube_idx"]
            task_ids[i, cube_idx] = 1.0
    
    return task_ids



def camera_images(env: ManagerBasedRLEnv, camera_cfg: SceneEntityCfg = SceneEntityCfg("camera")):
    """Get RGB images from the camera."""
    camera = env.scene.sensors[camera_cfg.name]
    
    # Get RGB data and normalize it
    rgb_data = camera.data.output["rgb"] / 255.0
    
    return rgb_data


"""
Reset functions
"""

def reset_robot_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose: list[float] = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot to a specific joint pose.
    
    Args:
        env: The environment instance
        env_ids: Tensor of environment IDs to reset
        pose: List of joint positions to set for the robot (default home pose if None)
        asset_cfg: Configuration for the robot
    """
    # Get the robot from the scene
    robot = env.scene[asset_cfg.name]
    
    # Use default pose if none provided
    if pose is None:
        pose = [0.0, -1.57, 0.0, -1.57, 0.0, 0.0, 0.0]  # Default home pose
    
    # Convert pose to tensor
    joint_pos = torch.tensor(pose, device=env.device).repeat(len(env_ids), 1)
    joint_vel = torch.zeros_like(joint_pos)
    
    # Set into the physics simulation
    robot.set_joint_position_target(joint_pos, env_ids=env_ids)
    robot.set_joint_velocity_target(joint_vel, env_ids=env_ids)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    
    return {}


def sample_object_poses(
    num_objects: int,
    min_separation: float = 0.0,
    pose_range: dict[str, tuple[float, float]] = {},
    max_sample_tries: int = 5000,
):
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    pose_list = []

    for i in range(num_objects):
        for j in range(max_sample_tries):
            sample = [random.uniform(range[0], range[1]) for range in range_list]

            # Accept pose if it is the first one, or if reached max num tries
            if len(pose_list) == 0 or j == max_sample_tries - 1:
                pose_list.append(sample)
                break

            # Check if pose of object is sufficiently far away from all other objects
            separation_check = [math.dist(sample[:3], pose[:3]) > min_separation for pose in pose_list]
            if False not in separation_check:
                pose_list.append(sample)
                break

    return pose_list


def reset_cube_positions(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    cube_cfgs: list[SceneEntityCfg] = None,
    min_distance: float = 0.1,
):
    """Randomize cube positions on the table within the pick position bounds.
    
    Args:
        env: The environment instance
        env_ids: Tensor of environment IDs to reset
        table_cfg: Configuration for the table
        cube_cfgs: List of configurations for the cubes
        min_distance: Minimum distance between cubes in meters
    """
    # Set default cube configurations if not provided
    if cube_cfgs is None:
        cube_cfgs = [
            SceneEntityCfg("red_cube"),
            SceneEntityCfg("green_cube"),
            SceneEntityCfg("blue_cube")
        ]
    
    # Define pick bounds for cube placement
    pick_min_bounds = torch.tensor([0.37, -0.560, 0.77], device=env.device)
    pick_max_bounds = torch.tensor([0.87, -0.23, 0.77], device=env.device)
    
    # Calculate pose range dictionary for the sample_object_poses function
    pose_range = {
        "x": (pick_min_bounds[0].item(), pick_max_bounds[0].item()),
        "y": (pick_min_bounds[1].item(), pick_max_bounds[1].item()),
        "z": (pick_min_bounds[2].item(), pick_max_bounds[2].item()),
        "roll": (0.0, 0.0),  # No rotation for now
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    
    # Randomize poses for each environment
    for cur_env in env_ids.tolist():
        # Sample poses with minimum separation
        pose_list = sample_object_poses(
            num_objects=len(cube_cfgs),
            min_separation=min_distance,
            pose_range=pose_range,
            max_sample_tries=100
        )
        
        # Apply poses to each cube
        for i, cube_cfg in enumerate(cube_cfgs):
            cube = env.scene[cube_cfg.name]
            
            # Convert pose to tensors
            pose_tensor = torch.tensor([pose_list[i]], device=env.device)
            positions = pose_tensor[:, 0:3] + env.scene.env_origins[cur_env, 0:3]
            orientations = math_utils.quat_from_euler_xyz(
                pose_tensor[:, 3], pose_tensor[:, 4], pose_tensor[:, 5]
            )
            
            # Write pose to simulation
            cube.write_root_pose_to_sim(
                torch.cat([positions,orientations], dim=-1),
                env_ids=torch.tensor([cur_env], device=env.device)
            )
            
            # Reset velocities
            cube.write_root_velocity_to_sim(
                torch.zeros(1, 6, device=env.device),  # [lin_vel, ang_vel]
                env_ids=torch.tensor([cur_env], device=env.device)
            )
    
    
    

def set_pick_and_place_task(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    cube_cfgs: list[SceneEntityCfg] = None,
):
    """Set up pick and place tasks for the environment.
    
    Args:
        env: The environment instance
        env_ids: Tensor of environment IDs to set tasks for
        command_cfg: Configuration for commands (target positions)
        cube_cfgs: List of configurations for the cubes
    """
    # Set default cube configurations if not provided
    if cube_cfgs is None:
        cube_cfgs = [
            SceneEntityCfg("red_cube"),
            SceneEntityCfg("green_cube"),
            SceneEntityCfg("blue_cube")
        ]
    
    # Initialize task info dictionary if it doesn't exist
    if not hasattr(env, "task_info"):
        env.task_info = {}
    
    # Get target positions from command_cfg if available
    target_positions = None
    if hasattr(env, "command_data") and "object_pose" in env.command_data:
        target_positions = env.command_data["object_pose"]
    
    # For each environment, assign a random task
    for cur_env in env_ids.tolist():
        # Randomly select which cube to pick
        target_cube_idx = random.randint(0, len(cube_cfgs) - 1)
        target_cube_name = cube_cfgs[target_cube_idx].name
        
        # Get placement position from target positions if available
        if target_positions is not None:
            placement_position = target_positions[cur_env, :3]
        else:
            # Sample a placement position as fallback
            pose_range = {
                "x": (0.37, 0.87),
                "y": (0.23, 0.56),
                "z": (0.85, 0.77),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            }
            placement_pose = sample_object_poses(
                num_objects=1,
                pose_range=pose_range
            )[0]
            placement_position = torch.tensor(
                placement_pose[:3], 
                device=env.device
            )
        
        # Store task information
        env.task_info[cur_env] = {
            "target_cube": target_cube_name,
            "target_cube_idx": target_cube_idx,
            "placement_position": placement_position,
        }
    
    return {}