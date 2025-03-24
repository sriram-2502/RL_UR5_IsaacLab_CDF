
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# tasks/mdp.py

import numpy as np
import random
from observations import *

from isaaclab.managers import SceneEntityCfg




def task_success(
    env: ManagerBasedRLEnv,
    gripper_threshold: float = 0.2,
    distance_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Determine if the task has been successfully completed.
    
    Args:
        env: The RL environment instance
        gripper_threshold: Maximum gripper position to consider "open"
        distance_threshold: Maximum distance to consider "at target"
        asset_cfg: Configuration for the robot asset
        
    Returns:
        torch.Tensor: Boolean tensor indicating task success
    """
    terminations = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    
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
            
            # Task is successful if cube is at target and gripper is open
            if is_at_target and is_open:
                terminations[i] = True
    
    return terminations