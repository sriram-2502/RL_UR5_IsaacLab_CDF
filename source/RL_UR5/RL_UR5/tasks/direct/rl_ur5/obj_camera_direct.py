"""
Direct RL Environment for Object Camera Pose Tracking with UR5 Robot

This implements a multi-observation space environment compatible with skrl.
"""

from __future__ import annotations

import torch
import numpy as np
import math
import random
from typing import Dict, Any, Tuple, Optional, Sequence
import gymnasium as gym

# IsaacLab imports
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg, FrameTransformer, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
import isaaclab.utils.math as math_utils
from isaaclab.utils.math import sample_uniform
# Robot configuration
from isaaclab_assets.robots.ur5 import UR5_GRIPPER_CFG

# Custom utilities - with fallback
try:
    from .thresholds import *
except ImportError:
    # Define minimal thresholds if file not found
    TABLE_HEIGHT = 0.77
    CUBE_HEIGHT = 0.0382
    CUBE_WIDTH = 0.0286
    CUBE_LENGTH = 0.0635
    CUBE_START_HEIGHT = TABLE_HEIGHT + (CUBE_HEIGHT / 2)
    PLACEMENT_POS_THRESHOLD = 0.05
    GRIPPER_OPEN_THRESHOLD = 5.0
    GRIPPER_CLOSED_THRESHOLD = 25.0
    GRIPPER_CLOSING_THRESHOLD = 15.0
    POSITION_THRESHOLD = 0.05
    ORIENTATION_THRESHOLD = 0.9
    CUBE_HOVER_HEIGHT = 0.3
    PRE_GRASP_HEIGHT = 0.1
    VELOCITY_THRESHOLD = 0.05
    TORQUE_THRESHOLD = 1.0
    CUBE_MAX_HEIGHT = 1.0
    DISTANCE_SCALE = 0.1


##
# Environment Configuration
##

@configclass
class ObjCameraPoseTrackingDirectEnvCfg(DirectRLEnvCfg):
    """Configuration for the direct RL environment."""
    
    # Visualization settings - MOVED TO TOP to fix reference issue
    debug_vis = True  # Enable/disable debug visualization
    
    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg.prim_path = "/Visuals/FrameTransformer"
    


    # UR5 Robot
    robot_cfg: ArticulationCfg = UR5_GRIPPER_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # Table configuration
    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/table",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/adi2440/Desktop/ur5_isaacsim/usd/table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.7, 0.0, 0.0), 
            rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )

    # White plane configuration
    white_plane_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/white_plane",
        spawn=sim_utils.CuboidCfg(
            size=(0.42112, 2.81, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 1.0, 1.0),
                metallic=0.0,
                roughness=0.1,
                opacity=1.0
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.2, 0.0, 0.85925),
            rot=(0.70711, 0.0, 0.70711, 0.0)
        ),
    )

    # Frame transformer for end-effector
    ee_frame_cfg: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        debug_vis=debug_vis,  # Now this works since enable_debug_vis is defined above
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/ee_link",
                name="end_effector",
                offset=OffsetCfg(
                    pos=[0.12, 0.0, 0.0],
                ),
            ),
        ],
    )
    


    # Red cube - dynamic obstacle
    red_cube_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/red_cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.0572, 0.0635, 0.191),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=100.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),
                metallic=0.0,
                roughness=0.5
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.97),
        ),
    )
    
    # Camera
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.208,
            focus_distance=28.0,
            horizontal_aperture=5.76,
            vertical_aperture=3.24,
            clipping_range=(0.1, 1000.0)
        ),
        width=224,
        height=224,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.17, 0.06, 1.143),
            rot=(0.62933, 0.32239, 0.32239, 0.62933),
            convention="opengl"
        )
    )

    # Basic environment settings
    episode_length_s = 8.0
    decimation = 4
    action_scale = 0.5  # Reduced for smoother movements
    state_dim = 19
        
    # Observation and action spaces
    action_space = gym.spaces.Box(low=-20.0, high=20.0, shape=(6,))
    state_space = 0
    observation_space = gym.spaces.Dict({
        "image": gym.spaces.Box(low=float("-inf"), high=float("inf"), shape=(80, 100, 3)),
        "state": gym.spaces.Box(low=float("-inf"), high=float("inf"), shape=(state_dim,)),
    })
    
    # Simulation settings
    sim: SimulationCfg = SimulationCfg(
        dt=1.0/120.0,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    
    # Scene settings
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8, 
        env_spacing=4.0,
        replicate_physics=True,
    )
    
    # Viewer settings
    viewer = ViewerCfg(eye=(7.5, 7.5, 7.5), origin_type="world", env_index=0)
    
    # Action filter settings
    action_filter_order = 2
    action_filter_cutoff_freq = 5.0
    action_filter_damping_ratio = 0.707
    
    # Command/target pose settings
    target_pose_range = {
        "x": (0.3, 0.7),
        "y": (-0.4, 0.4),
        "z": (-0.2, 0.2),  # Above table (height 0.77)
        "roll": (0.0, 0.0),
        "pitch": (1.57, 1.57),
        "yaw": (0.0, 0.0),
    }
    command_resampling_time = 8.0
    
    # Obstacle movement settings
    obstacle_movement_type = "random_smooth"
    obstacle_center_pos = [0.5, 0.0, 0.97]
    obstacle_radius = 0.75
    obstacle_speed = 0.8
    obstacle_height_variation = 0.0
    obstacle_diagonal_bounds = {
        "x_min": 0.3,
        "x_max": 0.6,
        "y_min": -0.35,
        "y_max": 0.35,
    }
    obstacle_reset_bounds = {
        "x_min": 0.2,
        "x_max": 0.6,
        "y_min": -0.3,
        "y_max": 0.3,
        "z": 0.97,
    }
    
    # Reward settings
    reward_distance_weight = -2.0
    reward_distance_tanh_weight = 1.0
    reward_distance_tanh_std = 0.2
    reward_orientation_weight = -1.0
    reward_torque_weight = -0.0001
    reward_table_collision_weight = -10.0
    reward_obstacle_avoidance_weight = 2.0
    reward_obstacle_smooth_weight = 3.0
    reward_action_penalty_weight = -0.01
    
    # Termination settings
    position_threshold = 0.05
    orientation_threshold = 0.1
    velocity_threshold = 0.05
    torque_threshold = 1.0
    
    # Camera preprocessing settings
    camera_crop_top = 50
    camera_crop_bottom = 70
    camera_target_height = 80
    camera_target_width = 100
    
    # Noise settings
    joint_pos_noise_min = -0.01
    joint_pos_noise_max = 0.01
    joint_vel_noise_min = -0.001
    joint_vel_noise_max = 0.001
    
    # Reset settings
    robot_base_pose = [-0.568, -0.658,  1.602, -2.585, -1.6060665,  -1.64142667]
    robot_reset_noise_range = 0.01


class ObjCameraPoseTrackingDirectEnv(DirectRLEnv):
    """Direct RL environment for object camera pose tracking with multi-observation space."""
    
    cfg: ObjCameraPoseTrackingDirectEnvCfg
    
    def __init__(self, cfg: ObjCameraPoseTrackingDirectEnvCfg, render_mode: str | None = None, **kwargs):
        # Store config
        self.cfg = cfg
        
        # Initialize parent
        super().__init__(cfg, render_mode, **kwargs)
        
        # Initialize extras dictionary for logging
        self.extras = {"log": {}}
        
        # Joint names and indices
        self._joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        self._joint_indices, _ = self._robot.find_joints(self._joint_names)
        
        # Get DOF limits
        self._robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, self._joint_indices, 0].to(self.device)
        self._robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, self._joint_indices, 1].to(self.device)
        
        # Initialize buffers
        self._robot_dof_targets = torch.zeros(
            (self.num_envs, len(self._joint_indices)), device=self.device
        )
        self._target_poses = torch.zeros((self.num_envs, 7), device=self.device)
        self._command_time_left = torch.zeros(self.num_envs, device=self.device)
        self._obstacle_time = torch.zeros(self.num_envs, device=self.device)
        
        # Initialize action filter
        self._setup_action_filter()
        
        # Performance tracking
        self._episode_sums = {
            "position_error": torch.zeros(self.num_envs, device=self.device),
            "total_reward": torch.zeros(self.num_envs, device=self.device),
            "success_count": torch.zeros(self.num_envs, device=self.device),
        }
        
        # Log initial information
        print(f"[INFO] Environment initialized with {self.num_envs} environments")
        print(f"[INFO] Action scale: {self.cfg.action_scale}")
        print(f"[INFO] Target pose range X: {self.cfg.target_pose_range['x']}")
        print(f"[INFO] Target pose range Y: {self.cfg.target_pose_range['y']}")
        print(f"[INFO] Target pose range Z: {self.cfg.target_pose_range['z']}")
        
        # Setup debug visualization if enabled
        self.set_debug_vis(self.cfg.debug_vis)


    def close(self):
        """Cleanup for the environment."""
        super().close()
        
    def _setup_scene(self):
        """Set up the scene with robots, table, obstacles, cameras, etc."""
        # --- spawn all prims in the source environment only ---
        self._robot = Articulation(self.cfg.robot_cfg)
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        self._ee_frame = FrameTransformer(self.cfg.ee_frame_cfg)
        self._red_cube = RigidObject(self.cfg.red_cube_cfg)
        
        # Create static assets
        self._table = RigidObject(self.cfg.table_cfg)
        self._white_plane = RigidObject(self.cfg.white_plane_cfg)

        # --- clone source → env_1…env_N (env_0 keeps its prims) ---
        self.scene.clone_environments(copy_from_source=False)

        # --- register handles in IsaacLab's scene registry ---
        self.scene.articulations["robot"] = self._robot
        self.scene.sensors["tiled_camera"] = self._tiled_camera
        self.scene.sensors["ee_frame"] = self._ee_frame
        self.scene.rigid_objects["red_cube"] = self._red_cube
        
        # Add static assets to scene registry
        self.scene.rigid_objects["table"] = self._table
        self.scene.rigid_objects["white_plane"] = self._white_plane

        # --- add static geometry and lighting ---
        # Ground plane
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/ground", ground_cfg)

        # Multiple lights for better scene illumination
        # Main dome light
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9))
        light_cfg.func("/World/DomeLight", light_cfg)
        
        # Additional directional light for shadows
        dir_light_cfg = sim_utils.DistantLightCfg(intensity=1000.0, color=(1.0, 1.0, 0.9), angle=0.53)
        dir_light_cfg.func("/World/DirectionalLight", dir_light_cfg)

        
    def _setup_action_filter(self):
        """Initialize action filter states and coefficients."""
        num_joints = len(self._joint_indices)
        self._action_filter_x1 = torch.zeros((self.num_envs, num_joints), device=self.device)
        self._action_filter_x2 = torch.zeros((self.num_envs, num_joints), device=self.device)
        self._action_filter_y1 = torch.zeros((self.num_envs, num_joints), device=self.device)
        self._action_filter_y2 = torch.zeros((self.num_envs, num_joints), device=self.device)
        
        # Calculate filter coefficients
        if self.cfg.action_filter_order == 2:
            omega = 2.0 * math.pi * self.cfg.action_filter_cutoff_freq
            dt = self.cfg.sim.dt
            k = omega * dt
            a1 = k * k
            a2 = k * 2.0 * self.cfg.action_filter_damping_ratio
            a3 = a1 + a2 + 1.0
            
            self._filter_b0 = a1 / a3
            self._filter_b1 = 2.0 * a1 / a3
            self._filter_b2 = a1 / a3
            self._filter_a1 = (2.0 * a1 - 2.0) / a3
            self._filter_a2 = (a1 - a2 + 1.0) / a3
            
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Apply actions before physics step."""
        # Store raw actions
        self.actions = actions.clone().clamp(-20.0, 20.0)
        self._previous_actions = self.actions.clone()
        
        # # Apply action filtering
        filtered_actions = self._apply_action_filter(self.actions)
        
        # # Scale actions
        self.actions = filtered_actions * self.cfg.action_scale
        
        # Update command timer
        self._command_time_left -= self.physics_dt
        
        # --- resample target poses when timer runs out ---
        expired_mask = self._command_time_left <= 0.0
        if torch.any(expired_mask):
            expired_ids = torch.nonzero(expired_mask, as_tuple=False).squeeze(-1)
            env_ids = expired_ids.cpu().tolist()
            self._sample_commands(env_ids)
            # reset their countdown
            self._command_time_left[expired_mask] = self.cfg.command_resampling_time
        
        # Update obstacle
        self._update_obstacle_position()
        
    def _apply_action(self) -> None:
        """Apply the processed actions to the robot."""
        # Get current joint positions
        # current_joint_pos = self._robot.data.joint_pos[:, self._joint_indices]
        
        # Add actions to current positions 
        # self._robot_dof_targets =  current_joint_pos + self.actions
        self._robot_dof_targets =  self.actions[:, self._joint_indices]*self.cfg.action_scale

        # # Clamp to joint limits
        # self._robot_dof_targets = torch.clamp(
        #     self._robot_dof_targets,
        #     self._robot_dof_lower_limits,
        #     self._robot_dof_upper_limits
        # )
        
        # Set joint position targets
        self._robot.set_joint_position_target(
            self._robot_dof_targets, joint_ids=self._joint_indices
        )
        
    def _apply_action_filter(self, actions: torch.Tensor) -> torch.Tensor:
        """Apply second-order Butterworth filter to actions."""
        if self.cfg.action_filter_order == 2:
            filtered_actions = (
                self._filter_b0 * actions +
                self._filter_b1 * self._action_filter_x1 +
                self._filter_b2 * self._action_filter_x2 -
                self._filter_a1 * self._action_filter_y1 -
                self._filter_a2 * self._action_filter_y2
            )
            
            # Update filter memory
            self._action_filter_x2 = self._action_filter_x1.clone()
            self._action_filter_x1 = actions.clone()
            self._action_filter_y2 = self._action_filter_y1.clone()
            self._action_filter_y1 = filtered_actions.clone()
            
            return filtered_actions
        else:
            return actions

    def _sample_commands(self, env_ids: Sequence[int]) -> None:
        """Randomize the target poses for the given env indices."""
        num = len(env_ids)
        if num == 0:
            return

        # sample positions within configured ranges
        x = sample_uniform(
            self.cfg.target_pose_range["x"][0],
            self.cfg.target_pose_range["x"][1],
            (num,), self.device
        )
        y = sample_uniform(
            self.cfg.target_pose_range["y"][0],
            self.cfg.target_pose_range["y"][1],
            (num,), self.device
        )
        z = sample_uniform(
            self.cfg.target_pose_range["z"][0],
            self.cfg.target_pose_range["z"][1],
            (num,), self.device
        )

        # sample orientations (roll, pitch, yaw)
        roll = sample_uniform(
            self.cfg.target_pose_range["roll"][0],
            self.cfg.target_pose_range["roll"][1],
            (num,), self.device
        )
        pitch = sample_uniform(
            self.cfg.target_pose_range["pitch"][0],
            self.cfg.target_pose_range["pitch"][1],
            (num,), self.device
        )
        yaw = sample_uniform(
            self.cfg.target_pose_range["yaw"][0],
            self.cfg.target_pose_range["yaw"][1],
            (num,), self.device
        )
        quat = math_utils.quat_from_euler_xyz(roll, pitch, yaw)

        # write into the buffer
        self._target_poses[env_ids, :3] = torch.stack([x, y, z], dim=-1)
        self._target_poses[env_ids, 3:7] = quat
        
        # Debug print for first environment
        if 0 in env_ids and len(env_ids) <= 4:  # Only log for small resets
            idx = env_ids.index(0)
            print(f"[DEBUG] New target for env 0: pos=[{x[idx].item():.3f}, {y[idx].item():.3f}, {z[idx].item():.3f}]")

    def _update_obstacle_position(self):
        """Update the dynamic obstacle position."""
        self._obstacle_time += self.physics_dt
        
        # Get current obstacle positions
        obstacle_positions = self._red_cube.data.root_pos_w.clone()
        
        for i in range(self.num_envs):
            time = self._obstacle_time[i].item()
            
            if self.cfg.obstacle_movement_type == "random_smooth":
                # Smooth random movement using multiple sine waves
                x = self.cfg.obstacle_center_pos[0] + 0.1 * (
                    math.sin(self.cfg.obstacle_speed * time) + 
                    0.5 * math.sin(1.7 * self.cfg.obstacle_speed * time)
                )
                y = self.cfg.obstacle_center_pos[1] + 0.1 * (
                    math.cos(0.8 * self.cfg.obstacle_speed * time) + 
                    0.3 * math.cos(2.3 * self.cfg.obstacle_speed * time)
                )
                z = self.cfg.obstacle_center_pos[2] + self.cfg.obstacle_height_variation * math.sin(
                    1.5 * self.cfg.obstacle_speed * time
                )
                
                # Update position in world frame
                obstacle_positions[i, 0] = x + self.scene.env_origins[i, 0]
                obstacle_positions[i, 1] = y + self.scene.env_origins[i, 1]
                obstacle_positions[i, 2] = z + self.scene.env_origins[i, 2]
        
        # Apply new positions
        self._red_cube.write_root_pose_to_sim(
            torch.cat([
                obstacle_positions,
                self._red_cube.data.root_quat_w
            ], dim=-1)
        )
        
        # Reset velocities for smooth motion
        self._red_cube.write_root_velocity_to_sim(
            torch.zeros((self.num_envs, 6), device=self.device)
        )
        
    def _get_observations(self) -> dict:
        """Compute and return observations as a dictionary."""
        # Get state observations
        state_obs = self._get_state_observations()
        
        # Get camera observations
        camera_obs = self._get_camera_observations()
        
        obs = {"image": camera_obs, "state": state_obs}
        observations = {"policy": obs}
        
        return observations

    def _get_state_observations(self) -> torch.Tensor:
        """Get state-based observations."""
        # Get joint positions with noise
        joint_pos = self._robot.data.joint_pos[:, self._joint_indices]
        if self.cfg.joint_pos_noise_max > 0:
            joint_pos_noise = torch.rand_like(joint_pos) * (
                self.cfg.joint_pos_noise_max - self.cfg.joint_pos_noise_min
            ) + self.cfg.joint_pos_noise_min
            joint_pos_noisy = joint_pos + joint_pos_noise
        else:
            joint_pos_noisy = joint_pos
        
        # Get joint velocities with noise
        joint_vel = self._robot.data.joint_vel[:, self._joint_indices]
        if self.cfg.joint_vel_noise_max > 0:
            joint_vel_noise = torch.rand_like(joint_vel) * (
                self.cfg.joint_vel_noise_max - self.cfg.joint_vel_noise_min
            ) + self.cfg.joint_vel_noise_min
            joint_vel_noisy = joint_vel + joint_vel_noise
        else:
            joint_vel_noisy = joint_vel
        
        # Get target pose (already in robot base frame)
        target_pose = self._target_poses
        
        # Concatenate all state observations
        state_obs = torch.cat([
            joint_pos_noisy,      # 6 dims
            joint_vel_noisy,      # 6 dims
            target_pose,          # 7 dims
        ], dim=-1)
        
        return state_obs
    
    def _get_camera_observations(self) -> torch.Tensor:
        """Get and preprocess camera observations."""
        # Get camera data
        camera_data = self._tiled_camera.data.output["rgb"] / 255.0  # Shape: (num_envs, H, W, C)
        
        # Mean subtraction for normalization
        mean_tensor = torch.mean(camera_data, dim=(1, 2), keepdim=True)
        camera_data = camera_data - mean_tensor
        
        # Crop image (top and bottom)
        cropped = camera_data[
            :,
            self.cfg.camera_crop_top:-self.cfg.camera_crop_bottom,
            :,
            :
        ]
        
        # Resize to target size using interpolation
        # Convert to NCHW format for processing
        cropped = cropped.permute(0, 3, 1, 2)  # (N, C, H, W)

        # Resize using torch interpolation
        resized = torch.nn.functional.interpolate(
            cropped,
            size=(self.cfg.camera_target_height, self.cfg.camera_target_width),
            mode='bilinear',
            align_corners=False
        )
        
        return resized
    
    def _get_rewards(self) -> torch.Tensor:
        """Compute and return rewards with detailed logging."""
        # Initialize total rewards
        rewards = torch.zeros(self.num_envs, device=self.device)
        
        # Get end-effector position and orientation
        ee_position = self._ee_frame.data.target_pos_w[..., 0, :]
        ee_quat = self._ee_frame.data.target_quat_w[..., 0, :]
        
        # Transform target pose to world frame
        robot_pos = self._robot.data.root_state_w[:, :3]
        robot_quat = self._robot.data.root_state_w[:, 3:7]
        
        des_pos_b = self._target_poses[:, :3]
        des_quat_b = self._target_poses[:, 3:7]
        
        des_pos_w, _ = math_utils.combine_frame_transforms(
            robot_pos, robot_quat, des_pos_b
        )
        des_quat_w = math_utils.quat_mul(robot_quat, des_quat_b)
        
        # 1. Position tracking reward
        position_error = torch.norm(ee_position - des_pos_w, dim=-1)
        position_reward = self.cfg.reward_distance_weight * position_error
        rewards += position_reward
        
        # 2. Position tracking tanh reward
        position_reward_tanh = 1.0 - torch.tanh(position_error / self.cfg.reward_distance_tanh_std)
        position_reward_tanh_scaled = self.cfg.reward_distance_tanh_weight * position_reward_tanh
        rewards += position_reward_tanh_scaled
        
        # 3. Orientation tracking reward
        orientation_error = math_utils.quat_error_magnitude(ee_quat, des_quat_w)
        orientation_reward = self.cfg.reward_orientation_weight * orientation_error
        rewards += orientation_reward
        
        # # # 4. Joint torque penalty
        # if hasattr(self._robot.data, 'applied_torque') and self._robot.data.applied_torque is not None:
        #     joint_torques = self._robot.data.applied_torque[:, self._joint_indices]
        #     torque_penalty = torch.sum(torch.square(joint_torques), dim=1)
        #     torque_reward = self.cfg.reward_torque_weight * torque_penalty
        #     rewards += torque_reward
        # else:
        #     torque_reward = torch.zeros_like(rewards)
        
        # 5. Table collision penalty
        ee_height = ee_position[:, 2]
        table_height = TABLE_HEIGHT
        safety_margin = 0.05
        
        table_penalty = torch.where(
            ee_height < (table_height + safety_margin),
            torch.ones_like(ee_height) * self.cfg.reward_table_collision_weight,
            torch.zeros_like(ee_height)
        )
        rewards += table_penalty
        
        # 6. Obstacle avoidance rewards
        obstacle_reward = self._compute_obstacle_avoidance_rewards()* self.cfg.reward_obstacle_avoidance_weight
        rewards += obstacle_reward
        
        # # 7. Action penalty (to encourage smooth movements)
        # if hasattr(self, '_last_actions'):
        #     action_penalty = torch.sum(torch.square(self._last_actions), dim=1)
        #     action_reward = self.cfg.reward_action_penalty_weight * action_penalty
        #     rewards += action_reward
        # else:
        #     action_reward = torch.zeros_like(rewards)
        
        # 8. Success bonus rewards
        close_bonus = torch.zeros_like(position_error)
        close_bonus = torch.where(position_error < 0.1, torch.ones_like(position_error) * 0.5, close_bonus)
        close_bonus = torch.where(position_error < 0.05, torch.ones_like(position_error) * 1.0, close_bonus)
        close_bonus = torch.where(position_error < 0.02, torch.ones_like(position_error) * 2.0, close_bonus)
        rewards += close_bonus
        
        # # Update episode tracking
        # self._episode_sums["position_error"] += position_error
        # self._episode_sums["total_reward"] += rewards
        # success_mask = position_error < self.cfg.position_threshold
        # self._episode_sums["success_count"] += success_mask.float()
        
        # # Log individual reward components (less frequently to avoid spam)
        # if self.common_step_counter % 100 == 0:  # Log every 100 steps
        #     self.extras["log"] = {
        #         "position_error": position_error.mean().item(),
        #         "position_reward": position_reward.mean().item(),
        #         "position_reward_tanh": position_reward_tanh_scaled.mean().item(),
        #         "orientation_error": orientation_error.mean().item(),
        #         "orientation_reward": orientation_reward.mean().item(),
        #         # "torque_penalty": torque_reward.mean().item(),
        #         "table_collision_penalty": table_penalty.mean().item(),
        #         "obstacle_avoidance_reward": obstacle_reward.mean().item(),
        #         "close_bonus": close_bonus.mean().item(),
        #         "total_reward": rewards.mean().item(),
        #         "ee_height": ee_height.mean().item(),
        #         "target_distance": position_error.mean().item(),
        #         "success_rate": success_mask.float().mean().item(),
        #     }
        
        return rewards
    
    def _compute_obstacle_avoidance_rewards(self) -> torch.Tensor:
        """Compute obstacle avoidance rewards for the entire arm."""
        rewards = torch.zeros(self.num_envs, device=self.device)
        safe_distance =0.2
        danger_distance = 0.1
        max_penalty = -5.0
        # Get obstacle position
        obstacle_position = self._red_cube.data.root_pos_w[:, :3]
        
        # Body names for UR5 arm
        arm_body_names = [
            "base_link", "shoulder_link", "upper_arm_link", "forearm_link",
            "wrist_1_link", "wrist_2_link", "wrist_3_link", "ee_link"
        ]

        total_penalty = torch.zeros(self.num_envs, device=self.device)

        # Default weights (higher weight = more important to avoid collision)
        body_weights = {
            "base_link": 0.1,      # Base is less likely to hit obstacle
            "shoulder_link": 0.3,
            "upper_arm_link": 0.5,
            "forearm_link": 0.7,
            "wrist_1_link": 0.8,
            "wrist_2_link": 0.9,
            "wrist_3_link": 0.9,
            "ee_link": 1.0         # End-effector most important
        }
    
        
        robot_body_names = self._robot.body_names
        for body_name in arm_body_names:
            if body_name in robot_body_names:
                body_idx = robot_body_names.index(body_name)
                body_position = self._robot.data.body_pos_w[:, body_idx, :]
                # Calculate distance between this body part and obstacle
                distance = torch.norm(body_position - obstacle_position, p=2, dim=-1)
                
                # Calculate penalty for this body part
                body_penalty = torch.zeros_like(distance)
                
                # Apply penalty only when within safe distance
                within_safe_distance = distance < safe_distance
                
                # Exponential penalty that increases as distance decreases
                normalized_distance = torch.clamp(
                    (distance - danger_distance) / (safe_distance - danger_distance), 
                    0.0, 1.0
                )
                penalty_magnitude = max_penalty * (1.0 - normalized_distance) ** 2
                
                body_penalty = torch.where(within_safe_distance, penalty_magnitude, body_penalty)
                
                # Apply body weight and accumulate
                body_weight = body_weights.get(body_name, 1.0)
                total_penalty += body_weight * body_penalty
        

        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute and return termination flags."""
        # Time limit truncation
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # Pose tracking success termination
        ee_position = self._ee_frame.data.target_pos_w[..., 0, :]
        ee_quat = self._ee_frame.data.target_quat_w[..., 0, :]
        
        # Transform target pose to world frame
        robot_pos = self._robot.data.root_state_w[:, :3]
        robot_quat = self._robot.data.root_state_w[:, 3:7]
        
        des_pos_b = self._target_poses[:, :3]
        des_quat_b = self._target_poses[:, 3:7]
        
        des_pos_w, _ = math_utils.combine_frame_transforms(
            robot_pos, robot_quat, des_pos_b
        )
        des_quat_w = math_utils.quat_mul(robot_quat, des_quat_b)
        
        # Check position and orientation errors
        position_error = torch.norm(ee_position - des_pos_w, p=2, dim=-1)
        orientation_error = math_utils.quat_error_magnitude(ee_quat, des_quat_w)
        
        # Check joint velocities for stability
        joint_velocities = torch.norm(self._robot.data.joint_vel, p=2, dim=-1)
        
        # Success criteria
        position_success = position_error < self.cfg.position_threshold
        orientation_success = orientation_error < self.cfg.orientation_threshold
        velocity_success = joint_velocities < self.cfg.velocity_threshold
        
        # Task success
        task_success = position_success & orientation_success & velocity_success
        
        # # Early termination for safety
        # # Check for table collision
        # ee_height = ee_position[:, 2]
        # table_collision = ee_height < (TABLE_HEIGHT + 0.02)  # Very close to table
        
        # # Check for extreme joint angles or NaN values
        # joint_positions = self._robot.data.joint_pos[:, self._joint_indices]
        # invalid_joints = torch.any(torch.isnan(joint_positions), dim=1) | torch.any(torch.isinf(joint_positions), dim=1)
        
        # # Combine early termination conditions
        # early_termination = table_collision | invalid_joints
        
        return task_success , time_out
    
    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset specified environments."""
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
            
        super()._reset_idx(env_ids)
        
        # Print episode statistics for completed environments
        if len(env_ids) > 0 and hasattr(self, '_episode_sums'):
            avg_position_error = self._episode_sums["position_error"][env_ids].mean().item()
            avg_reward = self._episode_sums["total_reward"][env_ids].mean().item()
            success_rate = self._episode_sums["success_count"][env_ids].mean().item()
            
            if self.common_step_counter % 1000 == 0:  # Log every 1000 steps
                print(f"[INFO] Episode complete - Avg position error: {avg_position_error:.4f}, "
                      f"Avg reward: {avg_reward:.2f}, Success rate: {success_rate:.2f}")
        
        # Reset episode tracking
        if hasattr(self, '_episode_sums'):
            for key in self._episode_sums:
                self._episode_sums[key][env_ids] = 0.0
        
        # Reset robot joint positions
        num_resets = len(env_ids)
        
        # Base joint positions
        base_pose = torch.tensor(
            self.cfg.robot_base_pose,
            device=self.device,
            dtype=torch.float32
        )
        
        # Add noise
        joint_pos = base_pose.unsqueeze(0).repeat(num_resets, 1)
        if self.cfg.robot_reset_noise_range > 0:
            joint_pos += sample_uniform(
                -self.cfg.robot_reset_noise_range,
                self.cfg.robot_reset_noise_range,
                joint_pos.shape,
                self.device
            )
        joint_vel = torch.zeros_like(joint_pos)
        
        # Set joint state
        self._robot.write_joint_state_to_sim(
            joint_pos, joint_vel,
            joint_ids=self._joint_indices,
            env_ids=env_ids
        )
        
        # Reset obstacle position
        x = torch.rand(num_resets, device=self.device) * (
            self.cfg.obstacle_reset_bounds["x_max"] - self.cfg.obstacle_reset_bounds["x_min"]
        ) + self.cfg.obstacle_reset_bounds["x_min"]
        
        y = torch.rand(num_resets, device=self.device) * (
            self.cfg.obstacle_reset_bounds["y_max"] - self.cfg.obstacle_reset_bounds["y_min"]
        ) + self.cfg.obstacle_reset_bounds["y_min"]
        
        z = torch.full((num_resets,), self.cfg.obstacle_reset_bounds["z"], device=self.device)
        
        # Create position tensor
        new_positions = torch.stack([x, y, z], dim=-1)
        new_positions_world = new_positions + self.scene.env_origins[env_ids, :3]
        
        # Default orientation
        default_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
        new_orientations = default_quat.unsqueeze(0).repeat(num_resets, 1)
        
        # Combine position and orientation
        new_poses = torch.cat([new_positions_world, new_orientations], dim=-1)
        
        # Write to simulation
        self._red_cube.write_root_pose_to_sim(new_poses, env_ids=env_ids)
        self._red_cube.write_root_velocity_to_sim(
            torch.zeros((num_resets, 6), device=self.device),
            env_ids=env_ids
        )
        
        # Reset target poses
        self._sample_target_poses_for_reset(env_ids)
        
        # Reset action filter states
        self._action_filter_x1[env_ids] = 0.0
        self._action_filter_x2[env_ids] = 0.0
        self._action_filter_y1[env_ids] = 0.0
        self._action_filter_y2[env_ids] = 0.0
        
        # Reset timers
        self._command_time_left[env_ids] = self.cfg.command_resampling_time
        self._obstacle_time[env_ids] = 0.0
        
    def _sample_target_poses_for_reset(self, env_ids: Sequence[int]):
        """Sample new target poses for reset environments."""
        num_resets = len(env_ids)
        
        # Sample target poses
        x = torch.rand(num_resets, device=self.device) * (
            self.cfg.target_pose_range["x"][1] - self.cfg.target_pose_range["x"][0]
        ) + self.cfg.target_pose_range["x"][0]
        
        y = torch.rand(num_resets, device=self.device) * (
            self.cfg.target_pose_range["y"][1] - self.cfg.target_pose_range["y"][0]
        ) + self.cfg.target_pose_range["y"][0]
        
        z = torch.rand(num_resets, device=self.device) * (
            self.cfg.target_pose_range["z"][1] - self.cfg.target_pose_range["z"][0]
        ) + self.cfg.target_pose_range["z"][0]
        
        # Fixed orientation for now
        roll = torch.full((num_resets,), self.cfg.target_pose_range["roll"][0], device=self.device)
        pitch = torch.full((num_resets,), self.cfg.target_pose_range["pitch"][0], device=self.device)
        yaw = torch.full((num_resets,), self.cfg.target_pose_range["yaw"][0], device=self.device)
        
        # Convert euler to quaternion
        target_quat = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
        
        # Update buffers
        self._target_poses[env_ids, :3] = torch.stack([x, y, z], dim=-1)
        self._target_poses[env_ids, 3:7] = target_quat


    def _set_debug_vis_impl(self, debug_vis:bool):
        # Createe markers for visualizing the goal poses
        if debug_vis:
            if not hasattr(self, "target_pos_visualizer"):
                # Create a separate marker config for target visualization with different color
                target_marker_cfg = FRAME_MARKER_CFG.copy()
                target_marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)  # Slightly larger
                target_marker_cfg.prim_path = "/Visuals/Command/target_position"
                self.target_pos_visualizer = VisualizationMarkers(target_marker_cfg)
            # Set target visibility to true
            self.target_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "target_pos_visualizer"):
                self.target_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self,event):
        #update the markers
        robot_pos = self._robot.data.root_state_w[:, :3]
        robot_quat = self._robot.data.root_state_w[:, 3:7]
        
        des_pos_b = self._target_poses[:, :3]  # Extract position only
        
        # Transform to world frame
        des_pos_w, _ = math_utils.combine_frame_transforms(
            robot_pos, robot_quat, des_pos_b
        )
        
        # Convert orientations to quaternions
        des_quat_b = self._target_poses[:, 3:7]
        des_quat_w = math_utils.quat_mul(robot_quat, des_quat_b)


        # Visualize the world positions
        self.target_pos_visualizer.visualize(translations=des_pos_w, orientations=des_quat_w)



# Factory function for creating the environment
def create_obj_camera_pose_tracking_env(
    cfg: ObjCameraPoseTrackingDirectEnvCfg = None,
    render_mode: str = None,
    **kwargs
) -> ObjCameraPoseTrackingDirectEnv:
    """Factory function to create the environment with default config if none provided."""
    if cfg is None:
        cfg = ObjCameraPoseTrackingDirectEnvCfg()
    
    return ObjCameraPoseTrackingDirectEnv(cfg, render_mode=render_mode, **kwargs)