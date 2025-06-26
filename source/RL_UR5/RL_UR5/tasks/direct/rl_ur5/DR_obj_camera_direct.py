"""
Direct RL Environment for Object Camera Pose Tracking with UR5 Robot
Enhanced with Domain Randomization for Sim2Real Transfer

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
            pos=(0.65, 0.0, 0.0), 
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
            size=(0.191,0.0572, 0.0635),
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
    episode_length_s = 4.0
    decimation = 4
    action_scale = 0.5  # Reduced for smoother movements
    state_dim = 19
        
    # Observation and action spaces
    action_space = gym.spaces.Box(low=-3.5, high=3.5, shape=(6,))
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

    # Command/target pose settings
    target_pose_range = {
        "x": (0.3, 0.8),
        "y": (0.4, 0.5),
        "z": (-0.2, 0.2),
        "roll": (0.0, 0.0),
        "pitch": (1.57, 1.57),
        "yaw": (0.0, 0.0),
    }
    command_resampling_time = 3.0
    
    # Obstacle movement settings
    obstacle_movement_type = "random_smooth"
    obstacle_center_pos = [0.5, 0.0, 1.0]
    obstacle_radius = 0.75
    obstacle_speed = 0.8
    obstacle_height_variation = 0.5
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
    reward_distance_weight = -1.5
    reward_distance_tanh_weight = 1.0
    reward_distance_tanh_std = 0.1
    reward_orientation_weight = -1.2
    reward_torque_weight = -0.0005
    reward_table_collision_weight = -10.0
    reward_obstacle_avoidance_weight = 10.0
    reward_obstacle_smooth_weight = 3.0
    reward_action_penalty_weight = -0.01
    
    # Action filter settings
    action_filter_order = 2
    action_filter_cutoff_freq = 2.0
    action_filter_damping_ratio = 0.707
    
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
    
    # ============ ORIGINAL NOISE SETTINGS (KEPT FOR REFERENCE) ============
    # joint_pos_noise_min = -0.01
    # joint_pos_noise_max = 0.01
    # joint_vel_noise_min = -0.001
    # joint_vel_noise_max = 0.001
    
    # ============ NEW DOMAIN RANDOMIZATION SETTINGS ============
    # Enable/disable domain randomization
    enable_domain_randomization = True
    
    # Joint observation noise (enhanced from original)
    joint_pos_noise_std = 0.015  # Standard deviation for Gaussian noise
    joint_vel_noise_std = 0.01   # Standard deviation for velocity noise
    
    # Camera domain randomization
    camera_brightness_range = (0.7, 1.3)     # Multiplicative brightness factor
    camera_contrast_range = (0.8, 1.2)       # Contrast adjustment
    camera_noise_std = 0.02                  # Gaussian noise on pixels
    camera_blur_probability = 0.3            # Probability of applying blur
    camera_blur_kernel_range = (3, 7)        # Blur kernel size range
    
    # Camera pose randomization (simulates mounting errors)
    camera_pos_noise_std = 0.02              # Position noise in meters
    camera_rot_noise_std = 0.05              # Rotation noise in radians
    
    # Action domain randomization
    action_noise_std = 0.01                  # Noise added to actions
    action_delay_range = (0, 3)              # Delay in timesteps (0-3 steps)
    action_delay_probability = 0.2           # Probability of action delay
    
    # Control frequency randomization (simulates timing variations)
    control_freq_variation = 0.1             # ±10% frequency variation
    
    # Dynamics randomization
    joint_friction_range = (0.05, 0.3)       # Joint friction multiplier
    joint_damping_range = (0.8, 1.2)         # Joint damping multiplier
    link_mass_range = (0.9, 1.1)             # Link mass multiplier
    payload_mass_range = (0.0, 0.5)          # Random payload at end-effector (kg)
    
    # Kinematic calibration errors
    joint_offset_std = 0.002                 # Joint position offset (rad)
    link_length_std = 0.005                  # Link length errors (m)
    
    # End-effector pose observation noise
    ee_pos_noise_std = 0.01                  # End-effector position noise
    ee_quat_noise_std = 0.02                 # End-effector quaternion noise
    
    # Target pose noise (simulates perception errors)
    target_pos_noise_std = 0.005             # Target position noise
    target_quat_noise_std = 0.01             # Target quaternion noise
    
    # Physics simulation variations
    gravity_variation = 0.02                 # ±2% gravity variation
    sim_dt_variation = 0.1                   # ±10% timestep variation
    
    # Reset settings
    robot_base_pose = [-0.568, -0.858,  1.402, -2.585, -1.6060665,  1.64142667]
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
        
        # ============ NEW: Initialize domain randomization buffers ============
        self._setup_domain_randomization()
        
        # Performance tracking
        self._episode_sums = {
            "position_error": torch.zeros(self.num_envs, device=self.device),
            "total_reward": torch.zeros(self.num_envs, device=self.device),
            "success_count": torch.zeros(self.num_envs, device=self.device),
        }
        
        # Log initial information
        print(f"[INFO] Environment initialized with {self.num_envs} environments")
        print(f"[INFO] Action scale: {self.cfg.action_scale}")
        print(f"[INFO] Domain Randomization: {'ENABLED' if self.cfg.enable_domain_randomization else 'DISABLED'}")
        print(f"[INFO] Target pose range X: {self.cfg.target_pose_range['x']}")
        print(f"[INFO] Target pose range Y: {self.cfg.target_pose_range['y']}")
        print(f"[INFO] Target pose range Z: {self.cfg.target_pose_range['z']}")
        
        # Setup debug visualization if enabled
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_domain_randomization(self):
        """Initialize domain randomization buffers and parameters."""
        if not self.cfg.enable_domain_randomization:
            return
            
        # Action delay buffers
        self._action_delay_buffer = torch.zeros(
            (self.num_envs, self.cfg.action_delay_range[1], len(self._joint_indices)), 
            device=self.device
        )
        self._action_delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        # Joint calibration errors (constant per episode)
        self._joint_offset_errors = torch.zeros(
            (self.num_envs, len(self._joint_indices)), device=self.device
        )
        
        # Camera calibration errors (constant per episode)
        self._camera_pos_errors = torch.zeros((self.num_envs, 3), device=self.device)
        self._camera_rot_errors = torch.zeros((self.num_envs, 3), device=self.device)
        
        # Dynamics parameters (per environment)
        self._joint_friction_multipliers = torch.ones(
            (self.num_envs, len(self._joint_indices)), device=self.device
        )
        self._joint_damping_multipliers = torch.ones(
            (self.num_envs, len(self._joint_indices)), device=self.device
        )
        self._payload_masses = torch.zeros(self.num_envs, device=self.device)
        
        # Control frequency variations
        self._control_freq_multipliers = torch.ones(self.num_envs, device=self.device)
        
        # Initialize randomization on first reset
        self._randomize_domain_parameters(torch.arange(self.num_envs, device=self.device))

    def _randomize_domain_parameters(self, env_ids):
        """Randomize domain parameters for specified environments."""
        if not self.cfg.enable_domain_randomization or len(env_ids) == 0:
            return
            
        num_envs = len(env_ids)
        
        # Joint calibration errors
        self._joint_offset_errors[env_ids] = torch.randn(
            (num_envs, len(self._joint_indices)), device=self.device
        ) * self.cfg.joint_offset_std
        
        # Camera calibration errors
        self._camera_pos_errors[env_ids] = torch.randn(
            (num_envs, 3), device=self.device
        ) * self.cfg.camera_pos_noise_std
        
        self._camera_rot_errors[env_ids] = torch.randn(
            (num_envs, 3), device=self.device
        ) * self.cfg.camera_rot_noise_std
        
        # Joint friction and damping
        self._joint_friction_multipliers[env_ids] = sample_uniform(
            self.cfg.joint_friction_range[0],
            self.cfg.joint_friction_range[1],
            (num_envs, len(self._joint_indices)),
            self.device
        )
        
        self._joint_damping_multipliers[env_ids] = sample_uniform(
            self.cfg.joint_damping_range[0],
            self.cfg.joint_damping_range[1],
            (num_envs, len(self._joint_indices)),
            self.device
        )
        
        # Payload masses
        self._payload_masses[env_ids] = sample_uniform(
            self.cfg.payload_mass_range[0],
            self.cfg.payload_mass_range[1],
            (num_envs,),
            self.device
        )
        
        # Control frequency variations
        self._control_freq_multipliers[env_ids] = sample_uniform(
            1.0 - self.cfg.control_freq_variation,
            1.0 + self.cfg.control_freq_variation,
            (num_envs,),
            self.device
        )
        
        # Apply dynamics randomization to robot
        self._apply_dynamics_randomization(env_ids)

    def _apply_dynamics_randomization(self, env_ids):
        """Apply randomized dynamics parameters to the robot."""
        if not self.cfg.enable_domain_randomization or len(env_ids) == 0:
            return
            
        # Apply joint friction and damping
        for i, joint_idx in enumerate(self._joint_indices):
            # Get current joint properties
            friction = self._robot.data.joint_friction_coefficient[:, joint_idx]
            damping = self._robot.data.joint_damping_coefficient[:, joint_idx]
            
            # Apply multipliers
            friction[env_ids] *= self._joint_friction_multipliers[env_ids, i]
            damping[env_ids] *= self._joint_damping_multipliers[env_ids, i]
            
            # Write back to simulation
            self._robot.write_joint_properties_to_sim(
                friction=friction,
                damping=damping,
                joint_ids=[joint_idx],
                env_ids=env_ids
            )

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
        self.actions = actions.clone().clamp(-3.5, 3.5)
        self._previous_actions = self.actions.clone()
        
        # ============ NEW: Apply action domain randomization ============
        if self.cfg.enable_domain_randomization:
            # Add action noise
            action_noise = torch.randn_like(self.actions) * self.cfg.action_noise_std
            self.actions += action_noise
            
            # Apply action delays
            self.actions = self._apply_action_delays(self.actions)
        
        # Apply action filtering
        filtered_actions = self._apply_action_filter(self.actions)
        
        # Scale actions
        self.actions = filtered_actions * self.cfg.action_scale
        
        # Update command timer
        self._command_time_left -= self.physics_dt
        
        # ============ NEW: Apply control frequency variation ============
        if self.cfg.enable_domain_randomization:
            self._command_time_left -= self.physics_dt * (self._control_freq_multipliers - 1.0)
        
        # --- resample target poses when timer runs out ---
        expired_mask = self._command_time_left <= 0.0
        if torch.any(expired_mask):
            expired_ids = torch.nonzero(expired_mask, as_tuple=False).squeeze(-1)
            env_ids = expired_ids.cpu().tolist()
            self._sample_commands(env_ids)
            # reset their countdown
            self._command_time_left[expired_mask] = self.cfg.command_resampling_time
        
        # IF robot is stuck at the table, reset it
        self._reset_robot_when_stuck_at_table()

        # Update obstacle
        self._update_obstacle_position()

    def _apply_action_delays(self, actions: torch.Tensor) -> torch.Tensor:
        """Apply randomized action delays to simulate control latency."""
        if not self.cfg.enable_domain_randomization:
            return actions
            
        # Update action delay buffer
        self._action_delay_buffer = torch.roll(self._action_delay_buffer, shifts=1, dims=1)
        self._action_delay_buffer[:, 0] = actions
        
        # Get delayed actions based on per-environment delay
        delayed_actions = actions.clone()
        for i in range(self.num_envs):
            delay = self._action_delay_steps[i]
            if delay > 0:
                delayed_actions[i] = self._action_delay_buffer[i, delay]
        
        return delayed_actions
        
    def _apply_action(self) -> None:
        """Apply the processed actions to the robot."""
        # Get current joint positions
        # current_joint_pos = self._robot.data.joint_pos[:, self._joint_indices]
        
        # Add actions to current positions 
        self._robot_dof_targets = self.actions
        
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
        """Get state-based observations with domain randomization."""
        # Get joint positions
        joint_pos = self._robot.data.joint_pos[:, self._joint_indices]
        
        # ============ NEW: Apply joint calibration errors ============
        if self.cfg.enable_domain_randomization:
            joint_pos = joint_pos + self._joint_offset_errors
        
        # ============ NEW: Apply enhanced joint position noise ============
        if self.cfg.enable_domain_randomization:
            joint_pos_noise = torch.randn_like(joint_pos) * self.cfg.joint_pos_noise_std
            joint_pos_noisy = joint_pos + joint_pos_noise
        else:
            # Fallback to original uniform noise
            joint_pos_noise = torch.rand_like(joint_pos) * 0.02 - 0.01
            joint_pos_noisy = joint_pos + joint_pos_noise
        
        # Get joint velocities
        joint_vel = self._robot.data.joint_vel[:, self._joint_indices]
        
        # ============ NEW: Apply enhanced joint velocity noise ============
        if self.cfg.enable_domain_randomization:
            joint_vel_noise = torch.randn_like(joint_vel) * self.cfg.joint_vel_noise_std
            joint_vel_noisy = joint_vel + joint_vel_noise
        else:
            # Fallback to original uniform noise
            joint_vel_noise = torch.rand_like(joint_vel) * 0.002 - 0.001
            joint_vel_noisy = joint_vel + joint_vel_noise
        
        # Get target pose
        target_pose = self._target_poses
        
        # ============ NEW: Apply target pose noise (perception errors) ============
        if self.cfg.enable_domain_randomization:
            target_pos_noise = torch.randn((self.num_envs, 3), device=self.device) * self.cfg.target_pos_noise_std
            target_quat_noise = torch.randn((self.num_envs, 4), device=self.device) * self.cfg.target_quat_noise_std
            
            target_pose_noisy = target_pose.clone()
            target_pose_noisy[:, :3] += target_pos_noise
            target_pose_noisy[:, 3:7] += target_quat_noise
            # Normalize quaternions
            target_pose_noisy[:, 3:7] = math_utils.normalize(target_pose_noisy[:, 3:7])
        else:
            target_pose_noisy = target_pose
        
        # Concatenate all state observations
        state_obs = torch.cat([
            joint_pos_noisy,      # 6 dims
            joint_vel_noisy,      # 6 dims
            target_pose_noisy,    # 7 dims
        ], dim=-1)
        
        return state_obs
    
    def _get_camera_observations(self) -> torch.Tensor:
        """Get and preprocess camera observations with domain randomization."""
        # Get camera data
        camera_data = self._tiled_camera.data.output["rgb"] / 255.0  # Shape: (num_envs, H, W, C)
        
        # ============ NEW: Apply camera domain randomization ============
        if self.cfg.enable_domain_randomization:
            camera_data = self._apply_camera_domain_randomization(camera_data)
        
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
    
    def _apply_camera_domain_randomization(self, camera_data: torch.Tensor) -> torch.Tensor:
        """Apply various camera domain randomization techniques."""
        # camera_data shape: (num_envs, H, W, C)
        
        # 1. Brightness adjustment
        brightness_factors = sample_uniform(
            self.cfg.camera_brightness_range[0],
            self.cfg.camera_brightness_range[1],
            (self.num_envs, 1, 1, 1),
            self.device
        )
        camera_data = camera_data * brightness_factors
        
        # 2. Contrast adjustment
        contrast_factors = sample_uniform(
            self.cfg.camera_contrast_range[0],
            self.cfg.camera_contrast_range[1],
            (self.num_envs, 1, 1, 1),
            self.device
        )
        mean = torch.mean(camera_data, dim=(1, 2, 3), keepdim=True)
        camera_data = (camera_data - mean) * contrast_factors + mean
        
        # 3. Gaussian noise
        noise = torch.randn_like(camera_data) * self.cfg.camera_noise_std
        camera_data = camera_data + noise
        
        # 4. Random blur (applied to subset of environments)
        if self.cfg.camera_blur_probability > 0:
            blur_mask = torch.rand(self.num_envs, device=self.device) < self.cfg.camera_blur_probability
            blur_envs = torch.nonzero(blur_mask).squeeze(-1)
            
            if len(blur_envs) > 0:
                for env_id in blur_envs:
                    kernel_size = random.randint(
                        self.cfg.camera_blur_kernel_range[0], 
                        self.cfg.camera_blur_kernel_range[1]
                    )
                    # Ensure kernel size is odd
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                    
                    # Apply Gaussian blur
                    img = camera_data[env_id].unsqueeze(0).permute(0, 3, 1, 2)  # (1, C, H, W)
                    blurred = torch.nn.functional.gaussian_blur(
                        img, kernel_size=(kernel_size, kernel_size)
                    )
                    camera_data[env_id] = blurred.permute(0, 2, 3, 1).squeeze(0)
        
        # Clamp to valid range
        camera_data = torch.clamp(camera_data, 0.0, 1.0)
        
        return camera_data
    
    def _get_rewards(self) -> torch.Tensor:
        """Compute and return rewards with detailed logging."""
        # Initialize total rewards
        rewards = torch.zeros(self.num_envs, device=self.device)
        
        # Get end-effector position and orientation
        ee_position = self._ee_frame.data.target_pos_w[..., 0, :]
        ee_quat = self._ee_frame.data.target_quat_w[..., 0, :]
        
        # ============ NEW: Apply EE observation noise for reward computation ============
        if self.cfg.enable_domain_randomization:
            ee_pos_noise = torch.randn_like(ee_position) * self.cfg.ee_pos_noise_std * 0.5
            ee_position = ee_position + ee_pos_noise
        
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
        
        # 4. Joint torque penalty
        if hasattr(self._robot.data, 'applied_torque') and self._robot.data.applied_torque is not None:
            joint_torques = self._robot.data.applied_torque[:, self._joint_indices]
            torque_penalty = torch.sum(torch.square(joint_torques), dim=1)
            torque_reward = self.cfg.reward_torque_weight * torque_penalty
            rewards += torque_reward
        else:
            torque_reward = torch.zeros_like(rewards)
        
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
        obstacle_reward = self._compute_obstacle_avoidance_rewards() * self.cfg.reward_obstacle_avoidance_weight
        rewards += obstacle_reward
        
        return rewards
    
    def _compute_obstacle_avoidance_rewards(self) -> torch.Tensor:
        """Compute obstacle avoidance rewards for the entire arm."""
        rewards = torch.zeros(self.num_envs, device=self.device)
        safe_distance = 0.2
        danger_distance = 0.1
        max_penalty = -10.0
        
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
            "shoulder_link": 0.9,
            "upper_arm_link": 0.9,
            "wrist_1_link": 0.5,
            "wrist_2_link": 0.5,
            "wrist_3_link": 0.5,
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

    def _reset_robot_when_stuck_at_table(self):
        """Reset robot to a safe position when it gets stuck at the table."""
        # Default safe poses (well above table)
        safe_poses = [
            [-0.71055204, -1.3046993, 1.9, -2.23, -1.59000665, 1.76992667],
            [-0.568, -0.658, 1.602, -2.585, -1.6060665, -1.64142667],  # Alternative safe pose
        ]
        
        # Get end-effector position
        ee_position = self._ee_frame.data.target_pos_w[..., 0, :]
        ee_height = ee_position[:, 2]
        
        # Check which environments have robots stuck at table
        table_height = TABLE_HEIGHT
        safety_margin = 0.05
        stuck_at_table = ee_height < (table_height + safety_margin)
        
        # Get environment IDs that are stuck
        stuck_env_ids = torch.nonzero(stuck_at_table, as_tuple=False).squeeze(-1)
        
        if len(stuck_env_ids) == 0:
            return
        
        # Reset stuck robots to safe positions
        for env_id in stuck_env_ids:
            # Choose a random safe pose
            import random
            base_pose = random.choice(safe_poses)
            
            # Convert pose to tensor and add noise
            joint_pos_base = torch.tensor(base_pose, device=self.device, dtype=torch.float32)
            noise_range = 0.02
            noise = torch.rand(len(base_pose), device=self.device) * 2 * noise_range - noise_range
            joint_pos = joint_pos_base + noise
            joint_vel = torch.zeros_like(joint_pos)
            
            # Clamp to joint limits
            joint_pos = torch.clamp(
                joint_pos,
                self._robot_dof_lower_limits,
                self._robot_dof_upper_limits
            )
            
            # Convert env_id to tensor
            env_id_tensor = torch.tensor([env_id], device=self.device, dtype=torch.long)
            
            # Reset robot to safe position
            self._robot.set_joint_position_target(
                joint_pos.unsqueeze(0), 
                joint_ids=self._joint_indices, 
                env_ids=env_id_tensor
            )
            self._robot.write_joint_state_to_sim(
                joint_pos.unsqueeze(0), 
                joint_vel.unsqueeze(0), 
                joint_ids=self._joint_indices, 
                env_ids=env_id_tensor
            )
            
            # Update target to prevent immediate re-collision
            self._robot_dof_targets[env_id] = joint_pos
            
            # Log the reset
            if len(stuck_env_ids) <= 2:  # Avoid spam
                print(f"[INFO] Reset stuck robot in environment {env_id.item()}")

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
        
        return task_success, time_out
    
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
        
        # ============ NEW: Randomize domain parameters on reset ============
        self._randomize_domain_parameters(env_ids)
        
        # ============ NEW: Randomize action delays ============
        if self.cfg.enable_domain_randomization:
            # Randomly assign action delays
            delay_mask = torch.rand(len(env_ids), device=self.device) < self.cfg.action_delay_probability
            self._action_delay_steps[env_ids] = torch.where(
                delay_mask,
                torch.randint(
                    self.cfg.action_delay_range[0], 
                    self.cfg.action_delay_range[1] + 1,
                    (len(env_ids),), 
                    device=self.device
                ),
                torch.zeros(len(env_ids), device=self.device, dtype=torch.long)
            )
            # Clear action delay buffer
            self._action_delay_buffer[env_ids] = 0.0
        
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
        # Create markers for visualizing the goal poses
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
        # Update the markers
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