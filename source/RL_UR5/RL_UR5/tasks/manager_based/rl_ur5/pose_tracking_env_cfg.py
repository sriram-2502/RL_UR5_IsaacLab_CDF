from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg  
from isaaclab_assets.robots.ur5 import UR5_GRIPPER_CFG
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from . import mdp


    
marker_cfg = FRAME_MARKER_CFG.copy()
marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
marker_cfg.prim_path = "/Visuals/FrameTransformer"


##
# Scene definition
##

@configclass
class PoseTrackingSceneCfg(InteractiveSceneCfg):
    """Configuration for a UR5 pick and place scene."""

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    # UR5 Robot
    robot = UR5_GRIPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


    # Key change: Move the frame transformer creation after the robot is created
    # This will be set up in the post_reset phase instead
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ee_link",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.12, 0.0, 0.0],
                    ),
                ),
            ],
        )
    
    # red_cube_frame: FrameTransformerCfg = FrameTransformerCfg(
    #         prim_path="{ENV_REGEX_NS}/Robot/base_link",
    #         debug_vis=True,
    #         visualizer_cfg=marker_cfg,
    #         target_frames=[
    #             FrameTransformerCfg.FrameCfg(
    #                 prim_path="{ENV_REGEX_NS}/red_cube",
    #                 name="red_cube_frames",
    #                 offset=OffsetCfg(
    #                     pos=[0.0, 0.0, 0.0],
    #                 ),
    #             ),
    #         ],
    #     )


    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"/home/adi2440/Desktop/ur5_isaacsim/usd/table.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.68, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # Red cube - plastic material
    red_cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/red_cube",
        spawn=sim_utils.CuboidCfg(
            size=( 0.0286, 0.0635, 0.0382),  # Dimensions in meters
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=100.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.01),  # 10 grams
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),  # Red
                metallic=0.0,  # Non-metallic for plastic
                roughness=0.5  # Medium roughness for plastic
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, -0.4, 0.77),  # Will be randomized during reset
        ),
    )

    # Green cube - plastic material
    green_cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/green_cube",
        spawn=sim_utils.CuboidCfg(
            size=( 0.0286, 0.0635, 0.0382),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=100.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.01),  # 10 grams
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 1.0, 0.0),  # Green
                metallic=0.0,  # Non-metallic for plastic
                roughness=0.5  # Medium roughness for plastic
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.6, -0.4, 0.77),  # Will be randomized during reset
        ),
    )

    # Blue cube - plastic material
    blue_cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/blue_cube",
        spawn=sim_utils.CuboidCfg(
            size=( 0.0286, 0.0635, 0.0382),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=100.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.01),  # 10 grams
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.0, 1.0),  # Blue
                metallic=0.0,  # Non-metallic for plastic
                roughness=0.5  # Medium roughness for plastic
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.7, -0.4, 0.77),  # Will be randomized during reset
        ),
    )
    
    # # CORRECT STRUCTURE - Change to this
    # tiled_camera_left: TiledCameraCfg = TiledCameraCfg(
    #     prim_path="{ENV_REGEX_NS}/camera_left",  # Move prim_path here
    #     data_types=["rgb"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=2.208,
    #         focus_distance=28.0,
    #         horizontal_aperture=5.76,
    #         vertical_aperture=3.24,
    #         clipping_range=(0.1, 1000.0)
    #     ),
    #     width=256,
    #     height=144,
    #     offset=TiledCameraCfg.OffsetCfg(
    #         pos=(1.2, -0.06, 1.2),
    #         rot=(0.3536, 0.3536, 0.1464, 0.8536),
    #         convention="world"
    #     )
    # )

    # tiled_camera_right: TiledCameraCfg = TiledCameraCfg(
    #     prim_path="{ENV_REGEX_NS}/camera_right",  # Move prim_path here
    #     data_types=["rgb"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=2.208,
    #         focus_distance=28.0,
    #         horizontal_aperture=5.76,
    #         vertical_aperture=3.24,
    #         clipping_range=(0.1, 1000.0)
    #     ),
    #     width=256,
    #     height=144,
    #     offset=TiledCameraCfg.OffsetCfg(
    #         pos=(1.2, 0.06, 1.2),
    #         rot=(0.3536, 0.3536, 0.1464, 0.8536),
    #         convention="world"
    #     )
    # )

    # Lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=3000.0),
    )

##
# MDP settings
##

## Commands class is for defining the final object location
@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    tracking_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="ee_link",  
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.3, 0.6), pos_y=(-0.56, 0.56), pos_z=(0.0, 0.3), roll=(0.0, 0.0), pitch=(1.57, 1.57), yaw=(-3.14, 3.14)
        ),
    )



@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # Joint position control for UR5 robot
    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
            "robotiq_85_left_knuckle_joint",
        ],
        scale=0.5,  # Scale for each joint
        use_default_offset=True,
    )

    # gripper_action = mdp.BinaryJointPositionActionCfg(            
    #     asset_name="robot",
    #     joint_names=["robotiq_85_left_knuckle_joint"],
    #     open_command_expr={"robotiq_85_left_knuckle_joint": 0.0},
    #     close_command_expr={"robotiq_85_left_knuckle_joint": 30.0},
    # )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Joint state observations - using joint_names instead of joint_ids
        joint_positions = ObsTerm(
            func=mdp.joint_pos_rel,noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot",
                                                joint_names=["shoulder_pan_joint",
                                                            "shoulder_lift_joint",
                                                            "elbow_joint",
                                                            "wrist_1_joint",
                                                            "wrist_2_joint",
                                                            "wrist_3_joint",
                                                            "robotiq_85_left_knuckle_joint"]
                                                )
                    }
        )
        
        # Joint velocity observations - also using joint_names
        joint_velocities = ObsTerm(
            func=mdp.joint_vel_rel,noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", 
                                                joint_names=["shoulder_pan_joint",
                                                            "shoulder_lift_joint",
                                                            "elbow_joint",
                                                            "wrist_1_joint",
                                                            "wrist_2_joint",
                                                            "wrist_3_joint",
                                                            "robotiq_85_left_knuckle_joint"]
                                                )
                    }
        )
        
        # End-effector pose
        # ee_pose = ObsTerm(func=mdp.end_effector_pose)
        
        # Cube positions
        # cube_positions = ObsTerm(func=mdp.cube_positions)
        
        # Target position (for the end-effector)
        target_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "tracking_pose"})
        
        # Task ID (which cube to pick)
        # task_id = ObsTerm(func=mdp.task_id)

        # Actions
        actions = ObsTerm(func=mdp.last_action)

    # @configclass
    # class SubtaskCfg(ObsGroup):
    #     """Observations for tracking the subtask progress."""
        
    #     # Stage 0: Alignment above cube
    #     alignment_complete = ObsTerm(
    #         func=mdp.alignment_above_cube_complete,
    #         params={
    #             "robot_cfg": SceneEntityCfg(name="robot", joint_ids=[], fixed_tendon_ids=[], body_ids=[], object_collection_ids=[]),
    #             "ee_frame_cfg": SceneEntityCfg(name="ee_frame", joint_ids=[], fixed_tendon_ids=[], body_ids=[], object_collection_ids=[]),
    #             "height_threshold": 0.3,
    #             "alignment_threshold": 0.0,
    #         },
    #     )
        
    #     # Stage 1: Grasp cube
    #     cube_grasped = ObsTerm(
    #         func=mdp.cube_grasped,
    #         params={
    #             "robot_cfg": SceneEntityCfg(name="robot", joint_ids=[], fixed_tendon_ids=[], body_ids=[], object_collection_ids=[]),
    #             "ee_frame_cfg": SceneEntityCfg(name="ee_frame", joint_ids=[], fixed_tendon_ids=[], body_ids=[], object_collection_ids=[]),
    #             "distance_threshold": 0.01,
    #             "gripper_threshold": 25.0,
    #         },
    #     )
        
    #     # Stage 2: Place cube at target
    #     cube_placed = ObsTerm(
    #         func=mdp.cube_placed_at_target,
    #         params={
    #             "robot_cfg": SceneEntityCfg(name="robot", joint_ids=[], fixed_tendon_ids=[], body_ids=[], object_collection_ids=[]),
    #             "distance_threshold": 0.05,
    #             "gripper_threshold": 5.0,
    #         },
    #     )
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    # @configclass
    # class RGBCameraPolicyCfg(ObsGroup):
    #     """Observations for policy group with RGB images."""
    #             # Camera images - updated for clarity
    #     camera_images_left = ObsTerm(
    #         func=mdp.image,
    #         params={"sensor_cfg": SceneEntityCfg("tiled_camera_left"), "data_type": "rgb"},
    #     )

    #     camera_images_right = ObsTerm(
    #         func=mdp.image,
    #         params={"sensor_cfg": SceneEntityCfg("tiled_camera_right"), "data_type": "rgb"},
    #     )



    # observation groups
    policy: PolicyCfg = PolicyCfg()
    # subtask_terms: SubtaskCfg = SubtaskCfg()
    # rgb_camera: RGBCameraPolicyCfg = RGBCameraPolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # Reset robot joints using the working function
    reset_robot = EventTerm(
        func=mdp.reset_robot_pose,
        mode="reset",
        params={
            'pose': [-0.2321, -2.0647, 1.9495, 0.8378, 1.5097, 0.0, 0.0],
        },
    )

    # reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    # # Randomize cube positions - updated to avoid slice objects
    # reset_cubes = EventTerm(
    #     func=mdp.reset_cube_positions,
    #     mode="reset",
    #     params={
    #         # Explicitly define each cube configuration with asset_name and empty joint_ids
    #         "cube_cfgs": [
    #             SceneEntityCfg(name="red_cube", joint_ids=[],fixed_tendon_ids=[], body_ids=[],object_collection_ids=[]),
    #             SceneEntityCfg(name="green_cube", joint_ids=[],fixed_tendon_ids=[], body_ids=[],object_collection_ids=[]),
    #             SceneEntityCfg(name="blue_cube", joint_ids=[],fixed_tendon_ids=[], body_ids=[],object_collection_ids=[]),
    #         ],
    #         "min_distance": 0.1,  # Minimum distance between cubes
    #     },
    # )
    
    # # # Randomly select target cube and set placement position from commands
    # set_task = EventTerm(
    #     func=mdp.set_pick_and_place_task,
    #     mode="reset",
    #     params={
    #         "cube_cfgs": [
    #             SceneEntityCfg(name="red_cube", joint_ids=[],fixed_tendon_ids=[], body_ids=[],object_collection_ids=[]),
    #             SceneEntityCfg(name="green_cube", joint_ids=[],fixed_tendon_ids=[], body_ids=[],object_collection_ids=[]),
    #             SceneEntityCfg(name="blue_cube", joint_ids=[],fixed_tendon_ids=[], body_ids=[],object_collection_ids=[]),
    #         ],
    #     },
    # )

    # init_task_stages = EventTerm(
    #     func=mdp.initialize_task_stages,
    #     mode="reset",
    # )
    
    # # Update task stages during each step
    # update_task_stages = EventTerm(
    #     func=mdp.update_task_stages,
    #     mode="step",
    # )

    #Think about adding noise to joint states for better generalization

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Distance-based reward to approach the tracking pose
    distance_to_tracking_pose = RewTerm(
        func=mdp.position_command_error,
        weight=-2.5,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "tracking_pose"},
    )

    distance_to_tracking_pose_tanh = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot"), "std": 0.2, "command_name": "tracking_pose"},
    )

    # # In the RewardsCfg class in rl_ur5_env_cfg.py
    orientation_reward = RewTerm(
        func=mdp.orientation_command_error,
        weight=-1.0,  # Adjust weight as needed
        params={
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),"command_name": "tracking_pose",
        },
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.00001)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.00001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    
    # In the RewardsCfg class in rl_ur5_env_cfg.py
    joint_torques_penalty = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-0.0000005,  # Adjust weight as needed
    )



@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    joint_torque = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_torques_penalty", "weight": -0.00005, "num_steps": 1000}
    )

    position_weight = CurrTerm(
    func=mdp.modify_reward_weight, 
    params={"term_name": "distance_to_tracking_pose", "weight": -0.5, "num_steps": 1000}
    )
##
# Environment configuration
##

@configclass
class PoseTrackingEnvCfg(ManagerBasedRLEnvCfg):
    """Environment configuration for UR5 pick and place task."""
    
    # Scene settings
    scene: PoseTrackingSceneCfg = PoseTrackingSceneCfg(num_envs=8, env_spacing=4.0)
    
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    # curriculum: CurriculumCfg = CurriculumCfg()
    curriculum = None

    # Unused managers
    commands: CommandsCfg = CommandsCfg()  # Add the command configuration


    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # General settings
        self.decimation = 4
        self.episode_length_s = 12  # Longer episodes for pick and place

        # make a smaller scene for play
        self.scene.num_envs = 8
        self.scene.env_spacing = 4.0
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # Simulation settings

        self.sim.dt = 1 / 120   #120Hz
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        # self.sim.physx.num_position_iterations = 4
        # self.sim.physx.num_velocity_iterations = 1
        # self.sim.physx.contact_offset = 0.005
        # self.sim.physx.rest_offset = 0.0
        # self.sim.physx.bounce_threshold_velocity = 0.2
        # self.sim.physx.bounce_threshold_velocity = 0.01
        # self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        # self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        # self.sim.physx.friction_correlation_distance = 0.00625