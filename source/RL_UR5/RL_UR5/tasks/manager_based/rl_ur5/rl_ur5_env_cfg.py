import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import TiledCameraCfg

from . import mdp

# Create custom UR5 robot configuration
UR5_GRIPPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/adi2440/Desktop/ur5_isaacsim/usd/ur5_moveit.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=None,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.88),  # Position the base of the robot
        joint_pos={
            "shoulder_pan_joint": -0.2321,
            "shoulder_lift_joint": -2.0647,
            "elbow_joint": 1.9495,
            "wrist_1_joint": 0.8378,
            "wrist_2_joint": 1.5097,
            "wrist_3_joint": 0.0,
            "robotiq_85_left_knuckle_joint": 0.0
        }
    ),
    actuators={
        "arm_actuator": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint"
            ],
            effort_limit=400.0,
            velocity_limit=10.0,
            stiffness=100000000.0,
            damping=0.0,
        ),
        "gripper_actuator": ImplicitActuatorCfg(
            joint_names_expr=["robotiq_85_left_knuckle_joint"],
            effort_limit=100.0,
            velocity_limit=10.0,
            stiffness=10000000.0,
            damping=0.0,
        ),
    },
)




##
# Scene definition
##

@configclass
class RlUr5SceneCfg(InteractiveSceneCfg):
    """Configuration for a UR5 pick and place scene."""

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(10.0, 10.0)),
    )
    # UR5 Robot
    robot = UR5_GRIPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/World")
    
    # Table configuration - greyish white
    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/table",
        spawn=sim_utils.CuboidCfg(
            size=(1.83, 0.91, 0.76),  # L x W x H in meters (6 x 3 x 2.5 ft)
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=100.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=50.0),  # Heavy table
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.9, 0.9, 0.9),  # Greyish white
                metallic=0.1,
                roughness=0.8
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.6, 0.0, 0.38),  # Positioned so top surface is at z=0.76m
        ),
    )

    # Red cube - plastic material
    red_cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/red_cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.0635, 0.0191, 0.0286),  # Dimensions in meters
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
            size=(0.0635, 0.0191, 0.0286),
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
            size=(0.0635, 0.0191, 0.0286),
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
    
    # CORRECT STRUCTURE - Change to this
    camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/camera",  # Move prim_path here
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.208,
            focus_distance=28.0,
            horizontal_aperture=5.76,
            vertical_aperture=3.24,
            clipping_range=(0.1, 1000.0)
        ),
        width=256,
        height=144,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.2, 0.06, 1.2),
            rot=(0.3536, 0.3536, 0.1464, 0.8536),
            convention="world"
        )
    )

    # Lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=3000.0),
    )

##
# MDP settings
##

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # Joint position control for UR5 robot
    joint_positions = mdp.JointPositionActionCfg(
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
        scale=1.0,  # Scale for each joint
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Joint state observations
        joint_positions = ObsTerm(func=mdp.joint_positions)
        joint_velocities = ObsTerm(func=mdp.joint_velocities)
        
        # End-effector pose
        ee_pose = ObsTerm(func=mdp.end_effector_pose)
        
        # Cube positions
        cube_positions = ObsTerm(func=mdp.cube_positions)
        
        # Target position (for placing the cube)
        target_position = ObsTerm(func=mdp.target_position)
        
        # Task ID (which cube to pick)
        task_id = ObsTerm(func=mdp.task_id)
        
        # Camera images - updated for clarity
        camera_images = ObsTerm(
            func=mdp.camera_images,
            params={"camera_cfg": SceneEntityCfg(name="camera")},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False  # Keep observations separate in a dict

    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        """Observations for policy group with RGB images."""

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False


    # observation groups
    policy: PolicyCfg = PolicyCfg()
    rgb_camera: RGBCameraPolicyCfg = RGBCameraPolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # Reset robot joints using the working function
    reset_robot = EventTerm(
        func=mdp.reset_robot_pose,
        mode="reset",
        params={
            'pose':[0.0, 0.0, 0.88, -0.2321, -2.0647, 1.9495, 0.8378, 1.5097, 0.0, 0.0],
        },
    )
    
    # Randomize cube positions - updated to avoid slice objects
    reset_cubes = EventTerm(
        func=mdp.reset_cube_positions,
        mode="reset",
        params={
            "table_cfg": SceneEntityCfg(name="table"),
            # Explicitly define each cube configuration with asset_name and empty joint_ids
            "cube_cfgs": [
                SceneEntityCfg(name="red_cube", joint_ids=[],fixed_tendon_ids=[], body_ids=[],object_collection_ids=[]),
                SceneEntityCfg(name="green_cube", joint_ids=[],fixed_tendon_ids=[], body_ids=[],object_collection_ids=[]),
                SceneEntityCfg(name="blue_cube", joint_ids=[],fixed_tendon_ids=[], body_ids=[],object_collection_ids=[]),
            ],
            "min_distance": 0.1,  # Minimum distance between cubes
        },
    )
    
    # Randomly select target cube and placement position
    set_task = EventTerm(
        func=mdp.set_pick_and_place_task,
        mode="reset",
        params={
            "table_cfg": SceneEntityCfg(name="table"),
            "cube_cfgs": [
                SceneEntityCfg(name="red_cube", joint_ids=[],fixed_tendon_ids=[], body_ids=[],object_collection_ids=[]),
                SceneEntityCfg(name="green_cube", joint_ids=[],fixed_tendon_ids=[], body_ids=[],object_collection_ids=[]),
                SceneEntityCfg(name="blue_cube", joint_ids=[],fixed_tendon_ids=[], body_ids=[],object_collection_ids=[]),
            ],
        },
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Distance-based reward for approaching the cube
    distance_to_cube = RewTerm(
        func=mdp.distance_to_target_cube,
        weight=-1.0,
    )
    
    # Reward for being close to the cube
    approach_reward = RewTerm(
        func=mdp.approach_reward,
        weight=1.0,
        params={"distance_threshold": 0.05},
    )
    
    # Reward for grasping the cube
    grasp_reward = RewTerm(
        func=mdp.grasp_reward,
        weight=2.0,
        params={"gripper_threshold": 0.4},
    )
    
    # Reward for placing the cube at the target
    placement_reward = RewTerm(
        func=mdp.placement_reward,
        weight=5.0,
        params={"distance_threshold": 0.1},
    )
    
    # Reward for successfully completing the task
    success_reward = RewTerm(
        func=mdp.success_reward,
        weight=10.0,
        params={"gripper_threshold": 0.1, "distance_threshold": 0.05},
    )
    
    # Penalty for excessive movement
    movement_penalty = RewTerm(
        func=mdp.movement_penalty,
        weight=-0.01,
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # Task success (cube at target and gripper open)
    task_success = DoneTerm(
        func=mdp.task_success,
        params={"gripper_threshold": 0.1, "distance_threshold": 0.05},
    )

##
# Environment configuration
##

@configclass
class RlUr5EnvCfg(ManagerBasedRLEnvCfg):
    """Environment configuration for UR5 pick and place task."""
    
    # Scene settings
    scene: RlUr5SceneCfg = RlUr5SceneCfg(num_envs=8, env_spacing=4.0)
    
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Unused managers
    commands = None
    curriculum = None

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # General settings
        self.decimation = 2
        self.episode_length_s = 30  # Longer episodes for pick and place
        
        # Viewer settings
        self.viewer.eye = (2.0, 2.0, 1.5)
        self.viewer.lookat = (0.5, 0.0, 0.5)
        
        # Simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physx.num_position_iterations = 4
        self.sim.physx.num_velocity_iterations = 1
        self.sim.physx.contact_offset = 0.005
        self.sim.physx.rest_offset = 0.0
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625