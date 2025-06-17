# tasks/__init__.py

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-UR5-DirectObjCamera-PPO",
    entry_point=f"{__name__}.obj_camera_direct:ObjCameraPoseTrackingDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.obj_camera_direct:ObjCameraPoseTrackingDirectEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_camera.yaml",
    },
)

gym.register(
    id="Isaac-UR5-DirectPoseTracking-PPO",
    entry_point=f"{__name__}.simple_pose_tracking_direct:SimpleCameraPoseTrackingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.simple_pose_tracking_direct:SimpleCameraPoseTrackingEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_camera.yaml",
    },
)