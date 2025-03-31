# tasks/__init__.py

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-RL-UR5-PickAndPlace-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_ur5_env_cfg:RlUr5EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-UR5-PoseTracking-PPO",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pose_tracking_env_cfg:PoseTrackingEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-UR5-PoseTracking-SAC",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pose_tracking_env_cfg:PoseTrackingEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
    disable_env_checker=True,
)