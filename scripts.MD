## Script for training the RL algoritm
```
conda activate ur5_env
cd  --checkpoint logs/rl_games/pick_place_ur5/2025-04-09_16-52-12/nn/pick_place_ur5.pth

```

## Script for playing the trained RL algorithm

```
conda activate ur5_env
python scripts/rl_games/play.py --task=Isaac-UR5-PickAndPlace-PPO --num_envs 2 --checkpoint logs/rl_games/pick_place_ur5/2025-04-09_17-00-04/nn/pick_place_ur5.pth

```

## Script for Tensorboard
```
tensorboard --logdir logs/rl_games/pick_place_ur5/2025-05-12_08-52-26/
```