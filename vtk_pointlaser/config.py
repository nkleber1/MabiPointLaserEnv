''' Training configuration parameters '''

# Parser Arguments
import argparse
# import torch
import os
import numpy as np
from datetime import datetime

class Config(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description='Reinforcement Learning Active Localization')

        # Setup GPU
        parser.add_argument('--gpu_index', type=int, default=0)
        parser.add_argument('--use_gpu', type=int, default=1)

        # Log Directories Setup
        parser.add_argument('--load', type=str, default=None, help='load a saved model')
        parser.add_argument('--log_name', type=str, default=datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M'),
                            help='Log Folder Name')
        parser.add_argument('--logdir', type=str, default='logs', help='Log Folder Location')
        parser.add_argument('--weights_logdir', type=str, default='weights')
        parser.add_argument('--env_logdir', type=str, default='env')
        parser.add_argument('--tensorboard_logdir', type=str, default='tensorboard')

        # Mesh Environment
        parser.add_argument('--mesh_file', type=str, default='r_map', help='mesh file name')
        parser.add_argument('--mesh_nr', type=int, default=1, help='Mesh file number')
        parser.add_argument('--dataset_dir', type=str, default='./meshes/train_data', help='Mesh environment directory')
        parser.add_argument('--mesh_dir', type=str, default='meshes', help='Mesh environment directory')
        parser.add_argument('--num_lasers', type=int, default=3, help='Number of lasers to use. 1, 2 or 3 possible')
        parser.add_argument('--rescale', type=int, default=400, help='Mesh rescale factor')
        parser.add_argument('--render', type=int, default=1, help='Render the current scene. 1 means True')
        parser.add_argument('--mesh_offset_min', type=int, default=100, help='Minimum offset from mesh boundary')
        parser.add_argument('--position_stddev', type=int, default=50, help='Initial position standard deviation')

        # VAE Setup
        parser.add_argument('--encodings_dir', type=str, default='map_encodings', help='Map directory')
        parser.add_argument('--vae_weights_dir', type=str, default='map_encoder/weights', help='VAE weights directory')
        parser.add_argument('--vae_latent_dims', type=int, default=16, help='Map Encoding dimension')

        # Training Setup
        parser.add_argument('--num_epochs', type=int, default=1200)
        parser.add_argument('--num_training_steps', type=int, default=400000)
        parser.add_argument('--num_episodes', type=int, default=100)
        parser.add_argument('--num_test_episodes', type=int, default=2000)
        parser.add_argument('--discount_factor', type=float, default=0.99)
        parser.add_argument('--steps_per_update', type=int, default=128)
        parser.add_argument('--steps_per_summary', type=int, default=4096)

        # RL Model Setup
        parser.add_argument('--algo', type=str, default='PPO2', help='Training algorithm to use')
        parser.add_argument('--normalized_env', type=int, default=1, help='Use vectorized environment')
        parser.add_argument('--use_LSTM', type=int, default=1, help='Use LSTM policy')
        parser.add_argument('--arch_size', type=str, default='M', help='Architecture Size')
        parser.add_argument('--num_actions', type=int, default=3, help='Action space dimension')
        parser.add_argument('--map_size', type=int, default=64, help='Map dimension')
        parser.add_argument('--n_actions', type=int, default=12, help='Discretization of action space')
        parser.add_argument('--action_limits', type=int, default=1, help='Action space bounds')
        parser.add_argument('--max_steps', type=int, default=10, help='Maximum no. of steps in one epoch')
        parser.add_argument('--var_eps_len', type=int, default=0, help='Variable Episode Length')
        parser.add_argument('--min_uncertainty', type=float, default=5, help='Uncertainty Goal')
        parser.add_argument('--init_policy_distr', type=np.array, default=np.array([0.25, 0.25, 0.25]),
                            help='Initial standard deviation of policy distribution') #unused
        parser.add_argument('--n_action', type=int, default=4, help='Discrete actions')
        parser.add_argument('--actor_kl', type=float, default=0.0075, help='Desired KL Divergence for Actor part') #unused
        parser.add_argument('--critic_kl', type=float, default=0.0075, help='Desired KL Divergence for Critic part') #unused
        parser.add_argument('--critic_lr', type=float, default=0.0005, help='Learning rate of Adam optimizer') #unused
        parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate of Adam optimizer')
        parser.add_argument('--ppo', type=int, default=0, help='To use PPO instead of TRPO. 0 means False') #unused
        parser.add_argument('--ppo_clip', type=float, default=0.15, help='Clipping bounds of PPO loss, corresponds to KL Div.') #unused

        # Obs params
        parser.add_argument('--use_mean_pos', type=int, default=1, help='Use Mean Position estimate in the state space')
        parser.add_argument('--use_measurements', type=int, default=1, help='Use Laser Measurements in the state space')
        parser.add_argument('--use_map_encodings', type=int, default=1, help='Use map_encodings in the state space')
        parser.add_argument('--use_map_height', type=int, default=1, help='Use map height in the state space')
        parser.add_argument('--use_joint_states', type=int, default=1, help='Use joint states in the state space')

        # Reward params
        parser.add_argument('--rew_alpha', type=float, default=1, help='Final reward weight for max eigen value reduction')
        parser.add_argument('--meas_cost', type=float, default=0.0, help='Cost for making a measurement')
        parser.add_argument('--use_uncert_rew', type=int, default=1, help='Use uncertainty reduction for reward')
        parser.add_argument('--uncert_rew_w', type=float, default=1.0, help='Weight for uncertainty reduction reward')
        parser.add_argument('--use_dist_rew', type=int, default=1, help='Use distance reduction for reward')
        parser.add_argument('--dist_rew_w', type=float, default=0.2, help='Weight for distance reduction reward')
        parser.add_argument('--use_eigval_rew', type=int, default=1, help='Use eigen value reduction for reward')
        parser.add_argument('--eigval_rew_w', type=float, default=0.5, help='Weight for eigen value reduction reward')
        parser.add_argument('--use_goal_rew', type=int, default=1, help='Use goal for reward')
        parser.add_argument('--goal_rew', type=float, default=100, help='Value for goal acheived')
        parser.add_argument('--incorrect_pose_punishment', type=float, default=0, help='Punishment for imposibele action')

        # PyBullet params
        parser.add_argument('--seed', type=int, default=None, help='Set random seed')
        parser.add_argument('--n_step', type=int, default=100, help='# of steps performed to reach target position')
        parser.add_argument('--sleep', type=float, default=0, help='sec sleep between each step (for debugging')
        parser.add_argument('--time_step', type=float, default=0.002, help='Set pyBullet time_step')
        parser.add_argument('--frame_skip', type=int, default=5, help='Set pyBullet frame_skip')
        parser.add_argument('--num_solver_iterations', type=int, default=5, help='Set pyBullet num_solver_iterations')
        parser.add_argument('--renderRobot', type=bool, default=False, help='Use connection_mode=pybullet.GUI')

        self.args = parser.parse_args()

    def get_arguments(self):
        return self.args
