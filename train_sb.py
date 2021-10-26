import os
import gym
import time
import numpy as np
from vtk_pointlaser.config import Config
# from sim_env.renderer import Renderer
from RL_model.tb_writer import TensorboardCallback
from RL_model.custom_policy import CustomPolicy, CustomLSTMPolicy
from stable_baselines import PPO2, SAC, TD3 #TRPO
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.td3.policies import MlpPolicy as TD3MlpPolicy
from stable_baselines.sac.policies import MlpPolicy as SACMlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from mabi_pointlaser.envs.bullet_env import BulletEnv


def rewards2return(rew, gamma):
    ret = 0
    for r in reversed(rew):
        ret = r + gamma*ret
    return ret


if __name__ == '__main__':

    args = Config().get_arguments()

    assert args.num_lasers <= 3, "Number of Lasers should be less than or equal to 3"
    #TODO Change dimensions using config file
    args.num_states = 6 + 3*args.use_mean_pos + args.use_measurements*args.num_lasers*(args.num_actions+1) + args.use_map_encodings*args.vae_latent_dims + 1*args.use_map_height

    # Create log directories
    if not os.path.exists(os.path.join(args.logdir, args.log_name)):
        os.makedirs(os.path.join(args.logdir, args.log_name))
        os.makedirs(os.path.join(args.logdir, args.log_name, args.weights_logdir))
        os.makedirs(os.path.join(args.logdir, args.log_name, args.tensorboard_logdir))
        os.makedirs(os.path.join(args.logdir, args.log_name, args.env_logdir))

    # Initialize TensorBoard
    tb = TensorboardCallback(args)

    # Import Env, Policy
    # Automatically normalize the input features and reward
    if args.normalized_env:
        env = DummyVecEnv([lambda: BulletEnv(args)])
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=20.)
    else: env = BulletEnv(args)

    policy_kwargs = dict(size=args.arch_size)
    if args.use_LSTM: policy = CustomLSTMPolicy
    else: policy = CustomPolicy

    # Define model
    if args.algo == "PPO2":
        model = PPO2(policy, env, gamma=args.discount_factor, n_steps=args.steps_per_update, ent_coef=0.01, learning_rate=args.lr, vf_coef=0.5, max_grad_norm=0.5, lam=0.95, nminibatches=1, noptepochs=4, cliprange=0.2, cliprange_vf=None, verbose=1, tensorboard_log=os.path.join(args.logdir, args.log_name, args.tensorboard_logdir), _init_setup_model=True, policy_kwargs=policy_kwargs, full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None)

    elif args.algo == "SAC":
        policy = SACMlpPolicy
        model = SAC(policy, env, gamma=args.discount_factor, learning_rate=args.lr, buffer_size=50000, learning_starts=100, train_freq=1, batch_size=64, tau=0.005, ent_coef='auto', target_update_interval=1, gradient_steps=1, target_entropy='auto', action_noise=None, random_exploration=0.0, verbose=1, tensorboard_log=os.path.join(args.logdir, args.log_name, args.tensorboard_logdir), _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None)

    elif args.algo == "TD3":
        policy = TD3MlpPolicy
        model = TD3(policy, env, gamma=args.discount_factor, learning_rate=args.lr, buffer_size=50000, learning_starts=100, train_freq=100, gradient_steps=100, batch_size=128, tau=0.005, policy_delay=2, action_noise=None, target_policy_noise=0.2, target_noise_clip=0.5, random_exploration=0.0, verbose=1, tensorboard_log=os.path.join(args.logdir, args.log_name, args.tensorboard_logdir), _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None)

    # elif args.algo == "TRPO":
    #     model = TRPO(policy, env, gamma=args.discount_factor, timesteps_per_batch=1024, max_kl=0.01, cg_iters=10, lam=0.98, entcoeff=0.0, cg_damping=0.01, vf_stepsize=0.0003, vf_iters=3, verbose=1, tensorboard_log=os.path.join(args.logdir, args.log_name, args.tensorboard_logdir), _init_setup_model=True, policy_kwargs=policy_kwargs, full_tensorboard_log=False, seed=None, n_cpu_tf_sess=1)


    model.learn(total_timesteps=args.num_training_steps, callback=tb)

    model.save(os.path.join(args.logdir, args.log_name, args.weights_logdir))
    env.save(os.path.join(args.logdir, args.log_name, args.env_logdir, "env.pkl"))

    print('\nTraining Finished Successfully !')

    # Test the learnt model and store useful metrics
    env.training = False
    env.norm_reward = False # reward normalization is not needed at test time
    env_ptr = model.get_env().envs[0]
    env_ptr._mesh_nr = 0
    args.dataset_dir = "meshes/eval_data"

    num_episodes = 0
    eps_rewards = []
    eps_rets = []
    obs = env.reset()

    print('\nStarting Model Evaluation: ')

    while num_episodes < args.num_test_episodes:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        eps_rewards.append(rewards[0])
        if dones[0]:
            ret = rewards2return(eps_rewards, args.discount_factor)
            print("Episode %d reward: %.3f" %(num_episodes+1, ret))
            eps_rets.append(ret)
            eps_rewards = []
            num_episodes+=1

    avg_test_vol = np.mean(env_ptr.volume_per_eps)
    avg_test_max_ax = np.mean(env_ptr.maxaxis_per_eps)
    avg_test_dist = np.mean(env_ptr.gt_dist)
    avg_rewards = np.mean(np.asarray(eps_rets))

    print("\n-------------------------------------\n")

    eval_out =   "Average Test Volume:             " + str(avg_test_vol) \
             + "\nAverage Max Axis:                " + str(avg_test_max_ax) \
             + "\nAverage Ground Truth Distance:   " + str(avg_test_dist) \
             + "\nAverage Episode Reward:          " + str(avg_rewards)

    print(eval_out)
    f_out = open(os.path.join(args.logdir, args.log_name, "scores.txt"), "w+")
    f_out.write(eval_out)
    f_out.close()

    f_out = open(os.path.join(args.logdir, args.log_name, "params.txt"), "w+")
    param_str = "\n".join(["{} : {}".format(k, v) for k, v in args.__dict__.items()])
    f_out.write(param_str)
    f_out.close()

    print("\nAll Done!")