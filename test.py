import os
from mabi_pointlaser.envs.bullet_env import BulletEnv
from stable_baselines import PPO2
from vtk_pointlaser.config import Config
from RL_model.tb_writer import TensorboardCallback
from RL_model.custom_policy import CustomPolicy, CustomLSTMPolicy

env = BulletEnv()
args = Config().get_arguments()

policy = CustomLSTMPolicy

policy_kwargs = dict(size=args.arch_size)
model = PPO2(policy, env, gamma=args.discount_factor, n_steps=args.steps_per_update, ent_coef=0.01, learning_rate=args.lr, vf_coef=0.5, max_grad_norm=0.5, lam=0.95, nminibatches=1, noptepochs=4, cliprange=0.2, cliprange_vf=None, verbose=1, tensorboard_log=os.path.join(args.logdir, args.log_name, args.tensorboard_logdir), _init_setup_model=True, policy_kwargs=policy_kwargs, full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None)
tb = TensorboardCallback(args)
model.learn(total_timesteps=args.num_training_steps, callback=tb)

model.save(os.path.join(args.logdir, args.log_name, args.weights_logdir))

print('\nTraining Finished Successfully !')
