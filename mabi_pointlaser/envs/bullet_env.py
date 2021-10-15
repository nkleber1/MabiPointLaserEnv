import gym, gym.spaces, gym.utils, gym.utils.seeding
from vtk_pointlaser.config import Config
from vtk_pointlaser import RandomizedPointlaserEnv
from mabi_pointlaser.resources.robot import Robot


class BulletEnv(gym.Env):
    """
	Base class for Bullet physics simulation environments in a Scene.
	These environments create single-player scenes and behave like normal Gym environments, if
	you don't use multiplayer.
	"""
    def __init__(self, render=True):
        # get args
        self.args = Config().get_arguments()

        # init render
        self.isRender = render

        # make robot
        self.robot = Robot(self.args)
        self.action_space = self.robot.action_space

        # make vtk
        self.vtk_env = RandomizedPointlaserEnv(self.args)
        self.observation_space = self.vtk_env.observation_space

        # same seed for robot, bullet_env and vtk_env
        self.np_random = None
        self.seed(self.args.seed)

        # episode history
        self.episode_steps = 0
        self.episode_reward = 0

    def seed(self, seed=None):
        '''
        Set random seed for BulletEnv and MabiRobot
        '''
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.robot.np_random = self.np_random
        self.vtk_env.np_random = self.np_random
        return [seed]

    def reset(self):
        '''
        Starts a new episode, withe random robot pose.
        :return: first obs (by vtk env)
        '''
        # Set Episode info to zero
        self.episode_steps = 0
        self.episode_reward = 0

        # sample new robot pose and apply if possible.
        while True:
            pose = self.robot.sample_pose()
            correct, q, joint_states = self.robot.reset(pose['x'], pose['q'])

            if correct:
                h_mm = pose['x'][2]
                obs = self.vtk_env.reset(h_mm, q)
                return obs

    def close(self):
        '''
        Disconnect client
        '''
        self.robot.close()

    def step(self, a):
        '''
        Perform a new step. (1) Check if action is possible under robot dynamics. (2) If, so hand action over to vtk_env
        :param a: action 3*[-1, 1] (new orientation in normed euler angels)
        :return: obs, reward, done, info by vtk_env (or punishment if action is impossible)
        '''
        self.episode_steps += 1
        correct, q, joint_states = self.robot.apply_action(a)
        if not correct:
            return None, self.args.incorrect_pose_punishment, True, {}

        obs, reward, done, info = self.vtk_env.step(q)
        self.episode_reward += reward
        return obs, reward, done, info


env = BulletEnv()
env.reset()
env.robot.add_debug_parameter()

while True:
    x, y, z = env.robot.get_debug_parameter()
    env.step([x, y, z])

