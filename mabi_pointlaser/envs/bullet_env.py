import gym, gym.spaces, gym.utils, gym.utils.seeding
import pybullet
import pybullet_data
from vtk_pointlaser.config import Config
from vtk_pointlaser import RandomizedPointlaserEnv
from pybullet_utils import bullet_client
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

        # init bullet
        self.physicsClientId = -1
        self._p = None
        self.isRender = render
        self.setup_client()

        # make robot
        self.robot = Robot(self._p)
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

    def setup_client(self):
        '''
        Set up BulletClient, Gravity, Plane  and setts PhysicsEngineParameter and debugging features.
        '''
        if self.isRender:
            self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._p = bullet_client.BulletClient()
        self.physicsClientId = self._p._client
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._p.setGravity(0, 0, -9.81)
        self._p.loadURDF("plane.urdf")
        self._p.setDefaultContactERP(0.9)
        self._p.setPhysicsEngineParameter(fixedTimeStep=self.args.time_step * self.args.frame_skip,
                                          numSolverIterations=self.args.num_solver_iterations,
                                          numSubSteps=self.args.frame_skip)

    def reset(self):
        '''
        Starts a new episode, withe random robot pose.
        :return: first obs (by vtk env)
        '''

        # Connect to Bullet-Client (if no client is connected)
        if self.physicsClientId < 0:
            self.setup_client()

        # Set Episode info to zero
        self.episode_steps = 0
        self.episode_reward = 0

        # sample new robot pose and apply if possible.
        while True:
            pose = self.robot.sample_pose()
            correct, q, joint_states = self.robot.reset(pose['x'], pose['q'])

            if correct:
                obs = self.vtk_env.reset(pose['x'][2], q)
                return obs

    def close(self):
        '''
        Disconnect client
        '''
        if self.physicsClientId >= 0:
            if self.physicsClientId >= 0:
                self._p.disconnect()
        self.physicsClientId = -1

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
action = env.action_space.sample()
env.step(action)
x = env._p.addUserDebugParameter("x", -1, 1, 0)
y = env._p.addUserDebugParameter("y", -1, 1, 0)
z = env._p.addUserDebugParameter("z", -1, 1, 0)
while True:
    x1 = env._p.readUserDebugParameter(x)
    y1 = env._p.readUserDebugParameter(y)
    z1 = env._p.readUserDebugParameter(z)
    action = env.action_space.sample()
    env.step([x1, y1, z1])
    # print(env.robot.sensor.measure())
