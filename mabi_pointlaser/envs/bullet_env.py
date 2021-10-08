import gym, gym.spaces, gym.utils, gym.utils.seeding
import pybullet
import pybullet_data
import time
# from vtk_env.pointlaser_env import PointlaserEnv
from pybullet_utils import bullet_client
from mabi_pointlaser.resources.robot import Robot
from math import pi


class BulletEnv(gym.Env):
    """
	Base class for Bullet physics simulation environments in a Scene.
	These environments create single-player scenes and behave like normal Gym environments, if
	you don't use multiplayer.
	"""

    def __init__(self, render=True):
        # Load all Elements
        self.physicsClientId = -1
        self._p = None

        self.np_random = None

        self.frame = 0
        self.done = 0
        self.reward = 0

        self.isRender = render
        self.setup_client()
        self.robot = Robot(self._p)
        self.seed()

        self.action_space = self.robot.action_space
        self.observation_space = self.robot.observation_space

    # do robots have args?
    # def configure(self, args):
    #     self.robot.args = args

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.robot.np_random = self.np_random  # use the same np_randomizer for robot as for env
        return [seed]

    def setup_client(self):
        time_step = 0.0020  # TODO Move to config.
        frame_skip = 5
        num_solver_iterations = 5

        if self.isRender:
            self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._p = bullet_client.BulletClient()
        self.physicsClientId = self._p._client
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)  # TODO What happens here?
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._p.setGravity(0, 0, -9.81)
        self._p.loadURDF("plane.urdf")
        self._p.setDefaultContactERP(0.9)  # TODO What is setDefaultContactERP
        # TODO What is setPhysicsEngineParameter
        self._p.setPhysicsEngineParameter(fixedTimeStep=time_step * frame_skip,
                                          numSolverIterations=num_solver_iterations, numSubSteps=frame_skip)

    def reset(self):
        # Connect to Bullet-Client (if no client is connected)
        if self.physicsClientId < 0:
            self.setup_client()

        self.frame = 0
        self.done = 0
        self.reward = 0

        s = self.robot.reset()  # TODO whats s and what is it good for? --> it is a state / obs
        self.step_simulation()
        return s

    def close(self):
        if self.physicsClientId >= 0:
            if self.physicsClientId >= 0:
                self._p.disconnect()
        self.physicsClientId = -1

    # TODO change reward function (!)
    def get_reward(self):
        pass

    # TODO change done function (!)
    def is_doen(self):
        pass

    def step_simulation(self, t_steps=100, sleep=0):
        for _ in range(t_steps):
            self._p.stepSimulation()
            time.sleep(sleep)

    def step(self, a, *args, **kwargs):
        self.robot.apply_action(a)
        self.step_simulation()

        state = self.robot.get_state()

        # obs = self.robot.get_observation()  # sets self.to_target_vec # TODO use robot observation
        obs = None

        # reward = self.get_reward()  # TODO use reward
        reward = 0

        done = self.is_doen()

        return obs, reward, done, {}


env = BulletEnv()
env.reset()
action = env.action_space.sample()
env.step(action)
x = env._p.addUserDebugParameter("x", -pi, pi, 0)
y = env._p.addUserDebugParameter("y", -pi, pi, 0)
z = env._p.addUserDebugParameter("z", -pi, pi, 0)
while True:
    for _ in range(100):
        x1 = env._p.readUserDebugParameter(x)
        y1 = env._p.readUserDebugParameter(y)
        z1 = env._p.readUserDebugParameter(z)
        action = env.action_space.sample()
        # env.step([x1, y1, z1])
        env.step([1, 1, 1])
        # print('\r', env.robot.get_obs(), end='', flush=True)
    env.reset()
