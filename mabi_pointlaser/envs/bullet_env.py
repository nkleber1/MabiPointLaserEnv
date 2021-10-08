import gym, gym.spaces, gym.utils, gym.utils.seeding
import numpy as np
import pybullet
import pybullet_data
import time
import random
import sys
# from vtk_env.pointlaser_env import PointlaserEnv
from pybullet_utils import bullet_client
from mabi_pointlaser.envs.robot import RobotWrapper
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

        # Camera Parameter and Render  # TODO What is meant? Observation or GUI?
        self.camera = Camera()
        self.isRender = render
        self._cam_dist = 3
        self._cam_yaw = 0
        self._cam_pitch = -30
        self._render_width = 320
        self._render_height = 240

        self.setup_client()
        self.robot = RobotWrapper(self._p)
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
        timestep = 0.0020  # TODO Move to config.
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
        self._p.setPhysicsEngineParameter(fixedTimeStep=timestep * frame_skip,
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

    def render(self, mode="human"):
        if mode == "human":
            self.isRender = True

        # Ignore rgb_array for now
        if mode != "rgb_array":
            return np.array([])

        base_pos = [0, 0, 0]
        if hasattr(self, 'robot'):
            if hasattr(self.robot, 'body_xyz'):
                base_pos = self.robot.body_xyz

        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=2)
        proj_matrix = self._p.computeProjectionMatrixFOV(
            fov=60, aspect=float(self._render_width) / self._render_height,
            nearVal=0.1, farVal=100.0)
        (_, _, px, _, _) = self._p.getCameraImage(
            width=self._render_width, height=self._render_height, viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL
        )
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def close(self):
        if self.physicsClientId >= 0:
            if self.physicsClientId >= 0:
                self._p.disconnect()
        self.physicsClientId = -1

    # HUD is not defined
    # def HUD(self, state, a, done):
    #     pass

    # TODO change reward function (!)
    def get_reward(self):
        potential_old = self.potential
        self.potential = self.robot.calc_potential()

        joint_vel = np.array([
            self.robot.shoulder_pan_joint.get_velocity(),
            self.robot.shoulder_lift_joint.get_velocity(),
            self.robot.upper_arm_roll_joint.get_velocity(),
            self.robot.elbow_flex_joint.get_velocity(),
            self.robot.forearm_roll_joint.get_velocity(),
            self.robot.wrist_flex_joint.get_velocity(),
            self.robot.wrist_roll_joint.get_velocity()
        ])

        action_product = np.matmul(np.abs(a), np.abs(joint_vel))
        action_sum = np.sum(a)

        electricity_cost = (
                -0.10 * action_product  # work torque*angular_velocity
                - 0.01 * action_sum  # stall torque require some energy
        )

        stuck_joint_cost = 0
        for j in self.robot.ordered_joints:
            if np.abs(j.current_relative_position()[0]) - 1 < 0.01:
                stuck_joint_cost += -0.1

        self.rewards = [float(self.potential - potential_old), float(electricity_cost), float(stuck_joint_cost)]
        return sum(self.rewards)

    def is_doen(self):
        # TODO check if done
        return False

    def step_simulation(self, t_steps=100, sleep=0):
        for _ in range(t_steps):
            self._p.stepSimulation()
            time.sleep(sleep)

    def step(self, a, *args, **kwargs):
        self.robot.apply_action(a)
        self.step_simulation()

        self.robot.get_state()  # TODO Define!

        # obs = self.robot.get_observation()  # sets self.to_target_vec # TODO use robot observation
        obs = None

        # reward = self.get_reward()  # TODO use reward
        reward = 0

        done = self.is_doen()

        # HUD is not defined
        # self.HUD(state, a, False)

        return obs, reward, done, {}

    def calc_potential(self):
        return -100 * np.linalg.norm(self.to_target_vec)

    def camera_adjust(self):
        x, y, z = self.robot.fingertip.pose().xyz()
        x *= 0.5
        y *= 0.5
        self.camera.move_and_look_at(0.3, 0.3, 0.3, x, y, z)


class Camera:
    def __init__(self):
        pass

    def move_and_look_at(self, i, j, k, x, y, z):
        lookat = [x, y, z]
        distance = 10
        yaw = 10
        self._p.resetDebugVisualizerCamera(distance, yaw, -20, lookat)


env = BulletEnv()
env.reset()
action = env.action_space.sample()
env.step(action)
x = env._p.addUserDebugParameter("x", -pi, pi, 0)
y = env._p.addUserDebugParameter("y", -pi, pi, 0)
z = env._p.addUserDebugParameter("z", -pi, pi, 0)
while True:
    for _ in range(10):
        x1 = env._p.readUserDebugParameter(x)
        y1 = env._p.readUserDebugParameter(y)
        z1 = env._p.readUserDebugParameter(z)
        action = env.action_space.sample()
        # env.step([x1, y1, z1])
        env.step([1, 1, 1])
        # print('\r', env.robot.get_obs(), end='', flush=True)
    print('reset')
    env.reset()
