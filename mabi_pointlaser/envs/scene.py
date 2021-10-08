import os
import sys
import time
import pybullet_data

sys.path.append(os.path.dirname(__file__))

import gym


class Scene:
    """A base class for single- and multiplayer scenes"""

    def __init__(self, bullet_client, timestep, frame_skip):
        self._p = bullet_client
        self.np_random, seed = gym.utils.seeding.np_random(None)

        # Never used?
        # self.dt = timestep * frame_skip  # TODO What is frame_skipp? -> Time delta?
        self.cpp_world = World(self._p, 9.81, timestep, frame_skip)

        self.test_window_still_open = True  # or never opened
        self.human_render_detected = False  # if user wants render("human"), we open test window
        self.laser = None

    def test_window(self):
        "Call this function every frame, to see what's going on. Not necessary in learning."
        self.human_render_detected = True
        return self.test_window_still_open

    def episode_restart(self, bullet_client):
        "This function gets overridden by specific scene, to reset specific objects into their start positions"
        self.cpp_world.clean_everything()
        #self.cpp_world.test_window_history_reset()

    def global_step(self):
        """
        The idea is: apply motor torques for all robots, then call global_step(), then collect
        observations from robots using step() with the same action.
        """
        self.cpp_world.step()



class World:

    def __init__(self, bullet_client, gravity, timestep, frame_skip):
        self._p = bullet_client
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.gravity = gravity
        self.timestep = timestep  # TODO What is timestep?
        self.frame_skip = frame_skip  # TODO What is frame_skipp?
        self.numSolverIterations = 5  # TODO What is numSolverIterations
        self.clean_everything()

    def clean_everything(self):
        # p.resetSimulation()
        self._p.setGravity(0, 0, -self.gravity)
        self._p.loadURDF("plane.urdf")
        #self._p.loadURDF('cube.urdf', basePosition=[-6, -6, 0], globalScaling=12, useFixedBase=1)
        self._p.setDefaultContactERP(0.9)  # TODO What is setDefaultContactERP
        # print("self.numSolverIterations=",self.numSolverIterations)
        # TODO What is setPhysicsEngineParameter
        self._p.setPhysicsEngineParameter(fixedTimeStep=self.timestep*self.frame_skip, numSolverIterations=self.numSolverIterations, numSubSteps=self.frame_skip)

    def step(self, t_steps=100, sleep=0):
        for _ in range(t_steps):
            self._p.stepSimulation()
            time.sleep(sleep)