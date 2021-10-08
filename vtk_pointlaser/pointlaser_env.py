'''
OpenAI Gym environment for taking pointlaser measurements to reduce
uncertainty.
'''

# Imports
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.neighbors import KDTree
# Relative Imports
from sim_env.bayes_filter import CustomBayesFilter
from sim_env.lasers import Lasers
from sim_env.measurement import LaserMeasurements
from sim_env.mesh import Mesh
from sim_env.belief import PositionBelief
from sim_env.utils import cov2corr
from sim_env.renderer import Renderer
from sim_env.local_info import get_local_info, correct_rotations_m1, local_info_robot

# Absolute range of XYZ euler angles
EULER_RANGE = np.array([np.pi, np.pi / 2, np.pi])

# To ignore 'Box bound precision' Warning
gym.logger.set_level(40)

class PointlaserEnv(gym.Env):
    def __init__(self, args, mesh_offset_min=100, position_stddev=50):
        # Configuration arguments
        self.args = args
        # Disable rendering
        self.rendering = False
        # Load mesh
        self._mesh_nr = args.mesh_nr
        self._mesh = Mesh(args.mesh_nr, args)
        # Laser cofiguration
        self._lasers = Lasers()
        # Minimum offset from mesh boundaries to sample positions
        self._mesh_offset = np.array([mesh_offset_min, self._lasers.range])
        # Initialize belief
        self._xbel = PositionBelief()
        # Ground Truth
        self._pose_gt = {}
        self._q = Rotation.from_euler('xyz', np.zeros(3))
        # Measurements
        self._meas = LaserMeasurements(self._lasers.num)
        # Filter update
        self._bayes_filter = CustomBayesFilter(self._mesh, self._lasers)
        # Maximum standard devitation
        self._max_stddev = position_stddev
        self.renderer = Renderer()
        # Numerical range of observations:
        # Normalized XYZ postion + Standard Deviations + Correlations
        self.observation_space = spaces.Box(
            low =np.hstack((np.array([0, 0, 0, 0, 0, 0, -1, -1, -1]),
                           -np.ones(args.num_lasers*4))),
            high=np.hstack((np.array([1, 1, 1, np.inf, np.inf, np.inf, +1, +1, +1,]),
                            np.ones(args.num_lasers*4))),
            dtype=np.float)
        # Numerical range of actions:
        # Normalized rotation
        self.action_space = spaces.Box(low=np.array([-1, -1, -1]),
                                       high=np.array([1, 1, 1]),
                                       dtype=np.float)
        self._n_actions = args.n_actions
        # Weighing factor for final reward
        self._alpha = 1.
        # store values for logging
        self.volume_per_eps = []
        self.maxaxis_per_eps = []
        self.curr_obs = np.zeros(9+args.num_lasers*4)
        # Seed RNG
        self.seed()

    def _normalize_position(self, xyz):
        '''
        Linearly normalize coordinates inside mesh to range 0 and 1.
        '''
        '''max_dim = (self._mesh.max_bounds - self._mesh.min_bounds).max(axis=1)
        return (xyz - self._mesh.min_bounds) / max_dim'''
        return (xyz - self._mesh.min_bounds) / (self._mesh.max_bounds -
                                                self._mesh.min_bounds)

    def _normalize_measurement(self, z):
        '''
        Linearly normalize measurements inside mesh to range 0 and 1.
        '''
        return z/self._lasers.range

    def _normalize_cov(self, cov):
        '''
        Return flat vector containing normalized standard deviations and
        correlation coefficients calculated from the covariance matrix.
        '''
        sigma, corr = cov2corr(self._xbel.cov)
        sigma_norm = sigma / self._max_stddev
        # sigma_norm = np.clip(sigma_norm, 0, 1)
        flat_corr = corr[np.triu_indices(3, k=1)]
        return np.hstack((sigma_norm, flat_corr))

    def _get_observation(self):
        '''
        Normalize belief parameters to get observation.
        '''
        mean_norm = self._normalize_position(self._xbel.mean)
        # return mean_norm
        cov_norm = self._normalize_cov(self._xbel.cov)
        laser_dir = np.dot(self._q.as_matrix(), self._lasers._directions).T.flatten()
        return np.hstack((mean_norm, cov_norm, laser_dir))

    def _get_reward(self):
        '''
        Return immediate reward for the step.
        '''
        # Reward is the information gain (reduction in uncertainty) where
        # uncertainty is defined as the volume of covariance ellipsoid
        uncertainty = self._xbel.uncertainty('det')
        reward = self._prev_uncertainty - uncertainty
        self._prev_uncertainty = uncertainty
        if self._is_done():
            # Final reward is the reduction in the major axis length of the
            # covariance ellipsoid from the start of the episode
            final_reward = self._initial_maxeigval - self._xbel.uncertainty(
                'max_eigval')
            reward += self._alpha * final_reward
        return reward

    def _is_done(self):
        '''
        Termination condition of episode.
        '''
        uncertainty = self._xbel.uncertainty('max_eigval')
        done = (self._current_step >= self.args.max_steps)
        if self.args.var_eps_len:
            done = bool(done or (uncertainty < self.args.min_uncertainty))
        if done:
            self.store_values()
        return done

    def store_values(self):
        self.volume_per_eps.append(self._xbel.uncertainty('det'))
        self.maxaxis_per_eps.append(self._xbel.uncertainty('max_eigval'))

    @staticmethod
    def _action2rot(action):
        '''
        Get rotation from a normalized action.
        '''
        euler = action * EULER_RANGE
        return Rotation.from_euler('xyz', euler)

    def _transform_action(self, action):
        '''
        Transform to discrete actions and to euler rotations
        '''

        n = self._n_actions
        action_list = np.zeros([np.power(n+1, 3), 3])
        index = 0

        for i in range(n+1):
            ii = -1 + (2 / n) * i
            for j in range(n+1):
                jj = -1 + (2 / n) * j
                for k in range(n+1):
                    kk = -1 + (2 / n) * k
                    action_list[index, :] = (ii, jj, kk)
                    index += 1

        tree = KDTree(action_list, leaf_size=2)
        _, ind = tree.query(np.expand_dims(action, axis=0), k=1)
        action_disc = (action_list[ind]).squeeze()  # discrete action
        euler = action_disc * EULER_RANGE
        return Rotation.from_euler('xyz', euler), action_disc

    def step(self, action):
        # Increment count
        self._current_step += 1
        # Clip out of range actions
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # Transform to discrete action and rotation
        rot, action_disc = self._transform_action(action)
        # Convert to scipy Rotation object
        #rot = self._action2rot(action)
        self._q = rot
        self._pose_gt['q'] = rot
        # Take measurement from new orientation
        z = self._meas.update(self._lasers, self._pose_gt, self._mesh)
        z_norm = self._normalize_measurement(z)
        # Update belief
        self._bayes_filter.measurement_update(self._xbel, self._q, z)
        # Calculate reward
        reward = self._get_reward()
        obs = self._get_observation()
        self.curr_obs = np.hstack((obs,z_norm.squeeze()))
        done = self._is_done()
        info = {"action_disc": action_disc, "map": self._mesh._map}

        return self.curr_obs, reward, done, info

    def reset(self):
        '''
        Reset environment to start a new episode.
        '''
        self._current_step = 0
        # Reset VTK measurement visualization
        self._meas.reset()
        # Sample position
        pos = self._mesh.sample_position(self._mesh_nr, *self._mesh_offset)
        # Reset belief
        self._xbel.mean = pos
        # Initial diagonal covariance matrix
        self._xbel.cov = np.eye(3) * self._max_stddev**2
        # Store uncertainty for calculating reward
        self._initial_maxeigval = self._xbel.uncertainty('max_eigval')
        self._prev_uncertainty = self._xbel.uncertainty('det')
        # Sample a fixed ground truth position
        self._pose_gt['x'] = self._xbel.sample()
        # Reset orientation of lasers
        self._pose_gt['q'] = Rotation.from_euler('xyz', np.zeros(3))
        self._q = Rotation.from_euler('xyz', np.zeros(3))
        # Get observation from belief
        obs = self._get_observation()
        self.curr_obs = np.hstack((obs,np.zeros(self.args.num_lasers)))
        return self.curr_obs

    def render(self, renderer=None):
        if renderer == None: renderer = self.renderer
        if self.rendering == False:
            self.rendering = True
            self._mesh.renderer = renderer
            self._xbel.renderer = renderer
            self._meas.renderer = renderer
        renderer.render()

    def get_map(self):
        return self._mesh._map

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # modified functions for verification run on M1
    def reset_verification(self):
        self._current_step = 0
        self._meas.reset()
        pos = self._mesh.sample_position_verification(*self._mesh_offset)

        # get pre-defined correct rotations
        _, rotations = correct_rotations_m1(pos)

        self._xbel.mean = pos
        self._xbel.cov = np.eye(3) * self._max_stddev ** 2
        self._initial_maxeigval = self._xbel.uncertainty('max_eigval')
        self._prev_uncertainty = self._xbel.uncertainty('det')
        self._pose_gt['x'] = self._xbel.sample()
        # Reset orientation of lasers
        self._pose_gt['q'] = Rotation.from_euler('xyz', np.zeros(3))
        self._q = Rotation.from_euler('xyz', np.zeros(3))
        obs = self._get_observation()
        return obs, rotations

    def steps_verification(self, action):
        self._current_step += 1
        # Transform to rotation
        rot = self._action2rot(action)
        self._q = rot
        self._pose_gt['q'] = rot
        z = self._meas.update(self._lasers, self._pose_gt, self._mesh)
        self._bayes_filter.measurement_update(self._xbel, self._q, z)
        reward = self._get_reward()
        obs = self._get_observation()
        done = self._is_done()

        return np.hstack((obs, self._normalize_measurement(z))), reward, done

    # modified function for experiments on real robot
    def reset_robot(self, pos, cov=50):
        self._current_step = 0
        self._meas.reset()
        self._max_stddev = cov
        self._xbel.cov = np.eye(3) * self._max_stddev ** 2
        self._xbel.mean = pos * self.args.rescale  # rescale position coordinates to match mesh in simulation

        # retrieve local information. rescaling not needed since local info configured with received position values
        _, _, normal_idx, distances = local_info_robot(pos)

        # Store uncertainty for calculating reward
        self._initial_maxeigval = self._xbel.uncertainty('max_eigval')
        self._prev_uncertainty = self._xbel.uncertainty('det')

        # Sample a fixed ground truth position
        self._pose_gt['x'] = self._xbel.sample()
        # Reset orientation of lasers
        self._pose_gt['q'] = Rotation.from_euler('xyz', np.zeros(3))
        self._q = Rotation.from_euler('xyz', np.zeros(3))
        # Get observation from belief
        obs = self._get_observation()

        return np.hstack((obs,np.zeros(self.args.num_lasers))), normal_idx, distances
