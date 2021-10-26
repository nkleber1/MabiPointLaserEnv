'''
OpenAI Gym environment for taking pointlaser measurements to reduce
uncertainty.
'''

# Imports
import gym
import os
import numpy as np
from gym import spaces
from gym.utils import seeding
from scipy.spatial.transform import Rotation
from sklearn.neighbors import KDTree
# Relative Imports
from vtk_pointlaser.bayes_filter import CustomBayesFilter
from vtk_pointlaser.lasers import Lasers
from vtk_pointlaser.measurement import LaserMeasurements
from vtk_pointlaser.mesh import Mesh
from vtk_pointlaser.belief import PositionBelief
from vtk_pointlaser.utils import cov2corr
from vtk_pointlaser.renderer import Renderer
from vtk_pointlaser.local_info import get_local_info, correct_rotations_m1, local_info_robot

# Absolute range of XYZ euler angles
EULER_RANGE = np.array([np.pi, np.pi / 2, np.pi])

# To ignore 'Box bound precision' Warning
gym.logger.set_level(40)  # TODO Whats that?


class RandomizedPointlaserEnv():
    def __init__(self, args):
        # Configuration arguments
        self.args = args
        # Load mesh
        self._mesh_nr = 0
        # Initialize Laser cofiguration
        self._lasers = Lasers()
        # Initialize belief class
        self._xbel = PositionBelief()
        # Initialize Ground Truth dict
        self._pose_gt = {}
        self._q = Rotation.from_euler('xyz', np.zeros(3))
        # Initialize Measurements class
        self._measurements = LaserMeasurements(self._lasers.num)
        # Initialize Filter update class
        self._bayes_filter = CustomBayesFilter(self._lasers)
        # Maximum standard devitation
        self._max_stddev = args.position_stddev
        # Minimum offset from mesh boundaries to sample positions
        self._mesh_offset = np.array([args.mesh_offset_min, self._lasers.range])
        # Disable rendering
        self.rendering = False
        self.renderer = Renderer()

        # Numerical range of observations:
        # [Normalized XYZ postion, Standard Deviations, Correlations, Observations, Map Encoding, map_height]
        low = np.array([0, 0, 0, -1, -1, -1])
        high = np.array([np.inf, np.inf, np.inf, +1, +1, +1,])
        if args.use_mean_pos:
            low = np.hstack((low, np.zeros(3)))
            high = np.hstack((high, np.ones(3)))
        if args.use_measurements:
            low = np.hstack((low, -np.ones(args.num_lasers*4)))
            high = np.hstack((high, np.ones(args.num_lasers*4)))
        if args.use_map_encodings:
            low = np.hstack((low, -np.inf*np.ones(args.vae_latent_dims)))
            high = np.hstack((high, np.inf*np.ones(args.vae_latent_dims)))
        if args.use_map_height:
            low = np.hstack((low, np.zeros(1)))
            high = np.hstack((high, np.inf*np.ones(1)))
        self.observation_space_low = low
        self.observation_space_high = high

        # Numerical range of actions:
        # Normalized rotation
        self.action_space = spaces.Box(low=np.array([-1, -1, -1]),
                                       high=np.array([1, 1, 1]),
                                       dtype=np.float)
        self._n_actions = args.n_actions
        # Weighing factor for final reward
        self._alpha = args.rew_alpha
        # store values for logging
        self.volume_per_eps = []
        self.maxaxis_per_eps = []
        self.gt_dist = []
        # Seed RNG
        self.np_random = None

    def _normalize_position(self, xyz):
        '''
        Linearly normalize coordinates inside mesh to range 0 and 1.
        '''
        max_dim = (self._mesh.max_bounds - self._mesh.min_bounds).max()
        return (xyz - self._mesh.min_bounds) / max_dim
        #return (xyz - self._mesh.min_bounds) / (self._mesh.max_bounds - self._mesh.min_bounds)

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
        obs = self._normalize_cov(self._xbel.cov)
        mean_norm = self._normalize_position(self._xbel.mean) # return mean_norm
        laser_dir = np.dot(self._q.as_matrix(), self._lasers._directions).T.flatten()
        if self.args.use_mean_pos: obs = np.hstack((obs, mean_norm))
        if self.args.use_measurements: obs = np.hstack((obs, laser_dir))

        return obs

    def _get_reward(self):
        '''
        Return immediate reward for the step.
        '''
        # Reward is the information gain (reduction in uncertainty) where
        # uncertainty is defined as the volume of covariance ellipsoid
        uncertainty = self._xbel.uncertainty('det')
        max_eigval = self._xbel.uncertainty('max_eigval')
        dist = np.linalg.norm(self._xbel.mean-self._pose_gt['x'])
        reward = self.args.use_uncert_rew*self.args.uncert_rew_w*(self._prev_uncertainty - uncertainty) + self.args.use_dist_rew*self.args.dist_rew_w*(self._prev_dist - dist) + self.args.use_eigval_rew*self.args.eigval_rew_w*(self._prev_max_eigval - max_eigval) - self.args.meas_cost
        self._prev_max_eigval = max_eigval
        self._prev_uncertainty = uncertainty
        self._prev_dist = dist
        if self._is_done():
            # Final reward is the reduction in the major axis length of the
            # covariance ellipsoid from the start of the episode
            final_reward = self._initial_maxeigval - self._xbel.uncertainty(
                'max_eigval')
            reward += self._alpha * final_reward
        if self.args.use_goal_rew and bool((uncertainty < self.args.min_uncertainty)):
            reward += self.args.goal_rew
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
        self.gt_dist.append(np.linalg.norm(self._xbel.mean-self._pose_gt['x']))

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

    def step(self, action, pos_noise=None):
        # Increment count
        self._current_step += 1
        # Transform to discrete action and rotation
        # rot, action_disc = self._transform_action(action)
        rot = Rotation.from_euler('xyz', action)
        # Convert to scipy Rotation object
        #rot = self._action2rot(action)
        self._q = rot
        self._pose_gt['q'] = rot
        # add noise to self._pose_gt['q']
        # pose = self._pose_gt  # TODO change gt instead since the true position of the robotic arm is changing.
        if pos_noise is not None:
            self._pose_gt['x'] += pos_noise
        # Take measurement from new orientation
        z = self._measurements.update(self._lasers, pose, self._mesh)
        z_norm = self._normalize_measurement(z)
        # Update belief
        self._bayes_filter.measurement_update(self._xbel, self._q, z)
        # Calculate reward
        reward = self._get_reward()
        obs = self._get_observation()
        if self.args.use_measurements: obs = np.hstack((obs, z_norm.squeeze()))
        if self.args.use_map_encodings: obs = np.hstack((obs, self._curr_map_encoding))
        if self.args.use_map_height: obs = np.hstack((obs, self._curr_mesh_h))
        self.curr_obs = np.hstack((obs,z_norm.squeeze(),self._curr_map_encoding,self._curr_mesh_h))
        done = self._is_done()
        info = {"action": action}

        return obs, reward, done, info

    def reset(self, h, q):
        '''
        Reset environment to start a new episode.
        :param h: height of the sensor (from pyBullet Simulation)
        :param q: starting orientation of the sensor (from pyBullet Simulation)
        :return: obs
        '''
        self._current_step = 0
        # Reset VTK measurement visualization
        self._measurements.reset()

        # Import new mesh
        num_meshes = len(os.listdir(os.path.join(self.args.dataset_dir, self.args.mesh_dir)))
        self._mesh_nr = 1 + (self._mesh_nr%num_meshes)
        self._mesh = Mesh(self._mesh_nr, self.args)
        self._curr_map_encoding = self._mesh._map_encoding
        self._curr_mesh_h = np.asarray([(self._mesh.max_bounds-self._mesh.min_bounds)[2]]).reshape(1)
        self._bayes_filter._mesh = self._mesh

        # Sample position
        pos = self._mesh.sample_position(*self._mesh_offset)
        # us height from pyBullet
        pos[2] = h * 1000  # m to mm
        # Reset belief
        self._xbel.mean = pos
        # Initial diagonal covariance matrix
        self._xbel.cov = np.eye(3) * self._max_stddev**2
        # Store uncertainty for calculating reward
        self._initial_maxeigval = self._xbel.uncertainty('max_eigval')
        self._prev_uncertainty = self._xbel.uncertainty('det')
        self._prev_max_eigval = self._initial_maxeigval
        # Sample a fixed ground truth position
        self._pose_gt['x'] = self._xbel.sample()
        self._prev_dist = np.linalg.norm(self._xbel.mean-self._pose_gt['x'])
        # Reset orientation of lasers
        self._pose_gt['q'] = Rotation.from_euler('xyz', q)
        self._q = Rotation.from_euler('xyz', q)
        # Get observation from belief
        obs = self._get_observation()
        if self.args.use_measurements: obs = np.hstack((obs, np.zeros(self.args.num_lasers)))
        if self.args.use_map_encodings: obs = np.hstack((obs, self._curr_map_encoding))
        if self.args.use_map_height: obs = np.hstack((obs, self._curr_mesh_h))
        self.curr_obs = np.hstack((obs, np.zeros(self.args.num_lasers), self._curr_map_encoding, self._curr_mesh_h))
        return obs

    def render(self, renderer=None, reset=False):
        if renderer == None: renderer = self.renderer
        if self.rendering == False:
            self.rendering = True
            reset = True
        if reset == True:
            self._mesh.renderer = renderer
            self._xbel.renderer = renderer
            self._measurements.renderer = renderer
        renderer.render()

    def get_map(self):
        return self._mesh._map

    # modified functions for verification run on M1
    def reset_verification(self, h, q):
        self._current_step = 0
        self._measurements.reset()
        pos = self._mesh.sample_position_verification(*self._mesh_offset)
        pos[2] = h

        # get pre-defined correct rotations
        _, rotations = correct_rotations_m1(pos)

        self._xbel.mean = pos
        self._xbel.cov = np.eye(3) * self._max_stddev ** 2
        self._initial_maxeigval = self._xbel.uncertainty('max_eigval')
        self._prev_uncertainty = self._xbel.uncertainty('det')
        self._pose_gt['x'] = self._xbel.sample()
        # Reset orientation of lasers
        self._pose_gt['q'] = Rotation.from_euler('xyz', q)
        self._q = Rotation.from_euler('xyz', q)
        obs = self._get_observation()
        return obs, rotations

    def steps_verification(self, action):
        self._current_step += 1
        # Transform to rotation
        rot = self._action2rot(action)
        self._q = rot
        self._pose_gt['q'] = rot
        z = self._measurements.update(self._lasers, self._pose_gt, self._mesh)
        self._bayes_filter.measurement_update(self._xbel, self._q, z)
        reward = self._get_reward()
        obs = self._get_observation()
        done = self._is_done()

        return np.hstack((obs, self._normalize_measurement(z))), reward, done

    # modified function for experiments on real robot
    def reset_robot(self, pos, q, cov=50):
        self._current_step = 0
        self._measurements.reset()
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
        self._pose_gt['q'] = Rotation.from_euler('xyz', q)
        self._q = Rotation.from_euler('xyz', q)
        # Get observation from belief
        obs = self._get_observation()

        return np.hstack((obs,np.zeros(self.args.num_lasers))), normal_idx, distances
