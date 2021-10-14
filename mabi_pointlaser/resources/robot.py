import pybullet
import gym, gym.spaces, gym.utils
import numpy as np
from math import pi
import time
from scipy.spatial.transform import Rotation as R

MAX_DISTANCE = 100
EULER_RANGE = np.array([np.pi, np.pi / 2, np.pi])

class Robot:
    """
    Base class for mujoco .xml based agents.
    """

    def __init__(self, bullet_client, add_ignored_joints=False, base_position=None, base_orientation=None,
                 fixed_base=1):
        self._p = bullet_client

        #  Not needed?
        # self.robot_body = None
        self.add_ignored_joints = add_ignored_joints

        # Define action space
        self.action_space = gym.spaces.box.Box(
            low=np.array([-1, -1, -1]),
            high=np.array([1, 1, 1]))

        # Robot set-up information
        self.model_urdf = "C:/Users/nilsk/Projects/MabiPointLaserEnv/mabi_pointlaser/resources" \
                          "/mabi_urdf/mabi.urdf"  # TODO use os and path
        self.basePosition = base_position if base_position is not None else [0, 0, 0]
        self.baseOrientation = base_orientation if base_orientation is not None else [0, 0, 0, 1]
        self.fixed_base = fixed_base

        self.ordered_joints = []
        self.robotID = self._p.loadURDF(self.model_urdf,
                                        basePosition=self.basePosition,
                                        baseOrientation=self.baseOrientation,
                                        useFixedBase=self.fixed_base,
                                        flags=pybullet.URDF_USE_SELF_COLLISION | pybullet.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)  # TODO what is URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS?

        self.parts = dict()
        self.jdict = dict()
        self.ordered_joints = list()

        for j in range(self._p.getNumJoints(self.robotID)):  # iterate over all joints of the robot
            self._p.setJointMotorControl2(self.robotID, j, pybullet.POSITION_CONTROL, positionGain=0.1,
                                          velocityGain=0.1, force=0)  # TODO What is setJointMotorControl2()?
            joint_info = self._p.getJointInfo(self.robotID, j)
            joint_name = joint_info[1].decode("utf8")
            part_name = joint_info[12].decode("utf8")

            # Generate body part object
            self.parts[part_name] = BodyPart(self._p, part_name, self.robotID, j)

            if joint_name[:6] == "ignore":
                ignored_joint = Joint(self._p, joint_name, self.robotID, j)
                ignored_joint.disable_motor()
                if self.add_ignored_joints:
                    self.jdict[joint_name] = ignored_joint
                    self.ordered_joints.append(ignored_joint)
                    self.jdict[joint_name].power_coef = 0.0
                continue

            if joint_name[:8] != "jointfix":
                self.jdict[joint_name] = Joint(self._p, joint_name, self.robotID, j)
                self.ordered_joints.append(self.jdict[joint_name])

                self.jdict[joint_name].power_coef = 100.0

        # parts
        self.shoulder = self.parts["SHOULDER"]
        self.arm = self.parts["ARM"]
        self.elbow = self.parts["ELBOW"]
        self.forearm = self.parts["FOREARM"]
        self.wrist_1 = self.parts["WRIST_1"]
        self.wrist_2 = self.parts["WRIST_2"]
        self.end = self.parts["EE_FORCETORQUESENSOR"]
        self.ray_x = self.parts["ray_x"]
        self.ray_y = self.parts["ray_y"]
        self.ray_z = self.parts["ray_z"]
        self.transmitter = self.parts["transmitter"]
        self.receiver_x = self.parts["receiver_x"]
        self.receiver_y = self.parts["receiver_y"]
        self.receiver_z = self.parts["receiver_z"]

        # joints
        self.shoulder_rot_joint = self.jdict["SH_ROT"]
        self.shoulder_fle_joint = self.jdict["SH_FLE"]
        self.elbow_fle_joint = self.jdict["EL_FLE"]
        self.elbow_rot_joint = self.jdict["EL_ROT"]
        self.wrist_fle_joint = self.jdict["WR_FLE"]
        self.wrist_rot_joint = self.jdict["WR_ROT"]
        self.laser_x = self.jdict["laser_x"]
        self.laser_y = self.jdict["laser_y"]
        self.laser_z = self.jdict["laser_z"]

        self.endEffectorPosition = None
        self.endEffectorOrientation = None
        self.joint_states = None
        self.target_pos = None
        self.target_q = None

        self.flex_joints = [self.shoulder_rot_joint.ID, self.shoulder_fle_joint.ID, self.elbow_fle_joint.ID,
                            self.elbow_rot_joint.ID, self.wrist_fle_joint.ID, self.wrist_rot_joint.ID]

        self.sensor = Sensor(self._p, self.robotID, self.transmitter.ID, self.receiver_x.ID, self.receiver_y.ID,
                             self.receiver_z.ID)

    def reset(self, pos=None, q=None):
        self.target_pos = pos if pos is not None else [1, 0, 1]
        q = q if q is not None else [0, 0, 0]
        self.eef_to_target(q)
        return self.get_state()

    def apply_action(self, a):
        assert (np.isfinite(a).all())
        self.eef_to_target(a)
        return self.get_state()

    def eef_to_target(self, q):
        euler = q * EULER_RANGE
        print(euler[1])
        self.target_q = self._p.getQuaternionFromEuler([euler[0], euler[1], euler[2]])
        self.joint_states = self._p.calculateInverseKinematics(self.robotID, self.transmitter.ID,
                                                               targetOrientation=self.target_q,
                                                               targetPosition=self.target_pos,
                                                               residualThreshold=0.000001,
                                                               maxNumIterations=1000000)
        pybullet.setJointMotorControlArray(self.robotID, self.flex_joints, pybullet.POSITION_CONTROL,
                                           targetPositions=self.joint_states)
        self.step_simulation()

    def get_state(self):
        end_effector_pos = self.transmitter.get_position()
        q = self._p.getEulerFromQuaternion(self.end.get_orientation())
        measure_successful = self.sensor.measure_successful()
        target = np.array(self.target_pos)
        mse = ((end_effector_pos - target) ** 2).mean()
        pos_correct = False
        if mse < 0.05:
            pos_correct = True
        return all(measure_successful) and pos_correct, q, self.joint_states

    def step_simulation(self, t_steps=100, sleep=0):
        for _ in range(t_steps):
            self._p.stepSimulation()
            time.sleep(sleep)

    def sample_pose(self):
        # TODO actually sample
        pos = [1, 0, 1]
        q = [0, 0, 0]
        return {'x': pos, 'q': q}


class BodyPart:
    def __init__(self, bullet_client, body_name, robot_id, body_part_id):  # TODO why is body_name needed?
        self.robotID = robot_id
        self.body_name = body_name
        self._p = bullet_client
        self.ID = body_part_id

    def get_pose(self):
        pos, q, _, _, _, _ = self._p.getLinkState(self.robotID, self.ID)
        return [pos, q]

    def get_position(self):
        return np.array(self.get_pose()[0])

    def get_orientation(self):
        return np.array(self.get_pose()[1])


class Joint:
    def __init__(self, bullet_client, joint_name, robot_id, joint_id):
        self.robotID = robot_id
        self._p = bullet_client
        self.ID = joint_id
        self.joint_name = joint_name

        joint_info = self._p.getJointInfo(self.robotID, self.ID)
        self.jointType = joint_info[2]
        self.lowerLimit = joint_info[8]
        self.upperLimit = joint_info[9]
        self.jointHasLimits = self.lowerLimit < self.upperLimit
        self.jointMaxVelocity = joint_info[11]

    # TODO delete unused methods!
    def set_state(self, x, vx):
        self._p.resetJointState(self.robotID, self.ID, x, vx)

    def current_position(self):  # just some synonym method
        return self.get_state()

    def current_relative_position(self):
        pos, vel = self.get_state()
        if self.jointHasLimits:
            pos_mid = 0.5 * (self.lowerLimit + self.upperLimit)
            pos = 2 * (pos - pos_mid) / (self.upperLimit - self.lowerLimit)

        if self.jointMaxVelocity > 0:
            vel /= self.jointMaxVelocity
        elif self.jointType == 0:  # JOINT_REVOLUTE_TYPE
            vel *= 0.1
        else:
            vel *= 0.5
        return (pos, vel)

    def get_state(self):
        x, vx, _, _ = self._p.getJointState(self.robotID, self.ID)
        return x, vx

    def get_position(self):
        x, _ = self.get_state()
        return x

    def get_orientation(self):
        _, r = self.get_state()
        return r

    def get_velocity(self):
        _, vx = self.get_state()
        return vx


class Sensor:
    def __init__(self, bullet_client, robot_id, transmitter_id, receiver_x_id, receiver_y_id, receiver_z_id):
        self._p = bullet_client
        self.robotID = robot_id
        self.transmitterID = transmitter_id
        self.receiver_x_id = receiver_x_id
        self.receiver_y_id = receiver_y_id
        self.receiver_z_id = receiver_z_id
        self.laser_visual = False

    def _measure(self, receiver):
        pos_transmitter = self._p.getLinkState(self.robotID, self.transmitterID)[0]
        pos_receiver = self._p.getLinkState(self.robotID, receiver)[0]
        return self._p.rayTest(pos_transmitter, pos_receiver)[0][2] * 10

    def measure(self):
        x_distance = self._measure(self.receiver_x_id)
        y_distance = self._measure(self.receiver_y_id)
        z_distance = self._measure(self.receiver_z_id)
        return [x_distance, y_distance, z_distance]

    def _measure_successful(self, receiver):
        pos_transmitter = self._p.getLinkState(self.robotID, self.transmitterID)[0]
        pos_receiver = self._p.getLinkState(self.robotID, receiver)[0]
        return self._p.rayTest(pos_transmitter, pos_receiver)[0][0] != self.robotID

    def measure_successful(self):
        # TODO How to deal with successful measurements? A) Error B) Distance from self measurements
        x_collision = self._measure_successful(self.receiver_x_id)
        y_collision = self._measure_successful(self.receiver_y_id)
        z_collision = self._measure_successful(self.receiver_z_id)
        return [x_collision, y_collision, z_collision]


# class Lasers:
#     def __init__(self, bullet_client):
#         args = Config().get_arguments()
#         self._p = bullet_client
#
#         if args.num_lasers == 1:
#             # Single pointlaser along X axis
#             self._directions = np.array([[1, 0, 0]]).T
#         elif args.num_lasers == 2:
#             # Two lasers along X, Y axes
#             self._directions = np.array([[1, 0, 0], [0, 1, 0]]).T
#         else:
#             # Three lasers along X, Y, Z axes
#             self._directions = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]).T
#
#         # Number of lasers
#         self.num = self._directions.shape[1]
#         # Range
#         self.range = 10  # (m)
#         # Measurement noise
#         self.sigma = 20.  # (mm)
#         # Initialize at origin
#         self._x = np.zeros(3)
#
#     def relative_endpoints(self, q):
#         '''
#         Get endpoints of laser rays based on given laser array orientation.
#         '''
#         rotation_matrix = q.as_matrix()
#         endpoints = self.range * np.dot(rotation_matrix, self._directions)
#         print('relative_endpoints', endpoints)
#         return endpoints
#
#     def update(self, pose, mesh):
#         '''
#         Cast laser rays from the given sensor array pose and return points of intersection.
#         Note: Returns point at max laser_range if no intersection is found.
#         '''
#         # Get endpoints of line segment along pointlaser direction
#         epoints = pose['x'][:, None] + self.relative_endpoints(pose['q'])
#         distance = np.zeros(self.num)
#         # Perform intersections for each laser
#         for i in range(self.num):
#             # intersections[:, i] = mesh.intersect(pose['x'], epoints[:, i])
#             distance[i] = self._p.rayTest(pose['x'], epoints[:, i])[0][2] * self.range
#         # Store a copy of current position
#         np.copyto(self._x, pose['x'])
#         return distance  # np.linalg.norm(intersections - pose['x'][:, None], axis=0)
#