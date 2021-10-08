import pybullet
import gym, gym.spaces, gym.utils
import numpy as np
from math import pi, sqrt

MAX_DISTANCE = 100


class Robot:
    """
    Base class for mujoco .xml based agents.
    """

    def __init__(self, bullet_client, add_ignored_joints=False, base_position=None, base_orientation=None, fixed_base=1):
        self._p = bullet_client

        #  Not needed?
        # self.robot_body = None
        self.add_ignored_joints = add_ignored_joints

        # Define action and observation spac
        self.action_space = gym.spaces.box.Box(
            low=np.array([-pi, -pi, -pi]),
            high=np.array([pi, pi, pi]))
        self.observation_space = gym.spaces.box.Box(
            low=np.array([0, 0, 0]),
            high=np.array([MAX_DISTANCE, MAX_DISTANCE, MAX_DISTANCE]))

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

        self.flex_joints = [self.shoulder_rot_joint.ID, self.shoulder_fle_joint.ID, self.elbow_fle_joint.ID,
                            self.elbow_rot_joint.ID, self.wrist_fle_joint.ID, self.wrist_rot_joint.ID]

        self.sensor = Sensor(self._p, self.robotID, self.transmitter.ID, self.receiver_x.ID, self.receiver_y.ID,
                             self.receiver_z.ID)

    def reset(self, end_effector_position=None, end_effector_orientation=None):
        self.endEffectorTargetPosition = end_effector_position if end_effector_position is not None else [1, 0, 1]
        o = end_effector_orientation if end_effector_orientation is not None else [0, 0, 0]
        self.endEffectorTargetOrientation = self._p.getQuaternionFromEuler([o[0], o[1], o[2]])
        target_position_joints = self._p.calculateInverseKinematics(self.robotID, self.end.ID,
                                                                    targetOrientation=self.endEffectorTargetOrientation,
                                                                    targetPosition=self.endEffectorTargetPosition,
                                                                    residualThreshold=0.0000001, maxNumIterations=1000)

        pybullet.setJointMotorControlArray(self.robotID, self.flex_joints, pybullet.POSITION_CONTROL,
                                           targetPositions=target_position_joints)

        # get observation
        s = self.get_state()  # optimization: calc_state() can calculate something in self.* for calc_potential() to use

        return s

    def apply_action(self, a):
        assert (np.isfinite(a).all())
        self.endEffectorTargetOrientation = self._p.getQuaternionFromEuler([a[0], a[1], a[2]])
        target_position_joints = self._p.calculateInverseKinematics(self.robotID, self.end.ID,
                                                                    targetOrientation=self.endEffectorTargetOrientation,
                                                                    targetPosition=self.endEffectorTargetPosition,
                                                                    residualThreshold=0.000001,
                                                                    maxNumIterations=1000000)
        pybullet.setJointMotorControlArray(self.robotID, self.flex_joints, pybullet.POSITION_CONTROL,
                                           targetPositions=target_position_joints)


        measure_successful = self.sensor.measure_successful()
        self.endEffectorPosition = self.end.get_position()
        self.endEffectorOrientation = self.end.get_orientation()

        return self.endEffectorPosition, self.endEffectorOrientation, measure_successful
        # TODO Deal with not equal config

    def get_state(self):
        real_end_effector_position = self._p.getLinkState(self.robotID, self.end.ID)[0]
        return real_end_effector_position

    def get_obs(self):
        # print(self.sensor.measure_successful())
        return self.sensor.measure()


class BodyPart:
    def __init__(self, bullet_client, body_name, robot_id, body_part_id):  # TODO why is body_name needed?
        self.robotID = robot_id
        self.body_name = body_name
        self._p = bullet_client
        self.ID = body_part_id

    def get_pose(self):
        position, orientation, _, _, _, _ = self._p.getLinkState(self.robotID, self.ID)
        return [position, orientation]

    def get_position(self):
        return self.get_pose()[0]

    def get_orientation(self):
        return self.get_pose()[1]


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

    def set_position(self, position):
        self._p.setJointMotorControl2(self.robotID, self.ID, pybullet.POSITION_CONTROL,
                                      targetPosition=position)

    def set_velocity(self, velocity):
        self._p.setJointMotorControl2(self.robotID, self.ID, pybullet.VELOCITY_CONTROL,
                                      targetVelocity=velocity)

    def set_motor_torque(self, torque):  # just some synonym method
        self.set_torque(torque)

    def set_torque(self, torque):
        self._p.setJointMotorControl2(bodyIndex=self.robotID, jointIndex=self.ID,
                                      controlMode=pybullet.TORQUE_CONTROL,
                                      force=torque)  # , positionGain=0.1, velocityGain=0.1)

    def reset_current_position(self, position, velocity):  # just some synonym method
        self.reset_position(position, velocity)

    def reset_position(self, position, velocity):
        self._p.resetJointState(self.robotID, self.ID, targetValue=position,
                                targetVelocity=velocity)
        self.disable_motor()

    def disable_motor(self):
        self._p.setJointMotorControl2(self.robotID, self.ID,
                                      controlMode=pybullet.POSITION_CONTROL, targetPosition=0, targetVelocity=0,
                                      positionGain=0.1, velocityGain=0.1, force=0)


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
