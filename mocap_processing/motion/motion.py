import numpy as np

from basecode.utils import basics
from basecode.math import mmMath

from mocap_processing.utils import constants
from mocap_processing.utils import utils


class Joint(object):
    def __init__(self, name="joint", dof=3):
        self.name = name
        self.parent_joint = None
        self.child_joint = []
        self.index_child_joint = {}
        self.xform_global = mmMath.I_SE3()
        self.xform_from_parent_joint = mmMath.I_SE3()
        self.info = {"dof": dof}  # set ball joint by default

    def get_child_joint(self, key):
        return self.child_joint[utils.get_index(self.index_child_joint, key)]

    def get_child_joint_recursive(self):
        """
        This could have duplicated joints if there exists loops in the chain
        """
        joints = []
        for j in self.child_joint:
            joints.append(j)
            joints += j.get_all_child_joint()
        return joints

    def add_child_joint(self, joint):
        assert isinstance(joint, Joint)
        assert joint.name not in self.index_child_joint.keys()
        self.index_child_joint[joint.name] = len(self.child_joint)
        self.child_joint.append(joint)
        joint.set_parent_joint(self)

    def set_parent_joint(self, joint):
        assert isinstance(joint, Joint)
        self.parent_joint = joint
        self.xform_global = np.dot(
            self.parent_joint.xform_global, self.xform_from_parent_joint,
        )


class Skeleton(object):
    def __init__(
        self,
        name="skeleton",
        v_up=np.array([0.0, 1.0, 0.0]),
        v_face=np.array([0.0, 0.0, 1.0]),
        v_up_env=np.array([0.0, 1.0, 0.0]),
    ):
        self.name = name
        self.joints = []
        self.index_joint = {}
        self.root_joint = None
        self.num_dofs = 0
        self.v_up = v_up
        self.v_face = v_face
        self.v_up_env = v_up_env

    def num_joint(self):
        return len(self.joints)

    def get_index_joint(self, key):
        return utils.get_index(self.index_joint, key)

    def get_joint(self, key):
        return self.joints[self.get_index_joint(key)]

    def add_joint(self, joint, parent_joint):
        if parent_joint is None:
            assert self.num_joint() == 0
            self.root_joint = joint
        else:
            parent_joint = self.get_joint(parent_joint)
            parent_joint.add_child_joint(joint)
        self.index_joint[joint.name] = len(self.joints)
        self.joints.append(joint)
        self.num_dofs += joint.info["dof"]

    def num_joints(self):
        return len(self.joints)


class Pose(object):
    def __init__(self, skel, data=None):
        """
        Construct Pose for a given skeleton and pose data.
        Pose data must be provided with an np.array of shape (num_joints, 4, 4)
        """
        assert isinstance(skel, Skeleton)
        if data is None:
            data = [constants.eye_T for _ in range(skel.num_joint())]
        assert skel.num_joint() == len(data)
        self.skel = skel
        self.data = data

    def get_transform(self, key, local):
        skel = self.skel
        if local:
            return self.data[skel.get_index_joint(key)]
        else:
            joint = skel.get_joint(key)
            T = np.dot(
                joint.xform_from_parent_joint, self.data[skel.get_index_joint(joint)]
            )
            while joint.parent_joint is not None:
                T_j = np.dot(
                    joint.parent_joint.xform_from_parent_joint,
                    self.data[skel.get_index_joint(joint.parent_joint)],
                )
                T = np.dot(T_j, T)
                joint = joint.parent_joint
            return T

    def set_transform(self, key, T, local, do_ortho_norm=True):
        if local:
            T1 = T
        else:
            T0 = self.skel.get_joint(key).xform_global
            T1 = np.dot(mmMath.invertSE3(T0), T)
        if do_ortho_norm:
            """
            This insures that the rotation part of
            the given transformation is valid
            """
            Q, p = mmMath.T2Qp(T1)
            Q = mmMath.post_process_Q(Q, normalize=True, half_space=False)
            T1 = mmMath.Qp2T(Q, p)
        self.data[self.skel.get_index_joint(key)] = T1

    def get_root_transform(self):
        root_idx = self.skel.get_index_joint(self.skel.root_joint)
        return self.get_transform(root_idx, local=False)

    def set_root_transform(self, T, local):
        root_idx = self.skel.get_index_joint(self.skel.root_joint)
        self.set_transform(root_idx, T, local)

    def get_facing_transform(self):
        d, p = self.get_facing_direction_position()
        z = d
        y = self.skel.v_up_env
        x = np.cross(y, z)
        return mmMath.Rp2T(np.array([x, y, z]).transpose(), p)

    def get_facing_position(self):
        d, p = self.get_facing_direction_position()
        return p

    def get_facing_direction(self):
        d, p = self.get_facing_direction_position()
        return d

    def get_facing_direction_position(self):
        R, p = mmMath.T2Rp(self.get_root_transform())
        d = np.dot(R, self.skel.v_face)
        d = d - mmMath.projectionOnVector(d, self.skel.v_up_env)
        p = p - mmMath.projectionOnVector(p, self.skel.v_up_env)
        return d / np.linalg.norm(d), p

    def to_matrix(self, local=True):
        """
        Returns pose data in transformation matrix format, with shape
        (num_joints, 4, 4)
        """
        transforms = []
        for joint in self.skel.joints:
            transforms.append(self.get_transform(joint, local))
        return np.array(transforms)

    @classmethod
    def from_matrix(cls, data, skel, local=True):
        """
        Expects pose data in transformation matrix format, with shape
        (num_joints, 4, 4)
        """
        num_joints, T_0, T_1 = data.shape
        assert num_joints == skel.num_joints(), (
            "Data for all joints not provided"
        )
        assert T_0 == 4 and T_1 == 4, (
            "Data not provided in 4x4 transformation matrix format. Use "
            "mocap_processing.utils.constants.eye_T for template identity "
            "matrix"
        )
        pose = cls(skel)
        for joint_id in range(len(skel.joints)):
            pose.set_transform(joint_id, data[joint_id], local)
        return pose


def interpolate_pose(alpha, pose1, pose2):
    skel = pose1.skel
    data = []
    for j in skel.joints:
        R1, p1 = mmMath.T2Rp(pose1.get_transform(j, local=True))
        R2, p2 = mmMath.T2Rp(pose2.get_transform(j, local=True))
        R, p = (
            mmMath.slerp(R1, R2, alpha), 
            mmMath.linearInterpol(p1, p2, alpha),
        )
        data.append(mmMath.Rp2T(R, p))
    return Pose(pose1.skel, data)


class Motion(object):
    def __init__(
        self, name="motion", skel=None,
    ):
        self.name = name
        self.skel = skel
        self.times = []
        self.poses = []
        self.info = {}

    def clear(self):
        self.times = []
        self.poses = []
        self.info = {}

    def set_skeleton(self, skel):
        self.skel = skel

    def add_one_frame(self, t, pose_data):
        self.times.append(t)
        self.poses.append(Pose(self.skel, pose_data))

    def frame_to_time(self, frame):
        return self.times[frame]

    def time_to_frame(self, time):
        return basics.bisect(self.times, time)

    def get_time_by_frame(self, frame):
        assert frame < self.num_frames()
        return self.times[frame]

    def get_pose_by_frame(self, frame):
        assert frame < self.num_frames()
        return self.poses[frame]

    def get_pose_by_time(self, time):
        time = basics.clamp(time, self.times[0], self.times[-1])
        frame1 = self.time_to_frame(time)
        frame2 = min(frame1 + 1, self.num_frames() - 1)
        if frame1 == frame2:
            return self.poses[frame1]
        t1 = self.times[frame1]
        t2 = self.times[frame2]
        if time == t1:
            return self.poses[frame1]
        if time == t2:
            return self.poses[frame2]
        assert (t2 - t1) > 1.0e-5
        alpha = basics.clamp((time - t1) / (t2 - t1), 0.0, 1.0)
        return interpolate_pose(alpha, self.poses[frame1], self.poses[frame2])

    def num_frames(self):
        return len(self.times)

    def length(self):
        return self.times[-1] - self.times[0]

    def to_matrix(self, local=True):
        """
        Returns pose data in transformation matrix format, with shape
        (seq_len, num_joints, 4, 4)
        """
        data = []
        for pose in self.poses:
            data.append(pose.to_matrix(local))
        return np.array(data)

    def rotations(self, local=True):
        """
        Returns joint rotations in rotation matrix format, with shape
        (seq_len, num_joints, 3, 3)
        """
        return self.to_matrix(local)[..., :3, :3]

    def positions(self, local=True):
        """
        Returns joint positions with shape (seq_len, num_joints, 3)
        """
        return self.to_matrix(local)[..., :3, 3]

    @classmethod
    def from_matrix(cls, data, skel, local=True, fps=None):
        """
        Expects pose data in transformation matrix format, with shape
        (seq_len, num_joints, 4, 4)
        """
        assert data.ndim == 4, (
            "Data must be provided in transformation matrix format, with shape"
            " (seq_len, num_joints, 4, 4)"
        )
        seq_len, num_joints, T_0, T_1 = data.shape
        assert num_joints == skel.num_joints(), (
            "Data for all joints not provided"
        )
        assert T_0 == 4 and T_1 == 4, (
            "Data not provided in 4x4 transformation matrix format. Use "
            "mocap_processing.utils.constants.eye_T for template identity "
            "matrix"
        )
        if fps is None:
            fps = 60.0
        motion = cls(skel)
        motion.times = (1 / fps) * np.arange(seq_len)
        for pose_data in data:
            pose = Pose.from_matrix(pose_data, skel, local)
            motion.poses.append(pose)
        return motion