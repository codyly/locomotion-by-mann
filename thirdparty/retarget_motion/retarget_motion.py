"""Run from motion_imitation/retarget_motion to find data correctly."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

import numpy as np

import pybullet
from pybullet_utils import transformations
from thirdparty.motion_imitation.utilities import pose3d
from thirdparty.motion_imitation.utilities import motion_util
from thirdparty.retarget_motion import retarget_config_a1 as config


POS_SIZE = 3
ROT_SIZE = 4
DEFAULT_ROT = np.array([0, 0, 0, 1])
FORWARD_DIR = np.array([1, 0, 0])

GROUND_URDF_FILENAME = "a1.urdf"

# reference motion
FRAME_DURATION = 0.01667
REF_COORD_ROT = transformations.quaternion_from_euler(0.5 * np.pi, 0, 0)
REF_POS_OFFSET = np.array([0, 0, 0])
REF_ROOT_ROT = transformations.quaternion_from_euler(0, 0, 0.47 * np.pi)

REF_PELVIS_JOINT_ID = 0
REF_NECK_JOINT_ID = 3
REF_HIP_JOINT_IDS = [6, 16, 11, 20]
REF_TOE_JOINT_IDS = [10, 19, 15, 23]


def build_markers(num_markers):
    marker_radius = 0.02

    markers = []
    for i in range(num_markers):
        if (i == REF_NECK_JOINT_ID) or (i == REF_PELVIS_JOINT_ID) or (i in REF_HIP_JOINT_IDS):
            col = [0, 0, 1, 1]
        elif i in REF_TOE_JOINT_IDS:
            col = [1, 0, 0, 1]
        else:
            col = [0, 1, 0, 1]

        virtual_shape_id = pybullet.createVisualShape(
            shapeType=pybullet.GEOM_SPHERE, radius=marker_radius, rgbaColor=col
        )
        body_id = pybullet.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=virtual_shape_id,
            basePosition=[0, 0, 0],
            useMaximalCoordinates=True,
        )
        markers.append(body_id)

    return markers


def get_joint_limits(robot):
    num_joints = pybullet.getNumJoints(robot)
    joint_limit_low = []
    joint_limit_high = []

    for i in range(num_joints):
        joint_info = pybullet.getJointInfo(robot, i)
        joint_type = joint_info[2]

        if joint_type == pybullet.JOINT_PRISMATIC or joint_type == pybullet.JOINT_REVOLUTE:
            joint_limit_low.append(joint_info[8])
            joint_limit_high.append(joint_info[9])

    return joint_limit_low, joint_limit_high


def get_root_pos(pose):
    return pose[0:POS_SIZE]


def get_root_rot(pose):
    return pose[POS_SIZE : (POS_SIZE + ROT_SIZE)]


def get_joint_pose(pose):
    return pose[(POS_SIZE + ROT_SIZE) :]


def set_root_pos(root_pos, pose):
    pose[0:POS_SIZE] = root_pos
    return


def set_root_rot(root_rot, pose):
    pose[POS_SIZE : (POS_SIZE + ROT_SIZE)] = root_rot
    return


def set_joint_pose(joint_pose, pose):
    pose[(POS_SIZE + ROT_SIZE) :] = joint_pose
    return


def set_pose(robot, pose):
    num_joints = pybullet.getNumJoints(robot)
    root_pos = get_root_pos(pose)
    root_rot = get_root_rot(pose)
    pybullet.resetBasePositionAndOrientation(robot, root_pos, root_rot)

    for j in range(num_joints):
        j_info = pybullet.getJointInfo(robot, j)
        j_state = pybullet.getJointStateMultiDof(robot, j)

        j_pose_idx = j_info[3]
        j_pose_size = len(j_state[0])
        j_vel_size = len(j_state[1])

        if j_pose_size > 0:
            j_pose = pose[j_pose_idx : (j_pose_idx + j_pose_size)]
            j_vel = np.zeros(j_vel_size)
            pybullet.resetJointStateMultiDof(robot, j, j_pose, j_vel)

    return


def set_maker_pos(marker_pos, marker_ids):
    num_markers = len(marker_ids)
    assert num_markers == marker_pos.shape[0]

    for i in range(num_markers):
        curr_id = marker_ids[i]
        curr_pos = marker_pos[i]

        pybullet.resetBasePositionAndOrientation(curr_id, curr_pos, DEFAULT_ROT)

    return


def process_ref_joint_pos_data(joint_pos):
    proc_pos = joint_pos.copy()
    num_pos = joint_pos.shape[0]

    for i in range(num_pos):
        curr_pos = proc_pos[i]
        curr_pos = pose3d.QuaternionRotatePoint(curr_pos, REF_COORD_ROT)
        curr_pos = pose3d.QuaternionRotatePoint(curr_pos, REF_ROOT_ROT)
        curr_pos = curr_pos * config.REF_POS_SCALE + REF_POS_OFFSET
        proc_pos[i] = curr_pos

    return proc_pos


def retarget_root_pose(ref_joint_pos, style=None):

    trunk_len = 0.267

    pelvis_pos = ref_joint_pos[REF_PELVIS_JOINT_ID]
    neck_pos = ref_joint_pos[REF_NECK_JOINT_ID]

    left_shoulder_pos = ref_joint_pos[REF_HIP_JOINT_IDS[0]]
    right_shoulder_pos = ref_joint_pos[REF_HIP_JOINT_IDS[2]]
    left_hip_pos = ref_joint_pos[REF_HIP_JOINT_IDS[1]]
    right_hip_pos = ref_joint_pos[REF_HIP_JOINT_IDS[3]]

    forward_dir = neck_pos - pelvis_pos
    forward_dir += config.FORWARD_DIR_OFFSET
    forward_dir = forward_dir / np.linalg.norm(forward_dir)

    delta_shoulder = left_shoulder_pos - right_shoulder_pos
    delta_hip = left_hip_pos - right_hip_pos
    dir_shoulder = delta_shoulder / np.linalg.norm(delta_shoulder)
    dir_hip = delta_hip / np.linalg.norm(delta_hip)

    left_dir = 0.5 * (dir_shoulder + dir_hip)

    up_dir = np.cross(forward_dir, left_dir)
    up_dir = up_dir / np.linalg.norm(up_dir)

    left_dir = np.cross(up_dir, forward_dir)
    left_dir[2] = 0.0  # make the base more stable
    left_dir = left_dir / np.linalg.norm(left_dir)

    rot_mat = np.array(
        [
            [forward_dir[0], left_dir[0], up_dir[0], 0],
            [forward_dir[1], left_dir[1], up_dir[1], 0],
            [forward_dir[2], left_dir[2], up_dir[2], 0],
            [0, 0, 0, 1],
        ]
    )

    # root_pos = 0.5 * (pelvis_pos + neck_pos)
    root_pos = 0.25 * (left_shoulder_pos + right_shoulder_pos + left_hip_pos + right_hip_pos)
    root_pos[2] += 0.075
    base_vec = np.array([trunk_len / 2, 0, 0])
    t_vec = rot_mat[:3, :3] @ base_vec
    if style[3] > 0:
        root_pos[2] = max(np.abs(t_vec[2]) + 0.23, root_pos[2])
    elif style[2] > 0:
        root_pos[2] = max(np.abs(t_vec[2]) + 0.2, root_pos[2] * (1.0 - 0.2 * style[2]))
    else:
        root_pos[2] = max(np.abs(t_vec[2]) + 0.2, root_pos[2])
    root_rot = transformations.quaternion_from_matrix(rot_mat)
    root_rot = transformations.quaternion_multiply(root_rot, config.INIT_ROT)
    root_rot = root_rot / np.linalg.norm(root_rot)

    return root_pos, root_rot, forward_dir, rot_mat


def retarget_pose(robot, default_pose, ref_joint_pos, style=None):
    joint_lim_low, joint_lim_high = get_joint_limits(robot)

    root_pos, root_rot, forward_dir, rot_mat = retarget_root_pose(ref_joint_pos, style=style)
    root_pos += config.SIM_ROOT_OFFSET

    pybullet.resetBasePositionAndOrientation(robot, root_pos, root_rot)

    toe_offsets = rot_mat[:3, :3] @ np.array([0, 1, 0]) * 0.075

    inv_init_rot = transformations.quaternion_inverse(config.INIT_ROT)
    heading_rot = motion_util.calc_heading_rot(transformations.quaternion_multiply(root_rot, inv_init_rot))

    tar_toe_pos = []
    tar_toe_ids = []

    for i in range(len(REF_TOE_JOINT_IDS)):
        ref_toe_id = REF_TOE_JOINT_IDS[i]
        ref_hip_id = REF_HIP_JOINT_IDS[i]
        sim_hip_id = config.SIM_HIP_JOINT_IDS[i]
        sim_knee_id = config.SIM_KNEE_JOINT_IDS[i]
        sim_toe_id = config.SIM_TOE_JOINT_IDS[i]
        toe_offset_local = config.SIM_TOE_OFFSET_LOCAL[i]

        ref_toe_pos = ref_joint_pos[ref_toe_id]
        ref_hip_pos = ref_joint_pos[ref_hip_id]

        hip_link_state = pybullet.getLinkState(robot, sim_hip_id, computeForwardKinematics=True)
        sim_hip_pos = np.array(hip_link_state[4])

        toe_offset_world = pose3d.QuaternionRotatePoint(toe_offset_local, heading_rot)

        ref_hip_toe_delta = ref_toe_pos - ref_hip_pos

        if style[2] > 0:
            sim_tar_toe_pos = sim_hip_pos + (1 + 0.25 * style[2]) * ref_hip_toe_delta
            sim_tar_toe_pos[2] = max(0, min(sim_hip_pos[2] - 0.1, ref_toe_pos[2]))
        else:
            sim_tar_toe_pos = sim_hip_pos + ref_hip_toe_delta
            sim_tar_toe_pos[2] = min(sim_hip_pos[2] - 0.02, ref_toe_pos[2])

        sim_tar_toe_pos += toe_offset_world

        front = 1 if forward_dir.dot(sim_hip_pos - root_pos) > 0 else -1
        indicator = np.array(sim_hip_pos - (front * forward_dir * 0.267 / 2 + root_pos))
        flag = 1 if indicator.dot(toe_offsets) > 0 else -1
        sim_tar_toe_pos[:2] += flag * toe_offsets[:2] * 0.5

        f = forward_dir
        v2 = sim_tar_toe_pos - sim_hip_pos
        v2 = v2 / np.linalg.norm(v2)

        y = 1
        a = np.array([[f[0], f[2]], [v2[0], v2[2]]])
        b = np.array([-f[1], -v2[1]])

        x, z = np.linalg.solve(a, b).tolist()

        v1 = np.array([x, y, z])
        v1 = v1 / np.linalg.norm(v1)

        xx = 1
        a = np.array([[v1[1], v1[2]], [v2[1], v2[2]]])
        b = np.array([-v1[0], -v2[0]])
        yy, zz = np.linalg.solve(a, b).tolist()
        v3 = np.array([xx, yy, zz])
        v3 = v3 / np.linalg.norm(v3)

        t = np.sqrt(max(0, 0.2 ** 2 - np.linalg.norm(sim_tar_toe_pos - sim_hip_pos) ** 2 / 4))

        m = (sim_tar_toe_pos + sim_hip_pos) / 2

        sim_knee_pos = m - np.sign(v3.dot(f)) * v3 * t

        if style is not None and style[3:].sum() > 0:

            front = 1 if forward_dir.dot(sim_hip_pos - root_pos) > 0 else -1
            indicator = np.array(sim_hip_pos - (front * forward_dir * 0.267 / 2 + root_pos))
            flag = 1 if indicator.dot(toe_offsets) > 0 else -1

            cache = np.array([sim_tar_toe_pos[0], sim_tar_toe_pos[1], sim_tar_toe_pos[2]])
            sim_tar_toe_pos[:2] = sim_tar_toe_pos[:2] * max(style[1], style[2]) + (
                sim_hip_pos[:2] + flag * toe_offsets[:2]
            ) * (1 - max(style[1], style[2]))

            f = forward_dir
            v2 = sim_tar_toe_pos - sim_hip_pos
            v2 = v2 / np.linalg.norm(v2)

            y = 1
            a = np.array([[f[0], f[2]], [v2[0], v2[2]]])
            b = np.array([-f[1], -v2[1]])

            x, z = np.linalg.solve(a, b).tolist()

            v1 = np.array([x, y, z])
            v1 = v1 / np.linalg.norm(v1)

            xx = 1
            a = np.array([[v1[1], v1[2]], [v2[1], v2[2]]])
            b = np.array([-v1[0], -v2[0]])
            yy, zz = np.linalg.solve(a, b).tolist()
            v3 = np.array([xx, yy, zz])
            v3 = v3 / np.linalg.norm(v3)

            t = np.sqrt(max(0, 0.2 ** 2 - np.linalg.norm(sim_tar_toe_pos - sim_hip_pos) ** 2 / 4))

            m = (sim_tar_toe_pos + sim_hip_pos) / 2

            sim_knee_pos = m - np.sign(v3.dot(f)) * v3 * t

            cnt = 1

            while sim_knee_pos[2] < 0:
                cnt += 0.05
                sim_tar_toe_pos[:2] = cache[:2] * max(style[1], style[2]) / cnt + (
                    sim_hip_pos[:2] + flag * toe_offsets[:2]
                ) * (1 - max(style[1], style[2]) / cnt)

                f = forward_dir
                v2 = sim_tar_toe_pos - sim_hip_pos
                v2 = v2 / np.linalg.norm(v2)

                y = 1
                a = np.array([[f[0], f[2]], [v2[0], v2[2]]])
                b = np.array([-f[1], -v2[1]])

                x, z = np.linalg.solve(a, b).tolist()

                v1 = np.array([x, y, z])
                v1 = v1 / np.linalg.norm(v1)

                xx = 1
                a = np.array([[v1[1], v1[2]], [v2[1], v2[2]]])
                b = np.array([-v1[0], -v2[0]])
                yy, zz = np.linalg.solve(a, b).tolist()
                v3 = np.array([xx, yy, zz])
                v3 = v3 / np.linalg.norm(v3)

                t = np.sqrt(max(0, 0.2 ** 2 - np.linalg.norm(sim_tar_toe_pos - sim_hip_pos) ** 2 / 4))

                m = (sim_tar_toe_pos + sim_hip_pos) / 2

                sim_knee_pos = m - np.sign(v3.dot(f)) * v3 * t

        tar_toe_ids.append(sim_toe_id)
        tar_toe_pos.append(sim_tar_toe_pos)
        tar_toe_ids.append(sim_knee_id)
        tar_toe_pos.append(sim_knee_pos)

    joint_lim_high = [0.40, 4.1, -0.91, 0.40, 4.1, -0.91, 0.40, 4.1, -0.91, 0.40, 4.1, -0.91]
    joint_lim_low = [-0.40, -1.0, -1.9, -0.40, -1.0, -1.9, -0.40, -1.0, -1.9, -0.40, -1.0, -1.9]
    joint_pose = pybullet.calculateInverseKinematics2(
        robot,
        tar_toe_ids,
        tar_toe_pos,
        jointDamping=config.JOINT_DAMPING,
        lowerLimits=joint_lim_low,
        upperLimits=joint_lim_high,
        restPoses=default_pose,
        maxNumIterations=40,
    )
    joint_pose = np.array(joint_pose)

    pose = np.concatenate([root_pos, root_rot, joint_pose])

    return pose


def update_camera(robot):
    base_pos = np.array(pybullet.getBasePositionAndOrientation(robot)[0])
    [yaw, pitch, dist] = pybullet.getDebugVisualizerCamera()[8:11]
    pybullet.resetDebugVisualizerCamera(dist, yaw, pitch, base_pos)
    return


def load_ref_data(JOINT_POS_FILENAME, FRAME_START, FRAME_END):
    joint_pos_data = np.loadtxt(JOINT_POS_FILENAME, delimiter=",")

    start_frame = 0 if (FRAME_START is None) else FRAME_START
    end_frame = joint_pos_data.shape[0] if (FRAME_END is None) else FRAME_END
    joint_pos_data = joint_pos_data[start_frame:end_frame]

    return joint_pos_data


def retarget_motion(robot, joint_pos_data):
    num_frames = joint_pos_data.shape[0]

    for f in range(num_frames):
        ref_joint_pos = joint_pos_data[f]
        ref_joint_pos = np.reshape(ref_joint_pos, [-1, POS_SIZE])
        ref_joint_pos = process_ref_joint_pos_data(ref_joint_pos)

        curr_pose = retarget_pose(robot, config.DEFAULT_JOINT_POSE, ref_joint_pos)
        set_pose(robot, curr_pose)

        if f == 0:
            pose_size = curr_pose.shape[-1]
            new_frames = np.zeros([num_frames, pose_size])

        new_frames[f] = curr_pose

    new_frames[:, 0:2] -= new_frames[0, 0:2]

    return new_frames


def retarget_motion_once(robot, joint_pos_data, style=None):
    ref_joint_pos = joint_pos_data
    ref_joint_pos = np.reshape(ref_joint_pos, [-1, POS_SIZE])
    ref_joint_pos = process_ref_joint_pos_data(ref_joint_pos)

    curr_pose = retarget_pose(robot, config.DEFAULT_JOINT_POSE, ref_joint_pos, style)
    set_pose(robot, curr_pose)

    return curr_pose


def output_motion(frames, out_filename):
    with open(out_filename, "w") as f:
        f.write("{\n")
        f.write('"LoopMode": "Wrap",\n')
        f.write('"FrameDuration": ' + str(FRAME_DURATION) + ",\n")
        f.write('"EnableCycleOffsetPosition": true,\n')
        f.write('"EnableCycleOffsetRotation": true,\n')
        f.write("\n")

        f.write('"Frames":\n')

        f.write("[")
        for i in range(frames.shape[0]):
            curr_frame = frames[i]

            if i != 0:
                f.write(",")
            f.write("\n  [")

            for j in range(frames.shape[1]):
                curr_val = curr_frame[j]
                if j != 0:
                    f.write(", ")
                f.write("%.5f" % curr_val)

            f.write("]")

        f.write("\n]")
        f.write("\n}")

    return


def prepare_markers(p, joint_pos_data):
    p.removeAllUserDebugItems()

    num_markers = joint_pos_data // POS_SIZE
    marker_ids = build_markers(num_markers)

    return marker_ids


def update(ref_joint_pos, marker_ids, robot, p):
    ref_joint_pos = np.reshape(ref_joint_pos, [-1, POS_SIZE])
    ref_joint_pos = process_ref_joint_pos_data(ref_joint_pos)

    set_maker_pos(ref_joint_pos, marker_ids)

    update_camera(robot)
    p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
