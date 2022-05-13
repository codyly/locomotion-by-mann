import argparse
import os
import time

import numpy as np
import pybullet
import pybullet_data as pd

from animation import common as C
from animation import utils as U
from thirdparty.retarget_motion import retarget_motion as retarget_utils
from thirdparty.retarget_motion import retarget_config_a1 as config

parser = argparse.ArgumentParser(description="Visualize generated motion clips")
parser.add_argument("-f", "--file", type=str, help="motion clip file")
parser.add_argument("-t", "--threshold", type=float, help="heigh threshold for on-ground detection", default=0.033)
parser.add_argument("-o", "--output", type=str, help="output path for corrected motion clip file")
args = parser.parse_args()

if not os.path.exists(args.file):
    raise FileNotFoundError("target motion clip file not found")

motion_clip = np.loadtxt(args.file, dtype=float)

duration = motion_clip.shape[0] * 1.0 / C.SYS_FREQ

config = retarget_utils.config

generator = iter(motion_clip)

p = pybullet
p.connect(p.DIRECT)
# p.connect(p.GUI, options="--width=800 --height=600")
# p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

p.setAdditionalSearchPath(pd.getDataPath())
p.resetSimulation()
p.setGravity(0, 0, 0)

planeId = p.loadURDF("plane.urdf")

init_pose = None
timer = 0

try:
    # record horizontal displacement
    prev_loc = np.zeros(2)
    prev_vec = np.array([1, 0, 0])
    d = 0
    angle = 0

    stuck_toe_pose = None
    stuck_state = 0
    stuck_ik_joint_angles = None
    stuck_recovery_iter = 0
    target_stuck_recovery_iter = 5
    all_on_ground_avg_toe_height = args.threshold

    toe_ids = [5, 15, 10, 20]

    corrected_mocap = []

    joint_lim_high = [0.40, 4.1, -0.91, 0.40, 4.1, -0.91, 0.40, 4.1, -0.91, 0.40, 4.1, -0.91]
    joint_lim_low = [-0.40, -1.0, -1.9, -0.40, -1.0, -1.9, -0.40, -1.0, -1.9, -0.40, -1.0, -1.9]
    prev_toe_pose = None

    stuck_toe_pose = np.zeros([4])
    stuck_state = 0

    while timer < duration:
        pose = next(generator)[1:]

        # correct quaternion
        w = pose[3]
        pose[3:6] = pose[4:7]
        pose[6] = w

        if init_pose is None:
            init_pose = pose.copy()
            bullet_robot = p.loadURDF(config.URDF_FILENAME, config.INIT_POS, config.INIT_ROT)
            retarget_utils.set_pose(bullet_robot, pose)
            # retarget_utils.update_camera(bullet_robot, force_dist=1)

            prev_toe_pose = np.array(
                [pybullet.getLinkState(bullet_robot, toe_id, computeForwardKinematics=True)[4] for toe_id in toe_ids]
            )

        else:

            retarget_utils.set_pose(bullet_robot, pose)
            # retarget_utils.update_camera(bullet_robot, force_dist=1)
            toe_pose = np.array(
                [pybullet.getLinkState(bullet_robot, toe_id, computeForwardKinematics=True)[4] for toe_id in toe_ids]
            )

            # print("expected: ", np.round(toe_pose, 3))
            toe_height = toe_pose[:, 2]

            on_ground = np.logical_and(
                toe_height.max() < all_on_ground_avg_toe_height,
                np.abs(toe_pose[:, 2] - prev_toe_pose[:, 2]).max() < 0.01,
            )

            # slipped = np.logical_and(
            #     np.abs(toe_pose[:, 2] - prev_toe_pose[:, 2]) < 0.01, toe_pose[:, 2] < all_on_ground_avg_toe_height
            # )
            # slipped = np.logical_and(slipped, prev_toe_pose[:, 2] < all_on_ground_avg_toe_height)
            # slipped = np.logical_and(np.linalg.norm(toe_pose[:, :2] - prev_toe_pose[:, :2], axis=1) > 0.001, slipped)
            # # print(stuck_state)
            # # print(slipped)
            # # print(toe_height, np.abs(toe_pose[:, 2] - prev_toe_pose[:, 2]).max())

            # start_slipped = np.logical_and(slipped, stuck_state == 0)
            # has_stucked = stuck_state > 0
            # normal = np.logical_not(np.logical_or(start_slipped, has_stucked))

            # stuck_state[start_slipped] = 1

            # stuck_toe_pose[start_slipped] = prev_toe_pose[start_slipped]
            # stuck_toe_pose[normal] = toe_pose[normal]

            # print("target: ", stuck_toe_pose.round(3))

            # stuck_ik_joint_angles = np.array(
            #     pybullet.calculateInverseKinematics2(
            #         bullet_robot,
            #         toe_ids,
            #         stuck_toe_pose,
            #         jointDamping=config.JOINT_DAMPING,
            #         restPoses=config.DEFAULT_JOINT_POSE,
            #         lowerLimits=joint_lim_low,
            #         upperLimits=joint_lim_high,
            #         maxNumIterations=10,
            #     )
            # ).reshape(4, 3)

            # cur_pose = pose[7:].reshape(4, 3)

            # stuck_ik_joint_angles[normal] = cur_pose[normal]
            # stuck_ik_joint_angles[has_stucked] = U.lerp(cur_pose[has_stucked], stuck_ik_joint_angles[has_stucked], 0.5)
            # stuck_state[has_stucked] -= 1

            # # print(pose[7:].round(2))
            # pose[7:] = stuck_ik_joint_angles.flatten()

            # retarget_utils.set_pose(bullet_robot, pose)
            # retarget_utils.update_camera(bullet_robot, force_dist=1)

            if on_ground and stuck_state < 5:
                stuck_toe_pose = toe_pose
                stuck_state += 1
            elif on_ground and stuck_state >= 5:
                stuck_ik_joint_angles = pybullet.calculateInverseKinematics2(
                    bullet_robot,
                    toe_ids,
                    stuck_toe_pose,
                    jointDamping=config.JOINT_DAMPING,
                    restPoses=config.DEFAULT_JOINT_POSE,
                    lowerLimits=joint_lim_low,
                    upperLimits=joint_lim_high,
                    maxNumIterations=10,
                )
                pose[7:] = stuck_ik_joint_angles
                retarget_utils.set_pose(bullet_robot, pose)
                retarget_utils.update_camera(bullet_robot, force_dist=1)
            elif stuck_state >= 5:
                stuck_state = 0
                stuck_ik_joint_angles = U.lerp(pose[7:], stuck_ik_joint_angles, 0.9)
                pose[7:] = stuck_ik_joint_angles
                stuck_recovery_iter = target_stuck_recovery_iter
                retarget_utils.set_pose(bullet_robot, pose)
                retarget_utils.update_camera(bullet_robot, force_dist=1)
            elif stuck_recovery_iter > 0:
                stuck_ik_joint_angles = U.lerp(pose[7:], stuck_ik_joint_angles, 0.9)
                pose[7:] = stuck_ik_joint_angles
                stuck_recovery_iter -= 1
                retarget_utils.set_pose(bullet_robot, pose)
                retarget_utils.update_camera(bullet_robot, force_dist=1)

            # prev_toe_pose = stuck_ik_joint_angles
        prev_toe_pose = np.array(
            [pybullet.getLinkState(bullet_robot, toe_id, computeForwardKinematics=True)[4] for toe_id in toe_ids]
        )
        # print("actual: ", np.round(prev_toe_pose, 3))
        # stuck_toe_pose = prev_toe_pose

        w = pose[6]
        pose[4:7] = pose[3:6]
        pose[3] = w
        corrected_mocap.append(np.concatenate([[timer], pose]))

        # time.sleep(1 / C.SYS_FREQ)
        timer += 1 / C.SYS_FREQ

    np.savetxt(
        args.output + "/" + args.file.split("/")[-1][:-4] + "_corrected.txt", np.array(corrected_mocap), fmt="%.5f"
    )

except KeyboardInterrupt:
    p.disconnect()

finally:
    p.disconnect()
