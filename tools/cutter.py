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
parser.add_argument("-s", "--start", type=int, help="", default=0)
parser.add_argument("-o", "--output", type=str, help="output path for corrected motion clip file")
args = parser.parse_args()

if not os.path.exists(args.file):
    raise FileNotFoundError("target motion clip file not found")

motion_clip = np.loadtxt(args.file, dtype=float)[args.start :]

duration = motion_clip.shape[0] * 1.0 / C.SYS_FREQ

config = retarget_utils.config

generator = iter(motion_clip)

p = pybullet
p.connect(p.DIRECT)

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
    stuck_ik_joint_pose = None
    stuck_recovery_iter = 0
    target_stuck_recovery_iter = 5

    toe_ids = [5, 15, 10, 20]

    corrected_mocap = []

    joint_lim_high = [0.40, 4.1, -0.91, 0.40, 4.1, -0.91, 0.40, 4.1, -0.91, 0.40, 4.1, -0.91]
    joint_lim_low = [-0.40, -1.0, -1.9, -0.40, -1.0, -1.9, -0.40, -1.0, -1.9, -0.40, -1.0, -1.9]
    prev_toe_pose = None

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
            retarget_utils.update_camera(bullet_robot, force_dist=1)

        else:

            retarget_utils.set_pose(bullet_robot, pose)
            retarget_utils.update_camera(bullet_robot, force_dist=1)

        w = pose[6]
        pose[4:7] = pose[3:6]
        pose[3] = w
        corrected_mocap.append(np.concatenate([[timer], pose]))

        # time.sleep(1 / C.SYS_FREQ)
        timer += 1 / C.SYS_FREQ

    np.savetxt(args.output + "/" + args.file.split("/")[-1][:-4] + "_cut.txt", np.array(corrected_mocap), fmt="%.5f")

except KeyboardInterrupt:
    p.disconnect()

finally:
    p.disconnect()
