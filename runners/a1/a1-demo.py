import argparse
import os
import time

import numpy as np
import pybullet
import pybullet_data as pd

from animation import common as C
from thirdparty.retarget_motion import retarget_motion as retarget_utils

parser = argparse.ArgumentParser(description="Visualize generated motion clips")
parser.add_argument("-f", "--file", type=str, help="motion clip file")
args = parser.parse_args()

if not os.path.exists(args.file):
    raise FileNotFoundError("target motion clip file not found")

motion_clip = np.loadtxt(args.file, dtype=float)

duration = motion_clip.shape[0] * 1.0 / C.SYS_FREQ

config = retarget_utils.config

generator = iter(motion_clip)

p = pybullet
p.connect(p.GUI, options="--width=800 --height=600")
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.setAdditionalSearchPath(pd.getDataPath())
p.resetSimulation()
p.setGravity(0, 0, 0)

bullet_robot = p.loadURDF(config.URDF_FILENAME, config.INIT_POS, config.INIT_ROT)
planeId = p.loadURDF("plane.urdf")

# Set robot to default pose to bias knees in the right direction.
retarget_utils.set_pose(bullet_robot, np.concatenate([config.INIT_POS, config.INIT_ROT, config.DEFAULT_JOINT_POSE]))

timer = 0

try:
    # record horizontal displacement
    prev_loc = np.zeros(2)
    prev_vec = np.array([1, 0, 0])
    d = 0
    angle = 0

    while timer < duration:
        pose = next(generator)[1:]

        # correct quaternion
        w = pose[3]
        pose[3:6] = pose[4:7]
        pose[6] = w

        retarget_utils.set_pose(bullet_robot, pose)
        retarget_utils.update_camera(bullet_robot, force_dist=1)

        time.sleep(1 / C.SYS_FREQ)
        timer += 1 / C.SYS_FREQ


except KeyboardInterrupt:
    p.disconnect()

finally:
    p.disconnect()
