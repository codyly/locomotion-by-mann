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
parser.add_argument("-r", "--record", type=str, help="motion record video", default=None)
args = parser.parse_args()

if not os.path.exists(args.file):
    raise FileNotFoundError("target motion clip file not found")

motion_clip = np.loadtxt(args.file, dtype=float)

duration = motion_clip.shape[0] * 1.0 / C.SYS_FREQ

config = retarget_utils.config

generator = iter(motion_clip)

p = pybullet
if args.record is not None:
    p.connect(p.GUI, options=f'--width=800 --height=600 --mp4="{args.record}" --mp4fps=60')
else:
    p.connect(p.GUI, options="--width=800 --height=600")

p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.setAdditionalSearchPath(pd.getDataPath())
p.resetSimulation()
p.setGravity(0, 0, 0)

init_pose = None
# bullet_robot = p.loadURDF(config.URDF_FILENAME, config.INIT_POS, config.INIT_ROT)
planeId = p.loadURDF("plane.urdf")

# Set robot to default pose to bias knees in the right direction.
# retarget_utils.set_pose(bullet_robot, np.concatenate([config.INIT_POS, config.INIT_ROT, config.DEFAULT_JOINT_POSE]))

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

        cur_loc = pose[:2]
        d += np.linalg.norm(cur_loc - prev_loc)

        prev = [prev_loc[0], prev_loc[1], 0]
        cur = [cur_loc[0], cur_loc[1], 0]
        p.addUserDebugLine(prev, cur, lineColorRGB=[0, 0, 1], lineWidth=50.0, lifeTime=10000)
        prev_loc = cur_loc

        if init_pose is None:
            init_pose = pose.copy()
            bullet_robot = p.loadURDF(config.URDF_FILENAME, init_pose[:3], init_pose[3:7])

        retarget_utils.set_pose(bullet_robot, pose)
        retarget_utils.update_camera(bullet_robot, force_dist=1)

        time.sleep(1 / C.SYS_FREQ)
        timer += 1 / C.SYS_FREQ


except KeyboardInterrupt:
    p.disconnect()

finally:
    p.disconnect()
