import os
import time

import numpy as np
import pybullet
import pybullet_data as pd

from animation import common as C
from animation.animation import Animation
from thirdparty.retarget_motion import retarget_motion as retarget_utils

config = retarget_utils.config

p = pybullet
p.connect(p.GUI, options="--width=1920 --height=1080")
p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
pybullet.setAdditionalSearchPath(pd.getDataPath())
pybullet.resetSimulation()
pybullet.setGravity(0, 0, 0)

bullet_robot = pybullet.loadURDF(config.URDF_FILENAME, config.INIT_POS, config.INIT_ROT)

# Set robot to default pose to bias knees in the right direction.
retarget_utils.set_pose(bullet_robot, np.concatenate([config.INIT_POS, config.INIT_ROT, config.DEFAULT_JOINT_POSE]))

animation = Animation()

generator = animation.gen_frame()

markids = retarget_utils.prepare_markers(p, 81)

motion_clip = []

output_path = "outputs"
output_file = f"{animation.profile.name}.txt"

timer = 0

try:
    while timer < C.DURATION:
        joint_pos_data = np.array([next(generator) for _ in range(1)])

        pose = retarget_utils.retarget_motion_once(bullet_robot, joint_pos_data[0], style=animation.get_root_styles())

        retarget_utils.update(joint_pos_data[0], markids, bullet_robot, p)

        # correct quaternion
        w = pose[6]
        pose[4:7] = pose[3:6]
        pose[3] = w

        motion_clip.append(np.concatenate([[timer], pose]))

        time.sleep(1 / C.SYS_FREQ)
        timer += 1 / C.SYS_FREQ


except KeyboardInterrupt:
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    np.savetxt(os.path.join(output_path, output_file), motion_clip, fmt="%.5f")
    p.disconnect()

finally:
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    np.savetxt(os.path.join(output_path, output_file), motion_clip, fmt="%.5f")
    p.disconnect()
