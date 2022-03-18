import os
import time

import numpy as np
import pybullet
import pybullet_data as pd

from animation import profiles as P
from animation import common as C
from animation.animation_time import Animation
# from animation.animation import Animation
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

animation = Animation(profile='manual')
# animation = Animation(profile=P.acc_stop)

generator = animation.gen_frame()

markids = retarget_utils.prepare_markers(p, 81)

motion_clip = []

output_path = "outputs"
if animation.profile != 'manual':
    output_file = f"{animation.profile.name}.txt"
else:
    output_file = "manual.txt"

timer = 0

def get_key_pressed(relevant=None):
        pressed_keys = []
        events = p.getKeyboardEvents()
        key_codes = events.keys()
        for key in key_codes:
            pressed_keys.append(key)
        return pressed_keys 

import time
try:
    while True:
        t_start = time.time()
        if (timer % 1 <= 1 / C.SYS_FREQ):
            animation.print_once = True

        pressed_keys = [chr(ch) for ch in get_key_pressed()]
        # print("pressed keys: ", pressed_keys)
        animation.input_handler.input_keys(pressed_keys)
        
        t_input = time.time()
        frame = next(generator)
        joint_pos_data = np.array([frame])
        t_mann = time.time()
        pose = retarget_utils.retarget_motion_once(bullet_robot, joint_pos_data[0], style=animation.get_root_styles())
        t_retarget = time.time()
        retarget_utils.update(joint_pos_data[0], markids, bullet_robot, p)
        t_vis = time.time()
        # correct quaternion
        w = pose[6]
        pose[4:7] = pose[3:6]
        pose[3] = w

        # for saving
        motion_clip.append(np.concatenate([[timer], pose]))

        if (timer % 1 <= 1 / C.SYS_FREQ):
            print("[Timing] Total: %d out of %d"%((t_vis-t_start)*1000, 1000/C.SYS_FREQ))
            print("Input: %d(ms), MANN: %d(ms), Retarget: %d(ms), Vis: %d(ms)"%(
                (t_input-t_start)*1000, (t_mann-t_input)*1000,
                (t_retarget-t_mann)*1000, (t_vis-t_retarget)*1000
            ))
        if (1/C.SYS_FREQ > (t_vis-t_start)):
            print("updating:", 1/C.SYS_FREQ - (t_vis-t_start))
            time.sleep(1 / C.SYS_FREQ - (t_vis-t_start))
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
