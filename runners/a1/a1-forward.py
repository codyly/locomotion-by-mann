import argparse
import os
import time

import numpy as np
import pybullet
import pybullet_data as pd

from animation import common as C
from animation import utils as U
from animation.animation import Animation
from animation.profiles import ForwardProfile
from thirdparty.retarget_motion import retarget_motion as retarget_utils

parser = argparse.ArgumentParser(description="Generate forwarding gaits at customized speeds.")
parser.add_argument("-v", "--velocity", type=float, help="target velocity")
parser.add_argument("-o", "--output", type=str, help="output path", default="outputs")
parser.add_argument("-s", "--startup", action="store_true", help="whether use startup second")
args = parser.parse_args()

if not os.path.exists(args.output):
    os.makedirs(args.output)


config = retarget_utils.config

p = pybullet
p.connect(p.DIRECT)
p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
p.setAdditionalSearchPath(pd.getDataPath())
p.resetSimulation()
p.setGravity(0, 0, 0)

bullet_robot = p.loadURDF(config.URDF_FILENAME, config.INIT_POS, config.INIT_ROT)

# Set robot to default pose to bias knees in the right direction.
retarget_utils.set_pose(bullet_robot, np.concatenate([config.INIT_POS, config.INIT_ROT, config.DEFAULT_JOINT_POSE]))

profile = ForwardProfile(f"forward_profile", vel=args.velocity, startup=args.startup)
animation = Animation(profile=profile)

generator = animation.gen_frame(keep_straight=False)

motion_clip = []

output_path = args.output
output_file = f"{animation.profile.name}.txt"

timer = 0

try:
    # record horizontal displacement
    prev_loc = np.zeros(2)
    prev_vec = np.array([1, 0, 0])
    d = 0
    d1 = 0
    angle = 0

    while timer < C.DURATION + args.startup:
        joint_pos_data = np.array([next(generator) for _ in range(1)])

        pose = retarget_utils.retarget_motion_once(bullet_robot, joint_pos_data[0], style=animation.get_root_styles())

        quat = pose[3:7]
        vec = U.quat_rot_vec(quat, np.array([1, 0, 0]))
        vec[2] = 0
        vec = vec / np.linalg.norm(vec)
        angle += U.signed_angle(prev_vec, vec, up=np.array([0, 0, 1]), deg=True)
        prev_vec = vec

        # correct quaternion
        w = pose[6]
        pose[4:7] = pose[3:6]
        pose[3] = w

        cur_loc = pose[:2]
        d += np.linalg.norm(cur_loc - prev_loc)
        if timer > args.startup:
            d1 += np.linalg.norm(cur_loc - prev_loc)
        prev_loc = cur_loc

        motion_clip.append(np.concatenate([[timer], pose]))

        # time.sleep(1 / C.SYS_FREQ)
        timer += 1 / C.SYS_FREQ

    speed = d / (C.DURATION + args.startup)
    print(f"Locomotion Speed: {speed:.2f} m/s")

    speed1 = d1 / (C.DURATION)
    if args.startup:
        print(f"Non-startup Locomotion Speed: {speed1:.2f} m/s")

    int_part = int(speed)
    flt_part = round((speed - int_part) * 1000)
    int_part_1 = int(speed1)
    flt_part_1 = round((speed1 - int_part_1) * 1000)
    int_part_input = int(args.velocity)
    flt_part_input = round((args.velocity - int_part_input) * 1000)
    output_file = f"{animation.profile.name}_v_{int_part_input}_{flt_part_input:03d}_sp_{int_part}_{flt_part:03d}_angle_{int(angle)}.txt"
    if args.startup:
        output_file = "startup_" + output_file[:-4] + f"_sp1_{int_part_1}_{flt_part_1:03d}.txt"


except KeyboardInterrupt:
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(os.path.join(output_path, output_file)):
        np.savetxt(os.path.join(output_path, output_file), motion_clip, fmt="%.5f")
    p.disconnect()

finally:
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(os.path.join(output_path, output_file)):
        np.savetxt(os.path.join(output_path, output_file), motion_clip, fmt="%.5f")
    p.disconnect()
