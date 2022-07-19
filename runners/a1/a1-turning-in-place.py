import argparse
import os

import numpy as np
import pybullet
import pybullet_data as pd

from animation import common as C
from animation import utils as U
from animation import profiles as P
from animation.animation import Animation
from animation.profiles import TurningInPlaceProfile
from thirdparty.retarget_motion import retarget_motion as retarget_utils

parser = argparse.ArgumentParser(description="Generate forwarding gaits at customized speeds.")
parser.add_argument("-c", "--coeff", type=float, help="turning coefficient")
parser.add_argument("-m", "--method", type=str, help="turning mix operation name, no_move or move")
parser.add_argument("-o", "--output", type=str, help="output path", default="outputs")
parser.add_argument("-d", "--direction", type=str, help="turning direction", default="right")
parser.add_argument("-s", "--startup", type=bool, help="whether use startup second", default=True)
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

if args.method == "no_move":
    mix_method = P.NMV

elif args.method == "move":
    if args.direction == "left":
        mix_method = P.MLF
    else:
        mix_method = P.MRT

else:
    raise ValueError("invalid value of argument method.")

profile = TurningInPlaceProfile(
    f"turning_{args.direction}_in_place_profile_m_{args.method}_c_{args.coeff:.2f}", mix_method=mix_method, coeff=args.coeff, direction=args.direction, startup=args.startup
)
animation = Animation(profile=profile)

generator = animation.gen_frame()

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
    int_part_input = int(args.coeff)
    flt_part_input = round((args.coeff - int_part_input) * 1000)
    output_file = f"{animation.profile.name}_coeff_{int_part_input}_{flt_part_input:03d}_sp_{int_part}_{flt_part:03d}_angle_{int(angle)}.txt"
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
