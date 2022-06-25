import os
import time
from threading import Thread, Event
from collections import deque

import numpy as np
import pybullet
import pybullet_data as pd

from animation import common as C
from animation import profiles as P
from animation import utils as U
from animation.animation import Animation
from animation.locopath import LocoPath
from thirdparty.retarget_motion import retarget_motion as retarget_utils

config = retarget_utils.config

p = pybullet
p.connect(p.GUI, options="--width=1920 --height=1080")
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
p.setAdditionalSearchPath(pd.getDataPath())
p.resetSimulation()
p.setGravity(0, 0, 0)

bullet_robot = p.loadURDF(config.URDF_FILENAME, config.INIT_POS, config.INIT_ROT)
planeId = p.loadURDF("plane.urdf")
# Set robot to default pose to bias knees in the right direction.
retarget_utils.set_pose(bullet_robot, np.concatenate([config.INIT_POS, config.INIT_ROT, config.DEFAULT_JOINT_POSE]))

# animation = Animation(profile=P.TurningProfile("turning_right", 0))
animation = Animation(profile=P.ForwardProfile("aa", 0.8, startup=False), keyboard_input=False)

loco_path = LocoPath("outputs/trajectory/star.txt", scale_x=5, scale_y=5, num_frames=C.DURATION * C.SYS_FREQ + 1)
# loco_path = LocoPath("outputs/trajectory/a.txt", scale_x=5, scale_y=5, num_frames=C.DURATION * C.SYS_FREQ + 1)
# loco_path = LocoPath("outputs/trajectory/a.txt", scale_x=5, scale_y=5, num_frames=C.DURATION * C.SYS_FREQ + 1)

generator = animation.gen_frame(keep_straight=False, loco_path=loco_path)

markids = retarget_utils.prepare_markers(p, 81)

motion_clip = []

output_path = "outputs"
timer = 0

joint_pos_data = deque()
retarget_joint_pos_id = -1

new_pose = Event()
new_pose.clear()


def mann_gen():
    global joint_pos_data
    num_frames = 5 * C.DURATION * C.SYS_FREQ
    i = 0
    while i < num_frames:
        start_t = time.time()
        joint_pos_data.append(next(generator))
        wait_t = 1.0 / C.SYS_FREQ - (time.time() - start_t)
        i += 1
        new_pose.set()

        if wait_t > 0:
            time.sleep(wait_t)


mann_thread = Thread(target=mann_gen)

output_file = "broken.txt"

try:
    # record horizontal displacement

    mann_thread.start()

    prev_loc = np.zeros(2)
    prev_vec = np.array([1, 0, 0])
    d = 0
    angle = 0
    while retarget_joint_pos_id < 5 * C.DURATION * C.SYS_FREQ - 1:

        start_time = time.time()

        new_pose.wait()
        new_pose.clear()

        retarget_joint_pos_id += 1
        data = joint_pos_data.popleft()

        pose = retarget_utils.retarget_motion_once(bullet_robot, data, style=animation.get_root_styles())
        # print(pose[:3].round(2))
        retarget_utils.update(data, markids, bullet_robot, p)

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

        prev = [prev_loc[0], prev_loc[1], 0]
        cur = [cur_loc[0], cur_loc[1], 0]
        p.addUserDebugLine(prev, cur, lineColorRGB=[0, 0, 1], lineWidth=50.0, lifeTime=10000)
        prev_loc = cur_loc
        if retarget_joint_pos_id < C.DURATION * C.SYS_FREQ - 1:
            p1 = loco_path.get_pos(frame_id=retarget_joint_pos_id)
            p0 = loco_path.get_pos(frame_id=retarget_joint_pos_id - 1) if retarget_joint_pos_id > 0 else [0, 0, 0]
            p.addUserDebugLine(
                [p0[2], p0[0], 0], [p1[2], p1[0], 0], lineColorRGB=[1, 0, 0], lineWidth=500.0, lifeTime=10000
            )

        motion_clip.append(np.concatenate([[timer], pose]))
        end_time = time.time()

        wait_time = 1 / C.SYS_FREQ - (end_time - start_time)
        if wait_time > 0:
            time.sleep(wait_time)
        timer += 1 / C.SYS_FREQ

    speed = d / C.DURATION
    print(f"Locomotion Speed: {speed:.2f} m/s")

    int_part = int(speed)
    flt_part = round((speed - int_part) * 1000)

    print(f"Locomotion Turning: {angle: .5f} degree")
    output_file = f"{animation.profile.name}_sp_{int_part}_{flt_part:03d}_angle_{int(angle)}.txt"


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
