
import argparse
from cmath import phase
import os

import math 
import numpy as np 
import pybullet
import pybullet_data as pd

from animation import common as C
from animation import utils as U
from thirdparty.retarget_motion import retarget_motion as retarget_utils
from thirdparty.retarget_motion import retarget_config_a1 as config

class TrajectoryGenerator:
    def __init__(self, name, func=lambda t: t, T = math.pi * 2) -> None:
        self.name = name 
        self.func = func 
        self.T = T 
        np.testing.assert_almost_equal(self.func(0), self.func(T))
    
    def get(self, t):
        return self.func(t)
    

class TGGait:
    def __init__(self, tg: TrajectoryGenerator, phases = [0, 0, 0, 0]) -> None:
        self.init_phases = phases
        self.phases = phases
        self.tg = tg 
    
    def get(self):
        return [self.tg.get(t) for t in self.phases]
    
    def step(self, dt):
        self.phases = [(t+dt) % self.tg.T for t in self.phases]
        return self.get()

parser = argparse.ArgumentParser(description="Visualize generated motion clips")
parser.add_argument("-f", "--file", type=str, help="motion clip file")
# parser.add_argument("-t", "--threshold", type=float, help="heigh threshold for on-ground detection", default=0.033)
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

    fused_toe_pose = None
    fused_state = 0
    fused_ik_joint_angles = None
    fused_recovery_iter = 0
    target_fused_recovery_iter = 5

    toe_ids = [5, 15, 10, 20]

    corrected_mocap = []

    joint_lim_high = [0.40, 4.1, -0.91, 0.40, 4.1, -0.91, 0.40, 4.1, -0.91, 0.40, 4.1, -0.91]
    joint_lim_low = [-0.40, -1.0, -1.9, -0.40, -1.0, -1.9, -0.40, -1.0, -1.9, -0.40, -1.0, -1.9]
    prev_toe_pose = None

    fused_toe_pose = np.zeros([4])
    fused_state = 0

    A = 0.08
    tg = TrajectoryGenerator("sin", lambda t: A + A * math.sin(2*math.pi*t), T=1.0)
    walk_gait = TGGait(tg=tg, phases=[0.5, 0, 0, 0.5])
    num_cycles = 25
    dt = num_cycles * walk_gait.tg.T / motion_clip.shape[0]
    print(dt)
    lerp_t = 1

    lerp_p = np.ones(4)

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
            prev_toe_pose = np.array(
                [pybullet.getLinkState(bullet_robot, toe_id, computeForwardKinematics=True)[4] for toe_id in toe_ids]
            )

        else:
            retarget_utils.set_pose(bullet_robot, pose)
            
            # retarget_utils.update_camera(bullet_robot, force_dist=1)
            toe_pose = np.array(
                [pybullet.getLinkState(bullet_robot, toe_id, computeForwardKinematics=True)[4] for toe_id in toe_ids]
            )

            slipped = prev_toe_pose[:, 2] < 0.033

            slope = np.rad2deg(
                np.arctan(
                    np.abs(toe_pose[:, 2] - prev_toe_pose[:, 2])
                    / (1e-10 + np.linalg.norm(toe_pose[:, :2] - prev_toe_pose[:, :2], axis=1))
                )
            )
            slipped = np.logical_and(slipped, slope < 5.0)

            vz = np.abs(toe_pose[:, 2] - prev_toe_pose[:, 2])
            slipped = np.logical_and(slipped, vz < 0.01)

            # if np.any(slipped):
            #     lerp_t *= 0.01
            # else:
            #     lerp_t *= 1.1
            
            # lerp_t = min(1, lerp_t)
            

            # tggait_height = U.lerp(np.array(walk_gait.step(dt)), toe_pose[:, 2], lerp_t)

            # lerp_p[slipped] *= 0.01
            # lerp_p[np.logical_not(slipped)] *= 1.1
            
            # lerp_p = np.clip(0, 1, lerp_p)
            
            # tggait_height = U.lerp(np.array(walk_gait.step(dt)), toe_pose[:, 2], lerp_p)

            # print(lerp_p)

            toe_pose[:, 2] = np.array(walk_gait.step(dt))

            # print(toe_pose)
            
            fused_ik_joint_angles = pybullet.calculateInverseKinematics2(
                bullet_robot,
                toe_ids,
                toe_pose,
                jointDamping=config.JOINT_DAMPING,
                restPoses=config.DEFAULT_JOINT_POSE,
                lowerLimits=joint_lim_low,
                upperLimits=joint_lim_high,
                maxNumIterations=10,
            )
            
            prev_toe_pose = np.array(
                [pybullet.getLinkState(bullet_robot, toe_id, computeForwardKinematics=True)[4] for toe_id in toe_ids]
            )

            pose[7:] = fused_ik_joint_angles
            retarget_utils.set_pose(bullet_robot, pose)
            retarget_utils.update_camera(bullet_robot, force_dist=1)

        # prev_toe_pose = np.array(
        #     [pybullet.getLinkState(bullet_robot, toe_id, computeForwardKinematics=True)[4] for toe_id in toe_ids]
        # )

        w = pose[6]
        pose[4:7] = pose[3:6]
        pose[3] = w
        corrected_mocap.append(np.concatenate([[timer], pose]))

        # time.sleep(1 / C.SYS_FREQ)
        timer += 1 / C.SYS_FREQ
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    np.savetxt(
        args.output + "/" + args.file.split("/")[-1][:-4] + "_fused.txt", np.array(corrected_mocap), fmt="%.5f"
    )

except KeyboardInterrupt:
    p.disconnect()

finally:
    p.disconnect()
