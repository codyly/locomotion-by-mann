import argparse
import os

import numpy as np
import pybullet
import pybullet_data as pd
import pylab as p

from animation import common as C
from animation import profiles as P
from animation import utils as U
from animation.animation import Animation
from animation.controller import SimInputHandler, Controller, KeyboardInputHandler
from thirdparty.retarget_motion import retarget_motion as retarget_utils
from collections import defaultdict
import transforms3d as t3d

import matplotlib.pyplot as plt

config = retarget_utils.config

def convert_to_pose(pose6):
    pose = np.eye(4)
    pose[:3, 3] = pose6[0]
    mat = t3d.euler.quat2mat(pose6[1])
    pose[:3, :3] = mat
    return pose

class dummy_env():
    def __init__(self, render=False) -> None:
        self.render = render
        self.p = pybullet

        if self.render:
            self.p.connect(self.p.GUI, options="--width=1920 --height=1080")
            self.p.configureDebugVisualizer(self.p.COV_ENABLE_GUI, 1)
            self.p.configureDebugVisualizer(self.p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 1)
            self.p.configureDebugVisualizer(self.p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
            self.p.configureDebugVisualizer(self.p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
        else:
            self.p.connect(self.p.DIRECT)
            self.p.configureDebugVisualizer(self.p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)

        self.p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=0, cameraPitch=-45,
                                         cameraTargetPosition=[0., 0., 0.])
        
        self.p.setAdditionalSearchPath(pd.getDataPath())
        self.p.resetSimulation()
        self.p.setGravity(0, 0, 0)

        self.robot = self.p.loadURDF(config.URDF_FILENAME, config.INIT_POS, config.INIT_ROT)
        self.ground = self.p.loadURDF("plane.urdf")
        self.obstacles = []
        self.cam_model = None
        self.cam_line = None

    def reset(self,):
        profile = P.motion_wiki['sit']
        self.animation = Animation(profile=profile)
        self.generator = self.animation.gen_frame()
        retarget_utils.set_pose(self.robot, np.concatenate([config.INIT_POS, config.INIT_ROT, config.DEFAULT_JOINT_POSE]))

        self.clear_landscape()
        self.load_landscape()

    def load_landscape(self, n=1):
        # x: [0.5, 4.5]
        # y: [-1, 1]
        half_size_range = [0.05, 0.3]
        y_bound = [-0.5, 0.5]
        x_bound = [0.5, 3.]

        for _ in range(n):
            sz = np.random.uniform(half_size_range[0],
                                   half_size_range[1],
                                   size=(3, ))
            x = np.random.uniform(x_bound[0]+sz[0], x_bound[1]-sz[0], size=())
            y = np.random.uniform(y_bound[0]+sz[1], y_bound[1]-sz[1], size=())

            cube = self.add_cube(pos=[x, y, sz[2]],
                                 quat=[0, 0, 0, 1],
                                 half_size=sz)
            self.obstacles.append(cube)

    def add_cube(self, pos, quat, half_size, color=(0.5, 0.5, 0.5, 0.8)):
        CubeVis = self.p.createVisualShape(shapeType=self.p.GEOM_BOX,
                                           rgbaColor=color,
                                           halfExtents=half_size)
        CubeColId = self.p.createCollisionShape(shapeType=self.p.GEOM_BOX,
                                                halfExtents=half_size,
                                                )
        CubeObj = self.p.createMultiBody(baseMass=0.,
                                         basePosition=pos,
                                         baseOrientation=quat,
                                         baseCollisionShapeIndex=CubeColId,
                                         baseVisualShapeIndex=CubeVis,
                                         useMaximalCoordinates=False)
        return CubeObj

    def add_line(self, point_A, point_B, color):
        line = self.p.addUserDebugLine(point_A, point_B, lineColorRGB=color, lineWidth=2.0, lifeTime=0)
        return line

    def clear_landscape(self):
        for obj in self.obstacles:
            self.p.removeBody(obj,)

    def step(self, action, startup=False):
        '''
        action: np.array([int]) 
        '''

        if action[0] == 0:
            profile = P.ForwardProfile("walk", vel=1.0, startup=startup)
            profile_length = 80
        elif action[0] == 1:
            profile = P.Profile("jump", stages=[0.1, 0.06], ops=[P.JMP, P.FWD], startup=startup)
            profile_length = 70
        elif action[0] == 2:
            coeff = 1
            profile = P.TurningProfile( "turn_left", coeff=coeff, direction='left', startup=startup)
            profile_length = 25
        else:
            coeff = 1
            profile = P.TurningProfile( "turn_right", coeff=coeff, direction='right', startup=startup)
            profile_length = 25

        self.animation.controller = Controller(input_handler=SimInputHandler(profile=profile.inst, need_parse=True, looping=True))

        obs = defaultdict(list)
        for _ in range(profile_length):
            joint_pos_data = np.array([next(self.generator) for _ in range(1)])
            pose = retarget_utils.retarget_motion_once(self.robot, joint_pos_data[0], style=self.animation.get_root_styles())
            obs["pos_n"].append(pose)

            agent_pos, agent_quat = self.p.getBasePositionAndOrientation(self.robot)
            img, dep, seg = self.get_camera_data(agent_pos, agent_quat, 480, 640)

            _, axs = plt.subplots(1, 3, figsize=(10, 3))
            axs[0].imshow(img)
            axs[0].set_title("color")
            axs[0].axis("off")
            axs[1].imshow(dep)
            axs[1].set_title("depth")
            axs[1].axis("off")
            axs[2].imshow(seg, cmap="tab10")
            axs[2].set_title("segmentation")
            axs[2].axis("off")
            plt.show()


            obs["agent_pos"].append(agent_pos)
            obs["agent_quat"].append(agent_quat)
            obs["agent_color"].append(img)
            obs["agent_depth"].append(dep)

            self.p.stepSimulation()
            collision = self.check_collision()
            obs["collision"].append(collision)

        return obs

    def check_collision(self, pnt_th=-0.0001):
        for box in self.obstacles:
            contact_i = self.p.getContactPoints(self.robot, box)
            if len(contact_i) > 0:
                for point in contact_i:
                    dist = point[8]
                    if dist < pnt_th:
                        return True
        return False

    def visualize_trajectory(self, obs, samp=5):
        collisions = obs["collision"]
        body_poss = obs["agent_pos"]
        print("traj len: ", len(body_poss), len(collisions))
        assert len(collisions) == len(body_poss)

        self.traj = []
        for c, pos in zip(collisions[::samp], body_poss[::samp]):
            if c:
                color = (1, 0, 0, 0.5)
            else:
                color = (0, 1, 0, 0.5)

            sphere_vis = self.p.createVisualShape(shapeType=self.p.GEOM_SPHERE,
                                                  radius=0.03,
                                                  rgbaColor=color,
                                                  )
            sphere_obj = self.p.createMultiBody(baseMass=0,
                                                baseVisualShapeIndex=sphere_vis,
                                                basePosition=pos,
                                                useMaximalCoordinates=False)
            self.traj.append(sphere_obj)

    def get_camera_data(self, cam_pos, cam_quat, cam_H, cam_W):
        cam_dist = 100.
        farVal = 4.
        nearVal = 0.01
        fov = 90.
        cam_ratio = float(cam_W) / float(cam_H)

        # make camera tilt
        phi = np.deg2rad(30)
        delta_pos = np.array([0.24, 0, 0.15])

        cam_pos = np.array(cam_pos) + delta_pos

        cam_mat = self.p.getMatrixFromQuaternion(cam_quat)
        cam_mat = np.array(cam_mat).reshape((3, 3))


        rot_y = np.array([[np.cos(phi), 0, np.sin(phi)],
                          [0, 1, 0],
                          [-np.sin(phi), 0, np.cos(phi)]])
        cam_mat = rot_y @ cam_mat

        cam_tar = cam_mat @ np.array([cam_dist, 0, 0]) + cam_pos

        if self.cam_line is not None:
            self.p.removeUserDebugItem(self.cam_line)
        if self.cam_model is not None:
            self.p.removeBody(self.cam_model)

        self.cam_model = self.add_cube(cam_pos, cam_quat, half_size=(0.02, 0.01, 0.01))
        self.cam_line = self.add_line(cam_pos, cam_tar, (0, 1, 0))

        view_mat = self.p.computeViewMatrix(
            cameraEyePosition=cam_pos,
            cameraTargetPosition=cam_tar,
            cameraUpVector=[0, 0, 1.0]
        )

        proj_mat = self.p.computeProjectionMatrixFOV(
            fov=fov, aspect=cam_ratio, nearVal=nearVal, farVal=farVal)

        width, height, rgbImg, depthImg, segImg = self.p.getCameraImage(
            width=cam_W,
            height=cam_H,
            viewMatrix=view_mat,
            projectionMatrix=proj_mat)

        color_img = np.array(rgbImg)
        color_img = np.reshape(color_img, (cam_H, cam_W, 4))
        color_img = color_img[:, :, :3]

        depth_img = np.array(depthImg)
        depth_img = farVal * nearVal / (farVal - (farVal - nearVal) * depth_img)
        depth_img = np.reshape(depth_img, (cam_H, cam_W))

        return color_img, depth_img, segImg

    def terminate_env(self,):
        self.p.disconnect()
