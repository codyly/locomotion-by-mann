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

config = retarget_utils.config



class dummy_env():
    def __init__(self, render=False) -> None:
        self.render = render
        self.p = pybullet

        if self.render:
            self.p.connect(self.p.GUI, options="--width=1920 --height=1080")
            self.p.configureDebugVisualizer(self.p.COV_ENABLE_GUI, 0)
            self.p.configureDebugVisualizer(self.p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
            self.p.configureDebugVisualizer(self.p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
            self.p.configureDebugVisualizer(self.p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
        else:
            self.p.connect(self.p.DIRECT)
            self.p.configureDebugVisualizer(self.p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)

        self.p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=0, cameraPitch=-60,
                                         cameraTargetPosition=[0., 0., 0.])
        
        self.p.setAdditionalSearchPath(pd.getDataPath())
        self.p.resetSimulation()
        self.p.setGravity(0, 0, 0)

        self.robot = self.p.loadURDF(config.URDF_FILENAME, config.INIT_POS, config.INIT_ROT)
        self.ground = self.p.loadURDF("plane.urdf")
        self.obstacles = []
        
    def define_ground(self,):
        pass

    
    def reset(self,):
        profile = P.motion_wiki['sit']
        self.animation = Animation(profile=profile)
        self.generator = self.animation.gen_frame()
        retarget_utils.set_pose(self.robot, np.concatenate([config.INIT_POS, config.INIT_ROT, config.DEFAULT_JOINT_POSE]))

        self.clear_landscape()
        self.load_landscape()

    def load_landscape(self, n=2):
        # x: [0.5, 4.5]
        # y: [-1, 1]
        half_size_range = [0.01, 0.3]
        y_bound = [-0.5, 0.5]
        x_bound = [2, 3.]

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

    def add_cube(self, pos, quat, half_size):
        CubeVis = self.p.createVisualShape( shapeType=self.p.GEOM_BOX,
                                            rgbaColor=(0.5, 0.5, 0.5, 0.8),
                                            halfExtents=half_size)
        pos[2] = half_size[2]
        CubeColId = self.p.createCollisionShape(shapeType=self.p.GEOM_BOX,
                                                halfExtents=half_size,
                                                )
        CubeObj = self.p.createMultiBody(baseMass=0.1,
                                         basePosition=pos,
                                         baseOrientation=quat,
                                         baseCollisionShapeIndex=CubeColId,
                                         baseVisualShapeIndex=CubeVis,
                                         useMaximalCoordinates=False)
        return CubeObj

    def clear_landscape(self):
        for Obj in self.obstacles:
            self.p.removeBody(Obj,)

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
            obs["body_pose"].append(self.p.getBasePositionAndOrientation(self.robot))

            # todo check collision
            collision = self.check_collision()
            obs["collision"].append(collision)

        return obs


    def check_collision(self):
        contact_w_ground = self.p.getContactPoints(self.robot, self.ground)
        print("contact ground: ", len(contact_w_ground))

        for box in self.obstacles:
            contact_i = self.p.getContactPoints(self.robot, box)
            print("contacts: ", len(contact_i))

            if len(contact_i) > 0:
                for point in contact_i:
                    print(point)
                assert 0


    def terminate_env(self,):
        self.p.disconnect()
