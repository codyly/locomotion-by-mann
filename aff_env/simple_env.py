import argparse
import os

import numpy as np
import pybullet
import pybullet_data as pd

from animation import common as C
from animation import profiles as P
from animation import utils as U
from animation.animation import Animation
from animation.controller import SimInputHandler, Controller, KeyboardInputHandler
from thirdparty.retarget_motion import retarget_motion as retarget_utils


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
            
        
        self.p.setAdditionalSearchPath(pd.getDataPath())
        self.p.resetSimulation()
        self.p.setGravity(0, 0, 0)

        self.robot = self.p.loadURDF(config.URDF_FILENAME, config.INIT_POS, config.INIT_ROT)
        self.ground = self.p.loadURDF("plane.urdf")

        self.reset()

        
    def define_ground(self,):
        pass

    
    def reset(self,):
        profile = P.motion_wiki['sit']
        self.animation = Animation(profile=profile)
        self.generator = self.animation.gen_frame()
        retarget_utils.set_pose(self.robot, np.concatenate([config.INIT_POS, config.INIT_ROT, config.DEFAULT_JOINT_POSE]))



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




        for _ in range(profile_length):
            joint_pos_data = np.array([next(self.generator) for _ in range(1)])
            pose = retarget_utils.retarget_motion_once(self.robot, joint_pos_data[0], style=self.animation.get_root_styles())
        
        return pose



    def terminate_env(self,):
        self.p.disconnect()
