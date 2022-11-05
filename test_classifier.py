import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
import time
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.ticker as ticker

import argparse
import gc, sys
from tensorboardX import SummaryWriter

from networks.classifier import SkillClassifier
from collections import defaultdict
from aff_env.simple_env import dummy_env
import matplotlib.pyplot as plt
import re

def draw_box(corners, ax):
    links = []
    assert len(corners) == 8
    for i in range(8):
        for j in range(8):
            if i ^ j in [1, 2, 4]:
                links.append((i, j))

    for link in links:
        p0, p1 = link
        p = np.stack([corners[p0], corners[p1]])
        ax.plot(p[:, 0], p[:, 1], p[:, 2], "b-.", )

def get_gt(sensor_data):
    acc = 0
    for dat in sensor_data:
        acc += dat["collision"]
    return acc > 0

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def visualize(obs, pred, gt, skill_id):
    fig = plt.figure(figsize=plt.figaspect(0.25))

    ax4 = fig.add_subplot(1, 4, 4, projection='3d')
    # ax4.set_aspect('auto')

    boxes = obs["sensor_data"][0]["obstacles"]
    for corners in boxes:
        draw_box(corners, ax4)

    sensor_data = obs["sensor_data"]
    for dat in sensor_data:
        agent_pos = dat["agent_pos"]
        collision = dat["collision"]
        if collision:
            color = "r"
        else:
            color = "g"
        ax4.scatter(agent_pos[0], agent_pos[1], agent_pos[2], c=color, s=3)

    ax4.set_xlabel("X")
    ax4.set_ylabel("Y")

    ax4.set_xlim(-0.5, 3.5)
    ax4.set_ylim(-2, 2)
    ax4.set_zlim(-1., 3)
    ax4.view_init(20, -100)

    labels = ["Free", "Collision"]
    skill_names = ["Walk", "Jump"]

    ax4.set_title("Skill: {}\nPred: {:.2f}\nLabel: {}".format(skill_names[skill_id],
                                                              pred, labels[gt]))

    ax1 = fig.add_subplot(1, 4, 1)
    ax1.imshow(obs["sensor_data"][0]["agent_color"])

    ax2 = fig.add_subplot(1, 4, 2)
    ax2.imshow(obs["sensor_data"][0]["agent_depth"])

    ax3 = fig.add_subplot(1, 4, 3)
    ax3.imshow(obs["sensor_data"][0]["agent_segm"])



    plt.show()

def merge_defaultdict(d1, d2):
    res = defaultdict(list)
    for k in d1:
        res[k] = d1[k] + d2[k]
    return res

class Evaluator():
    def __init__(self, model, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint['model'], strict=True)
        model.eval()
        self.model = model

    def predict(self, obs):
        dep_map = obs["agent_depth"]
        seg_map = obs["agent_segm"] * 127. / 255.

        data_np = np.concatenate([np.expand_dims(dep_map, 0),
                                  np.expand_dims(seg_map, 0)],
                                  axis=0)
        data_np = np.expand_dims(data_np, axis=0).astype("float32")
        data = torch.from_numpy(data_np)
        ret = self.model(data).cpu()
        return ret

model: torch.nn.Module = SkillClassifier(num_cls=2, num_in_c=2)
ckpt_path = "./checkpoints/training_2022-11-04-01:55:21/model_00008000.ckpt"
evaluator = Evaluator(model=model, ckpt_path=ckpt_path)


def main(skill_id):
    env = dummy_env(render=True, cam_w=160, cam_h=120)

    for i in range(100):
        sensor_data = env.reset()
        pred = evaluator.predict(sensor_data)
        pred = pred.squeeze().detach().numpy()
        skill_pred = sigmoid(pred[skill_id])

        if skill_id == 1:
            action = [0]
            obs1 = env.step(action)
            action = [1]
            obs2 = env.step(action)
            obs = merge_defaultdict(obs1, obs2)
        elif skill_id == 0:
            action = [0]
            obs = env.step(action)
        else:
            raise NotImplemented

        gt = get_gt(obs["sensor_data"])

        visualize(obs, skill_pred, gt, skill_id=skill_id)

        # x = input("input key: ")
        # if x == "" or str(x).lower() == "y":
        #     pass
        # else:
        #     break

main(skill_id=0)