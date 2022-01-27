import os

import numpy as np
import torch
import torch.nn as nn
from numpy.random import random, RandomState
from MANN.expert_weights import ExpertWeights
from MANN.gating_network import Gating


class MANN(nn.Module):
    def __init__(
        self,
        rng=RandomState(23456),
        num_experts=8,
        input_size=480,
        output_size=363,
        hidden_size=512,
        hidden_size_gt=32,
        index_gating=[285, 286, 287, 345, 346, 347, 393, 394, 395, 441, 442, 443, 84, 85, 86, 87, 88, 89, 90],
        batch_size=1,
        keep_prob_ini=1.0,
        pretrained_path="MANN/trained",
    ):
        super().__init__()
        self.rng = rng

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        # gatingNN
        self.num_experts = num_experts
        self.gating_hidden_size = hidden_size_gt
        self.index_gating = index_gating

        # hyperpara
        self.batch_size = batch_size
        self.keep_prob = keep_prob_ini
        self.dropout_prob = 1 - keep_prob_ini

        self.init(pretrained_path)

    def init(self, pretrained_path=None):
        self.gating_input_size = len(self.index_gating)
        self.gating_nn = Gating(
            rng=self.rng,
            input_size=self.gating_input_size,
            output_size=self.num_experts,
            hidden_size=self.gating_hidden_size,
            keep_prob=self.keep_prob,
        )

        self.layer0 = ExpertWeights(
            rng=self.rng, shape=(self.num_experts, self.hidden_size, self.input_size), name="layer0"
        )
        self.layer1 = ExpertWeights(
            rng=self.rng, shape=(self.num_experts, self.hidden_size, self.hidden_size), name="layer1"
        )
        self.layer2 = ExpertWeights(
            rng=self.rng, shape=(self.num_experts, self.output_size, self.hidden_size), name="layer2"
        )

        self.input_mean = None
        self.input_std = None
        self.output_mean = None
        self.output_std = None

        if pretrained_path is not None and os.path.exists(pretrained_path):
            self.gating_nn.load_pretrained(pretrained_path)
            self.layer0.load_pretrained(0, pretrained_path)
            self.layer1.load_pretrained(1, pretrained_path)
            self.layer2.load_pretrained(2, pretrained_path)

            self.input_mean = np.fromfile(pretrained_path + "/Xmean.bin", dtype=np.float32)
            self.input_std = np.fromfile(pretrained_path + "/Xstd.bin", dtype=np.float32)
            self.output_mean = np.fromfile(pretrained_path + "/Ymean.bin", dtype=np.float32)
            self.output_std = np.fromfile(pretrained_path + "/Ystd.bin", dtype=np.float32)

    def forward(self, x):
        gating_nn_input = torch.transpose(Gating.getInput(x, self.index_gating), 0, 1)

        BC = self.gating_nn(gating_nn_input)

        w0 = self.layer0.get_nn_weight(BC, self.batch_size)
        w1 = self.layer1.get_nn_weight(BC, self.batch_size)
        w2 = self.layer2.get_nn_weight(BC, self.batch_size)

        b0 = self.layer0.get_nn_bias(BC, self.batch_size)
        b1 = self.layer1.get_nn_bias(BC, self.batch_size)
        b2 = self.layer2.get_nn_bias(BC, self.batch_size)

        H0 = torch.unsqueeze(x, dim=-1)  # ?*in -> ?*in*1
        H0 = nn.Dropout(p=self.dropout_prob)(H0)

        H1 = torch.matmul(w0, H0) + b0  # ?*out*in mul ?*in*1 + ?*out*1 = ?*out*1
        H1 = nn.ELU()(H1)
        H1 = nn.Dropout(p=self.dropout_prob)(H1)

        H2 = torch.matmul(w1, H1) + b1
        H2 = nn.ELU()(H2)
        H2 = nn.Dropout(p=self.dropout_prob)(H2)

        H3 = torch.matmul(w2, H2) + b2
        H3 = torch.squeeze(H3, -1)

        return H3


def test():
    mann = MANN()
    mann.eval()

    x = torch.from_numpy(random([1, 480]).astype(np.float32))

    assert mann(x).shape[1] == 363


if __name__ == "__main__":
    test()
