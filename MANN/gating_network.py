import numpy as np
import torch
import torch.nn as nn


class Gating(nn.Module):
    def __init__(self, rng, input_size, output_size, hidden_size, keep_prob):
        super().__init__()
        """rng"""
        self.init_rng = rng

        """dropout"""
        self.keep_prob = keep_prob
        self.dropout_prob = 1 - keep_prob

        """size"""
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        """parameters"""
        self.w0 = nn.Parameter(self.initial_weight([hidden_size, input_size]))
        self.w1 = nn.Parameter(self.initial_weight([hidden_size, hidden_size]))
        self.w2 = nn.Parameter(self.initial_weight([output_size, hidden_size]))

        self.b0 = nn.Parameter(self.initial_bias([hidden_size, 1]))
        self.b1 = nn.Parameter(self.initial_bias([hidden_size, 1]))
        self.b2 = nn.Parameter(self.initial_bias([output_size, 1]))

    def initial_weight(self, shape):
        rng = self.init_rng
        weight_bound = np.sqrt(6.0 / np.sum(shape[-2:]))
        weight = np.asarray(rng.uniform(low=-weight_bound, high=weight_bound, size=shape), dtype=np.float32)
        return torch.from_numpy(weight)

    def initial_bias(self, shape):
        return torch.zeros(shape, dtype=torch.float32)

    def forward(self, x):
        H0 = nn.Dropout(p=self.dropout_prob)(x)

        H1 = torch.matmul(self.w0, H0) + self.b0
        H1 = nn.ELU()(H1)
        H1 = nn.Dropout(p=self.dropout_prob)(H1)

        H2 = torch.matmul(self.w1, H1) + self.b1
        H2 = nn.ELU()(H2)
        H2 = nn.Dropout(p=self.dropout_prob)(H2)

        H3 = torch.matmul(self.w2, H2) + self.b2
        H3 = nn.Softmax(dim=0)(H3)

        return H3

    def load_pretrained(self, path):
        self.w0.data = torch.from_numpy(
            np.fromfile(path + "/wc0_w.bin", dtype=np.float32).astype(np.float32).reshape(self.w0.shape)
        )
        self.w1.data = torch.from_numpy(
            np.fromfile(path + "/wc1_w.bin", dtype=np.float32).astype(np.float32).reshape(self.w1.shape)
        )
        self.w2.data = torch.from_numpy(
            np.fromfile(path + "/wc2_w.bin", dtype=np.float32).astype(np.float32).reshape(self.w2.shape)
        )
        self.b0.data = torch.from_numpy(
            np.fromfile(path + "/wc0_b.bin", dtype=np.float32).astype(np.float32).reshape(self.b0.shape)
        )
        self.b1.data = torch.from_numpy(
            np.fromfile(path + "/wc1_b.bin", dtype=np.float32).astype(np.float32).reshape(self.b1.shape)
        )
        self.b2.data = torch.from_numpy(
            np.fromfile(path + "/wc2_b.bin", dtype=np.float32).astype(np.float32).reshape(self.b2.shape)
        )

    # get the velocity of joints, desired velocity and style
    @classmethod
    def getInput(cls, data, index_joint):
        gating_input = data[..., index_joint[0] : index_joint[0] + 1]
        for i in index_joint[1:]:
            gating_input = torch.cat([gating_input, data[..., i : i + 1]], axis=-1)
        return gating_input
