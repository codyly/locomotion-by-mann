import numpy as np
import torch
import torch.nn as nn


class ExpertWeights(object):
    def __init__(self, rng, shape, name) -> None:
        super().__init__()
        self.init_rng = rng
        self.name = name

        self.weight_shape = shape
        self.bias_shape = (shape[0], shape[1], 1)

        self.alpha = nn.Parameter(self.init_alpha())
        self.beta = nn.Parameter(self.init_beta())

    def init_alpha(self):
        shape = self.weight_shape
        rng = self.init_rng
        alpha_bound = np.sqrt(6.0 / np.prod(shape[-2:]))
        alpha_numpy = np.asarray(rng.uniform(low=-alpha_bound, high=alpha_bound, size=shape), dtype=np.float32)
        return torch.from_numpy(alpha_numpy)

    def init_beta(self):
        return torch.zeros(self.bias_shape, dtype=torch.float32)

    def get_nn_weight(self, control_weights, batch_size):
        a = torch.unsqueeze(self.alpha, dim=1)
        a = a.repeat(1, batch_size, 1, 1)
        w = torch.unsqueeze(torch.unsqueeze(control_weights, dim=-1), dim=-1)
        r = w * a

        return torch.sum(r, dim=0)

    def get_nn_bias(self, control_weights, batch_size):
        b = torch.unsqueeze(self.beta, dim=1)
        b = b.repeat(1, batch_size, 1, 1)
        w = torch.unsqueeze(torch.unsqueeze(control_weights, dim=-1), dim=-1)
        r = w * b

        return torch.sum(r, dim=0)

    def load_pretrained(self, cp_id, path):
        pretrained_alpha = torch.zeros(self.weight_shape)
        pretrained_beta = torch.zeros(self.bias_shape)
        for j in range(self.weight_shape[0]):
            pretrained_alpha[j] = torch.from_numpy(
                np.fromfile(path + f"/cp{cp_id}_a{j}.bin", dtype=np.float32)
                .astype(np.float32)
                .reshape(self.weight_shape[1:]),
            )
            pretrained_beta[j] = torch.from_numpy(
                np.fromfile(path + f"/cp{cp_id}_b{j}.bin", dtype=np.float32)
                .astype(np.float32)
                .reshape(self.bias_shape[1:]),
            )

        self.alpha.data = pretrained_alpha
        self.beta.data = pretrained_beta


from numpy.random import RandomState


def test():
    rng = RandomState(23456)
    except_weights = ExpertWeights(rng, (8, 512, 480), "layer0")
    except_weights.load_pretrained(0, "trained")


if __name__ == "__main__":
    test()
