from typing import List

import numpy as np
from scipy.spatial.transform import Rotation as R

from animation import common as C
from animation import utils as U


class Trajectory:
    def __init__(self) -> None:
        self.points: List[Point] = []


class Point:
    def __init__(self, traj_vec=None, index=0) -> None:
        self.index = index
        self.velocity = np.zeros(3)
        self.transformation = np.identity(4)
        self.set_direction(C.VEC_FORWARD)
        self.speed = 0
        if traj_vec is None:
            self.styles = C.STYLE_NOMOVE
        else:
            self.speed = traj_vec[6]
            self.styles = traj_vec[7:]

    def set_position(self, position):
        self.transformation[0, 3] = position[0]
        self.transformation[1, 3] = position[1]
        self.transformation[2, 3] = position[2]

    def get_position(self):
        return np.array([self.transformation[0, 3], self.transformation[1, 3], self.transformation[2, 3]])

    def set_direction(self, direction):
        if np.all(direction == 0):
            corrected_dir = C.VEC_FORWARD
        else:
            corrected_dir = direction

        self.set_rotation(U.quat_look_at(corrected_dir))

    def set_rotation(self, quat: np.ndarray) -> None:
        rmat_3_3 = R.from_quat(quat).as_matrix()

        self.transformation[:3, 0] = rmat_3_3.dot(C.VEC_RIGHT)
        self.transformation[:3, 1] = rmat_3_3.dot(C.VEC_UP)
        self.transformation[:3, 2] = rmat_3_3.dot(C.VEC_FORWARD)

    def get_direction(self) -> np.ndarray:
        fwd = self.transformation[:3, :3].dot(C.VEC_FORWARD)
        fwd[1] = 0
        fwd /= np.linalg.norm(fwd)
        return fwd


class PID:
    def __init__(self, p_gain=0.2, i_gain=0.8, d_gain=0.0):
        self.p_gain = p_gain
        self.i_gain = i_gain
        self.d_gain = d_gain
        self.integrator = 0
        self.differentiator = 0
        self.last_err = 0
        self.err = 0

    def update(self, target, current, update):
        self.err = target - current
        self.integrator += self.err * update
        self.differentiator = (self.err - self.last_err) / update
        self.last_err = self.err
        value = self.err * self.p_gain + self.integrator * self.i_gain + self.differentiator * self.d_gain

        return value

    def reset(self, err=0, integrator=0, differentiator=0, last_err=0):
        self.integrator = integrator
        self.differentiator = differentiator
        self.last_err = last_err
        self.err = err


def test():
    axis = np.array([0, 1, 0])
    angle = np.pi / 2

    quat = U.quat_from_axis_angle(axis, angle)

    rot = R.from_quat(quat)

    direction = rot.as_matrix().dot(np.array([0, 0, 1]))

    pt = Point()
    pt.set_direction(direction)

    print(pt.transformation)
    print(rot.as_matrix())


if __name__ == "__main__":
    test()
