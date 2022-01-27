import os.path
import numpy as np
from scipy.spatial.transform import Rotation as R

from animation import common as C


def build_path(path):
    for i in path:
        if not os.path.exists(i):
            os.makedirs(i)


def Normalize(X, axis, savefile=None):
    Xmean, Xstd = X.mean(axis=axis), X.std(axis=axis)
    for i in range(Xstd.size):
        if Xstd[i] == 0:
            Xstd[i] = 1
    X = (X - Xmean) / Xstd
    if savefile is not None:
        Xmean.tofile(savefile + "mean.bin")
        Xstd.tofile(savefile + "std.bin")
    return X


def normalize_from_file(X, path, name):

    Xmean = np.fromfile(path + "/" + name + "mean.bin", dtype=np.float32)
    Xstd = np.fromfile(path + "/" + name + "std.bin", dtype=np.float32)

    for i in range(Xstd.size):
        if Xstd[i] == 0:
            Xstd[i] = 1

    X = (X - Xmean) / Xstd

    return X


def normalize(X, Xmean, Xstd):

    X = (X - Xmean) / Xstd

    return X


def renormalize(X, X_mean, Xstd):

    X = X * Xstd + X_mean

    return X


def quat_look_at(forward: np.ndarray, upward: np.ndarray = C.VEC_UP) -> np.ndarray:
    forward = forward / np.linalg.norm(forward)
    upward = upward / np.linalg.norm(upward)

    dot = forward.dot(C.VEC_FORWARD)

    if np.abs(dot - (-1.0)) < 1e-6:
        rot_axis = upward
        rot_angle = np.pi

    elif np.abs(dot - (1.0)) < 1e-6:
        rot_axis = np.zeros(3)
        rot_angle = 0

    else:
        rot_angle = np.arccos(dot)

        rot_axis = np.cross(C.VEC_FORWARD, forward)
        rot_axis /= np.linalg.norm(rot_axis)

    return quat_from_axis_angle(rot_axis, rot_angle)


def quat_from_axis_angle(axis, angle):
    half = angle * 0.5
    s = np.sin(half)
    q = np.zeros(4)
    q[0] = s * axis[0]
    q[1] = s * axis[1]
    q[2] = s * axis[2]
    q[3] = np.cos(half)

    return q


def quat_rot_vec(quat, vec):
    r = R.from_quat(quat)
    rvec = r.as_matrix().dot(vec)
    return rvec / np.linalg.norm(rvec)


def mat_multi_pos(mat, vec):
    if vec.shape[0] == 3:
        v = np.concatenate([vec, [1]])
    else:
        v = vec

    res = mat.dot(v)
    if vec.shape[0] == 3:
        return res[:3]
    else:
        return res


def mat_multi_vec(mat, vec):
    if mat.shape[0] > 3:
        res = mat[:3, :3].dot(vec)
    else:
        res = mat.dot(vec)

    return res


def signed_angle(v1, v2, up=C.VEC_UP, deg=False):
    n1 = v1 / np.linalg.norm(v1)
    n2 = v2 / np.linalg.norm(v2)
    cos = np.dot(n1, n2)

    sign = np.sign(up.dot(np.cross(v1, v2)))

    if deg:
        return sign * np.rad2deg(np.arccos(cos))
    else:
        return sign * np.arccos(cos)


def lerp(v1, v2, t):
    # assert t <= 1 and t >= 0
    t = np.clip(t, 0, 1)
    return v1 + (v2 - v1) * t


def remap(v, v_min, v_max, n_min, n_max):
    if v_max != v_min:
        return (v - v_min) / (v_max - v_min) * (n_max - n_min) + n_min


def style_normalize(styles):
    s = 0
    for style in styles:
        s += np.abs(style)

    if s != 0:
        styles = np.abs(styles) / s

    return styles


def vec_normalize(vec):
    if np.linalg.norm(vec) == 0:

        return vec

    else:
        return vec / np.linalg.norm(vec)
