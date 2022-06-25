import numpy as np

from animation import common as C
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


class LocoPath:
    def __init__(self, path, scale_x=1, scale_y=1, num_frames=100) -> None:
        self.ptrs = np.loadtxt(path, dtype=np.float32)
        self.ptrs = -self.ptrs
        # self.ptrs[:, 1] = -self.ptrs[:, 1]

        scales = self.ptrs.max(axis=0) - self.ptrs.min(axis=0)
        self.ptrs[:, 0] = self.ptrs[:, 0] * scale_x / scales[0]
        self.ptrs[:, 1] = self.ptrs[:, 1] * scale_y / scales[1]

        # Define some points:
        points = self.ptrs

        # Linear length along the line:
        distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
        distance = np.insert(distance, 0, 0) / distance[-1]

        # Interpolation for different methods:
        interpolations_methods = ["slinear", "quadratic", "cubic"]
        alpha = np.linspace(0, 1, num_frames)

        interpolated_points = {}
        for method in interpolations_methods:
            interpolator = interp1d(distance, points, kind=method, axis=0)
            interpolated_points[method] = interpolator(alpha)

        self.interp_ptrs = interpolated_points
        self.num_frames = num_frames

        # # Graph:
        plt.figure(figsize=(7, 7))
        for method_name, curve in interpolated_points.items():
            plt.plot(*curve.T, "-", label=method_name)

        plt.plot(*points.T, "ok", label="original points")
        plt.axis("equal")
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("y")
        # plt.show()

    def get_pos(self, frame_id, method="cubic"):
        return np.array([self.interp_ptrs[method][frame_id][1], 0, self.interp_ptrs[method][frame_id][0]])

    def get_fwd(self, frame_id, method="cubic"):
        if frame_id >= self.num_frames - 1:
            return C.VEC_FORWARD

        v = self.get_pos(frame_id=frame_id + 1, method=method) - self.get_pos(frame_id=frame_id, method=method)

        return v / np.linalg.norm(v)


if __name__ == "__main__":
    t = LocoPath(path="./outputs/trajectory/star.txt")
