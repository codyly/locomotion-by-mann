import numpy as np
import torch

from animation.kinematics import Trajectory, Point, PID
from animation.controller import SimInputHandler, Controller, BulletInputHandler
from animation import common as C
from animation import utils as U
from animation import profiles as P
from MANN.mann_network import MANN


class Animation:
    def __init__(self, profile=P.trot) -> None:
        self.profile = profile
        self.root_point_id = 6
        self.input_num = 12
        self.output_num = 6
        self.bone_num = 27
        self.frame_rate = 60
        self.trajectory_dim_input = 13
        self.trajectory_dim_output = 6
        self.future_num = 5
        self.bone_dim_input = 12
        self.bone_dim_output = 12

        self.target_gain = 0.25
        self.target_decay = 0.05
        self.trajectory_control = True
        self.trajectory_correction = 1.0
        self.target_direction = C.VEC_FORWARD
        self.target_velocity = np.array([0, 0, 0])

        self.pid = PID()

        if profile == 'manual':
            # Animate based on Keyboard input from PyBullet
            self.input_handler = BulletInputHandler()
        else:
            # Animate predefined instruction set
            self.input_handler = SimInputHandler(profile=self.profile.inst, need_parse=True, looping=True)
        self.controller = Controller(input_handler=self.input_handler)

        self.mann = MANN()
        self.mann.eval()

    def gen_input(self):
        base_input = np.zeros([self.input_num, self.trajectory_dim_input])
        bone_input = np.zeros([self.bone_num, self.bone_dim_input])
        cur_root = self.trajectory.points[self.root_point_id].transformation
        cur_root[1, 3] = 0  # TODO why remove x position?

        for i in range(self.input_num):
            pos = U.mat_multi_pos(np.linalg.inv(cur_root), self.trajectory.points[i].get_position())
            dir = U.mat_multi_vec(np.linalg.inv(cur_root[:3, :3]), self.trajectory.points[i].get_direction())
            vel = U.mat_multi_vec(np.linalg.inv(cur_root[:3, :3]), self.trajectory.points[i].velocity)
            base_input[i, 0] = pos[0]  #x
            base_input[i, 1] = pos[2]  #z  (y is upper height direction)
            base_input[i, 2] = dir[0]
            base_input[i, 3] = dir[2]
            base_input[i, 4] = vel[0]
            base_input[i, 5] = vel[2]
            base_input[i, 6] = self.trajectory.points[i].speed
            base_input[i, 7:] = self.trajectory.points[i].styles  # one-hot 6D vector -> Total 13 D base_input

        prev_root = self.trajectory.points[self.root_point_id - 1].transformation
        prev_root[1, 3] = 0

        for i in range(self.bone_num):
            pos = U.mat_multi_pos(np.linalg.inv(prev_root), self.bone_pos[i])
            fwd = U.mat_multi_vec(np.linalg.inv(prev_root[:3, :3]), self.bone_fwd[i])
            up = U.mat_multi_vec(np.linalg.inv(prev_root[:3, :3]), self.bone_up[i])
            vel = U.mat_multi_vec(np.linalg.inv(prev_root[:3, :3]), self.bone_vel[i])
            bone_input[i, 0:3] = pos
            bone_input[i, 3:6] = fwd
            bone_input[i, 6:9] = up
            bone_input[i, 9:] = vel

        net_input = np.concatenate([base_input.flatten(), bone_input.flatten()])

        self.current_transform_mat = cur_root.copy()
        self.previous_transform_mat = prev_root.copy()

        return net_input[np.newaxis, :]

    def get_curvature(self, start: int = 0, end: int = 11, step: int = 1):
        curvature: float = 0.0
        for i in range(step, end, step):
            v1 = self.trajectory.points[i].get_position() - self.trajectory.points[i - step].get_position()
            v2 = self.trajectory.points[i + step].get_position() - self.trajectory.points[i].get_position()

            if np.linalg.norm(v1) * np.linalg.norm(v2) != 0:
                curvature += U.signed_angle(v1, v2, up=C.VEC_UP, deg=True)

        curvature = np.abs(curvature)
        curvature = np.clip(curvature / 180, 0, 1)

        return curvature

    def pool_bias(self):
        styles = self.trajectory.points[self.root_point_id].styles
        bias = 0
        for i in range(C.NUM_STYLES):
            _bias = self.controller.styles[i].bias
            # TODO: support key multiplier

            bias += styles[i] * _bias
        return bias

    def predict_trajectory(self):
        # turn left: -1;  turn right: 1
        turn: float = self.controller.query_turn()

        # W/A/S/D
        move: np.ndarray = self.controller.query_move()

        style: np.ndarray = self.controller.query_styles()

        control: bool = (turn != 0.0) or (np.linalg.norm(move) != 0) or (style[1] != 0)

        # process control
        curvature: float = self.get_curvature(0, 11, 1)

        target: float = self.pool_bias()

        current_vel: float = np.linalg.norm(self.trajectory.points[self.root_point_id].velocity)
        bias: float = target

        if turn == 0:
            bias += self.pid.update(U.lerp(target, current_vel, curvature), current_vel, 1 / self.frame_rate)
        else:
            self.pid.reset()

        move = bias * U.vec_normalize(move)

        mag_move = np.linalg.norm(move)
        if mag_move == 0 and turn != 0:
            move = 2 / 3.0 * C.VEC_FORWARD
        else:
            if move[2] == 0 and turn != 0:
                move = np.array([move[0], 0, 1])
                move = U.vec_normalize(move)
                move = bias * move
            else:
                move = U.lerp(move, bias * C.VEC_FORWARD, current_vel / 6)

        if style[2] == 0:
            style[1] = max(style[1], np.clip(current_vel, 0, 1) ** 2)
            if style[1] > 0:
                move[2] = max(move[2], 0.1 * style[1])
        else:
            move[2] = bias
            move[0] = 0
            turn = 0
            if curvature > 0.25:
                style = C.STYLE_TROT
            else:
                if current_vel < 0.5:
                    style = C.STYLE_TROT
                else:
                    style = C.STYLE_JUMP

        if style[3] > 0 or style[4] > 0 or style[5] > 0:
            bias = 0
            if current_vel > 0.5:
                style[3:6] = 0

        rate = self.target_gain if control else self.target_decay
        self.target_direction = U.lerp(
            self.target_direction,
            U.quat_rot_vec(
                U.quat_from_axis_angle(C.VEC_UP, turn * 60), self.trajectory.points[self.root_point_id].get_direction()
            ),
            rate,
        )

        if np.linalg.norm(move) != 0:
            self.target_velocity = U.lerp(
                self.target_velocity,
                np.linalg.norm(move) * U.quat_rot_vec(U.quat_look_at(self.target_direction, C.VEC_UP), move),
                rate,
            )
            self.trajectory_correction = U.lerp(self.trajectory_correction, max(1, np.abs(turn)), rate)
        else:
            self.target_velocity = U.lerp(self.target_velocity, np.zeros(3), rate)
            self.trajectory_correction = U.lerp(self.trajectory_correction, max(0, np.abs(turn)), rate)

        trajectory_pos_blend = np.zeros([self.input_num, 3])
        trajectory_pos_blend[self.root_point_id] = self.trajectory.points[self.root_point_id].get_position()

        for i in range(self.root_point_id + 1, self.input_num):
            bias_pos = 0.75
            bias_dir = 1.25
            bias_vel = 1.0
            weight = (i - self.root_point_id) / self.future_num
            scale_pos = 1 - (1 - weight) ** bias_pos
            scale_dir = 1 - (1 - weight) ** bias_dir
            scale_vel = 1 - (1 - weight) ** bias_vel
            scale = 1 / (self.input_num - self.root_point_id - 1)
            trajectory_pos_blend[i] = trajectory_pos_blend[i - 1] + U.lerp(
                self.trajectory.points[i].get_position() - self.trajectory.points[i - 1].get_position(),
                scale * self.target_velocity,
                scale_pos,
            )
            self.trajectory.points[i].set_direction(
                U.lerp(self.trajectory.points[i].get_direction(), self.target_direction, scale_dir)
            )
            self.trajectory.points[i].velocity = U.lerp(
                self.trajectory.points[i].velocity, self.target_velocity, scale_vel
            )

        for i in range(self.root_point_id + 1, self.input_num):
            self.trajectory.points[i].set_position(trajectory_pos_blend[i])

        for i in range(self.root_point_id, self.input_num):
            weight = (i - self.root_point_id) / self.future_num
            for j in range(C.NUM_STYLES):
                self.trajectory.points[i].styles[j] = U.lerp(
                    self.trajectory.points[i].styles[j],
                    style[j],
                    U.remap(weight, 0, 1, self.controller.styles[j].transition, 1),
                )
            self.trajectory.points[i].styles = U.style_normalize(self.trajectory.points[i].styles)
            rate = self.target_gain if control else self.target_decay
            self.trajectory.points[i].speed = U.lerp(
                self.trajectory.points[i].speed, np.linalg.norm(self.target_velocity), rate
            )

    def gen_frame(self):

        Xmean = self.mann.input_mean
        Xstd = self.mann.input_std

        Ymean = self.mann.output_mean
        Ystd = self.mann.output_std

        initial_data = np.loadtxt(C.MOCAP_SAMPLE_PATH, dtype=np.float32, max_rows=1)[np.newaxis, :]

        base_input = np.zeros([self.input_num, self.trajectory_dim_input])
        base_input[:, :7] = initial_data[0, : 12 * 7].reshape([self.input_num, 7])
        base_input[:, 6] = 0
        base_input[:, 7] = 1

        self.trajectory = Trajectory()
        for i in range(self.input_num):
            self.trajectory.points.append(Point(base_input[i], i))

        bone_input = np.zeros([self.bone_num, self.bone_dim_input])
        bone_input = initial_data[0, self.trajectory_dim_input * self.input_num :].reshape(
            [self.bone_num, self.bone_dim_input]
        )

        current_root = self.trajectory.points[self.root_point_id].transformation
        current_root[1, 3] = 0

        previous_root = self.trajectory.points[self.root_point_id].transformation
        previous_root[1, 3] = 0

        self.bone_pos = []
        self.bone_fwd = []
        self.bone_up = []
        self.bone_vel = []
        for i in range(self.bone_num):
            self.bone_pos.append(U.mat_multi_pos(previous_root, bone_input[i, 0:3]))
            self.bone_fwd.append(U.mat_multi_vec(previous_root, bone_input[i, 3:6]))
            self.bone_up.append(U.mat_multi_vec(previous_root, bone_input[i, 6:9]))
            self.bone_vel.append(np.zeros(3))
        import time
        while True:
            t_start = time.time()

            self.controller.get_input()
            t_input = time.time()

            self.predict_trajectory()
            t_traj = time.time()

            input_norm = U.normalize(self.gen_input(), Xmean, Xstd).astype(np.float32)
            input_mann = torch.from_numpy(input_norm)
            t_norm = time.time()

            output_mann = self.mann(input_mann)
            t_mann = time.time()
            output_numpy = output_mann.to("cpu").detach().numpy()
            output = U.renormalize(output_numpy, Ymean, Ystd)
            t_unnorm = time.time()

            # update past trajectory
            for i in range(self.root_point_id):
                self.trajectory.points[i].set_position(self.trajectory.points[i + 1].get_position())
                self.trajectory.points[i].set_direction(self.trajectory.points[i + 1].get_direction())
                self.trajectory.points[i].velocity = self.trajectory.points[i + 1].velocity
                self.trajectory.points[i].speed = self.trajectory.points[i + 1].speed
                self.trajectory.points[i].styles = self.trajectory.points[i + 1].styles

            # update root
            update_part_1 = (1.0 - self.trajectory.points[self.root_point_id].styles[0]) ** 0.25
            update_part_2 = (
                1.0
                - self.trajectory.points[self.root_point_id].styles[3]
                - self.trajectory.points[self.root_point_id].styles[4]
                - self.trajectory.points[self.root_point_id].styles[5]
            ) ** 0.5
            update = min(update_part_1, update_part_2)

            root_motion = update * output[0, -3:] / self.frame_rate
            translation = np.array([root_motion[0], 0, root_motion[2]])
            angle = np.deg2rad(root_motion[1])
            self.trajectory.points[self.root_point_id].set_position(
                U.mat_multi_pos(self.current_transform_mat, translation)
            )

            quat = U.quat_from_axis_angle(C.VEC_UP, angle)

            self.trajectory.points[self.root_point_id].set_direction(
                U.quat_rot_vec(quat, self.trajectory.points[self.root_point_id].get_direction())
            )
            self.trajectory.points[self.root_point_id].velocity = (
                U.mat_multi_vec(self.current_transform_mat, translation) * self.frame_rate
            )

            self.next_transform_mat = self.trajectory.points[self.root_point_id].transformation
            self.next_transform_mat[1, 3] = 0

            t_update_past_traj = time.time()

            # update future
            for i in range(self.root_point_id + 1, self.input_num):
                self.trajectory.points[i].set_position(
                    self.trajectory.points[i].get_position() + U.mat_multi_pos(self.next_transform_mat, translation)
                )
                self.trajectory.points[i].set_direction(U.quat_rot_vec(quat, self.trajectory.points[i].get_direction()))
                self.trajectory.points[i].velocity = (
                    self.trajectory.points[i].velocity
                    + U.mat_multi_vec(self.next_transform_mat, translation) * self.frame_rate
                )

            start = 0
            for i in range(self.root_point_id + 1, self.input_num):
                index = i
                next_sample_id = index

                next_pos = U.mat_multi_pos(
                    self.next_transform_mat,
                    np.array(
                        [
                            output[0, start + (next_sample_id - 6) * self.trajectory_dim_output + 0],
                            0,
                            output[0, start + (next_sample_id - 6) * self.trajectory_dim_output + 1],
                        ]
                    ),
                )

                next_dir = U.mat_multi_vec(
                    self.next_transform_mat,
                    np.array(
                        [
                            output[0, start + (next_sample_id - 6) * self.trajectory_dim_output + 2],
                            0,
                            output[0, start + (next_sample_id - 6) * self.trajectory_dim_output + 3],
                        ]
                    ),
                )

                next_vel = U.mat_multi_vec(
                    self.next_transform_mat,
                    np.array(
                        [
                            output[0, start + (next_sample_id - 6) * self.trajectory_dim_output + 4],
                            0,
                            output[0, start + (next_sample_id - 6) * self.trajectory_dim_output + 5],
                        ]
                    ),
                )

                direction = next_dir / np.linalg.norm(next_dir)
                pos = U.lerp(self.trajectory.points[i].get_position(), next_pos, self.trajectory_correction)
                vel = U.lerp(self.trajectory.points[i].velocity, next_vel, self.trajectory_correction)

                self.trajectory.points[i].set_position(pos)
                self.trajectory.points[i].velocity = vel
                self.trajectory.points[i].set_direction(direction)

            start += self.trajectory_dim_output * self.output_num

            for i in range(self.bone_num):
                pos = U.mat_multi_pos(
                    self.current_transform_mat,
                    output[0, start + i * self.bone_dim_output + 0 : start + i * self.bone_dim_output + 3],
                )
                forward = U.mat_multi_vec(
                    self.current_transform_mat,
                    output[0, start + i * self.bone_dim_output + 3 : start + i * self.bone_dim_output + 6],
                )
                up = U.mat_multi_vec(
                    self.current_transform_mat,
                    output[0, start + i * self.bone_dim_output + 6 : start + i * self.bone_dim_output + 9],
                )
                velocity = U.mat_multi_vec(
                    self.current_transform_mat,
                    output[0, start + i * self.bone_dim_output + 9 : start + i * self.bone_dim_output + 12],
                )

                # self.bone_pos[i] = po
                self.bone_pos[i] = (self.bone_pos[i] + velocity / self.frame_rate + pos) / 2
                self.bone_fwd[i] = forward
                self.bone_up[i] = up
                self.bone_vel[i] = velocity
        
            t_update_future_traj = time.time()

            start += self.bone_dim_output * self.bone_num
            if self.print_once:
                self.print_once = False
                print("\n[MANN Timing] %d(ms)"%((t_update_future_traj-t_start)*1000))
                print('input %d(ms) '% ((t_input-t_start)*1000),
                        'set traj %d(ms) '%((t_traj-t_input)*1000),
                        'normalize %d(ms) '%((t_norm-t_traj)*1000),
                        'mann %d(ms) '%((t_mann-t_norm)*1000),
                        'unnormalize %d(ms) '%((t_unnorm-t_mann)*1000),
                        'update past traj: %d(ms) '%((t_update_past_traj-t_unnorm)*1000),
                        'update future traj; %d(ms)'%((t_update_future_traj-t_update_past_traj)*1000)
                )

            yield np.concatenate(self.bone_pos)

    def get_root_styles(self):
        return self.trajectory.points[self.root_point_id].styles
