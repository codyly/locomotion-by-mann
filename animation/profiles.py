import numpy as np

from animation import common as C

NMV = " "
FWD = "w"
MLF = "a"
MRT = "d"
TLF = "q"
TRT = "e"
BWD = "s"
JMP = "j"
SIT = "i"
STD = "t"
LIE = "l"
DELIMITER = ","

NUM_QUERIES = C.NUM_QUERIES
BLOCK_ORDER = {NMV: 0, FWD: 1, JMP: 2, MLF: 3, MRT: 4}


class Profile:
    def __init__(self, name, stages, ops) -> None:
        self.name = name
        self.stages = stages
        self.ops = ops
        self.inst = self.build()

    def build(self):

        profile = []

        for i in range(len(self.stages)):
            for _ in range(int(self.stages[i] * NUM_QUERIES)):
                profile.append(self.ops[i])
                profile.append(DELIMITER)

        profile.append(NMV)

        return "".join(profile)


class ForwardProfile(Profile):
    def __init__(self, name, vel) -> None:
        self.name = name
        self.interp = ForwardGaitInterpolatior(velocity=vel)
        self.inst = self.averaging()
        print(self.inst[:60])

    def averaging(self) -> None:
        stage_lib = {self.interp.ops[i]: self.interp.stages[i] for i in range(len(self.interp.ops))}
        cmd_lib = sorted(self.interp.ops, reverse=True, key=lambda x: stage_lib[x])
        num_cmd_lib = {}
        num_sum = 0
        for cmd in cmd_lib[:-1]:
            num_cmd_lib[cmd] = round(C.NUM_QUERIES * stage_lib[cmd])
            num_sum += num_cmd_lib[cmd]
        num_cmd_lib[cmd_lib[-1]] = 600 - num_sum
        inst = [[cmd_lib[0]] for _ in range(num_cmd_lib[cmd_lib[0]])]
        num_base = len(inst)
        base_interval = C.DURATION / num_base
        for cmd in cmd_lib[1:]:
            num = num_cmd_lib[cmd]
            if num > 0:
                interval = num_base / num
                p = 0
                while num > 0:
                    if np.abs(p - interval * (num_cmd_lib[cmd] - num)) < 0.1:
                        inst[int(p)].append(cmd)
                        num -= 1
                    p += base_interval
                    if num > 0 and p >= num_base:
                        interval = num_base / num
                        p = interval - 1

        inst_out = []
        for i in range(num_base):
            for cmd in sorted(inst[i], key=lambda x: BLOCK_ORDER[x]):
                inst_out.append(cmd)
                inst_out.append(DELIMITER)
        inst_out.pop(-1)

        return "".join(inst_out)


class ForwardGaitInterpolatior:
    MAX_PACE_VELOCITY = 0.86
    MAX_JUMP_VELOCITY = 1.82

    def __init__(self, velocity, min_split=1.0 / C.SYS_FREQ) -> None:

        self.stages = []
        self.ops = []
        self.splits = int(1.0 / min_split)

        if self.splits > C.SYS_FREQ:
            raise ValueError("number of splits larger than control frequency!")

        self.min_split = min_split

        if velocity < self.MAX_PACE_VELOCITY:
            ratio = (self.MAX_PACE_VELOCITY - velocity) / self.MAX_PACE_VELOCITY
            nmv_interval = ratio
            fwd_interval = 1 - ratio
            self.stages = [nmv_interval, fwd_interval]
            self.ops = [NMV, FWD]

        elif velocity < self.MAX_JUMP_VELOCITY:
            ratio = (velocity - self.MAX_PACE_VELOCITY) / (self.MAX_JUMP_VELOCITY - self.MAX_PACE_VELOCITY)
            fwd_interval = 1 - ratio
            jmp_interval = ratio
            self.stages = [fwd_interval, jmp_interval]
            self.ops = [FWD, JMP]
        else:
            self.stages = [min_split, 1 - min_split]
            self.ops = [FWD, JMP]


class TurningProfile(Profile):
    def __init__(self, name, vel) -> None:
        self.name = name
        self.interp = TurningGaitInterpolatior(velocity=vel, angle=90)
        self.inst = self.averaging()
        print(self.inst[:60])

    def averaging(self) -> None:
        stage_lib = {self.interp.ops[i]: self.interp.stages[i] for i in range(len(self.interp.ops))}
        cmd_lib = sorted(self.interp.ops, reverse=True, key=lambda x: stage_lib[x])
        num_cmd_lib = {}
        num_sum = 0
        for cmd in cmd_lib[:-1]:
            num_cmd_lib[cmd] = round(C.NUM_QUERIES * stage_lib[cmd])
            num_sum += num_cmd_lib[cmd]
        num_cmd_lib[cmd_lib[-1]] = 600 - num_sum
        inst = [[cmd_lib[0]] for _ in range(num_cmd_lib[cmd_lib[0]])]
        num_base = len(inst)
        base_interval = C.DURATION / num_base
        for cmd in cmd_lib[1:]:
            num = num_cmd_lib[cmd]
            if num > 0:
                interval = num_base / num
                p = 0
                while num > 0:
                    if np.abs(p - interval * (num_cmd_lib[cmd] - num)) < 0.1:
                        inst[int(p)].append(cmd)
                        num -= 1
                    p += base_interval
                    if num > 0 and p >= num_base:
                        interval = num_base / num
                        p = interval - 1

        inst_out = []
        for i in range(num_base):
            for cmd in sorted(inst[i], key=lambda x: BLOCK_ORDER[x]):
                inst_out.append(cmd)
                inst_out.append(DELIMITER)
        inst_out.pop(-1)

        return "".join(inst_out)


class TurningGaitInterpolatior:
    MAX_PACE_VELOCITY = 0.86
    MAX_JUMP_VELOCITY = 1.82

    def __init__(self, velocity, angle, direction="right", min_split=1.0 / C.SYS_FREQ) -> None:

        self.stages = []
        self.ops = []
        self.splits = int(1.0 / min_split)
        self.turn_op = MLF if direction == "left" else MRT

        if self.splits > C.SYS_FREQ:
            raise ValueError("number of splits larger than control frequency!")

        self.min_split = min_split
        self.ops = [NMV, FWD, self.turn_op]
        self.stages = [0.5, 0.40, 0.1]


trot = Profile(name="trot", stages=[0.1, 0.3, 0.1], ops=[NMV, FWD, NMV])
full_speed_forwarding = Profile(name="full_speed_forwarding", stages=[0.01, 0.99], ops=[NMV, JMP])
full_speed_moving_left = Profile(name="full_speed_moving_left", stages=[0.01, 0.99], ops=[NMV, MLF])

acc_stop = Profile(name="acc_stop", stages=[0.1, 0.3, 0.3, 0.2, 0.1], ops=[NMV, FWD, JMP, BWD, NMV])

s_move = Profile(name="s_move", stages=[0.05, 0.1, 0.3, 0.3, 0.1, 0.05], ops=[NMV, FWD, MLF, MRT, FWD, NMV])

turn_around = Profile(
    name="turn_around", stages=[0.05, 0.1, 0.15, 0.3, 0.15, 0.1, 0.05], ops=[NMV, FWD, TLF, FWD, TRT, FWD, NMV]
)

sit_walk_sit_lie = Profile(
    name="sit_walk_sit_lie",
    stages=[0.05, 0.2, 0.1, 0.2, 0.15, 0.1, 0.1, 0.1],
    ops=[NMV, SIT, NMV, FWD, BWD, SIT, LIE, FWD],
)

lie_walk = Profile(
    name="lie_walk",
    stages=[0.05, 0.2, 0.1, 0.2, 0.15, 0.1, 0.1, 0.1],
    ops=[NMV, LIE, FWD, FWD, BWD, LIE, FWD, FWD],
)
