from collections import namedtuple
from tracemalloc import start

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
    def __init__(self, name="", stages=[], ops=[], startup=False) -> None:
        self.name = name
        self.stages = stages
        self.ops = ops
        self.inst = self.build()
        if startup:
            n = round(1.0 / np.sum(self.stages))
            startup_cmd = f"{FWD}{DELIMITER}" * C.SYS_FREQ
            self.inst = startup_cmd + self.inst * n

    def build(self):

        profile = []

        for i in range(len(self.stages)):
            for _ in range(int(self.stages[i] * NUM_QUERIES)):
                profile.append(self.ops[i])
                profile.append(DELIMITER)

        profile.append(NMV)

        return "".join(profile)
    
    def __str__(self) -> str:
        return self.name


class ForwardProfile(Profile):
    def __init__(self, name, vel, startup=False) -> None:
        super().__init__()
        self.v = vel
        self.name = name
        self.interp = ForwardGaitInterpolator(velocity=vel)
        self.inst = self.averaging()
        if startup:
            startup_cmd = f"{FWD}{DELIMITER}" * C.SYS_FREQ
            self.inst = startup_cmd + self.inst
        # print(self.inst[:60])

    def averaging(self) -> None:
        stage_lib = {self.interp.ops[i]: self.interp.stages[i] for i in range(len(self.interp.ops))}
        cmd_lib = sorted(self.interp.ops, reverse=True, key=lambda x: stage_lib[x])
        num_cmd_lib = {}
        num_sum = 0
        for cmd in cmd_lib[:-1]:
            num_cmd_lib[cmd] = round(C.NUM_QUERIES * stage_lib[cmd])
            num_sum += num_cmd_lib[cmd]
        num_cmd_lib[cmd_lib[-1]] = C.NUM_QUERIES - num_sum
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


class ForwardGaitInterpolator:
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
    def __init__(self, name, coeff, direction="right", startup=False) -> None:
        super().__init__()
        self.name = name
        self.interp = TurningGaitInterpolatior(coeff=coeff, direction=direction)
        self.inst = self.averaging()
        if startup:
            startup_cmd = f"{FWD}{DELIMITER}" * C.SYS_FREQ
            self.inst = startup_cmd + self.inst
        # print(self.inst[:60])

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

    def __init__(self, coeff, direction="right", min_split=1.0 / C.SYS_FREQ) -> None:

        self.stages = []
        self.ops = []
        self.splits = int(1.0 / min_split)
        self.turn_op = MLF if direction == "left" else MRT

        if self.splits > C.SYS_FREQ:
            raise ValueError("number of splits larger than control frequency!")

        self.min_split = min_split
        self.ops = [NMV, FWD, self.turn_op]
        if coeff == 0:
            self.stages = [0.5, 0.5 * (1 - coeff), 0.5 * coeff]

        self.stages = [0.5, 0.5 * (1 - coeff), 0.5 * coeff]


# trot = Profile(name="trot", stages=[1.0 / 60, 2.0 / 60], ops=[JMP, FWD])
trot = Profile(name="trot", stages=[1.5 / 100, 1.5 / 100], ops=[JMP, FWD])
full_speed_forwarding = Profile(name="full_speed_forwarding", stages=[0.01, 0.99], ops=[NMV, JMP])
full_speed_moving_left = Profile(name="full_speed_moving_left", stages=[0.01, 0.99], ops=[NMV, MLF])

acc_stop_slow = Profile(name="acc_stop", stages=[0.1, 0.3, 0.3, 0.2, 0.1], ops=[NMV, FWD, JMP, BWD, NMV])
acc_stop = Profile(name="acc_stop", stages=[0.1, 0.3, 0.5], ops=[FWD, FWD, JMP])

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

dynamic_jumping_dummy = Profile(name="dyn_jumping_dummy", stages=[0.1, 0.05], ops=[JMP, FWD], startup=True)


motion_wiki = {
    "walk": ForwardProfile("walk", 0.5, startup=True),
    "trot": ForwardProfile("trot", 1.28, startup=True),
    "jump": Profile("dynamic_jumping", stages=[0.1, 0.05], ops=[JMP, FWD], startup=True),
    "turn": TurningProfile("turn", 0.01, startup=True),
    "sit": Profile("sit", [1.0], [SIT], startup=True),
    "stand": Profile("stand", [1.0], [STD], startup=True),
    "lie": Profile("lie", [1.0], [LIE], startup=True),
    "turn-in-place": Profile("turn-in-place", [1.0], [TLF], startup=True),
}

motion_wiki_no_startup = {
    "walk": ForwardProfile("walk", 0.5, startup=False),
    "trot": ForwardProfile("trot", 1.28, startup=False),
    "jump": Profile("dynamic_jumping", stages=[0.1, 0.05], ops=[JMP, FWD], startup=False),
    "turn": TurningProfile("turn", 0.01, startup=False),
    "sit": Profile("sit", [1.0], [SIT], startup=False),
    "stand": Profile("stand", [1.0], [STD], startup=False),
    "lie": Profile("lie", [1.0], [LIE], startup=False),
    "turn-in-place": Profile("turn-in-place", [1.0], [TLF], startup=False),
}


def gen_bounding_profile():
    n = round(0.9 / 3 * C.SYS_FREQ)
    stages = [0.1]
    ops = [JMP]
    for _ in range(n):
        stages.append(1.0 / C.SYS_FREQ)
        stages.append(2.0 / C.SYS_FREQ)
        ops.append(JMP)
        ops.append(FWD)
    return Profile(name=f"bounding", stages=stages, ops=ops, startup=True)


def gen_dynamic_jumping_profile_trot(distribution):
    jmp, trot = distribution
    j1 = int(jmp)
    j2 = round((jmp - j1) * 1000)
    t1 = int(trot)
    t2 = round((trot - t1) * 1000)
    n = round(trot / 3 * C.SYS_FREQ)
    stages = [jmp]
    ops = [JMP]
    for _ in range(n):
        stages.append(1.0 / C.SYS_FREQ)
        stages.append(2.0 / C.SYS_FREQ)
        ops.append(JMP)
        ops.append(FWD)

    return Profile(name=f"dyn_jumping_j_{j1}_{j2:03d}_t_{t1}_{t2:03d}", stages=stages, ops=ops, startup=True)


class MixedProfile(Profile):
    def __init__(self, profiles, proportions) -> None:
        super().__init__()
        self.inst = []
        profile_length = len(profiles[0].inst.split(DELIMITER))
        profiles_len = [ int(profile_length * p) for p in proportions]

        for i in range(len(profiles)):
            self.inst.extend(profiles[i].inst.split(DELIMITER)[:profiles_len[i]])
        
        while profile_length > len(self.inst):
            self.inst.append(NMV)
        self.inst = DELIMITER.join(self.inst)

        # print(len(self.inst))
        self.name = []
        for i in range(len(profiles)):
            self.name.append(profiles[i].name)
            self.name.append(f"p_{proportions[i]:.2f}")
        self.name = '_'.join(self.name)


def random_profile_mixer():
    reliable_motions = ['walk', 'trot', 'turn', 'jump', 'sit', 'lie']
    num_motions = np.random.randint(1, 4)
    motions_selected = np.random.choice(reliable_motions, num_motions)
    print(motions_selected)

    proportions = []
    rem = 1.0
    for _ in range(num_motions-1):
        p = np.random.uniform(0.1, rem-0.1)
        rem -= p 
        proportions.append(p)
    proportions.append(rem)

    profiles = []
    for motion in motions_selected:
        startup = len(profiles) == 0 
        if motion in ['sit', 'lie']:
            profile = motion_wiki[motion] if startup else motion_wiki_no_startup[motion]
        elif motion == "walk":
            vel = np.random.uniform(0, 1)
            profile = ForwardProfile(f"walk_v_{vel:.2f}", vel=vel, startup=startup)
        elif motion == 'trot':
            vel = np.random.uniform(1.06, 1.28)
            profile = ForwardProfile(f"trot_v_{vel:.2f}", vel=vel, startup=startup)
        elif motion == 'jump':
            j = np.random.uniform(0.075, 0.125)
            w = np.random.uniform(0.05, 0.07)
            profile = Profile(f"jump_j_{j:.3f}_w_{w:.3f}", stages=[j, w], ops=[JMP, FWD], startup=startup)
        elif motion == "turn":
            coeff = np.random.uniform(0, 1)
            direction = np.random.choice(["left", "right"])
            profile = TurningProfile(f"turn_{direction}_c_{coeff:.2f}", coeff=coeff, direction=direction, startup=startup)
        
        profiles.append(profile)

    return MixedProfile(profiles, proportions)
    


if __name__ == "__main__":
    # from matplotlib import pyplot as plt

    # sampled_v = np.linspace(0, 2, 101)
    # profiles = [ForwardProfile("", v) for v in sampled_v]

    # data = {" ": [], "w": [], "j": []}

    # for p in profiles:
    #     print(f"v = {p.v:.3f}  inst. = [{p.inst[:100]}...]")
    #     # print(p.interp.ops, p.interp.stages)
    #     cmds = set(data.keys())
    #     for i in range(len(p.interp.ops)):
    #         data[p.interp.ops[i]].append(p.interp.stages[i] * 100)
    #         cmds.discard(p.interp.ops[i])
    #     for cmd in cmds:
    #         data[cmd].append(0)

    # plt.figure(figsize=(8, 6))
    # names = {" ": "NoMove", "w": "Forward", "j": "Jump"}
    # for cmd in data:
    #     plt.plot(sampled_v, data[cmd], label=f"{names[cmd]}")

    # plt.legend()
    # plt.xlabel("v")
    # plt.ylabel("proportion of commands / %")

    # plt.savefig("outputs/v.png")


    # MixedProfile(profiles[:2], [0.5, 0.5])
    print(random_profile_mixer())
