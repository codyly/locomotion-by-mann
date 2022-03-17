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

NUM_QUERIES = C.NUM_QUERIES  # SYS_FREQ * DURAION = 60 * 10


class Profile:
    def __init__(self, name, stages, ops) -> None:
        self.name = name
        self.stages = stages
        self.ops = ops
        self.inst = self.build()  # build instruction set for MANN using above 3 variables

    def build(self):

        profile = []

        for i in range(len(self.stages)):
            for _ in range(int(self.stages[i] * NUM_QUERIES)):
                profile.append(self.ops[i])
                profile.append(DELIMITER)

        profile.append(NMV)

        return "".join(profile)

# Profile does not have to go full(10s) duration. [0.1, 0.3, 0.1] -> Duration: [1s, 3s, 1s] = Queries: [60, 180, 60]
trot = Profile(name="trot", stages=[0.1, 0.3, 0.1], ops=[NMV, FWD, NMV])

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
