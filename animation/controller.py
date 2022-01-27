from copy import copy

import numpy as np

from animation import common as C
from animation import profiles as P

NOMOVE_KEY = ord(" ")
FORWARD_KEY = ord("w")
BACKWARD_KEY = ord("s")
MOVE_LEFT_KEY = ord("a")
MOVE_RIGHT_KEY = ord("d")

TURN_LEFT_KEY = ord("q")
TURN_RIGHT_KEY = ord("e")

JUMP_KEY = ord("j")
SIT_KEY = ord("i")
STAND_KEY = ord("t")
LIE_KEY = ord("l")

NOMOVE_PROFILE = [[NOMOVE_KEY]]
FORWARDING_PROFILE = [[FORWARD_KEY, TURN_LEFT_KEY]]


class SimInputHandler:
    """Simulate user input to generate certain motion profiles"""

    def __init__(self, profile=FORWARDING_PROFILE, looping=False, need_parse=False) -> None:
        if len(profile) == 0:
            raise ValueError("Input empty profile!")

        if not need_parse:
            self.profile = profile
        else:
            self.profile = []
            profile = profile.split(",")
            for step in profile:
                self.profile.append([])
                for ch in step:
                    self.profile[-1].append(ord(ch))

        self.looping = looping
        self.cur_id = 0

    def get_keys(self):
        if self.looping:
            while True:
                yield self.profile[self.cur_id]
                self.cur_id = (self.cur_id + 1) % len(self.profile)
        else:
            while True:
                while self.cur_id < len(self.profile):
                    yield self.profile[self.cur_id]
                    self.cur_id = self.cur_id + 1

                if self.cur_id >= len(self.profile):
                    print("Query time is larger than profile length!")

                yield [NOMOVE_KEY]


class Style:
    def __init__(self, name, multipliers, negations, bias, transition):
        self.name = name
        self.multipliers = multipliers
        self.bias = bias
        self.transition = transition
        self.negations = negations
        self.keys = list(self.negations.keys())
        self.key_len = len(self.keys)

        self.mkeys = list(self.multipliers.keys())
        self.mkey_len = len(self.mkeys)

    def query(self, current_keys):
        if self.key_len == 0:
            return False

        active = False

        for key in self.keys:
            if key in current_keys and self.negations[key] > 0:
                active = True

        for key in self.keys:
            if key in current_keys and self.negations[key] < 0:
                active = False

        return active


DEFAULT_STYLES = [
    Style(name="idle", multipliers={}, negations={NOMOVE_KEY: 1.0}, bias=1, transition=0.01),
    Style(
        name="move",
        multipliers={},
        negations={
            FORWARD_KEY: 1.0,
            MOVE_LEFT_KEY: 1.0,
            MOVE_RIGHT_KEY: 1.0,
            TURN_LEFT_KEY: 1.0,
            TURN_RIGHT_KEY: 1.0,
            JUMP_KEY: -1.0,
            BACKWARD_KEY: -1.0,
        },
        bias=1.1,
        transition=0.1,
    ),
    Style(name="jump", multipliers={}, negations={JUMP_KEY: 1.0}, bias=3, transition=0.05),
    Style(
        name="sit",
        multipliers={},
        negations={SIT_KEY: 1.0, STAND_KEY: -1.0, LIE_KEY: -1.0, NOMOVE_KEY: -1.0},
        bias=0,
        transition=0.025,
    ),
    Style(
        name="stand",
        multipliers={},
        negations={STAND_KEY: 1.0, SIT_KEY: -1.0, LIE_KEY: -1.0, NOMOVE_KEY: -1.0},
        bias=0,
        transition=0.025,
    ),
    Style(
        name="lie",
        multipliers={},
        negations={LIE_KEY: 1.0, SIT_KEY: -1.0, STAND_KEY: -1.0, NOMOVE_KEY: -1.0},
        bias=0,
        transition=0.025,
    ),
]


class Controller:
    """"""

    def __init__(self, input_handler) -> None:
        self.input_handler = input_handler
        self.profile = self.input_handler.get_keys()
        self.current_keys = None
        self.style_num = C.NUM_STYLES
        self.styles = copy(DEFAULT_STYLES)

    def get_input(self):
        self.current_keys = next(self.profile)

    def query_move(self):
        move = np.zeros(3)
        if FORWARD_KEY in self.current_keys:
            move[2] += 1
        if BACKWARD_KEY in self.current_keys:
            move[2] -= 1
        if MOVE_LEFT_KEY in self.current_keys:
            move[0] += 1
        if MOVE_RIGHT_KEY in self.current_keys:
            move[0] -= 1

        return move

    def query_turn(self):
        turn = 0.0
        if TURN_LEFT_KEY in self.current_keys:
            turn -= 1
        if TURN_RIGHT_KEY in self.current_keys:
            turn += 1
        return turn

    def query_styles(self):
        styles = np.zeros(self.style_num)
        for i in range(self.style_num):
            if self.styles[i].query(self.current_keys):
                styles[i] = 1
            else:
                styles[i] = 0

        return styles


def test():
    input_handler = SimInputHandler(profile=P.trot, need_parse=True, looping=False)
    controller = Controller(input_handler=input_handler)

    for _ in range(1):
        controller.get_input()
        print(controller.query_styles())


if __name__ == "__main__":
    test()
