import numpy as np

VEC_FORWARD = np.array([0, 0, 1])
VEC_UP = np.array([0, 1, 0])
VEC_RIGHT = np.array([1, 0, 0])

STYLE_NOMOVE = np.array([1, 0, 0, 0, 0, 0])
STYLE_TROT = np.array([0, 1, 0, 0, 0, 0])
STYLE_JUMP = np.array([0, 0, 1, 0, 0, 0])
STYLE_SIT = np.array([0, 0, 0, 1, 0, 0])
STYLE_STAND = np.array([0, 0, 0, 0, 1, 0])
STYLE_LAY = np.array([0, 0, 0, 0, 0, 1])


NUM_STYLES = 6
SYS_FREQ = 60
DURATION = 10
START_UP_DURATION = .1

NUM_QUERIES = SYS_FREQ * DURATION

MOCAP_SAMPLE_PATH = "animation/data/mocap-sample.txt"


def set_duration(use_startup):
    global DURATION
    DURATION -= START_UP_DURATION
    NUM_QUERIES = SYS_FREQ * DURATION
