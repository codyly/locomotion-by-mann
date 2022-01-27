import numpy as np

URDF_FILENAME = "a1/a1.urdf"
# URDF_FILENAME = "examples/unitree_ros/robots/a1_description/urdf/a1.urdf"

REF_POS_SCALE = 0.825
INIT_POS = np.array([0, 0, 0.37])
INIT_ROT = np.array([0, 0, 0, 1.0])

SIM_TOE_JOINT_IDS = [
    5,  # right hand
    15,  # right foot
    10,  # left hand
    20,  # left foot
]
SIM_HIP_JOINT_IDS = [1, 11, 6, 16]
SIM_KNEE_JOINT_IDS = [4, 14, 9, 19]
SIM_ROOT_OFFSET = np.array([0, 0, -0.06])
SIM_TOE_OFFSET_LOCAL = [
    np.array([0, -0.05, 0.0]),
    np.array([0, -0.05, 0.01]),
    np.array([0, 0.05, 0.0]),
    np.array([0, 0.05, 0.01]),
]

DEFAULT_JOINT_POSE = np.array([0, 0.5, -1.0, 0, 0.5, -1.0, 0, 0.5, -1.0, 0, 0.5, -1.0])
# DEFAULT_JOINT_POSE = np.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8])
JOINT_DAMPING = [0.1, 0.025, 0.01, 0.1, 0.025, 0.01, 0.1, 0.025, 0.01, 0.1, 0.025, 0.01]
# JOINT_DAMPING = [0.1, 0.05, 0.01, 0.1, 0.05, 0.01, 0.1, 0.05, 0.01, 0.1, 0.05, 0.01]

FORWARD_DIR_OFFSET = np.array([0, 0, 0])
