
import numpy as np
from aff_env.simple_env import dummy_env
import time

'''
0 for walk
1 for jump
2/3 for left and right turn, currently not well implemented

'''

def main():
    env = dummy_env(render=True)

    for i in range(100):
        env.reset()
        action =[0]
        obs1 = env.step(action)
        env.visualize_trajectory(obs1)
        action = [1]
        obs2 = env.step(action)
        env.visualize_trajectory(obs2)
        x = input("input key: ")
        if x == "" or str(x).lower() == "y":
            pass
        else:
            break



if __name__ == '__main__':
    main()