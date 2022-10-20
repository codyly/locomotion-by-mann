
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
        action = np.array([np.random.randint(2)])
        action =[0]
        env.step(action)
        action = [1]
        obs = env.step(action)
        print(len(obs["body_pose"]))




    


if __name__ == '__main__':
    main()