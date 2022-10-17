
import numpy as np
from aff_env.simple_env import dummy_env

'''
0 for walk
1 for jump
2/3 for left and right turn, currently not well implemented

'''

def main():
    env = dummy_env(render=True)
    env.reset()
    

    env.reset()
    for i in range(1):
        action = np.array([np.random.randint(4)]) 
        action =[0]
        pose = env.step(action)

    for i in range(1):
        action = np.array([np.random.randint(4)]) 
        action =[1]
        pose = env.step(action)

    for i in range(2):
        action = np.array([np.random.randint(4)]) 
        action =[0]
        pose = env.step(action)


    for i in range(2):
        action = np.array([np.random.randint(4)]) 
        action =[1]
        pose = env.step(action)


    


if __name__ == '__main__':
    main()