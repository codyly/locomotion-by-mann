import numpy as np
import sys

from aff_locomotion.aff_env.simple_env import dummy_env
import time
import os
import matplotlib.pyplot as plt
from PIL import Image
import re
import json

'''
0 for walk
1 for jump
2/3 for left and right turn, currently not well implemented

'''

def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale

def write_pfm(file, image, scale=1):
    file = open(file, 'wb')
    color = None
    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode() if color else 'Pf\n'.encode())
    file.write('%d %d\n'.encode() % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n'.encode() % scale)

    image_string = image.tostring()
    file.write(image_string)
    file.close()

def main(root_dir, num_samp, skill_id):
    env = dummy_env(render=False)

    save_dir = f"{root_dir}/skill_{skill_id}"
    os.makedirs(save_dir, exist_ok=True)

    for i in range(num_samp):
        env.reset()
        robot_pos, robot_quat = env.get_robot_pose()
        rgbImg, depthImg, segImg = env.get_camera_data(cam_pos=robot_pos, cam_quat=robot_quat,
                                                       cam_H=120, cam_W=160)

        # _, axs = plt.subplots(1, 3)
        # axs[0].imshow(rgbImg)
        # axs[1].imshow(depthImg)
        # axs[2].imshow(segImg)
        # plt.show()

        depth_name = "{}/depth_{:02d}_{:08d}.pfm".format(save_dir, skill_id, i)
        write_pfm(depth_name, depthImg)
        color_name = "{}/color_{:02d}_{:08d}.png".format(save_dir, skill_id, i)
        Image.fromarray(rgbImg).save(color_name)

        segImg[segImg == -1] = 0
        segImg *= 127
        segm_name = "{}/segm_{:02d}_{:08d}.png".format(save_dir, skill_id, i)
        Image.fromarray(segImg.astype("uint8")).save(segm_name)

        label = {"skill_id": skill_id}
        if skill_id == 0:
            obs = env.step([0])
            label["collision"] = list(obs["collision"])
        elif skill_id == 1:
            obs1 = env.step([0])
            obs2 = env.step([1])
            label["collision"] = list(obs1["collision"]) + list(obs2["collision"])

        json_name = "{}/label_{:02d}_{:08d}.json".format(save_dir, skill_id, i)
        with open(json_name, "w") as outfile:
            json.dump(label, outfile)


if __name__ == '__main__':
    main("../meta_controller/test", 100, 1)
