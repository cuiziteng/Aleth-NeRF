## Low-light Synthesis Code, write from MBLLEN (http://bmvc2018.org/contents/papers/0700.pdf)
## By Cui: cui@mi.t.u-tokyo.ac.jp

from telnetlib import EXOPL
from PIL import Image
import cv2
from cv2 import imwrite
import numpy as np
import os
from tqdm import tqdm
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--in_path', type=str, help="")
parser.add_argument('--out_path', type=str, help="")
config = parser.parse_args()

## Parameter Setting

exposure = 0.2  # exposure value, < 1.0   
gamma= 2.5  # gamma value, > 1.0     
noise = 0 # noise level 

def low_light_degradation(img, exposure, gamma, noise):
    sigma = 1e-3
    img = img
    noise = np.random.normal(0, noise/255.0, size=(img.shape[0], img.shape[1]))
    noise = np.stack([noise, noise, noise], axis=-1)

    # Low light transform
    img = ((img/255.0) ** gamma) * exposure + noise
    img = np.clip(img, 0, 1)
    return img*255.0


if __name__ == '__main__':

    input_path = config.in_path
    new_path = config.out_path

    os.makedirs(new_path, exist_ok=True)

    for file in tqdm(os.listdir(input_path)):
        img = cv2.imread(os.path.join(input_path, file))
        img_low = low_light_degradation(img, exposure, gamma, noise)
        cv2,imwrite(os.path.join(new_path, file), img_low)
        


