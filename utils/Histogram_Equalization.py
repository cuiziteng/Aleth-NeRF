## Histogram Equalization CODE for RGB images,
## By Cui: cui@mi.t.u-tokyo.ac.jp

from genericpath import exists
import os
import cv2
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--in_path', type=str, help="")
parser.add_argument('--out_path', type=str, help="")
config = parser.parse_args()


os.makedirs(config.outpath, exist_ok=True)

# histogram equalization
def HE(img):
    b,g,r = img[:,:,0], img[:,:,1], img[:,:,2]
    equ_b = cv2.equalizeHist(b)
    equ_g = cv2.equalizeHist(g)
    equ_r = cv2.equalizeHist(r)

    img_equ = np.stack([equ_b,equ_g,equ_r], axis=-1)
    return img_equ


for file in os.listdir(config.inpath):
    img_name = os.path.join(config.inpath, file)
    img = cv2.imread(img_name)
    img_equ = HE(img)
    cv2.imwrite(os.path.join(config.outpath, file), img_equ)
