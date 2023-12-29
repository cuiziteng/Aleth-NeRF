import cv2
import os
import numpy as np
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--in_path', type=str, default=r'/home/mil/cui/Aleth-NeRF_1/logs/aleth_nerf_exp_blender_buu_eta0.4con5.0/depth')
parser.add_argument('--out_path', type=str, default=r'/home/mil/cui/Aleth-NeRF_1/logs/aleth_nerf_exp_blender_buu_eta0.4con5.0/depth_color')
config = parser.parse_args()



def convertPNG(pngfile,outdir):
    # READ THE DEPTH
    im_depth = cv2.imread(pngfile)
    print(im_depth.shape)
    #im_color=cv2.applyColorMap(cv2.convertScaleAbs(im_depth,alpha=1),cv2.COLORMAP_JET)
    im_color=cv2.applyColorMap(im_depth,cv2.COLORMAP_JET)
    im=Image.fromarray(im_color)
    im.save(os.path.join(outdir,os.path.basename(pngfile)))

in_path = config.in_path
out_path = config.out_path

if not os.path.exists(out_path):
    os.makedirs(out_path)

for file in os.listdir(in_path):
    img_file = os.path.join(in_path, file)
    convertPNG(img_file, out_path)
