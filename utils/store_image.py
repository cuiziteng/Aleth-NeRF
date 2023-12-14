# ------------------------------------------------------------------------------------
# NeRF-Factory
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

import os
import torch
import imageio
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def to8b(x):
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)

def to8b_depth(x):
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)

def norm8b(x):
    x = (x - x.min()) / (x.max() - x.min())
    return to8b(x)


def store_image(dirpath, rgbs):
    for (i, rgb) in enumerate(rgbs):
        imgname = f"image{str(i).zfill(3)}.jpg"
        rgbimg = Image.fromarray(to8b(rgb.detach().cpu().numpy()))
        imgpath = os.path.join(dirpath, imgname)
        rgbimg.save(imgpath)

def store_image_LOL(name, rgbs):
    store_path = r'/data/uni0/users/cui/Aleth_NeRF_results/LOL_results_total'
    os.makedirs(store_path, exist_ok=True)
    for (i, rgb) in enumerate(rgbs):
        imgname = str(name)
        rgbimg = Image.fromarray(to8b(rgb.detach().cpu().numpy()))
        imgpath = os.path.join(store_path, imgname)
        rgbimg.save(imgpath)

def store_depth(dirpath, rgbs):
    for (i, rgb) in enumerate(rgbs):
        imgname = f"image{str(i).zfill(3)}.jpg"
        rgb = rgb[:,:,0]
        rgb = (rgb - torch.min(rgb))/ (torch.max(rgb) - torch.min(rgb))
        #rgb = 1.0-rgb
        rgbimg = Image.fromarray(to8b(rgb.detach().cpu().numpy()))
        imgpath = os.path.join(dirpath, imgname)
        rgbimg.save(imgpath)

def store_video(dirpath, rgbs):

    rgbimgs = [to8b(rgb.cpu().detach().numpy()) for rgb in rgbs]
    video_dir = os.path.join(dirpath, "videos")
    os.makedirs(video_dir, exist_ok=True)
    imageio.mimwrite(os.path.join(video_dir, "images.mp4"), rgbimgs, fps=8, quality=8)

def store_video_depth(dirpath, rgbs):
    for i, rgb in enumerate(rgbs):
        rgbs[i] = (rgb - torch.min(rgb))/ (torch.max(rgb) - torch.min(rgb))

    rgbimgs = [to8b_depth(rgb.cpu().detach().numpy()) for rgb in rgbs]
    video_dir = os.path.join(dirpath, "videos")
    os.makedirs(video_dir, exist_ok=True)
    imageio.mimwrite(os.path.join(video_dir, "images.mp4"), rgbimgs, fps=8, quality=8)
