# ------------------------------------------------------------------------------------
# NeRF-Factory
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Modified from NeRF++ (https://github.com/Kai-46/nerfplusplus)
# Copyright (c) 2020 the NeRF++ authors. All Rights Reserved.
# ------------------------------------------------------------------------------------

import os
from subprocess import check_output
from typing import *
from glob import glob
import imageio
import numpy as np


# Load Single Image data
def _load_data(basedir, scene_name, factor=None, width=None, height=None, load_imgs=True):
    poses_arr = np.load(os.path.join(basedir, "poses_bounds.npy"))  # Load Position Information
    
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])[:, :, 0:1]
    bds = poses_arr[:, -2:].transpose([1, 0])[:, 0:1]
    
    # Load Image
    image_path = os.path.join(basedir, scene_name)
    if 'lol' or 'Huawei' in basedir:
        image_gt_path = os.path.join(basedir.replace('low', 'high'), scene_name)
    elif 'NH-HAZE' in basedir:
        image_gt_path = os.path.join(basedir.replace('hazy', 'GT'), scene_name)
    
    factor = 4

    sh = imageio.imread(image_path).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1.0 / factor
    poses = np.concatenate((poses, poses), axis=-1)
    bds = np.concatenate((bds, bds), axis = -1)
    
    if not load_imgs:
        return poses, bds

    def imread(f):
        if f.endswith("png"):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)

    img = imread(image_path)[..., :3]/ 255.0    # input
    img_gt = imread(image_gt_path)[..., :3]/ 255.0  # target
    imgs = np.stack([img, img_gt], -1)
    
    return poses, bds, imgs


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3, :3].T, (pts - c2w[:3, 3])[..., np.newaxis])[..., 0]
    return tt


def poses_avg(poses):

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.0])
    hwf = c2w[:, 4:5]

    for theta in np.linspace(0.0, 2.0 * np.pi * rots, N + 1)[:-1]:
        c = np.dot(
            c2w[:3, :4],
            np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0])
            * rads,
        )
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses


def recenter_poses(poses):

    poses_ = poses + 0
    bottom = np.reshape([0, 0, 0, 1.0], [1, 4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses


def spherify_poses(poses, bds):

    p34_to_44 = lambda p: np.concatenate(
        [p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])], 1
    )

    rays_d = poses[:, :3, 2:3]
    rays_o = poses[:, :3, 3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(
            -np.linalg.inv((np.transpose(A_i, [0, 2, 1]) @ A_i).mean(0)) @ (b_i).mean(0)
        )
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)

    center = pt_mindist
    up = (poses[:, :3, 3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([0.1, 0.2, 0.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))

    sc = 1.0 / rad
    poses_reset[:, :3, 3] *= sc
    bds *= sc
    rad *= sc

    centroid = np.mean(poses_reset[:, :3, 3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad**2 - zh**2)
    new_poses = []

    for th in np.linspace(0.0, 2.0 * np.pi, 120):

        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0, 0, -1.0])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)

    new_poses = np.concatenate(
        [new_poses, np.broadcast_to(poses[0, :3, -1:], new_poses[:, :3, -1:].shape)], -1
    )
    poses_reset = np.concatenate(
        [
            poses_reset[:, :3, :4],
            np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape),
        ],
        -1,
    )

    return poses_reset, new_poses, bds


def transform_pose_llff(poses):
    ret = np.zeros_like(poses)
    ret[:] = poses[:]
    ret[:, 0, 1:3] *= -1
    ret[:, 1:, 3] *= -1
    ret[:, 1:3, 0] *= -1
    return ret


def load_image_data(
    datadir: str,
    scene_name: str,
    ndc_coord: bool=True,    # True
    recenter: bool=True,
    bd_factor: float=0.75,
    spherify: bool=False,
    llffhold: int=8,
    path_zflat: bool=False,
    near: Optional[float]=None,
    far: Optional[float]=None,
):
    
    poses, bds, imgs = _load_data(datadir, scene_name)

    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    images = imgs
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)

    # Rescale if bd_factor is provided, bd_factor 0.75
    sc = 1.0 if bd_factor is None else 1.0 / (bds.min() * bd_factor)
    poses[:, :3, 3] *= sc
    bds *= sc

    if recenter:    # True
        poses = recenter_poses(poses)

    if spherify:    # False
        poses, render_poses, bds = spherify_poses(poses, bds)

    else:
        c2w = poses_avg(poses)

        ## Get spiral
        # Get average pose
        up = normalize(poses[:, :3, 1].sum(0))

        # Find a reasonable "focus depth" for this dataset
        close_depth, inf_depth = bds.min() * 0.9, bds.max() * 5.0
        dt = 0.75
        mean_dz = 1.0 / (((1.0 - dt) / close_depth + dt / inf_depth))
        focal = mean_dz

        # Get radii for spiral path
        shrink_factor = 0.8
        zdelta = close_depth * 0.2
        tt = poses[:, :3, 3]
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        N_views = 120
        N_rots = 2

        if path_zflat:  # False
            zloc = -close_depth * 0.1
            c2w_path[:3, 3] = c2w_path[:3, 3] + zloc * c2w_path[:3, 2]
            rads[2] = 0.0
            N_rots = 1
            N_views /= 2

        render_poses = render_path_spiral(
            c2w_path, up, rads, focal, zdelta, zrate=0.5, rots=N_rots, N=N_views
        )

    c2w = poses_avg(poses)

    dists = np.sum(np.square(c2w[:3, 3] - poses[:, :3, 3]), -1)
    i_test = np.argmin(dists)
    print(i_test)
    images = images.astype(np.float32)
    poses = poses.astype(np.float32)
    print(images.shape)
    print(poses.shape)

    # Transforming the coordinate
    poses = transform_pose_llff(poses)
    render_poses = np.stack(render_poses)[:, :3, :4]
    render_poses = transform_pose_llff(render_poses)
    extrinsics = poses[:, :3, :4]
    print('extric', extrinsics.shape)
    
    if not isinstance(i_test, list):
        i_test = [i_test]

    num_frame = len(poses)
    hwf = poses[0, :3, -1]
    h, w, focal = hwf
    h, w = int(h), int(w)
    hwf = [h, w, focal]
    intrinsics = np.array(
        [
            [[focal, 0.0, 0.5 * w], [0.0, focal, 0.5 * h], [0.0, 0.0, 1.0]]
            for _ in range(num_frame)
        ]
    )

    # if llffhold > 0:
    #     i_test = np.arange(num_frame)[::llffhold]
    
    #i_train = i_val = i_test

    if near is None and far is None:
        near = np.ndarray.min(bds) * 0.9 if not ndc_coord else 0.0
        far = np.ndarray.max(bds) * 1.0 if not ndc_coord else 1.0

    image_sizes = np.array([[h, w] for i in range(num_frame)])
    
    #i_all = np.arange(num_frame)
    i_train = np.array([0])
    i_val = i_test = np.array([1])
    i_all = np.array([0,1])
    i_split = (i_train, i_val, i_test, i_all)
    
    if ndc_coord:
        ndc_coeffs = (2 * intrinsics[0, 0, 0] / w, 2 * intrinsics[0, 1, 1] / h)
    else:
        ndc_coeffs = (-1.0, -1.0)
    
    return (
        images,
        intrinsics, # pose相关
        extrinsics, #
        image_sizes,
        near,
        far,
        ndc_coeffs,
        i_split,
        render_poses,
    )
    

if __name__ == '__main__':
    c = load_image_data('data/single_img/fern')
