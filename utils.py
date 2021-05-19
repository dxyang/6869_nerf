import math
import os
import random
import sys

import cv2
import numpy as np
import torch

cwd = os.getcwd()
sys.path.append(f"{cwd}/nerf-pytorch")
from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data

def sample_rays_to_render(args, target, N_rand, H, W, visualizer=None):
    '''
    sampling rays
    a) random
    b) interest point
    c) interest region
    '''
    if args.sample_rays == "random":
        # randomly sample N_rand rays from all HxW possibilities
        coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)
        coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
        select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
        select_coords = coords[select_inds].long()  # (N_rand, 2)
    elif args.sample_rays == "feature_points":
        # use orb features to pick keypoints
        margin = 30
        orb = cv2.ORB_create(
            nfeatures=int(N_rand * 2),       # max number of features to retain
            edgeThreshold=margin,            # size of border where features are not detected
            patchSize=margin                 # size of patch used by the oriented BRIEF descriptor
        )
        target_with_orb_features = np.copy(target.cpu().numpy()) * 255

        target_with_orb_features_opencv = cv2.cvtColor(target_with_orb_features.astype(np.uint8), cv2.COLOR_RGB2BGR)
        kps = orb.detect(target_with_orb_features_opencv,None)

        random.shuffle(kps)
        select_coords = torch.zeros(N_rand, 2).long()
        cv2.imwrite('color_img.jpg', target_with_orb_features_opencv)

        if len(kps) < N_rand:
            print(f"less keypoints ({len(kps)}) than N_rand ({N_rand})")
            # randomly sample N_rand rays from all HxW possibilities
            coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)
            coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
            select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
            select_coords = coords[select_inds].long()  # (N_rand, 2)

        for i in range(min(len(kps), N_rand)):
            x = int(kps[i].pt[0])
            y = int(kps[i].pt[1])
            select_coords[i, 0] = y
            select_coords[i, 1] = x
            cv2.circle(target_with_orb_features_opencv,(x,y), 5, (0, 0, 255), thickness=1)

        if args.dbg:
            vis_img = cv2.cvtColor(target_with_orb_features_opencv.astype(np.uint8), cv2.COLOR_BGR2RGB)
            visualizer.plot_rgb(vis_img, "target_with_orb_features")

    elif args.sample_rays == "feature_regions":
        # use orb features to pick keypoints
        margin = 30
        orb = cv2.ORB_create(
            nfeatures=int(N_rand*2),       # max number of features to retain
            edgeThreshold=margin,            # size of border where features are not detected
            patchSize=margin                 # size of patch used by the oriented BRIEF descriptor
        )
        target_with_orb_features = np.copy(target.cpu().numpy()) * 255
        target_with_orb_features_opencv = cv2.cvtColor(target_with_orb_features.astype(np.uint8), cv2.COLOR_RGB2BGR)
        kps = orb.detect(target_with_orb_features_opencv,None)

        I = args.dilation
        kps_ij = [[int(kp.pt[1]), int(kp.pt[0])] for kp in kps]

        tmp = np.zeros((H, W)).astype("uint8")

        for i,j in kps_ij:
            tmp[i,j] = 255
        kern = np.ones((5,5))
        for i in range(I):
            tmp = cv2.dilate(tmp, kern)

        d_kps_ij = np.argwhere(tmp > 0)
        np.random.shuffle(d_kps_ij)

        select_coords = torch.from_numpy(d_kps_ij[0:N_rand])

        if args.dbg:
            for i in range(min(N_rand, len(kps_ij))):
                y,x = kps_ij[i]
                cv2.circle(target_with_orb_features_opencv,(x,y), 5, (0, 0, 255), thickness=1)
            #target_with_orb_features_opencv[tmp>1,2] = 255
            vis_img = cv2.cvtColor(target_with_orb_features_opencv.astype(np.uint8), cv2.COLOR_BGR2RGB)
            visualizer.plot_rgb(vis_img,"target_with_regions")
            visualizer.plot_rgb(cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB),"target_with_regions2")
    else:
        assert(False) # define a way to sample rays

    return select_coords

def load_data(args):
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.

        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

        # added by dxy - we like poses in 4x4
        bottom = np.expand_dims(np.expand_dims(np.array([0, 0, 0, 1.0]), axis=0), axis=0) #1x1x4
        bottom_batch = np.concatenate([bottom for _ in range(poses.shape[0])])
        poses = np.hstack((poses, bottom_batch))

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        assert(False)

    return images, poses, render_poses, hwf, near, far, i_train, i_val, i_test



def check_pose_error(That_A_B: np.array, T_A_B: np.array):
    T_Ahat_A = np.matmul(That_A_B, np.linalg.inv(T_A_B))
    rotation_error = np.arccos(
        np.clip(
            (np.trace(T_Ahat_A[:3, :3]) - 1.0) / 2.0,
            -1.0,
            1.0
        )
    ) * 180.0 / np.pi
    translation_error = np.linalg.norm(T_Ahat_A[:3, 3])
    return translation_error, rotation_error


def sample_unit_sphere():
    # http://corysimon.github.io/articles/uniformdistn-on-sphere/
    theta = 2 * np.pi * np.random.rand()
    phi = math.acos(1 - 2 * np.random.rand())
    x = math.sin(phi) * math.cos(theta)
    y = math.sin(phi) * math.sin(theta)
    z = math.cos(phi)

    return np.array([x, y, z])


def quat_from_axis_angle(axis: np.array, angle_rads: float):
    # don't divide by zero
    axis = np.squeeze(axis)
    length = np.linalg.norm(axis)
    if length == 0:
        if angle_rads == 0:
            assert False
        else:
            return np.array([0.0, 0.0, 0.0, 1.0]) #xyzw

    normalized_axis = axis / length
    sin_half_angle = math.sin(angle_rads * 0.5)

    return np.array([
        normalized_axis[0] * sin_half_angle,
        normalized_axis[1] * sin_half_angle,
        normalized_axis[2] * sin_half_angle,
        math.cos(angle_rads * 0.5),
    ])


def rotation_matrix_from_quat(q: np.array):
    q = q.squeeze() # xyzw
    assert np.isclose(np.linalg.norm(q) ** 2, 1.0) # quat should be normalized?

    tx = q[0] + q[0]
    ty = q[1] + q[1]
    tz = q[2] + q[2]

    twx = q[3] * tx
    twy = q[3] * ty
    twz = q[3] * tz

    txx = q[0] * tx
    txy = q[0] * ty
    txz = q[0] * tz

    tyy = q[1] * ty
    tyz = q[1] * tz
    tzz = q[2] * tz

    mat3 = np.zeros((3, 3))
    mat3[0][0] = 1 - (tyy + tzz)
    mat3[0][1] = txy - twz
    mat3[0][2] = txz + twy
    mat3[1][0] = txy + twz
    mat3[1][1] = 1 - (txx + tzz)
    mat3[1][2] = tyz - twx
    mat3[2][0] = txz - twy
    mat3[2][1] = tyz + twx
    mat3[2][2] = 1 - (txx + tyy)

    assert(is_rotation_matrix(mat3))

    return mat3


def is_rotation_matrix(mat3: np.array):

    if np.linalg.det(mat3) > 0:
        dist = np.linalg.norm(np.dot(np.array(mat3).T, mat3) - np.eye(3))
    else:
        dist = 1e+9

    return abs(dist) <= 1e-3


def rotation_matrix_from_axis_angle(axis: np.array, angle_rads: float):
    q = quat_from_axis_angle(axis, angle_rads)
    rot = rotation_matrix_from_quat(q)
    return rot
