import os, sys
import numpy as np
import cv2
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
import visdom
from os.path import join, isdir
from os import makedirs
from pathlib import Path

import matplotlib.pyplot as plt

from run_inerf_helpers import screwToMatrixExp4_torch, TestIfSO3

# some viz tools
from utils import check_pose_error, sample_unit_sphere, rotation_matrix_from_axis_angle
from vis import VisdomVisualizer, PlotlyScene, plot_transform

curr_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{curr_path}/nerf-pytorch")
from run_nerf_helpers import *
from run_nerf import batchify, run_network, batchify_rays, render, render_rays, render_path, create_nerf, raw2outputs

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000,
                        help='frequency of render_poses video saving')

    # debug visualization
    parser.add_argument("--dbg",   action='store_true',
                        help='debug plots and visualizations')
    parser.add_argument("--dbg_render_imgs", action='store_true',
                        help='debug render imgs periodically')
    # how to sample rays
    parser.add_argument("--sample_rays", type=str, default="random",
                        help='how to sample rays')
    parser.add_argument("--num_steps", type=int, default=300, help="Number of iterations to run")
    parser.add_argument("--batchsize", type=int, default=512, help="Number of rays to use")

    # testing
    parser.add_argument("--split", type=str, default="train", help="options are train, val, and test, default is train")
    parser.add_argument("--use_disparity", action="store_true", help="use disparity")
    #parser.add_argument("--use_disparity_only", action="store_true", help="use disparity ONLY") #todo
    parser.add_argument("--save_results", action="store_true", help="store training results in .txt files")

    return parser


def train():

    '''
    Setup
    '''
    parser = config_parser()
    args = parser.parse_args()
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    if args.dbg:
        vis = visdom.Visdom()
        visualizer = VisdomVisualizer(vis, "inerf")

    # Load data
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
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, _, _, _ = create_nerf(args)

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    '''
    let's pick a random image that we want to estimate the pose of
    '''
    indxs = dict()
    indxs["train"] = i_train
    indxs["val"] = i_val
    indxs["test"] = i_test

    print("Num images: ", len(indxs[args.split]))
    print("Use disparity? ", args.use_disparity)

    if args.save_results:
        obj_name = Path(args.config).stem
        if args.use_disparity:
            folder = join("results_with_disparity_bs_"+str(args.batchsize),obj_name)
        else:
            folder = join("results_bs_"+str(args.batchsize),obj_name)
            
        if not isdir(folder): makedirs(folder)
    
    for img_i in indxs[args.split]:
        img_i = np.random.choice(i_train)
        target = images[img_i]
        target = torch.from_numpy(target).float().to(device)

        pose = poses[img_i, :3,:4]
        T_world_camera = poses[img_i, :4,:4] # camera pose in world frame

        print(f"Image: {img_i}")
        print(f"GT camera pose: {pose}")

        if args.use_disparity:
            with torch.no_grad():
                Twc = torch.from_numpy(np.expand_dims(T_world_camera[:3, :4],0)).float().to(device)
                rgbs_rgt, disps_rgt = render_path(Twc, hwf, args.chunk, render_kwargs_test)

            rgb_rgt = rgbs_rgt[0]
            disp_rgt = disps_rgt[0]
            visualizer.plot_rgb((rgb_rgt * 255).astype(np.uint8), "rgb_rgt")
            visualizer.plot_rgb(cv2.cvtColor((np.copy(disp_rgt)*255).astype("float32"), cv2.COLOR_GRAY2RGB),"disparity")
            target_disp = torch.from_numpy(disp_rgt).float().to(device)

        # random init
        t_rng = 0.5
        r_rng = 20

        # random_axis = np.random.randn((3))
        # random_axis = random_axis / np.linalg.norm(random_axis)
        random_axis = sample_unit_sphere()
        random_angle_rads = np.deg2rad(np.random.uniform(-r_rng,r_rng))
        # random_translation = np.zeros((3))
        # random_angle_rads = (np.random.rand() - 0.5) * 2 * 20 * np.pi / 180 # random [-20, 20 degrees]
        # random_translation = (np.random.rand((3)) - 0.5) * 2 * 0.2 # random [-0.2, 0.2] meters
        # random_translation = np.array([random.uniform(-0.2, 0.2) for _ in range(3)]) # random [-0.2, 0.2] meters
        random_translation = np.random.uniform(-t_rng,t_rng,3)
        random_rot_mat3 = rotation_matrix_from_axis_angle(random_axis, random_angle_rads)

        T_rotated_original = np.eye(4)
        T_rotated_original[:3, :3] = random_rot_mat3

        T_translated_original = np.eye(4)
        T_translated_original[:3, 3] = random_translation

        T_offset_original = np.matmul(T_translated_original, T_rotated_original)

        T_world_cameraInit = np.matmul(T_offset_original, T_world_camera).astype(float)
        T_world_cameraInit = torch.from_numpy(T_world_cameraInit).float().to(device)
        print(f"offset: t: {random_translation}, rot: {random_angle_rads * 180 / np.pi} degrees around {random_axis}")

        if args.dbg:
            T_camera_world = np.linalg.inv(T_world_camera)
            T_cameraInit_world = np.linalg.inv(T_world_cameraInit.cpu().numpy())

            initial_scene = PlotlyScene(
                x_range=(-5, 5), y_range=(-5, 5), z_range=(-5, 5)
            )
            plot_transform(initial_scene.figure, np.eye(4), "cam frame", linelength=0.5, linewidth=10)
            plot_transform(initial_scene.figure, T_world_cameraInit.cpu().numpy(), "T_world_camInit", linelength=0.5, linewidth=10)
            plot_transform(initial_scene.figure, T_world_camera, "T_world_cam", linelength=0.5, linewidth=10)

            visualizer.plot_scene(initial_scene, "initial")
            visualizer.plot_rgb(target.cpu().numpy(), "target")

            if args.dbg_render_imgs:
                with torch.no_grad():
                    rgbs, _ = render_path(torch.unsqueeze(T_world_cameraInit, 0), hwf, args.chunk, render_kwargs_test)
                rgb = rgbs[0]
                visualizer.plot_rgb((rgb * 255).astype(np.uint8), "target_hat")

        '''
        we go from our R6 parameterization of an offset to an SE3 transform
        we can later premultiply T_0 by to get the new camera pose in world frame
        '''
        # sample a 6-vector of exponential coordinates
        screw_exp = torch.normal(mean=torch.zeros(6), std=1e-6 * torch.ones(6)).to(device)
        screw_exp.requires_grad = True

        # sanity check the initial pose error
        T_newWorld_oldWorld = screwToMatrixExp4_torch(screw_exp)
        T_world_cameraHat = torch.mm(T_newWorld_oldWorld, T_world_cameraInit)
        t_err, rot_err = check_pose_error(T_world_cameraHat.detach().cpu().numpy(), T_world_camera)
        print(f"beginning: trans error: {t_err}, rot error: {rot_err}")

        '''
        optimizer
        tl;dr we want to render rays of the camera pose and backpropagate the loss to update the
        6 parameters of our se3 representation, the omega and nu
        '''
        initial_lr = 0.01
        optimizer = torch.optim.Adam(params=[screw_exp], lr=initial_lr, betas=(0.9, 0.999))

        '''
        ~ main optimization loop ~
        '''
        N_rand = args.batchsize
        global_step = 0
        num_steps = args.num_steps
        while global_step < num_steps:
            # convert to se4
            T_newWorld_oldWorld = screwToMatrixExp4_torch(screw_exp)
            assert(TestIfSO3(T_newWorld_oldWorld.cpu().detach().numpy()[:3, :3])) # sanity check

            # premultiply our estimate as specified in the paper
            T_world_cameraHat = torch.mm(T_newWorld_oldWorld, T_world_cameraInit)
            # print(f"T_world_cameraHat: \n{T_world_cameraHat}")

            '''
            sampling rays
            a) random
            b) interest point
            c) interest region
            '''
            # generate all the rays through all the pixels
            rays_o, rays_d = get_rays(H, W, focal, T_world_cameraHat[:3, :4]) # (H, W, 3), (H, W, 3)

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

                I = 3
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

                for i in range(N_rand):
                    y,x = kps_ij[i]
                    cv2.circle(target_with_orb_features_opencv,(x,y), 5, (0, 0, 255), thickness=1)

                if args.dbg:
                    #target_with_orb_features_opencv[tmp>1,2] = 255
                    vis_img = cv2.cvtColor(target_with_orb_features_opencv.astype(np.uint8), cv2.COLOR_BGR2RGB)
                    visualizer.plot_rgb(vis_img,"target_with_regions")
                    visualizer.plot_rgb(cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB),"target_with_regions2")
            else:
                assert(False) # define a way to sample rays

            rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            batch_rays = torch.stack([rays_o, rays_d], 0)
            target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

            if args.use_disparity:
                disp_s = target_disp[select_coords[:, 0], select_coords[:, 1]]
            '''
            render and compare
            '''
            rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, rays=batch_rays,
                                            retraw=True,
                                            **render_kwargs_test)

            # plot the points we are visualizing
            rendered_rays = np.zeros_like(target.cpu().numpy())
            rgb_0_255 = rgb.detach().cpu().numpy() * 255
            rgb_0_255 = rgb_0_255.astype(np.uint8)
            rendered_rays[select_coords[:, 0].cpu().numpy(), select_coords[:, 1].cpu().numpy()] = rgb_0_255

            if args.dbg:
                visualizer.plot_rgb(rendered_rays, "rendered")

            optimizer.zero_grad()

            if args.use_disparity:
                target_s = torch.cat((target_s, disp_s.unsqueeze(1)), 1) # is this correct to include depth like this?
                rgb = torch.cat((rgb, disp.unsqueeze(1)), 1)
                
            img_loss = img2mse(rgb, target_s)
            loss = img_loss

            loss.backward()
            optimizer.step()

            ###   update learning rate   ###
            # The learning rate at step t is set as follow α_t = α_0 * 0.8^(t/100)
            decay_rate = 0.6
            decay_steps = 100
            new_lrate = initial_lr * (decay_rate ** (global_step / decay_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate

            '''
            prints and visualizations
            '''
            if global_step % 10 == 0:
                # current camera pose
                T_world_cameraHat_np = T_world_cameraHat.detach().cpu().numpy()

                # check pose error
                t_err, rot_err = check_pose_error(T_world_cameraHat_np, T_world_camera)
                print(f"iteration {global_step}, loss: {loss.cpu().detach().numpy()}, trans error: {t_err}, rot error: {rot_err}")

                with open(join(folder, f"{img_i:03d}.txt"), "a") as f:
                    f.write(f"iteration {global_step}, loss: {loss.cpu().detach().numpy()}, trans error: {t_err}, rot error: {rot_err}\n")

            if args.dbg and global_step % 10 == 0:
                # current camera pose
                T_world_cameraHat_np = T_world_cameraHat.detach().cpu().numpy()
                T_world_cameraHats = torch.from_numpy(np.expand_dims(T_world_cameraHat_np, 0)).to(device)

                optimization_scene = PlotlyScene(
                    x_range=(-5, 5), y_range=(-5, 5), z_range=(-5, 5)
                )
                plot_transform(optimization_scene.figure, np.eye(4), "camera frame", linelength=0.5, linewidth=10)
                plot_transform(optimization_scene.figure, T_world_camera, "T_world_camera", linelength=0.5, linewidth=10)
                plot_transform(optimization_scene.figure, T_world_cameraHat_np, "T_world_cameraHat", linelength=0.5, linewidth=10)
                visualizer.plot_scene(optimization_scene, "optimization")

                # clear gpu cuda ram since forward pass takes up some vram
                if args.dbg_render_imgs:
                    if global_step % 50 == 0:
                        del rgb
                        del disp
                        del acc
                        del extras
                        del loss
                        del img_loss
                        torch.cuda.empty_cache()

                        # render the whole image from the camera pose
                        with torch.no_grad():
                            rgbs, _ = render_path(T_world_cameraHats, hwf, args.chunk, render_kwargs_test)
                        rgb = rgbs[0]
                        visualizer.plot_rgb((rgb * 255).astype(np.uint8), "rgb_hat")

            global_step += 1
        del target
        del pose
        del T_world_cameraInit
        del T_world_cameraHats
        del screw_exp
        del rgb
        del disp
        del acc
        del extras
        del loss
        del img_loss

        if args.use_disparity:
            del target_disp
            del rgbs_rgt
            del disps_rgt
            del Twc
        torch.cuda.empty_cache()

if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()
