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
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
import visdom

from torch.utils.data import DataLoader, Dataset

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

class ImgPoseDataset(Dataset):
    def __init__(self, imgs, poses):
        self.imgs = imgs
        self.poses = poses
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self,idx):
        img = self.imgs[idx].T
        pose = self.poses[idx]
        return (img, pose)

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

    return parser

def train():
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

    mobilev2 = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=True)
        
    #model_cut = nn.Sequential(*list(mobilev2.children())[:-1],nn.Sequential(nn.Dropout(p=0.2,inplace=False),nn.Linear(mobilev2.classifier[1].in_features, 1000)))
    features = nn.Sequential(*list(mobilev2.children())[:-1])
    regressor = nn.Sequential(nn.Dropout(p=0.2, inplace=False),nn.Linear(mobilev2.classifier[1].in_features, 12))

    features.to('cuda')
    regressor.to('cuda')

    # freeze the feature extractor?
    for param in features.parameters():
        param.requires_grad = True

    for param in regressor.parameters():
        param.requires_grad = True

    poses = poses[:,:3,:4]
    images = torch.Tensor(images)
    poses = torch.Tensor(poses)

    bs = 4

    train_images = images[i_train]
    train_poses = poses[i_train]
    train_dataset = ImgPoseDataset(train_images, train_poses)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)

    val_images = images[i_val]
    val_poses = poses[i_val]
    val_dataset = ImgPoseDataset(val_images, val_poses)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    test_images = images[i_test]
    test_poses = poses[i_test]
    test_dataset = ImgPoseDataset(test_images, test_poses)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    optimizer = torch.optim.Adam(params=list(features.parameters())+list(regressor.parameters()), lr=1e-4)

    num_epochs = 4000

    print(mobilev2)
    
    regressor.train()
    for epoch in range(num_epochs):

        running_loss = 0.0
        running_trans_loss = 0.0
        for inputs, pose in tqdm(train_loader):
            inputs = inputs.to(device)
            gt_pose = pose.to(device)
            output = features(inputs)
            output = nn.functional.adaptive_avg_pool2d(output, (1,1))
            output = torch.flatten(output,1)
            output = regressor(output)

            output = output.reshape((bs,3,4))
            u,s,vt = torch.linalg.svd(output, full_matrices=False)
            output = torch.bmm(u,vt)

            loss_gt = torch.norm(output-gt_pose)
            #print("gt: ",gt_pose)
            #print("guess: ",output)
            loss_gt.backward()
            optimizer.step()
            running_loss += loss_gt.item()
            running_trans_loss += torch.norm(output[:,:3,3]-gt_pose[:,:3,3]).item()
        print(epoch)
        print(running_loss)
        print(running_trans_loss)
    
if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()
