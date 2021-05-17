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
import torchvision.models as models
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

def set_bn_grad_recursive(m, req_grad=True):
    if isinstance(m, nn.BatchNorm2d):
        m.requires_grad = req_grad
        print(m)

    for child in m.children():
        set_bn_grad_recursive(child, req_grad)

def set_grad_recursive(m, req_grad=True):
    m.requires_grad = req_grad

    for child in m.children():
        set_grad_recursive(child, req_grad)

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

    # load the pose regression model and freeze the bn weights
    mobilenet_v2 = models.mobilenet_v2(pretrained=True)
    num_ftrs = mobilenet_v2.classifier[1].in_features
    mobilenet_v2.classifier[1] = nn.Linear(num_ftrs, 12)
    mobilenet_v2.to(device)
    set_bn_grad_recursive(mobilenet_v2, req_grad = False)

    # data wrangling
    poses = poses[:,:3,:4]
    images = torch.Tensor(images.astype(np.float32))
    poses = torch.Tensor(poses.astype(np.float32))

    bs = 2

    train_images = images[i_train]
    train_poses = poses[i_train]
    train_dataset = ImgPoseDataset(train_images, train_poses)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)

    val_images = images[i_val]
    val_poses = poses[i_val]
    val_dataset = ImgPoseDataset(val_images, val_poses)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=True)

    test_images = images[i_test]
    test_poses = poses[i_test]
    test_dataset = ImgPoseDataset(test_images, test_poses)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    dataloaders = dict()
    dataloaders["train"] = train_loader
    dataloaders["val"] = val_loader

    optimizer = torch.optim.Adam(params=mobilenet_v2.parameters(), lr=1e-4)

    '''
    training loop
    '''
    num_epochs = 4000

    mobilenet_v2.train()

    lambda1 = 0.0 # photoloss
    lambda2 = 1.0 - lambda1 #gt loss

    global_step = 0
    for epoch in range(num_epochs):

        running_loss = 0.0
        running_trans_loss = 0.0

        print("="*10)
        for phase in ["train","val"]:
            if phase == "train":
                mobilenet_v2.train()
            else:
                mobilenet_v2.eval()

            for inputs, pose in tqdm(dataloaders[phase]):
                optimizer.zero_grad()

                # get the data
                inputs = inputs.to(device)
                gt_pose = pose.to(device)
                actual_bs = inputs.size()[0]

                with torch.set_grad_enabled(phase == "train"):

                    # forward pass through network
                    output = mobilenet_v2(inputs)

                    # turn R12 into 3x4 with SVD for SO3 manifold
                    output = output.reshape((actual_bs,3,4))
                    rotation_mat_hat = output[:, :3, :3]
                    translation_vec_hat = output[:, :3, 3]
                    u,s,vt = torch.linalg.svd(rotation_mat_hat, full_matrices=False)
                    rotation_svd_hat = torch.bmm(u,vt)
                    pose_svd_hat = torch.cat((rotation_svd_hat, translation_vec_hat.unsqueeze(2)), dim=2)

                    # get the loss
                    loss_gt = torch.norm(pose_svd_hat-gt_pose)

                    '''
                    render some images and eat all the gpu vram
                    - sample rays for computational reasons
                    '''
                    N_rand = 512
                    # generate all the rays through all the pixels
                    rgb_hats = []
                    rgb_targets = []
                    for bs_idx in range(actual_bs):
                        # one img at a time within this batch
                        pose_hat_oi = pose_svd_hat[bs_idx]
                        pose_gt_oi = gt_pose[bs_idx]
                        img_oi = inputs[bs_idx].permute(1, 2, 0) # HWC

                        rays_o, rays_d = get_rays(H, W, focal, pose_gt_oi) # (H, W, 3), (H, W, 3)

                        # use orb features to pick keypoints
                        margin = 30
                        orb = cv2.ORB_create(
                            nfeatures=int(N_rand*2),       # max number of features to retain
                            edgeThreshold=margin,            # size of border where features are not detected
                            patchSize=margin                 # size of patch used by the oriented BRIEF descriptor
                        )
                        target_with_orb_features = np.copy(img_oi.cpu().numpy()) * 255
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

                        rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                        rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                        batch_rays = torch.stack([rays_o, rays_d], 0)
                        target_s = img_oi[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

                        # render
                        rgb_hat, _, _, _ = render(H, W, focal, chunk=args.chunk, rays=batch_rays,
                                                        retraw=True,
                                                        **render_kwargs_test)
                        rgb_hats.append(rgb_hat)
                        rgb_targets.append(target_s)

                    # concatenate individual renders into a batch and calculate loss
                    batch_rgb_hats = torch.cat(rgb_hats, dim=0)
                    batch_rgb_targets = torch.cat(rgb_targets, dim=0)
                    loss_photo = torch.norm(batch_rgb_hats - batch_rgb_targets)

                    loss = lambda1*loss_photo + lambda2*loss_gt
                    #print("gt: ",gt_pose)
                    #print("guess: ",output)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item()
                running_trans_loss += torch.norm(output[:,:3,3]-gt_pose[:,:3,3]).item()
            print("Epoch: {} {} Loss: {:.4f} Trans. Loss: {:.4f}".format(epoch, phase, running_loss, running_trans_loss))

        if epoch%10 == 0 and epoch > 0:
            print("Saving...")
            torch.save(mobilenet_v2.state_dict(), os.path.join("snapshots",f'weights_{epoch}_valloss_{running_loss:.04f}.pt'))

        global_step +=1
        #print(epoch)
        #print(running_loss)
        #print(running_trans_loss)

if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()
