import torch,cv2
from torch.utils.data import Dataset
import json
from tqdm import tqdm
import os
from PIL import Image
from torchvision import transforms as T
from glob import glob
import cv2 as cv
from .ray_utils import *


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    
    rot_cam = np.array([[1,0,0],[0,0,-1],[0,1,0]])
    pose = np.eye(4, dtype=np.float32)
    # pose[:3, :3] =  R.transpose()
    pose[:3, :3] = rot_cam @ R.transpose()
    pose[:3, 3] = rot_cam @ (t[:3] / t[3])[:, 0]

    # breakpoint()
    return intrinsics, pose


def load_cam_from_txt(filename):
    lines = open(filename).read().splitlines()

    intrinsic = []
    extrinsic = []
    for idx in range(1,5):
        elements = lines[idx].split(' ')
        extrinsic.append([float(el) for el in elements])
    
    # import pdb;pdb.set_trace()
    for idx in range(7,10):
        elements = lines[idx].split(' ')
        intrinsic.append([float(el) for el in elements])
    # import pdb;pdb.set_trace()

    return np.array(intrinsic), np.array(extrinsic)

class DTU(Dataset):
    def __init__(self, datadir, split='train', downsample=1.0, is_stack=False, N_vis=-1):

        self.N_vis = N_vis
        self.root_dir = datadir
        self.split = split
        self.is_stack = is_stack
        self.downsample = downsample
        self.define_transforms()

        self.scene_bbox = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
        self.blender2opencv = np.eye(4)
        
        self.v_mean = np.array([0.0,0.0,0.0])
        self.read_meta()

        self.white_bg = True
        self.near_far = [0.1,3.0]
        
        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        self.downsample=downsample
        
        

    def read_depth(self, filename):
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)  # (800, 800)
        return depth
    def update_mean(self,val):
        self.v_mean = val
        self.read_meat()
    
    def read_meta(self):

        render_cameras_name = os.path.join(self.root_dir, 'cameras_sphere.npz')
        
        camera_dict = np.load(render_cameras_name)
        images_lis = sorted(glob(os.path.join(self.root_dir, 'image/*.png')))
        n_images = len(images_lis)
        images_np = np.stack([cv.cvtColor(cv.imread(im_name), cv.COLOR_BGR2RGB) for im_name in images_lis]) / 256.0
        masks_lis = sorted(glob(os.path.join(self.root_dir, 'mask/*.png')))
        masks_np = np.stack([cv.imread(im_name) for im_name in masks_lis]) / 256.0

        world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
        scale_mats_np = []

        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
        intrinsics_all = []
        poses = []

        rot_cam_axis = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
        for idx_,(scale_mat, world_mat) in enumerate(zip(scale_mats_np, world_mats_np)):

            P = world_mat @ scale_mat
        
            P = P[:3, :4]
            # breakpoint()
            intrinsics_, pose_ = load_K_Rt_from_P(None, P)
            pose_[:3,:3] = pose_[:3,:3] @ rot_cam_axis
            # print(pose[:3,3],pose_[:3,3])

            # import pdb;pdb.set_trace()
            intrinsics_all.append(intrinsics_)
            poses.append(pose_)

        
        images = images_np.astype(np.float32) 
        masks  = masks_np.astype(np.float32)   
        intrinsics_all = np.stack(intrinsics_all)  
        focal = intrinsics_all[0][0, 0]
        poses = np.stack(poses) 
        H, W = images.shape[1], images.shape[2]
        _mask = (masks[...,0]<0.5)
        images[_mask] = 1.0


        i_train = np.array(list(set(range(n_images)) - set(range(0, n_images, 8))))
        i_val = np.array(list(range(0, n_images, 8)))
        i_test = i_val
        i_cur = i_train if self.split == 'train' else i_test
        
        
        print(self.split,   i_cur)
        self._all_rgbs = images[i_cur]
        self.poses = poses[i_cur]

        self.img_wh = [W,H]

        self.intrinsics = intrinsics_all[0][:3,:3]

        # self.image_paths = []
        # self.poses = []
        self._all_rays = []
        # self.all_rgbs = []
        # self.all_masks = []
        # self.all_depth = []
        def get_rays_np(H, W, K, c2w):
            i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
            dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
            # Rotate ray directions from camera frame to the world frame
            rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
            # Translate camera frame's origin to the world frame. It is the origin of all rays.
            rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
            return rays_o, rays_d

        for c2w in self.poses:
            rays_o, rays_d = get_rays_np(H,W,self.intrinsics, c2w)  # both (h*w, 3)
            self._all_rays += [torch.cat([torch.Tensor(rays_o), torch.Tensor(rays_d)], -1)]  # (h*w, 6)
        # breakpoint()
        if not self.is_stack:
            self.all_rays = torch.stack(self._all_rays, 0).reshape(-1,6)
            self.all_rgbs = torch.Tensor(self._all_rgbs.reshape(-1,3))
        else:
            self.all_rays = torch.stack(self._all_rays, 0)   # (len(self.meta['frames]),h,w, 3)
            self.all_rgbs = torch.Tensor(self._all_rgbs)  # (len(self.meta['frames]),h,w,3)

            

    def define_transforms(self):
        self.transform = T.ToTensor()
        
    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):

        sample = {'rays': self.all_rays[idx],
                    'rgbs': self.all_rgbs[idx]}

        return sample
