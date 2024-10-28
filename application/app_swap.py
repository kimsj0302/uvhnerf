
import os
from tqdm.auto import tqdm
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from opt import config_parser



from renderer import *
from utils import *
import datetime

from app_mesh_util_swap import *
from dataLoader import dataset_dict
import sys
import imageio
import numpy as np

import cv2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

renderer = OctreeRender_trilinear_fast


class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr+=self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr:self.curr+self.batch]




@torch.no_grad()
def swap_mesh(args):
    # breakpoint()
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)
    white_bg = test_dataset.white_bg
    ndc_ray = args.ndc_ray

    if not os.path.exists(args.ckpt):
        print('the ckpt path does not exists!!')
        return

    
    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    #load texture
    if not os.path.exists(args.fill_tex):
        print('the texture image path does not exists!!')
        return
    
    if not os.path.exists(args.tex_mask):
        print('the tex_mask does not exists!!')
        return
    
    tex = cv2.imread(args.fill_tex, cv2.IMREAD_UNCHANGED) #65536
    tex = cv2.cvtColor(tex, cv2.COLOR_BGR2RGB)
    tex_f = tex[...,:2].astype(np.float32)/65536
    
    
    tex_mask = cv2.imread(args.tex_mask, cv2.IMREAD_UNCHANGED)
    tex_mask = tex_mask > 125
    

    logfolder = os.path.dirname(args.ckpt)
    df_path = os.path.join(args.datadir,args.deform_mesh)
    mesh = mesh_util(os.path.join(args.datadir,args.mesh_name),tex_f,tex_mask,args.disp,args.disp_up,args.disp_down,
                     data_name=args.dataset_name,voxel_size=args.voxel_size)
    
    
    if args.render_train:
        os.makedirs(f'{logfolder}/swap_imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation(train_dataset,tensorf,mesh,args, renderer,f'{logfolder}/swap_imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        print(f'======> {args.expname} train all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/{args.expname}/swap_imgs_test_all', exist_ok=True)
        evaluation(test_dataset,tensorf,mesh, args, renderer, f'{logfolder}/{args.expname}/swap_imgs_test_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)

    if args.render_path:
        c2ws = test_dataset.render_path
        os.makedirs(f'{logfolder}/{args.expname}/swap_imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset,tensorf,mesh,c2ws, renderer, f'{logfolder}/{args.expname}/swap_imgs_path_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)



if __name__ == '__main__':

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    args = config_parser()
    print(args)

    swap_mesh(args)
    

