import torch,os,imageio,sys
from tqdm.auto import tqdm
from dataLoader.ray_utils import get_rays
from models.tensoRF import TensorVM, TensorCP, raw2alpha, TensorVMSplit, AlphaGridMask
from utils import *
from dataLoader.ray_utils import ndc_rays_blender


# def OctreeRender_trilinear_fast(rays, tensorf, chunk=4096, N_samples=-1, ndc_ray=False, white_bg=True, is_train=False, device='cuda'):

#     rgbs, alphas, depth_maps, weights, uncertainties = [], [], [], [], []
#     N_rays_all = rays.shape[0]
#     for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
#         rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
    
#         rgb_map, depth_map = tensorf(rays_chunk, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray, N_samples=N_samples)

#         rgbs.append(rgb_map)
#         depth_maps.append(depth_map)
    
#     return torch.cat(rgbs), None, torch.cat(depth_maps), None, None


def OctreeRender_trilinear_fast(rays, tensorf, tensorf2, mesh, chunk=4096, N_samples=-1, ndc_ray=False, white_bg=True, is_train=False, step_size=0.001, h_ratio = 1.0, device='cuda:0'):

    rgbs, alphas, depth_maps, weights, uncertainties = [], [], [], [], []
    N_rays_all = rays.shape[0]
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
        

        
        with torch.no_grad():
            xyz_sampled, dists, ray_valid, viewdirs, z_vals = mesh.ray_sample(rays_chunk, step_size=step_size,N_samples=N_samples,is_train=is_train)
            
            _uvh, valids, newdirs, target_mask , _uvh_init = mesh.convert(xyz_sampled[ray_valid],viewdirs[ray_valid])
            
            
            
        if valids==None:
            
            tmp = torch.zeros_like(ray_valid).to(ray_valid.device)
            tmp2 = torch.zeros_like(viewdirs).to(ray_valid.device)
            rgb_map, depth_map = tensorf.forward_query(xyz_sampled, dists, tmp, tmp2, z_vals, is_train=is_train, white_bg=white_bg)
        elif target_mask.sum()==0:           
            
            ray_valid[ray_valid.clone()] = valids
            uvh_sampled = torch.zeros_like(xyz_sampled).to(xyz_sampled.device)
            uvh_sampled[ray_valid] = _uvh_init
            uvh_dirs = torch.zeros_like(xyz_sampled).to(xyz_sampled.device)
            uvh_dirs[ray_valid] = newdirs
            
            
            rgb_map, depth_map = tensorf.forward_query(uvh_sampled, dists, ray_valid, uvh_dirs, z_vals, is_train=is_train, white_bg=white_bg)
        else:
            sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
            # rgb_tet = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)
            
            ray_valid[ray_valid.clone()] = valids
            
            
            uvh_sampled_init = torch.zeros_like(xyz_sampled).to(xyz_sampled.device)
            uvh_sampled_init[ray_valid]= _uvh_init
            
            
            uvh_sampled = torch.zeros_like(xyz_sampled).to(xyz_sampled.device)
            uvh_sampled[ray_valid]= _uvh
            
            uvh_dirs = torch.zeros_like(xyz_sampled).to(xyz_sampled.device)
            uvh_dirs[ray_valid] = newdirs
            
            ray_valid_og = ray_valid.detach().clone()
            # breakpoint()
            ray_valid_og[ray_valid.clone()] = target_mask
            
            # ray_valid[ray_valid_og] = False
            
            
            sigma, rgb= tensorf.forward_query_partial(uvh_sampled_init, dists, ray_valid, uvh_dirs, z_vals, is_train=is_train, white_bg=white_bg)
            _, rgb_og = tensorf2.forward_query_partial(uvh_sampled, dists, ray_valid_og, uvh_dirs, z_vals, is_train=is_train, white_bg=white_bg)
            
            # rgb_tet[ray_valid_og] = uvh_sampled[ray_valid_og]
            # rgb_tet[ray_valid_og][:,-1]=0.0
            # tet_max = rgb_tet[ray_valid_og].max()
            # tet_min = rgb_tet[ray_valid_og].min()
            # rgb_tet[ray_valid_og] = (rgb_tet[ray_valid_og]-tet_min) / (tet_max-tet_min)
            # breakpoint()
            sigma_com = sigma 
            _, weight, _ = raw2alpha(sigma_com, dists * tensorf.distance_scale)
            rgb_com = rgb 
            # + rgb_og
            # rgb_com = torch.zeros_like(rgb).to(rgb.device)
            rgb_com[ray_valid_og] =  rgb_og[ray_valid_og]
            # rgb_com = rgb + rgb_tet
            # rgb_com = rgb
            
            acc_map = torch.sum(weight, -1)
            rgb_map = torch.sum(weight[..., None] * rgb_com, -2)

            rgb_map = rgb_map + (1. - acc_map[..., None])

            
            rgb_map = rgb_map.clamp(0,1)

            with torch.no_grad():
                depth_map = torch.sum(weight * z_vals, -1)

        
            
            
        rgbs.append(rgb_map)
        depth_maps.append(depth_map)
    return torch.cat(rgbs), None, torch.cat(depth_maps), None, None

@torch.no_grad()
def evaluation(test_dataset,tensorf, tensorf2,mesh,args, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
               white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda:0'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays.shape[0] // N_vis,1)
    idxs = list(range(0, test_dataset.all_rays.shape[0], img_eval_interval))
    for idx, samples in tqdm(enumerate(test_dataset.all_rays[0::img_eval_interval]), file=sys.stdout):

        W, H = test_dataset.img_wh
        rays = samples.view(-1,samples.shape[-1])

        rgb_map, _, depth_map, _, _ = renderer(rays, tensorf, tensorf2, mesh,chunk=4096, N_samples=N_samples,
                                        ndc_ray=ndc_ray, white_bg = white_bg, device=device)
        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)
        if len(test_dataset.all_rgbs):
            gt_rgb = test_dataset.all_rgbs[idxs[idx]].view(H, W, 3)
            loss = torch.mean((rgb_map - gt_rgb) ** 2)
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

            if compute_extra_metrics:
                ssim = rgb_ssim(rgb_map, gt_rgb, 1)
                l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'alex', tensorf.device)
                l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'vgg', tensorf.device)
                ssims.append(ssim)
                l_alex.append(l_a)
                l_vgg.append(l_v)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', rgb_map)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=10)
    imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=10)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))


    return PSNRs

@torch.no_grad()
def evaluation_path(test_dataset,tensorf, mesh,c2ws, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
                    white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda:0'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    for idx, c2w in tqdm(enumerate(c2ws)):

        W, H = test_dataset.img_wh

        c2w = torch.FloatTensor(c2w)
        rays_o, rays_d = get_rays(test_dataset.directions, c2w)  # both (h*w, 3)
        if ndc_ray:
            rays_o, rays_d = ndc_rays_blender(H, W, test_dataset.focal[0], 1.0, rays_o, rays_d)
        rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)

        rgb_map, _, depth_map, _, _ = renderer(rays, tensorf, mesh,chunk=8192, N_samples=N_samples,
                                        ndc_ray=ndc_ray, white_bg = white_bg, device=device)
        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', rgb_map)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=8)
    imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=8)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))


    return PSNRs

