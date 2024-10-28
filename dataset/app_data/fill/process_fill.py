import numpy as np

import igl
import sys
sys.path.append(".")
from app_tex_gen import *
path1 = "../../../dataset/dtu/dtu_scan24/fill/target24.obj"
## part1 -> original (region want to replate)
path2 = "../../../dataset/dtu/dtu_scan24/fill/uv_24.obj"
## part2 -> target (replace into this area)
# og = "../../dataset/dtu/dtu_scan97/tet/dtu97_fin.obj"

v1, tc1, _, f1, ft1,_ = igl.read_obj(path1)
v2, tc2, _, f2, ft2,_ = igl.read_obj(path2)
res = 1024
range_res = np.arange(res)
tex_map = np.stack(np.meshgrid(range_res, range_res),-1)/1024 # (N,N,2)







def points_in_triangle(vertices, resolution):
    # Determine bounding box of the triangle
    vertices = vertices * resolution
    
    vertices_int = np.round(vertices).astype(int)

    # Determine bounding box of the triangle
    min_x = min(vertices_int[:, 0])
    max_x = max(vertices_int[:, 0])
    min_y = min(vertices_int[:, 1])
    max_y = max(vertices_int[:, 1])
    # Generate grid of points within bounding box
    xx, yy = np.meshgrid(np.arange(min_x, max_x + 1), np.arange(min_y, max_y + 1))
    points = np.vstack((xx.flatten(), yy.flatten())).T

    # Check if each point is inside the triangle using barycentric coordinates
    v,w,u = barycentric_coordinates(vertices, points)
    result = (v>0) * (w>0) * (u>0)
    result_sub = (v>-5e-1) * (w>-5e-1) * (u>-5e-1)
    fin = np.stack([v[result],w[result],u[result],points[result,0],points[result,1]],-1)
    fin_sub = np.stack([v[result_sub],w[result_sub],u[result_sub],points[result_sub,0],points[result_sub,1]],-1)
    return fin,fin_sub

def barycentric_coordinates(V, P):
    # P : (N,2)
    # V : (3,2)
    A,B,C = V
    # Calculate the areas of the whole triangle and sub-triangles
    
    v0 = np.repeat(B[None,:] - A[None,:],repeats=P.shape[0],axis=0)
    v1 = np.repeat(C[None,:] - A[None,:],repeats=P.shape[0],axis=0)
    v2 = P - A[None,:]
    
    d00 = (v0 * v0).sum(-1)
    d01 = (v0 * v1).sum(-1)
    d11 = (v1 * v1).sum(-1)
    d20 = (v2 * v0).sum(-1)
    d21 = (v2 * v1).sum(-1)
    denom = d00 * d11 - d01 * d01
    v = (d11*d20-d01*d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    return u,v,w

target_list = []
target_list2 = []
re_map_list = []
re_map_list2 = []
from tqdm import trange
for idx in trange(ft1.shape[0]):
    # idx = 0
    res, res_sub = points_in_triangle(tc1[ft1[idx]],1024)
    # res[:,:3] = res[:,:3]@v1[f1[idx]]
    remap = res[:,:3]@tc2[ft2[idx]]
    remap2 = res_sub[:,:3]@tc2[ft2[idx]]
    # res_sub[:,:3] = res_sub[:,:3]@v1[f1[idx]]
    target_list.append(res)
    target_list2.append(res_sub)
    re_map_list.append(remap)
    re_map_list2.append(remap2)
result = np.concatenate(target_list)
result2 = np.concatenate(target_list2)
result_map = np.concatenate(re_map_list)
result_map2 = np.concatenate(re_map_list2)
# breakpoint()
# import point_cloud_utils as pcu
# pcu.save_mesh_v("tset.ply",result[:,:3])
source_uvw = result[:,:3]
source_loc = result[:,3:].astype(np.int32)

# source_xyz_sub = result_sub[:,:3]
source_loc_sub = result2[:,3:].astype(np.int32)

# S, I, C = igl.signed_distance(source_xyz,v2,f2)

# breakpoint()
# uv_ = np.concatenate([source_loc,np.zeros((source_loc.shape[0],1))],1)

# uv_tris = v1[f1[I]]

# bary_target = igl.barycentric_coordinates_tri(source_xyz,v2[f2[I][:,0]],v2[f2[I][:,1]],v2[f2[I][:,2]])
# uv_target = (bary_target[...,None] *tc2[ft2[I]]).sum(1)


# S_sub, I_sub, C_sub = igl.signed_distance(source_xyz_sub,v2,f2)

# bary_target_sub = igl.barycentric_coordinates_tri(C_sub,v2[f2[I_sub][:,0]],v2[f2[I_sub][:,1]],v2[f2[I_sub][:,2]])
# uv_target_sub = (bary_target_sub[...,None] *tc2[ft2[I_sub]]).sum(1)


map_int = (result_map*1024).astype(np.int32)
map_int2 = (result_map2*1024).astype(np.int32)

source_loc_sub[source_loc_sub>1023] = 1023
source_loc_sub[source_loc_sub<0] = 0
map_int2[map_int2>1023] = 1023
map_int2[map_int2<0] = 0
# test = np.zeros((1024,1024,3)).astype(np.uint8)
# breakpoint()

c_val = tex_gen()
# breakpoint()
tex_map[source_loc_sub[:,1],source_loc_sub[:,0]] = c_val[map_int2[:,1],map_int2[:,0],:2]
tex_map[source_loc[:,1],source_loc[:,0]] = c_val[map_int[:,1],map_int[:,0],:2]
tex_im = np.concatenate([tex_map,np.zeros((1024,1024,1))],-1)
im = (tex_im*65536).astype(np.uint16)

mask1 = np.zeros((1024,1024,3)).astype(np.uint8)
mask1[source_loc_sub[:,1],source_loc_sub[:,0]]=255

m_x = np.concatenate((np.abs(tex_im[1:]-tex_im[:-1]),np.zeros((1,1024,3))),0)
m_y = np.concatenate((np.abs(tex_im[:,1:]-tex_im[:,:-1]),np.zeros((1024,1,3))),1)
edge = np.logical_or((m_x > 0.01).any(-1) , (m_y>0.01).any(-1))

from numpngw import write_png
write_png('result2.png', im)
write_png('mask.png', mask1)

write_png('mask_edge.png', (edge*255).astype(np.uint8))
# breakpoint()


# # pcu.save_mesh_v("test2.ply",C)
# tex_map[source_loc_sub[:,1],source_loc_sub[:,0]] = uv_target_sub
# tex_map[source_loc[:,1],source_loc[:,0]] = uv_target
# tex_im = np.concatenate([tex_map,np.zeros((1024,1024,1))],-1)


# im = (tex_im*65536).astype(np.uint16)

# # breakpoint()
# m_x = np.concatenate((np.abs(tex_im[1:]-tex_im[:-1]),np.zeros((1,1024,3))),0)
# m_y = np.concatenate((np.abs(tex_im[:,1:]-tex_im[:,:-1]),np.zeros((1024,1,3))),1)
# edge = np.logical_or((m_x > 0.01).any(-1) , (m_y>0.01).any(-1))
# from numpngw import write_png
# write_png('result2.png', im)

# write_png('mask.png', (edge*255).astype(np.uint8))