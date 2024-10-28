import igl
import scipy as sp
import numpy as np
# from meshplot import plot, subplot, interact
import torch
import time
from tqdm import trange, tqdm
import os
import pickle
from typing import List
from torch_scatter import scatter_min,scatter_max
import nerfacc 


device_model = "cuda:0"
device_mesh = "cuda:1"

def cal_bc(triangle, query_points):
    """
    Calculate the barycentric coordinates for given triangle vertices and query points in 2D.

    Parameters:
        triangle (numpy.ndarray): Array of shape (3, 2) representing the vertices of the triangle.
        query_points (numpy.ndarray): Array of shape (N, 2) representing the query points.

    Returns:
        bary_coords (numpy.ndarray): Array of shape (N, 3) representing the barycentric coordinates for each query point.
    """
    if triangle.shape != (3, 2):
        raise ValueError("Triangle must have shape (3, 2)")
    
    if query_points.shape[1] != 2:
        raise ValueError("Query points must have shape (N, 2)")

    # Extract triangle vertices
    A, B, C = triangle

    # Calculate vectors of the triangle edges
    AB = B - A
    AC = C - A

    # Calculate the denominator for barycentric coordinates
    denom = AB[0] * AC[1] - AB[1] * AC[0]

    # Calculate vectors from triangle vertices to query points
    P = query_points - A

    # Calculate barycentric coordinates for all query points simultaneously
    b1 = (P[:, 0] * AC[1] - P[:, 1] * AC[0]) / denom
    b2 = (AB[0] * P[:, 1] - AB[1] * P[:, 0]) / denom
    b0 = 1 - b1 - b2

    # Combine barycentric coordinates into a single array
    bary_coords = np.column_stack((b0, b1, b2))

    return bary_coords

#Following https://math.stackexchange.com/questions/1993953/closest-points-between-two-lines
def check_disp(v,vn,f,disp_up,disp_down):
    edge_set = set()
    
    for (v1,v2,v3) in f:
        if v1>v2:
            edge_set.add((v1,v2))
        else:
            edge_set.add((v2,v1))
        
        if v1>v3:
            edge_set.add((v1,v3))
        else:
            edge_set.add((v3,v1))
        
        if v2>v3:
            edge_set.add((v2,v3))
        else:
            edge_set.add((v2,v3))
    
    edge_list = list(edge_set)
    edge = np.array(edge_list)
    
    P = v[edge]
    P1 = P[:,0]
    P2 = P[:,1]
    
    P_diff = P2-P1
    
    V = vn[edge]
    V1 = V[:,0]
    V2 = V[:,1]
    V3 = np.cross(V1,V2)
    
    mat_V = np.stack([V1,-V2,V3],-1)
    det_V = np.linalg.det(mat_V)
    
    par_mask = (det_V!=0)
    
    
    _t_vals = np.matmul(np.linalg.inv(mat_V[par_mask]),P_diff[par_mask,:,None])[...,0]
    t_vals = _t_vals[...,:2]
    
    max_disp = np.full((v.shape[0],),disp_up)
    min_disp = np.full((v.shape[0],),-1*disp_down)
    
    edge_rs = edge.reshape(-1)
    t_vals_rs = t_vals.reshape(-1)
    
    
    mask = np.logical_and(t_vals_rs <=  disp_up,t_vals_rs >= -1 * disp_down)
    edge_m = edge[par_mask].reshape(-1)[mask]
    t_vals_m = t_vals_rs[mask]
    
    
    
    for idx in range(edge_m.shape[0]):
        val = t_vals_m[idx]
        e_id = edge_m[idx]
        if val > 0 :
            max_disp[e_id] = np.min([max_disp[e_id],val])
        else:
            min_disp[e_id] = np.max([min_disp[e_id],val])
    
    max_disp = np.full((v.shape[0],),disp_up)
    min_disp = np.full((v.shape[0],),-1*disp_down)    
        
    # breakpoint()
    return max_disp,min_disp
    
def dup_edge(remap_f_fn,f,tc):
    # return two texture coordinate which are duplicated edge (covering chart boundary)
    # (N,2,4)
    e = igl.edges(f)
    e_remap = remap_f_fn[e]
    e_remap.sort()
    uni, cnt = np.unique(e_remap, return_counts=True,axis=0)
    dup_data = uni[cnt>1]
    ll = [np.where(np.logical_and((e_remap[:,0]==d_data[0]),(e_remap[:,1]==d_data[1] )))[0][:2] for d_data in dup_data]
    # breakpoint()
    du_ids = np.stack(ll,0)
    edge_bound = e[du_ids]
    ## edge bodunary order checking
    tex_chart =[]
    for edge in edge_bound:
        tc1 = np.concatenate((tc[edge[0,0]],tc[edge[0,1]]))
        if remap_f_fn[edge[0,0]] == remap_f_fn[edge[1,0]]:
            # the order is correct
            tc2 = np.concatenate((tc[edge[1,0]],tc[edge[1,1]]))
        else:
            # the order is incorrect
            tc2 = np.concatenate((tc[edge[1,1]],tc[edge[1,0]]))
        tex_chart.append(np.stack((tc1,tc2),0))    
    tc_edge = np.stack(tex_chart,0)
    return tc_edge
    
class Remapper:
    def __init__(self,division=3,margin_uv=0.01, margin_h=0.01):
        self.division = division
        self.margin_uv = margin_uv
        self.margin_h = margin_h
        
        self.grid_size = division
        self.height_size = 1 / (division**2)
        
        self.uv_ratio = 1.0 - 2 * margin_uv
        self.h_ratio = 1.0 - 2 * margin_h

        
        
    
    def map(self,uvh):
        # change uv coordiante
        uv = uvh[...,:-1]
        loc = torch.floor(uv * self.division) / self.division
        uv_reloc = (uv - loc) * self.division # (0,1)
        uv_margin = uv_reloc * self.uv_ratio + self.margin_uv #(0,1-m)
        
        # change h cooridnate
        loc_level = (loc[:,0]*self.division +loc[:,1])*self.division * self.height_size # [0,1,...,(DIV-1)*(DIV-1)] / (DIV*DIV)
        _h = uvh[:,-1]
        h_margin = _h * (1 - 2*self.division**2 * self.margin_h) / (self.division**2)  + self.margin_h + loc_level
        
        uvh_margin = torch.cat([uv_margin,h_margin[:,None]],1)
        # breakpoint()
        return uvh_margin
        
        
        
        
class mesh_util:
    def __init__(self,obj_path,uv,uv_mask,edge_mask,disp,disp_up,disp_down,data_name="blender",voxel_size=0.02):
        
        v, tc, _, f, _ , fn = igl.read_obj(obj_path)
        
        def mesh2normal(v_,f_):
            
            rdv = igl.remove_duplicate_vertices(v_,f_, 1e-7)
            t_vn = igl.per_vertex_normals(rdv[0],rdv[2][f_])  
            _vn = t_vn[rdv[2]]
            return _vn
        
        vn = mesh2normal(v,f)
        
        # breakpoint()
        
        
        
        self.uv = torch.Tensor(uv).to(device_mesh).permute(2,0,1)
        self.uv_mask = torch.Tensor(uv_mask).to(device_mesh)
        self.edge_mask = torch.Tensor(edge_mask).to(device_mesh)
        self.remaper = Remapper()
        
        print(data_name)
        if data_name == "blender":
            print("LOAD blender DATASET")
            # self.xyz2mesh = torch.Tensor([[0,1,0],[0,0,1],[1,0,0]]).to(torch.float32).cuda()
            self.xyz2mesh = np.array([[0,1,0],[0,0,1],[1,0,0]]).astype(np.float32)
        elif data_name == "dtu":
            print("LOAD dtu DATASET")
            # self.xyz2mesh = torch.Tensor([[0,1,0],[0,0,1],[1,0,0]]).to(torch.float32).cuda()
            self.xyz2mesh = np.array([[1,0,0],[0,0,1],[0,-1,0]]).astype(np.float32)
        elif data_name == "tankstemple":
            print("LOAD tanks-and-temple DATASET")
            mesh_data = np.loadtxt(os.path.join(os.path.split(obj_path)[0],'mesh_data.txt'))
            scale = mesh_data[0]
            scale_mat = np.eye(3)/scale
            coord_mat = np.array([[0,0,1],[0,-1,0],[1,0,0]]).astype(np.float32)
            self.xyz2mesh = np.array([[1,0,0],[0,1,0],[0,0,1]]).astype(np.float32)

            # igl.write_obj("test.obj",test_v,f)
        elif data_name == "tankstemple2":
            print("LOAD tanks-and-temple (NeuralAngleo) DATASET")
            # mesh_data = np.loadtxt(os.path.join(os.path.split(obj_path)[0],'mesh_data.txt'))
            # scale = mesh_data[0]
            # scale_mat = np.eye(3)/scale
            # coord_mat = np.array([[0,0,1],[0,-1,0],[1,0,0]]).astype(np.float32)
            # self.xyz2mesh = np.array([[1,0,0],[0,1,0],[0,0,1]]).astype(np.float32)
            
            self.xyz2mesh = np.array([[0,1,0],[0,0,1],[1,0,0]]).astype(np.float32)
        else:
            raise Exception("dataset not supported")
        
        
                
        
        rmv_datas = igl.remove_duplicate_vertices(v,f,1e-7)
        
        remap_v_fn = rmv_datas[1]
        remap_f_fn = rmv_datas[2]
        
        new_f = remap_f_fn[f]
        
        ind  = np.argsort(new_f,axis=1)
        
        
        f_up = f[np.arange(f.shape[0])[:,None],ind]
        
        new_f.sort()
        
        ### Use original info
        
        
        uvs = tc[f_up]
        uv_u = np.concatenate((uvs,np.ones((uvs.shape[0],uvs.shape[1],1))),-1)
        uv_d = np.concatenate((uvs,np.zeros((uvs.shape[0],uvs.shape[1],1))),-1)
        
        self.uv_t1 = torch.Tensor(np.stack((uv_u[:,0],uv_u[:,1],uv_u[:,2],uv_d[:,0]),-1))
        self.uv_t2 = torch.Tensor(np.stack((uv_u[:,1],uv_u[:,2],uv_d[:,0],uv_d[:,1]),-1))
        self.uv_t3 = torch.Tensor(np.stack((uv_u[:,2],uv_d[:,0],uv_d[:,1],uv_d[:,2]),-1))
        self.uvh = torch.stack((self.uv_t1,self.uv_t2,self.uv_t3), 1).reshape(-1,3,4)
        
        
        
        
        new_v = rmv_datas[0]
        new_vn = vn[remap_v_fn]
        
        
        if disp_up < 0 :
            disp_up = disp
        if disp_down < 0 :
            disp_down = disp
        
        max_disp, min_disp = check_disp(new_v,new_vn,new_f,disp_up,disp_down)
        
        
        if data_name == "tankstemple":
            new_v =  new_v @ scale_mat @ coord_mat  + mesh_data[1:]
            new_vn = new_vn @ coord_mat
        else:
            # breakpoint()
            new_v = new_v @ self.xyz2mesh
            new_vn = new_vn @ self.xyz2mesh
        
        
        # _v_u = (new_v+disp*new_vn)
        # _v_d = (new_v-disp*new_vn)
        
        _v_u = (new_v+max_disp[...,None]*new_vn)
        _v_d = (new_v+min_disp[...,None]*new_vn)
               
        
        v_u = _v_u[remap_f_fn][f_up]
        v_d = _v_d[remap_f_fn][f_up]
        ## unmapping
        
        self.v_u = v_u.astype(np.float32)
        self.v_d = v_d.astype(np.float32)
        v_up0 = self.v_u[:,0]
        v_up1 = self.v_u[:,1]
        v_up2 = self.v_u[:,2]
        
        v_down0 = self.v_d[:,0]
        v_down1 = self.v_d[:,1]
        v_down2 = self.v_d[:,2]
        
        self.xyz_t1 = torch.Tensor(np.stack((v_up0,v_up1,v_up2,v_down0),-1))
        self.xyz_t2 = torch.Tensor(np.stack((v_up1,v_up2,v_down0,v_down1),-1))
        self.xyz_t3 = torch.Tensor(np.stack((v_up2,v_down0,v_down1,v_down2),-1))
        self.xyz_t = torch.stack((self.xyz_t1,self.xyz_t2,self.xyz_t3), 1).reshape(-1,3,4)
        self.xyz_t_cuda = self.xyz_t.to(device_mesh)
        
        
        self.vs = voxel_size
        
        
        
        vertex_normal = new_vn[remap_f_fn][f_up]
        
        v_n0 = vertex_normal[:,0]
        v_n1 = vertex_normal[:,1]
        v_n2 = vertex_normal[:,2]
        
        
        normal_t1 = torch.Tensor(np.stack((v_n0,v_n1,v_n2,v_n0),-1))
        normal_t2 = torch.Tensor(np.stack((v_n1,v_n2,v_n0,v_n1),-1))
        normal_t3 = torch.Tensor(np.stack((v_n2,v_n0,v_n1,v_n2),-1))
        normal_t = torch.stack((normal_t1,normal_t2,normal_t3), 1).reshape(-1,3,4)
        self.normal= normal_t.to(device_mesh)
        
        self.setup_grid()
    

    def convert(self,pts,vds):
        
        
        
        pts_flat = pts.reshape(-1,3).to(device_mesh)
        
        
        pts_cov = pts_flat
        
        
        
        pg = torch.floor((pts_cov -self.bbox_torch[1])/self.vs).to(torch.long)
                
        val_vol = (pg[:, 0] < self.v_grid.shape[0]) & (pg[:, 0] >= 0) & \
             (pg[:, 1] < self.v_grid.shape[1]) & (pg[:, 1] >= 0) & \
             (pg[:, 2] < self.v_grid.shape[2]) & (pg[:, 2] >= 0)

        
        pg_vol = pg[val_vol]
        pg_idx = self.v_grid[pg_vol[:,0],pg_vol[:,1],pg_vol[:,2]]
        
        ## REMOVE empty area
        val_empty = (pg_idx > -0.5)
        
        
        
        pg_fill = pg_idx[val_empty]
        
        pts_vol = pts_cov[val_vol]
        pts_fill = pts_vol[val_empty]
        
        tet_id_fill = self.vox2tet[pg_fill]
        tet_mask_fill = self.vox_mask[pg_fill]
        pts_id = torch.arange(pts_fill.shape[0]).to(device_mesh)[...,None].repeat(1,tet_id_fill.shape[-1])
        
        tet_id_masked = tet_id_fill[tet_mask_fill]
        pts_id_masked = pts_id[tet_mask_fill]
        
        
        # Ax = b
        # A^t A x = A^t b
        # x = (A^t A)^-1 A^t b = M A^t b = M b'
        mtx_M = self.tet_mtx[tet_id_masked]
        
        mtx_b = pts_fill[pts_id_masked]-self.xyz_t_cuda[tet_id_masked,...,3]
        
        mtx_tran = self.tran_mtx[tet_id_masked]
        
        
        
        mtx_b_p = torch.matmul(mtx_tran,mtx_b[...,None])
        ini_sol = torch.matmul(mtx_M,mtx_b_p)[...,0]
        
        ini_cost = torch.abs(1-ini_sol.sum(1)) + torch.abs(ini_sol).sum(1)
        
        
        tight_val = ini_cost < (1.0+1e-6)
        tight_val,tight_inv = scatter_max((tet_id_masked+1) * tight_val,pts_id_masked)
        tight_mask = tight_val > 0.5
        
        inv_tight = tight_inv[tight_mask]
        
        
        
        inv_map = inv_tight
        mask_bary = tight_mask
        
        if mask_bary.sum() == 0:
            return None, None, [], [], []
        
        
        
        _bary = ini_sol[inv_map]
        
            
        
        bary = torch.cat((_bary,(1-_bary.sum(1))[...,None]),-1)
        
        
        uvh_masked = self.uvh[tet_id_masked[inv_map]].to(device_mesh)
        
        uvh_initial = torch.matmul(uvh_masked,bary[...,None])[...,0]
        
        uv_initial = uvh_initial[...,:2]
        h_initial = uvh_initial[...,-1:]
        grid = uv_initial * 2.0 -1.0
        # uv_mod = self.uv.clone()
        # uv_mod[...,0] = 1.0 - uv_mod[...,0]
        # breakpoint()
        result = torch.nn.functional.grid_sample(self.uv[None,...],grid[None,None,...],align_corners=True)
        result_mask = torch.nn.functional.grid_sample(self.uv_mask[None,None,...],grid[None,None,...],align_corners=True,mode='nearest')
        result_edge_mask = torch.nn.functional.grid_sample(self.edge_mask[None,None,...],grid[None,None,...],align_corners=True)
        new_uv = result[0,:,0,:].permute(1,0)
        new_uv_mask = result_mask[0,0,0,:]>0.0
        new_edge_mask = result_edge_mask[0,0,0,:]
        uvh_target = torch.cat((new_uv,h_initial),-1).to(uvh_initial.device)
        
        
        if new_edge_mask.max()>0.0:
            
            result_nn = torch.nn.functional.grid_sample(self.uv[None,...],grid[None,None,...],align_corners=True,mode='nearest')
            new_uv_nn = result_nn[0,:,0,:].permute(1,0)
            uvh_target[new_edge_mask>0.0] = torch.cat((new_uv_nn,h_initial),-1).to(uvh_initial.device)[new_edge_mask>0.0]
        
        '''
        self.uv = torch.Tensor(uv).to(device_mesh).permute(2,0,1)
        self.uv_mask = torch.Tensor(uv_mask).to(device_mesh)
        self.edge_mask = torch.Tensor(edge_mask).to(device_mesh)
        '''
            
        
        
        _uvh_initial = uvh_initial.detach().clone()
        uvh = self.remaper.map(uvh_target)
        uvh_init = self.remaper.map(_uvh_initial)
        
        valids = torch.zeros((pts_cov.shape[0],)).to(torch.bool)
        valids_idxs = torch.arange(pts_cov.shape[0])[val_vol][val_empty][mask_bary]
        valids[valids_idxs] = True
        
        ## re-calcualte direction
        # valid_mtx_M = mtx_M[inv_map]
        # valid_pts = pts_fill[pts_id_masked][inv_map]
        # valid_xyz_t = self.xyz_t_cuda[tet_id_masked[inv_map],...,3]
        # valid_vds = vds[valids].to(device_mesh)
        # valid_delta = valid_pts + 1e-6 * valid_vds
        # valid_mtx_b = valid_delta-valid_xyz_t
        # valid_mtx_tran = mtx_tran[inv_map]
        
        # valid_mtx_b_p = torch.matmul(valid_mtx_tran,valid_mtx_b[...,None])
        # delta_sol = torch.matmul(valid_mtx_M,valid_mtx_b_p)[...,0]
        # delta_bary = torch.cat((delta_sol,(1-delta_sol.sum(1))[...,None]),-1)
        # delta_uvh = torch.matmul(uvh_masked,delta_bary[...,None])[...,0]
        # uvh_dirs = delta_uvh - uvh
        # uvh_dirs = uvh_dirs / (torch.norm(uvh_dirs,dim=1)[:,None]+1e-7)
        
        '''
        #Reflection Direction
        
        normal_masked = self.normal[tet_id_masked[inv_map]].to(device_mesh)
        
        normals = torch.matmul(normal_masked,bary[...,None])[...,0]
        
        # breakpoint()
        
        w_0 = -vds[valids].to(device_mesh)
        
        new_vds = 2*torch.sum(w_0*normals,-1)[:,None]*normals - w_0
        
        new_vds = new_vds / (torch.norm(new_vds,dim=1,keepdim=True)+1e-7)
        '''
        
        
        return uvh.to(device_model), valids.to(device_model), vds[valids].to(device_model), new_uv_mask.to(device_model), uvh_init.to(device_model)
      
    def setup_grid(self):
        
        t1_np = self.xyz_t1.detach().cpu().numpy()
        t2_np = self.xyz_t2.detach().cpu().numpy()
        t3_np = self.xyz_t3.detach().cpu().numpy()
        
        bbox = torch.stack([self.xyz_t.max(-1)[0].max(0)[0],self.xyz_t.min(-1)[0].min(0)[0]],0).detach().cpu().numpy()
        self.bbox = bbox
        self.bbox_torch = torch.Tensor(self.bbox).to(device_mesh)
        
        size = np.floor((bbox[0]-bbox[1])/self.vs).astype(np.int16)
        self.v_grid = torch.full((size[0]+1,size[1]+1,size[2]+1),-1)
        
        global counter
        counter = 0
        val = []        
        
        count_dict = dict()
        ##
        inv_dict = dict()
        ##
        
        def insert(pt_div,id):
            global counter
            for g in pt_div:
                x,y,z = g[0],g[1],g[2]
                if self.v_grid[x,y,z] == -1:
                    val.append([])
                    self.v_grid[x,y,z] = counter
                    count_dict[counter] = 0 
                    inv_dict[counter] = (x,y,z)
                    counter = counter + 1
                
                cur_c = int(self.v_grid[x,y,z])
                val[cur_c].append(id)
                count_dict[cur_c] = count_dict[cur_c] + 1
                
        def process(pt,id):
            pt_min = pt.min(1)
            pt_max = pt.max(1)
            pt_min_c = np.floor((pt_min-bbox[1])/self.vs).astype(np.int16)
            pt_max_c = np.floor((pt_max-bbox[1])/self.vs).astype(np.int16)
            pt_r0 = np.arange(pt_min_c[0],pt_max_c[0]+1)
            pt_r1 = np.arange(pt_min_c[1],pt_max_c[1]+1)
            pt_r2 = np.arange(pt_min_c[2],pt_max_c[2]+1)
            pt_rs = cartesian((pt_r0,pt_r1,pt_r2))
            insert(pt_rs,id)
        
        for f_t in trange(t1_np.shape[0]):
            pt1 = t1_np[f_t]
            pt2 = t2_np[f_t]
            pt3 = t3_np[f_t]
            process(pt1,f_t*3)
            process(pt2,f_t*3 + 1)
            process(pt3,f_t*3 + 2)
        
        
        
        
            
        
        # breakpoint()
        
        max_tet = np.array(list(count_dict.values())).max()
        print("max_tet : ",max_tet)
        # breakpoint()
        self.vox2tet = torch.zeros((len(val),max_tet)).to(torch.long) #(# of voxel, # of max tet id)
        self.vox_mask = torch.zeros((len(val),max_tet)).to(torch.bool)
        
        for idx in range(len(val)):
            idx_len = len(val[idx])
            self.vox2tet[idx,0:idx_len] = torch.Tensor(val[idx])
            self.vox_mask[idx,0:idx_len] = True
        
        _u = self.xyz_t[...,0]-self.xyz_t[...,3]
        _v = self.xyz_t[...,1]-self.xyz_t[...,3]
        _w = self.xyz_t[...,2]-self.xyz_t[...,3]
        d00 = (_u*_u).sum(1)
        d01 = (_u*_v).sum(1)
        d02 = (_u*_w).sum(1)
        d10 = (_v*_u).sum(1)
        d11 = (_v*_v).sum(1)
        d12 = (_v*_w).sum(1)
        d20 = (_w*_u).sum(1)
        d21 = (_w*_v).sum(1)
        d22 = (_w*_w).sum(1)
        
        m00 =  d11*d22 - d12 * d21
        m01 = -(d01*d22 -d02*d21) 
        m02 = d01*d12 - d02*d11
        m10 = -(d10*d22-d12*d20)
        m11 = d00*d22-d02*d20
        m12 = -(d00*d12-d02*d10)
        m20 = d10*d21-d11*d20
        m21 = -(d00*d21-d01*d20)
        m22 = d00*d11 - d01*d10
        det = 1/(d00*d11*d22 + d01*d12*d20 + d02*d10*d21 - d02*d11*d20 - d01*d10*d22 -d00*d12*d21)
        met = torch.stack([m00,m01,m02,m10,m11,m12,m20,m21,m22],-1) * det[:,None]
        self.tet_mtx = met.reshape(-1,3,3)
        self.tran_mtx =torch.cat((_u,_v,_w),-1).reshape(-1,3,3)
        
        self.v_grid = self.v_grid.to(device_mesh)
        self.vox2tet = self.vox2tet.to(device_mesh)
        self.vox_mask = self.vox_mask.to(device_mesh)
        self.tet_mtx = self.tet_mtx.to(device_mesh)
        self.tran_mtx = self.tran_mtx.to(device_mesh)
        self.inv_dict = inv_dict
        ##
        self.binary_gird = (self.v_grid>(-0.5)).to(device_model)
        
        max_loc = torch.floor((self.bbox_torch[0] -self.bbox_torch[1])/self.vs).to(torch.long)*self.vs + self.bbox_torch[1] + self.vs
        self.aabbs = torch.cat((self.bbox_torch[1],max_loc)).to(device_model)
        # self.save_voxel_mesh()
        # breakpoint()
        ##
    
        
        
    def get_aabbs(self):
        aabbs = []
        for count in self.inv_dict.keys():
            _x,_y,_z = self.inv_dict[count]
            x, y, z = _x*self.vs+self.bbox[1][0], _y*self.vs+self.bbox[1][1], _z*self.vs+self.bbox[1][2]
            xp,yp,zp = x + self.vs , y + self.vs , z + self.vs
            aabbs.append([x,y,z,xp,yp,zp])
        return torch.Tensor(aabbs).to(device_mesh)
        
    @torch.no_grad()
    def filter_rays(self,all_rays, all_rgbs, chunk=1024*5):
        print("Initial ray filering")
        tt = time.time()
        
        aabbs = self.get_aabbs()
        aabbs.requires_grad  = False
        N = torch.tensor(all_rays.shape[:-1]).prod()
        idx_chunks = torch.split(torch.arange(N), chunk)
        filters = []
        for idx_chunk in tqdm(idx_chunks):
            rays_chunk = all_rays[idx_chunk].to(device_mesh)

            rays_o, rays_d = rays_chunk[..., :3], rays_chunk[..., 3:6]
            
            _,_, hits = nerfacc.ray_aabb_intersect(rays_o,rays_d,aabbs)
            # breakpoint()
            filters.append(hits.any(-1))
        filter_mask = torch.cat(filters).view(all_rgbs.shape[:-1]).detach().cpu()
        
        print(f'Ray filtering done! {time.time()-tt} s.')
        
        return all_rays[filter_mask], all_rgbs[filter_mask]
    
        
    
    @torch.no_grad()
    def ray_sample(self, rays_chunk, step_size = 0.001, N_samples=-1, is_train=False):
        
        # sample points
        rays_o = rays_chunk[:, :3]
        rays_d = rays_chunk[:, 3:6]
        # breakpoint()
        _, sample, _ = nerfacc.traverse_grids(rays_o,rays_d,self.binary_gird[None,...],self.aabbs[None,...].to(device_model),step_size=step_size) 
        vals = sample.vals
        ri = sample.ray_indices
        pack = sample.packed_info
        
        
        pack_arr = pack[:,1].view(-1).long()

        # Create a sequence of indices for each row in the mask
        indices = torch.arange(0, pack_arr.max().item()).unsqueeze(0).repeat(pack_arr.shape[0],1).to(pack_arr.device)

        # Create the mask using boolean comparison and clamp to ensure correct size
        ray_valid = (indices < pack_arr.unsqueeze(1)) 
        
        z_vals = indices * step_size
        if is_train:
            diff = (torch.rand_like(z_vals) - 0.5)*step_size
            z_vals += diff
            vals += diff[ray_valid]
        
        
        ray_pts_flat= rays_o[ri] + rays_d[ri] * vals[...,None]
        
        
        ray_pts = torch.zeros((ray_valid.shape[0],ray_valid.shape[1],3)).to(ray_pts_flat.device)
        ray_pts[ray_valid] = ray_pts_flat
        
        
        
        xyz_sampled = ray_pts
        viewdirs = rays_d.view(-1, 1, 3).expand(xyz_sampled.shape)
        
        
        dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
        
        
        return xyz_sampled, dists, ray_valid, viewdirs, z_vals
            
    def boundary_sample(self, N_samples=1000):
        ## sample query points
        sample_idx = torch.randint(self.tc_edge.shape[0], (N_samples,)).to(device_mesh)
        sample_ratio = torch.rand((N_samples,)).to(device_mesh)
        sample_height = torch.rand((N_samples,)).to(device_mesh)
        phi = np.pi * torch.rand((N_samples,)).to(device_mesh)
        theta = 2 * np.pi * torch.rand((N_samples,)).to(device_mesh)
        _x = torch.sin(phi) * torch.cos(theta)
        _y = torch.sin(phi) * torch.sin(theta)
        _z = torch.cos(phi)
        sample_angle = torch.stack([_x,_y,_z],-1)
        
        ## query points to input uvh
        sample_tc_edge = self.tc_edge[sample_idx]
        sample_uv1 = sample_tc_edge[:,0,:2] * sample_ratio[:,None] + sample_tc_edge[:,0,2:] * (1.0 - sample_ratio[:,None])
        sample_uvh1 = torch.cat([sample_uv1,sample_height[...,None]],1)
        
        sample_uv2 = sample_tc_edge[:,1,:2] * sample_ratio[:,None] + sample_tc_edge[:,1,2:] * (1.0 - sample_ratio[:,None])
        sample_uvh2 = torch.cat([sample_uv2,sample_height[...,None]],1)
        
        return sample_uvh1.to(device_model), sample_uvh2.to(device_model), sample_angle.to(device_model)
        


def loss_edge_fn(mesh,query_fn,N_samples=1000):
    uvh1,uvh2,dirs = mesh.boundary_sample(N_samples)
    uvh1 += torch.rand_like(uvh1) * 1e-5
    uvh2 += torch.rand_like(uvh2) * 1e-5
    rgb1,sig1 = query_fn(uvh1)
    rgb2,sig2 = query_fn(uvh2)
    loss_rgb = torch.mean((rgb1 - rgb2) ** 2)
    loss_sigma = torch.mean((torch.exp(-sig1) - torch.exp(-sig2) ) ** 2)
    return loss_sigma + loss_rgb
    

 
def cartesian(arrays, out=None):
    """
    Generate a Cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the Cartesian product of.
    out : ndarray
        Array to place the Cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing Cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    #m = n / arrays[0].size
    m = int(n / arrays[0].size)
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
        #for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
    return out        
