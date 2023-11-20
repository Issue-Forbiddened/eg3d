import os
import json
import torch
import numpy as np
from tqdm import tqdm
import dnnlib
import legacy
from torch_utils import misc
from training.triplane import TriPlaneGenerator,TriPlaneGenerator_Modified
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics, UniformCameraPoseSampler
device=torch.device('cuda:0')
weight_dtype=torch.float32
with open('/home1/jo_891/data1/eg3d/eg3d/only_scale/stats_dict.json','r') as f:
    stats_dict=json.load(f)
    f.close()
mean_load=torch.tensor(stats_dict['mean'],device=device,dtype=weight_dtype).reshape(1,96,256,256)
std_load=torch.tensor(stats_dict['std'],device=device,dtype=weight_dtype).reshape(1,96,256,256)


with open('/home1/jo_891/data1/eg3d/eg3d/control3diff_trained_clip_large_noise/stats_dict.json','r') as f:
    stats_dict1=json.load(f)
mean_load1=torch.tensor(stats_dict1['mean'],device=device,dtype=weight_dtype).reshape(1,96,256,256)
std_load1=torch.tensor(stats_dict1['std'],device=device,dtype=weight_dtype).reshape(1,96,256,256)

mse_mean=((mean_load-mean_load1)**2).mean()
mse_std=((std_load-std_load1)**2).mean()
print(mse_mean,mse_std)

network_pkl='/home1/jo_891/data1/eg3d/ffhq512-128.pkl'

with dnnlib.util.open_url(network_pkl) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

# Specify reload_modules=True if you want code modifications to take effect; otherwise uses pickled code
print("Reloading Modules!")
G_new = TriPlaneGenerator_Modified(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
misc.copy_params_and_buffers(G, G_new, require_all=True)
G_new.neural_rendering_resolution = G.neural_rendering_resolution
G_new.rendering_kwargs = G.rendering_kwargs
G = G_new

fov_deg=18.837
intrinsics = FOV_to_intrinsics(fov_deg, device=device)
truncation_psi=0.7
truncation_cutoff=14

total_sample=50000
minibatch=64
total_batch=total_sample//minibatch
sample_count=0
mean_100k = torch.zeros([96, 256, 256], device=device,dtype=torch.float64)
M2_100k = torch.zeros([96, 256, 256], device=device,dtype=torch.float64)
mean_1000k = torch.zeros([96, 256, 256], device=device,dtype=torch.float64)
M2_1000k = torch.zeros([96, 256, 256], device=device,dtype=torch.float64)

normalize_fn_100k = lambda x: (x - mean_load1) / std_load1
normalize_fn_1000k = lambda x: (x - mean_load) / std_load

with torch.no_grad():
    for idx in tqdm(range(total_batch),desc='getting scale:'):
        z_generator=torch.randn(minibatch, G.z_dim, device=device)
        cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
        cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
        cam2world_pose=UniformCameraPoseSampler.sample(np.pi/2, np.pi/2 , horizontal_stddev=np.pi/4,vertical_stddev=np.pi/4,
            lookat_position=torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device), 
            radius=G.rendering_kwargs.get('avg_camera_radius', 2.7), device=device,batch_size=minibatch)
        conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device,batch_size=minibatch)
        camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9).repeat(cam2world_pose.shape[0],1)], 1)
        conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9).repeat(cam2world_pose.shape[0],1)], 1)

        ws = G.mapping(z_generator, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        planes=G.get_planes(ws) # (batch_size,96,256,256)
        planes= planes.to(torch.float64)
        planes_normalized_100k = normalize_fn_100k(planes)
        planes_normalized_1000k = normalize_fn_1000k(planes)
        # planes=torch.tanh(planes).to(torch.float64)

        if not torch.isnan(planes).any():
            sample_count += minibatch
            delta_100k = planes_normalized_100k - mean_100k
            mean_100k += delta_100k.sum(dim=0) / sample_count
            M2_100k += (1/sample_count)*((sample_count-minibatch)/(sample_count)*((delta_100k**2).sum(dim=0))-M2_100k)
            delta_1000k = planes_normalized_1000k - mean_1000k
            mean_1000k += delta_1000k.sum(dim=0) / sample_count
            M2_1000k += (1/sample_count)*((sample_count-minibatch)/(sample_count)*((delta_1000k**2).sum(dim=0))-M2_1000k)

print(f'mean_100k:{mean_100k.mean().item()},std:{torch.sqrt(M2_100k).mean().item()}')
print(f'mean_1000k:{mean_1000k.mean().item()},std:{torch.sqrt(M2_1000k).mean().item()}')