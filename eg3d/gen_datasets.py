# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Generate images and shapes using pretrained network pickle."""

import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm
import mrcfile


import legacy
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils import misc
from training.triplane import TriPlaneGenerator,TriPlaneGenerator_Modified


#----------------------------------------------------------------------------

def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        if m := range_re.match(p):
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

#----------------------------------------------------------------------------

def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

#----------------------------------------------------------------------------

def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--shapes', help='Export shapes as .mrc files viewable in ChimeraX', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--shape-res', help='', type=int, required=False, metavar='int', default=512, show_default=True)
@click.option('--fov-deg', help='Field of View of camera in degrees', type=int, required=False, metavar='float', default=18.837, show_default=True)
@click.option('--shape-format', help='Shape Format', type=click.Choice(['.mrc', '.ply']), default='.mrc')
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
# lora_dim, int, default 0
@click.option('--lora_dim', help='Dimension of the LoRA', type=int, required=False, metavar='int', default=0, show_default=True)
# lora_alpha, float, default 1.0
@click.option('--lora_alpha', help='Alpha value of the LoRA', type=float, required=False, metavar='float', default=1.0, show_default=True)
def generate_images(
    network_pkl: str,
    seeds: List[int],
    truncation_psi: float,
    truncation_cutoff: int,
    outdir: str,
    shapes: bool,
    shape_res: int,
    fov_deg: float,
    shape_format: str,
    class_idx: Optional[int],
    reload_modules: bool,
    lora_dim: int,
    lora_alpha: float,
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate an image using pre-trained FFHQ model.
    CUDA_VISIBLE_DEVICES=1 python gen_datasets.py --outdir=./dataset_v1_test --trunc=0.7 --seeds=5000-5099 --network=/root/eg3d/eg3d/pretrained_models/ffhqrebalanced512-128.pkl --reload_modules=True   
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    # Specify reload_modules=True if you want code modifications to take effect; otherwise uses pickled code
    if reload_modules:
        print("Reloading Modules!")
        if 'lora_dim' in G.init_kwargs:
            # G.init_kwargs['lora_dim'] = lora_dim
            # G.init_kwargs['lora_alpha'] = lora_alpha
            G_init_kwargs_new=G.init_kwargs.copy()
            G_init_kwargs_new['lora_dim']=lora_dim
            G_init_kwargs_new['lora_alpha']=lora_alpha
            G_init_kwargs_new['mapping_kwargs']['lora_dim']=lora_dim
            G_init_kwargs_new['mapping_kwargs']['lora_alpha']=lora_alpha
            G_new = TriPlaneGenerator_Modified(*G.init_args, **G_init_kwargs_new).eval().requires_grad_(False).to(device)
        else:
            G_new = TriPlaneGenerator_Modified(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=False)
        G_new.neural_rendering_resolution = G.neural_rendering_resolution
        G_new.rendering_kwargs = G.rendering_kwargs

        G = G_new

    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, 'triplane'), exist_ok=True)
    os.makedirs(os.path.join(outdir, 'image'), exist_ok=True)

    outdir_triplane=os.path.join(outdir, 'triplane')
    outdir_image=os.path.join(outdir, 'image')

    cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0.2], device=device), radius=2.7, device=device)
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)

    num_image_per_identity = 3
    angle_p_max=0.5
    angle_p_min=-0.5
    angle_y_max=0.7
    angle_y_min=-0.7
    import tqdm
    bar=tqdm.tqdm(total=len(seeds)*num_image_per_identity)
    camera_param_list=[]
    # Generate images.
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)

        for image_idx in range(num_image_per_identity):
            if image_idx==0:
                angle_p=-0.2
                angle_y=0.
            else:
                angle_y=np.random.uniform(angle_y_min, angle_y_max)
                angle_p=np.random.uniform(angle_p_min, angle_p_max)
            cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
            cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
            cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
            conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
            camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
            conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

            ws = G.mapping(z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)

            planes=G.get_planes(ws)
            if image_idx==0:
                torch.save(planes[0].to(torch.float16), os.path.join(outdir_triplane, f'id_{seed}_planes.pt'))

            render_return=G.render_from_planes(camera_params,planes)
            img = render_return['image']

            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(os.path.join(outdir_image, f'id_{seed}_img_{image_idx}.png'))

            depth_image= render_return['image_depth']
            # normalize
            depth_image = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min())
            depth_image = (depth_image * 255).to(torch.uint8).squeeze(0).cpu().numpy()

            PIL.Image.fromarray(depth_image[0], 'L').save(os.path.join(outdir_image, f'id_{seed}_depth_{image_idx}.png'))

            camera_param_list.append((f'id_{seed}_img_{image_idx}.png',camera_params.squeeze(-1).cpu().numpy().tolist()))

            bar.update(1)
    import json
    with open(os.path.join(outdir, f'camera_params_{seed}.txt'), 'w') as f:
        tosave={}
        tosave['labels']=camera_param_list
        json.dump(tosave, f)


        

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
