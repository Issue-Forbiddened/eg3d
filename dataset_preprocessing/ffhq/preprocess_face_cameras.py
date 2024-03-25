# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

#############################################################

# Usage: python dataset_preprocessing/ffhq/preprocess_ffhq_cameras.py --source /data/ffhq --dest /data/preprocessed_ffhq_images

#############################################################

import json
import numpy as np
from PIL import Image, ImageOps
import os
from tqdm import tqdm
import argparse
import torch
import sys
from multiprocessing import Pool
sys.path.append('../../eg3d')
from camera_utils import create_cam2world_matrix
from concurrent.futures import ProcessPoolExecutor

COMPRESS_LEVEL=0
    
def fix_intrinsics(intrinsics):
    intrinsics = np.array(intrinsics).copy()
    assert intrinsics.shape == (3, 3), intrinsics
    intrinsics[0,0] = 2985.29/700
    intrinsics[1,1] = 2985.29/700
    intrinsics[0,2] = 1/2
    intrinsics[1,2] = 1/2
    assert intrinsics[0,1] == 0
    assert intrinsics[2,2] == 1
    assert intrinsics[1,0] == 0
    assert intrinsics[2,0] == 0
    assert intrinsics[2,1] == 0
    return intrinsics

# For our recropped images, with correction
def fix_pose(pose):
    COR = np.array([0, 0, 0.175])
    pose = np.array(pose).copy()
    location = pose[:3, 3]
    direction = (location - COR) / np.linalg.norm(location - COR)
    pose[:3, 3] = direction * 2.7 + COR
    return pose

# Used in original submission
def fix_pose_orig(pose):
    pose = np.array(pose).copy()
    location = pose[:3, 3]
    radius = np.linalg.norm(location)
    pose[:3, 3] = pose[:3, 3]/radius * 2.7
    return pose

# Used for original crop images
def fix_pose_simplify(pose):
    cam_location = torch.tensor(pose).clone()[:3, 3]
    normalized_cam_location = torch.nn.functional.normalize(cam_location - torch.tensor([0, 0, 0.175]), dim=0)
    camera_view_dir = - normalized_cam_location
    camera_pos = 2.7 * normalized_cam_location + np.array([0, 0, 0.175])
    simple_pose_matrix = create_cam2world_matrix(camera_view_dir.unsqueeze(0), camera_pos.unsqueeze(0))[0]
    return simple_pose_matrix.numpy()

def flip_yaw(pose_matrix):
    flipped = pose_matrix.copy()
    flipped[0, 1] *= -1
    flipped[0, 2] *= -1
    flipped[1, 0] *= -1
    flipped[2, 0] *= -1
    flipped[0, 3] *= -1
    return flipped

def process_image(args, cameras):
    source, dest, filename, mode = args

    if filename not in cameras:
        return None

    pose = cameras[filename]['pose']
    intrinsics = cameras[filename]['intrinsics']

    if mode == 'cor':
        pose = fix_pose(pose)
    elif mode == 'orig':
        pose = fix_pose_orig(pose)
    elif mode == 'simplify':
        pose = fix_pose_simplify(pose)
    intrinsics = fix_intrinsics(intrinsics)
    label = np.concatenate([pose.reshape(-1), intrinsics.reshape(-1)]).tolist()

    # image_path = os.path.join(source, filename)
    # img = Image.open(image_path)

    # os.makedirs(os.path.dirname(os.path.join(dest, filename)), exist_ok=True)
    # img.save(os.path.join(dest, filename))

    # flipped_img = ImageOps.mirror(img)
    flipped_pose = flip_yaw(pose)
    flipped_label = np.concatenate([flipped_pose.reshape(-1), intrinsics.reshape(-1)]).tolist()
    base, ext = os.path.splitext(filename)
    flipped_filename = base + '_mirror' + ext
    # flipped_img.save(os.path.join(dest, flipped_filename))

    return [[filename, label], [flipped_filename, flipped_label]]
from tqdm import tqdm

# Other parts of your code remain the same

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str)
    parser.add_argument("--dest", type=str)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--mode", type=str, default="orig", choices=["orig", "cor", "simplify"])
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    # 一次性读取 JSON 文件
    camera_dataset_file = os.path.join(args.source, 'cameras.json')
    with open(camera_dataset_file, "r") as f:
        cameras = json.load(f)

    filenames = list(cameras.keys())
    if args.max_images:
        filenames = filenames[:args.max_images]

    tasks = [(args.source, args.dest, f, args.mode) for f in filenames]

    # 使用进程池来并行处理图片
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        # tqdm 可以显示进度条
        results = list(tqdm(executor.map(process_image, tasks, [cameras] * len(tasks)), total=len(tasks)))


    dataset = {'labels': []}
    for result in results:
        if result:
            dataset['labels'].extend(result)

    with open(os.path.join(args.dest, 'dataset.json'), "w") as f:
        json.dump(dataset, f)
