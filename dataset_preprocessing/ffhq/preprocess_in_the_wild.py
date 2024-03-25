# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import argparse
import concurrent.futures
parser = argparse.ArgumentParser()
parser.add_argument('--indir', type=str, required=True)
# gpu_ids 
parser.add_argument('--gpu_ids', type=str, default='0,1,2,3', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
args = parser.parse_args()

# # rm args.indir/*_mirror*
# command = f"rm {args.indir}/*_mirror*"
# print(command)
# os.system(command)

# # run mtcnn needed for Deep3DFaceRecon
# command = "python batch_mtcnn.py --in_root " + args.indir
# print(command)
# os.system(command)


# 这是将要在每个线程中执行的函数
def run_command(rank):
    command = f"python batch_mtcnn.py --in_root {args.indir}"
    print(command)
    os.system(command)

# 使用ThreadPoolExecutor创建一个线程池
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    # 将每个命令作为一个任务提交给线程池
    futures = [executor.submit(run_command, i) for i in range(4)]

    # 等待所有任务完成
    for future in concurrent.futures.as_completed(futures):
        future.result()  # 获取任务结果，这里不做特别处理

out_folder = args.indir.split("/")[-2] if args.indir.endswith("/") else args.indir.split("/")[-1]

os.makedirs(f'Deep3DFaceRecon_pytorch/checkpoints/pretrained/results/{out_folder}/',exist_ok=True)

# mkdir arg.indir/tmp_files, unlink Deep3DFaceRecon_pytorch/checkpoints/pretrained/results/{out_folder}/epoch_20_000000, ln -s arg.indir/tmp_files Deep3DFaceRecon_pytorch/checkpoints/pretrained/results/{out_folder}/epoch_20_000000
command = f"mkdir {args.indir}/tmp_files; unlink Deep3DFaceRecon_pytorch/checkpoints/pretrained/results/{out_folder}/epoch_20_000000; ln -s {args.indir}/tmp_files Deep3DFaceRecon_pytorch/checkpoints/pretrained/results/{out_folder}/epoch_20_000000"
print(command)
os.system(command)

# # run Deep3DFaceRecon
# os.chdir('Deep3DFaceRecon_pytorch')
# command = f"CUDA_VISIBLE_DEVICES={args.gpu_ids} python test.py --img_folder=" + args.indir + f" --gpu_ids={args.gpu_ids} --name=pretrained --epoch=20"
# print(command)
# os.system(command)
# os.chdir('..')

# 这是将要在每个线程中执行的函数
def run_command(rank):
    command = f"python crop_images_in_the_wild.py --indir={args.indir} --rank={rank} --world_size=16"
    print(command)
    os.system(command)

# 使用ThreadPoolExecutor创建一个线程池
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
    # 将每个命令作为一个任务提交给线程池
    futures = [executor.submit(run_command, i) for i in range(16)]

    # 等待所有任务完成
    for future in concurrent.futures.as_completed(futures):
        future.result()  # 获取任务结果，这里不做特别处理

# # convert the pose to our format
# command = f"python 3dface2idr_mat.py --in_root Deep3DFaceRecon_pytorch/checkpoints/pretrained/results/{out_folder}/epoch_20_000000 --out_path {os.path.join(args.indir, 'crop', 'cameras.json')}"
# # command = f"python 3dface2idr_mat.py --in_root /root/eg3d/dataset_preprocessing/ffhq/Deep3DFaceRecon_pytorch/checkpoints/pretrained/results/epoch_20_000000 --out_path {os.path.join(args.indir, 'crop', 'cameras.json')}"
# print(command)
# os.system(command)

# # additional correction to match the submission version
# command = f"python preprocess_face_cameras.py --source {os.path.join(args.indir, 'crop')} --dest {args.indir} --mode orig"
# print(command)
# os.system(command)


