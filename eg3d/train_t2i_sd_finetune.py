# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Generate images and shapes using pretrained network_pkl pickle."""

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
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics, UniformCameraPoseSampler
from torch_utils import misc
from training.triplane import TriPlaneGenerator,TriPlaneGenerator_Modified

import diffusers
import torch.nn as nn

import accelerate
from diffusers.models.unet_2d_blocks import get_down_block,UNetMidBlock2D
from diffusers.utils import is_torch_version
from diffusers.optimization import get_scheduler
import wandb
from accelerate import Accelerator
import logging
from diffusers.training_utils import EMAModel
from accelerate.logging import get_logger
from diffusers import UNet2DConditionModel,StableDiffusionImageVariationPipeline,StableDiffusionPipeline
import shutil
import argparse
from accelerate.utils import ProjectConfiguration
from packaging import version
import torch.nn.functional as F
import math
import json
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
import lpips
from torchvision import transforms
from transformers import  CLIPVisionModelWithProjection,CLIPTextModel,CLIPTokenizer,CLIPModel,AutoProcessor
from training.dual_discriminator import DualDiscriminator_Mine

import cv2
import pdb
import time
from copy import deepcopy

from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

from diffusers.utils.import_utils import is_xformers_available

from functools import partial

from diffusers import ControlNetModel

# import matplotlib.pyplot as plt
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




class Encoder(ModelMixin,ConfigMixin):
    #111
    @register_to_config
    def __init__(
        noise_scheduler,
        in_channels=3,
        down_block_types=("DownEncoderBlock2D",),
        block_out_channels=(64,),
        layers_per_block=2,
        norm_num_groups=32,
        act_fn="silu",
        down_sampling_ratio :int =1,
    ):
        super().__init__()
        noise_scheduler.layers_per_block = layers_per_block

        noise_scheduler.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        noise_scheduler.mid_block = None
        noise_scheduler.down_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=noise_scheduler.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                resnet_eps=1e-6,
                downsample_padding=0,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=output_channel,
                temb_channels=None,
            )
            noise_scheduler.down_blocks.append(down_block)

        # # mid
        # noise_scheduler.mid_block = UNetMidBlock2D(
        #     in_channels=block_out_channels[i],
        #     resnet_eps=1e-6,
        #     resnet_act_fn=act_fn,
        #     output_scale_factor=1,
        #     resnet_time_scale_shift="default",
        #     attention_head_dim=output_channel,
        #     resnet_groups=norm_num_groups,
        #     temb_channels=None,
        # )

        noise_scheduler.gradient_checkpointing = False

    def forward(noise_scheduler, x):
        sample = x
        
        # interpolate to 256
        sample = F.interpolate(sample, size=(256, 256), mode="bilinear", align_corners=False)

        sample = noise_scheduler.conv_in(sample)

        down_block_res_samples=tuple()

        if noise_scheduler.training and noise_scheduler.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            # down
            if is_torch_version(">=", "1.11.0"):
                for down_block in noise_scheduler.down_blocks:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(down_block), sample, use_reentrant=False
                    )
                    if getattr(down_block,'has_cross_attention',False):
                        down_block_res_samples=down_block_res_samples+(res_sample)
            else:
                for down_block in noise_scheduler.down_blocks:
                    sample,res_sample = torch.utils.checkpoint.checkpoint(create_custom_forward(down_block), sample)
                    if hasattr(down_block,'attention'):
                        if len(down_block.attention)>0:
                            down_block_res_samples=down_block_res_samples+(res_sample[1],)

        else:
            # down
            for down_block in noise_scheduler.down_blocks:
                sample,res_sample = down_block(sample)
                # down_block_res_samples=down_block_res_samples+(sample.flatten(-2,-1).permute(0,2,1),)
                if hasattr(down_block,'attentions'):
                        if len(down_block.attentions)>0:
                            down_block_res_samples=down_block_res_samples+(res_sample[1].flatten(-2,-1).permute(0,2,1),)
        

        return down_block_res_samples

def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr

def compute_sigma(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    return sigma

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    # parser.add_argument(
    #     "--pretrained_model_name_or_path",
    #     type=str,
    #     default=None,
    #     required=True,
    #     help="Path to pretrained model or model identifier from huggingface.co/models.",
    # )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="control3diff_trained",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=500)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=5.0,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=0., help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default='epsilon',
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="./logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=3,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="control3diff",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        '--network_pkl', help='Network pickle filename', required=False,default='/root/eg3d/eg3d/pretrained_models/ffhqrebalanced512-128.pkl'
    )
    parser.add_argument(
        '--truncation_psi', type=float, help='Truncation psi', default=0.7
    )
    parser.add_argument(
        '--truncation_cutoff', type=int, help='Truncation cutoff', default=14
    )
    parser.add_argument(
        '--fov-deg', help='Field of View of camera in degrees', type=float, required=False, default=18.837
    )
    parser.add_argument(
        '--log_step_interval', help='Log step interval', type=int, required=False, default=1000
    )
    parser.add_argument(
        '--num_inference_steps', help='Number of inference steps', type=int, required=False, default=50
    )
    parser.add_argument("--wandb_offline", action="store_true", help="Whether to run wandb offline.")

    parser.add_argument("--test_video", action="store_true", help="Whether to test video.")
    # recon
    parser.add_argument("--recon", action="store_true", help="Whether to recon.")
    # multiview
    parser.add_argument("--multiview", action="store_true", help="Whether to multiview.")
    # adv
    parser.add_argument("--adv", action="store_true", help="Whether to adversarial training.")

    # verify_text str
    parser.add_argument("--verify_text", type=str, default='', help="The text to verify.")

    # additional_sample int
    parser.add_argument("--additional_sample", type=int, default=0, help="The additional sample to generate.")

    # scaled bool, default false
    parser.add_argument("--scaled", action="store_true", help="Whether to scaled.")

    # lora bool, default false
    parser.add_argument("--lora", action="store_true", help="Whether to lora.")

    # controlnet bool, default false
    parser.add_argument("--controlnet", action="store_true", help="Whether to controlnet.")

    # backbone bool, default false
    parser.add_argument("--backbone", action="store_true", help="Whether to backbone.")

    # freeze_attentions bool, default false
    parser.add_argument("--freeze_attentions", action="store_true", help="Whether to freeze_attentions.")

    # dataset_path str
    parser.add_argument("--dataset_path", type=str, default='', help="The dataset path.")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args



@torch.enable_grad()
def latent_guidance(latents,camera_params,G,lpips_fn,gt_dict,inv_norm_fn):
    latents=latents.detach().requires_grad_(True)
    latents_inv_normed=inv_norm_fn(latents)
    # render using latents
    return_dict=G.render_from_planes(camera_params,latents_inv_normed)
    loss=torch.tensor(0.0,device=latents.device)
    for k,v in return_dict.items():
        if k =='image':
            loss=loss+lpips_fn(v,gt_dict[k].detach()).mean()
            loss=loss+F.mse_loss(v,gt_dict[k].detach()).mean()
    gradient=torch.autograd.grad(loss,latents)[0]
    for param in G.parameters():
        if param.grad is not None:
            param.grad.zero_()
    return gradient


def forward_process(latent2img_fn,timesteps,noise_scheduler,latents,noise):
    forward_process_list=[]
    noise=noise.detach().clone()
    latents=latents.detach().clone()
    forward_process_list.append(latent2img_fn(latents))

    for i, t in enumerate(timesteps):
        if i%5==0 or i==len(timesteps)-1:
            forward_process_list.append(latent2img_fn(noise_scheduler.add_noise(latents, noise, t)))
    return forward_process_list

def reverse_process(latent2img_fn,timesteps,noise_scheduler,latents,unet,encoder_states,
                    camera_params,G,latent_guidance=None,lpips_fn=None,eg3doutput=None,
                    inv_normalize_fn=None,reverse_type='normal',return_pred=False,return_latents=False,i_end=50):
    reverse_process_list=[]
    latents=latents.detach().clone()
    reverse_process_list.append(latent2img_fn(latents))

    # if reverse_type=='cfg':
    encoder_states_none=[torch.zeros_like(encoder_state) for encoder_state in encoder_states]

    for i, t in enumerate(timesteps):
        # normal
        if reverse_type=='normal':
            pred=unet(latents,t,encoder_states,return_dict=False)[0] 
            latents=noise_scheduler.step(pred, t, latents, return_dict=False)[0]

        elif reverse_type=='langevin_correct':
            step_size=0.25
            if i<45:
                sigmas=compute_sigma(noise_scheduler, t)
                for langevin_step_idx in range(10):
                    pred=unet(latents,t,encoder_states,return_dict=False)[0] 
                    pred_non_con=unet(latents,t,encoder_states_none,return_dict=False)[0]
                    guidance_grad=latent_guidance(pred,camera_params,G,lpips_fn,eg3doutput,inv_normalize_fn)
                    pred=pred+2.0*(pred-pred_non_con)
                    pred=pred-1e4*guidance_grad*sigmas.expand_as(guidance_grad)
                    pred_dict=noise_scheduler.step(pred, t, latents, return_dict=True)
                    pred_epsilon=pred_dict.pred_epsilon
                    latents=latents+(-0.5*step_size*pred_epsilon+torch.randn_like(pred_epsilon)*(step_size**0.5))*sigmas.expand_as(pred_epsilon)
            else:
                pred=unet(latents,t,encoder_states,return_dict=False)[0] 
                pred_non_con=unet(latents,t,encoder_states_none,return_dict=False)[0]
                pred=pred+2.0*(pred-pred_non_con)
                
            latents=noise_scheduler.step(pred, t, latents, return_dict=False)[0]

        elif reverse_type=='langevin_correct1':
            step_size=0.25
            if i<45:
                sigmas=compute_sigma(noise_scheduler, t)
                for langevin_step_idx in range(10):
                    pred=unet(latents,t,encoder_states,return_dict=False)[0] 
                    pred_non_con=unet(latents,t,encoder_states_none,return_dict=False)[0]
                    pred=pred+2.0*(pred-pred_non_con)
                    pred_dict=noise_scheduler.step(pred, t, latents, return_dict=True)
                    pred_epsilon=pred_dict.pred_epsilon
                    latents=latents+(-0.5*step_size*pred_epsilon+torch.randn_like(pred_epsilon)*(step_size**0.5))*sigmas.expand_as(pred_epsilon)
            else:
                pred=unet(latents,t,encoder_states,return_dict=False)[0] 
                pred_non_con=unet(latents,t,encoder_states_none,return_dict=False)[0]
                pred=pred+2.0*(pred-pred_non_con)

            latents=noise_scheduler.step(pred, t, latents, return_dict=False)[0]

        elif reverse_type=='langevin_correct2':
            step_size=0.25
            if i<40:
                sigmas=compute_sigma(noise_scheduler, t)
                for langevin_step_idx in range(10):
                    pred=unet(latents,t,encoder_states,return_dict=False)[0] 
                    guidance_grad=latent_guidance(pred,camera_params,G,lpips_fn,eg3doutput,inv_normalize_fn)
                    pred=pred-1e4*guidance_grad*sigmas.expand_as(guidance_grad)
                    pred_dict=noise_scheduler.step(pred, t, latents, return_dict=True)
                    pred_epsilon=pred_dict.pred_epsilon
                    latents=latents+(-0.5*step_size*pred_epsilon+torch.randn_like(pred_epsilon)*(step_size**0.5))*sigmas.expand_as(pred_epsilon)
            else:
                pred=unet(latents,t,encoder_states,return_dict=False)[0] 
                
            latents=noise_scheduler.step(pred, t, latents, return_dict=False)[0]

        elif reverse_type=='cfg':
            pred_cfg=unet(latents,t,encoder_states,return_dict=False)[0]
            pred_non_con=unet(latents,t,encoder_states_none,return_dict=False)[0]
            pred_cfg=pred_cfg+4.0*(pred_cfg-pred_non_con)
            latents=noise_scheduler.step(pred_cfg, t, latents, return_dict=False)[0]
            pred=pred_cfg

        # vgg guidance
        elif reverse_type=='vgg_guidance_0':
            assert lpips_fn is not None, 'lpips_fn is None'
            assert latent_guidance is not None, 'latent_guidance is None'
            assert eg3doutput is not None, 'eg3doutput is None'
            assert inv_normalize_fn is not None, 'inv_normalize_fn is None'
            pred_guidance=unet(latents,t,encoder_states,return_dict=False)[0]
            guidance_grad=latent_guidance(pred_guidance,camera_params,G,lpips_fn,eg3doutput,inv_normalize_fn)
            sigmas=compute_sigma(noise_scheduler, t)
            pred_guidance=pred_guidance-7e5*guidance_grad*sigmas.expand_as(guidance_grad)
            latents=noise_scheduler.step(pred_guidance, t, latents, return_dict=False)[0]
            pred=pred_guidance
        # vgg guidance
        elif reverse_type=='vgg_guidance_1':
            assert lpips_fn is not None, 'lpips_fn is None'
            assert latent_guidance is not None, 'latent_guidance is None'
            assert eg3doutput is not None, 'eg3doutput is None'
            assert inv_normalize_fn is not None, 'inv_normalize_fn is None'
            pred_guidance=unet(latents,t,encoder_states,return_dict=False)[0]
            guidance_grad=latent_guidance(pred_guidance,camera_params,G,lpips_fn,eg3doutput,inv_normalize_fn)
            snr=compute_snr(noise_scheduler, t)
            pred_guidance=pred_guidance-7e5*guidance_grad*torch.sigmoid(-torch.log(snr)).expand_as(guidance_grad)
            latents=noise_scheduler.step(pred_guidance, t, latents, return_dict=False)[0]
            pred=pred_guidance

        # vgg guidance
        elif reverse_type=='vgg_guidance_2':
            assert lpips_fn is not None, 'lpips_fn is None'
            assert latent_guidance is not None, 'latent_guidance is None'
            assert eg3doutput is not None, 'eg3doutput is None'
            assert inv_normalize_fn is not None, 'inv_normalize_fn is None'
            pred_guidance=unet(latents,t,encoder_states,return_dict=False)[0]
            guidance_grad=latent_guidance(pred_guidance,camera_params,G,lpips_fn,eg3doutput,inv_normalize_fn)

            pred_guidance=pred_guidance-7e4*guidance_grad
            latents=noise_scheduler.step(pred_guidance, t, latents, return_dict=False)[0]
            pred=pred_guidance

        # vgg guidance
        elif reverse_type=='vgg_guidance_3':
            assert lpips_fn is not None, 'lpips_fn is None'
            assert latent_guidance is not None, 'latent_guidance is None'
            assert eg3doutput is not None, 'eg3doutput is None'
            assert inv_normalize_fn is not None, 'inv_normalize_fn is None'
            pred_guidance=unet(latents,t,encoder_states,return_dict=False)[0]
            guidance_grad=latent_guidance(pred_guidance,camera_params,G,lpips_fn,eg3doutput,inv_normalize_fn)

            pred_guidance=pred_guidance-7e5*guidance_grad
            latents=noise_scheduler.step(pred_guidance, t, latents, return_dict=False)[0]
            pred=pred_guidance

        # vgg guidance
        elif reverse_type=='vgg_guidance_4':
            assert lpips_fn is not None, 'lpips_fn is None'
            assert latent_guidance is not None, 'latent_guidance is None'
            assert eg3doutput is not None, 'eg3doutput is None'
            assert inv_normalize_fn is not None, 'inv_normalize_fn is None'
            pred_guidance=unet(latents,t,encoder_states,return_dict=False)[0]
            guidance_grad=latent_guidance(pred_guidance,camera_params,G,lpips_fn,eg3doutput,inv_normalize_fn)
            snr=compute_snr(noise_scheduler, t)
            pred_guidance=pred_guidance-7e4*guidance_grad*torch.sigmoid(-torch.log(snr)).expand_as(guidance_grad)
            latents=noise_scheduler.step(pred_guidance, t, latents, return_dict=False)[0]
            pred=pred_guidance

        # vgg guidance
        elif reverse_type=='vgg_guidance_5':
            assert lpips_fn is not None, 'lpips_fn is None'
            assert latent_guidance is not None, 'latent_guidance is None'
            assert eg3doutput is not None, 'eg3doutput is None'
            assert inv_normalize_fn is not None, 'inv_normalize_fn is None'
            pred_guidance=unet(latents,t,encoder_states,return_dict=False)[0]
            guidance_grad=latent_guidance(pred_guidance,camera_params,G,lpips_fn,eg3doutput,inv_normalize_fn)
            sigmas=compute_sigma(noise_scheduler, t)
            pred_guidance=pred_guidance-7e4*guidance_grad*sigmas.expand_as(guidance_grad)
            latents=noise_scheduler.step(pred_guidance, t, latents, return_dict=False)[0]
            pred=pred_guidance

        # vgg guidance
        elif reverse_type=='vgg_guidance_6':
            assert lpips_fn is not None, 'lpips_fn is None'
            assert latent_guidance is not None, 'latent_guidance is None'
            assert eg3doutput is not None, 'eg3doutput is None'
            assert inv_normalize_fn is not None, 'inv_normalize_fn is None'
            pred_guidance=unet(latents,t,encoder_states,return_dict=False)[0]
            guidance_grad=latent_guidance(pred_guidance,camera_params,G,lpips_fn,eg3doutput,inv_normalize_fn)
            sigmas=compute_sigma(noise_scheduler, t)
            guidance_grad_normalized=guidance_grad/guidance_grad.std()
            # step_size is exponential interpolation with ratio i/len(timesteps)
            step_size=0.1**(i/len(timesteps)+1)
            pred_guidance=pred_guidance-step_size*guidance_grad_normalized*sigmas.expand_as(guidance_grad)
            latents=noise_scheduler.step(pred_guidance, t, latents, return_dict=False)[0]
            pred=pred_guidance

        # vgg guidance
        elif reverse_type=='vgg_guidance_7':
            assert lpips_fn is not None, 'lpips_fn is None'
            assert latent_guidance is not None, 'latent_guidance is None'
            assert eg3doutput is not None, 'eg3doutput is None'
            assert inv_normalize_fn is not None, 'inv_normalize_fn is None'
            pred_guidance=unet(latents,t,encoder_states,return_dict=False)[0]
            guidance_grad=latent_guidance(pred_guidance,camera_params,G,lpips_fn,eg3doutput,inv_normalize_fn)
            sigmas=compute_sigma(noise_scheduler, t)
            guidance_grad_normalized=guidance_grad/guidance_grad.std()
            # step_size is exponential interpolation with ratio i/len(timesteps)
            step_size=0.1**(i/len(timesteps))
            pred_guidance=pred_guidance-step_size*guidance_grad_normalized*sigmas.expand_as(guidance_grad)
            latents=noise_scheduler.step(pred_guidance, t, latents, return_dict=False)[0]
            pred=pred_guidance

        # vgg guidance
        elif reverse_type=='vgg_guidance_8':
            assert lpips_fn is not None, 'lpips_fn is None'
            assert latent_guidance is not None, 'latent_guidance is None'
            assert eg3doutput is not None, 'eg3doutput is None'
            assert inv_normalize_fn is not None, 'inv_normalize_fn is None'
            pred_guidance=unet(latents,t,encoder_states,return_dict=False)[0]
            guidance_grad=latent_guidance(pred_guidance,camera_params,G,lpips_fn,eg3doutput,inv_normalize_fn)
            sigmas=compute_sigma(noise_scheduler, t)
            pred_guidance_guided=pred_guidance-7e5*guidance_grad*sigmas.expand_as(guidance_grad)
            original_pred_std=pred_guidance.std()
            guided_pred_std=pred_guidance_guided.std()
            pred_guidance_rescale=pred_guidance_guided*original_pred_std/guided_pred_std
            pred_guidance_interpolate=0.5*pred_guidance_guided+0.5*pred_guidance_rescale
            latents=noise_scheduler.step(pred_guidance_interpolate, t, latents, return_dict=False)[0]
            pred=pred_guidance_interpolate

        # vgg guidance
        elif reverse_type=='vgg_guidance_9':
            assert lpips_fn is not None, 'lpips_fn is None'
            assert latent_guidance is not None, 'latent_guidance is None'
            assert eg3doutput is not None, 'eg3doutput is None'
            assert inv_normalize_fn is not None, 'inv_normalize_fn is None'
            pred_guidance=unet(latents,t,encoder_states,return_dict=False)[0]
            guidance_grad=latent_guidance(pred_guidance,camera_params,G,lpips_fn,eg3doutput,inv_normalize_fn)
            sigmas=compute_sigma(noise_scheduler, t)
            pred_guidance_guided=pred_guidance-7e5*guidance_grad*sigmas.expand_as(guidance_grad)
            original_pred_std=pred_guidance.std()
            guided_pred_std=pred_guidance_guided.std()
            pred_guidance_rescale=pred_guidance_guided*original_pred_std/guided_pred_std
            pred_guidance_interpolate=0.2*pred_guidance_guided+0.8*pred_guidance_rescale
            latents=noise_scheduler.step(pred_guidance_interpolate, t, latents, return_dict=False)[0]
            pred=pred_guidance_interpolate

        if i%10==0 or i==len(timesteps)-1:
            if return_pred:
                reverse_process_list.append(latent2img_fn(pred))
            reverse_process_list.append(latent2img_fn(latents))
        if i==i_end:
            latents=pred
            break
    if not return_latents:
        return reverse_process_list
    else:
        return reverse_process_list,latents

def expand_unet(unet,unet_id):
    original_conv_in=unet.conv_in

    new_conv_in = nn.Conv2d(in_channels=96, out_channels=original_conv_in.out_channels, 
                            kernel_size=original_conv_in.kernel_size, 
                            stride=original_conv_in.stride, padding=original_conv_in.padding,
                            dtype=original_conv_in.weight.dtype,device=original_conv_in.weight.device)
    original_weights_conv_in = original_conv_in.weight.data
    new_weights = original_weights_conv_in.repeat(1, 24, 1, 1)  # 重复权重以匹配新的输入通道数

    # 将新的权重赋值给新的卷积层
    new_conv_in.weight.data = new_weights
    # new_conv_in.bias.data=original_conv_in.bias.data

    unet.conv_in=new_conv_in


    original_conv_out=unet.conv_out
    new_conv_out = nn.Conv2d(in_channels=original_conv_out.in_channels, out_channels=96, 
                            kernel_size=original_conv_out.kernel_size, 
                            stride=original_conv_out.stride, padding=original_conv_out.padding,
                            dtype=original_conv_out.weight.dtype,device=original_conv_out.weight.device
                            )
    original_weights_conv_out = original_conv_out.weight.data
    new_weights = original_weights_conv_out.repeat(24, 1, 1, 1)  # 重复权重以匹配新的输出通道数
    new_conv_out.weight.data = new_weights
    # new_conv_out.bias.data=original_conv_out.bias.data

    unet.conv_out=new_conv_out

    original_config=UNet2DConditionModel.load_config(unet_id,subfolder='unet')
    new_config=deepcopy(original_config)
    new_config['in_channels']=new_config['out_channels']=96
    unet.register_to_config(**new_config)

    return unet

def save_expanded_unet(unet,save_path):
    # save unet.conv_in and unet.conv_out 's state_dict
    torch.save(unet.conv_in.state_dict(), os.path.join(save_path,'unet_conv_in.pt'))
    torch.save(unet.conv_out.state_dict(), os.path.join(save_path,'unet_conv_out.pt'))

def load_expanded_unet(unet, load_path, device='cpu'):
    # Load the saved state_dicts into unet.conv_in and unet.conv_out
    # unet.conv_in.load_state_dict(torch.load('unet_conv_in.pt', map_location=device))
    # unet.conv_out.load_state_dict(torch.load('unet_conv_out.pt', map_location=device))
    if os.path.exists(os.path.join(load_path,'unet_conv_in.pt')):
        unet.conv_in.load_state_dict(torch.load(os.path.join(load_path,'unet_conv_in.pt'), map_location=device))
        unet.conv_out.load_state_dict(torch.load(os.path.join(load_path,'unet_conv_out.pt'), map_location=device))
        print('loaded unet conv_in and conv_out from {}'.format(load_path))
    else:
        print('no unet conv_in and conv_out in {}'.format(load_path))

def pred_x0_fn(
    noise_scheduler,
    model_output: torch.FloatTensor,
    timestep: int,
    sample: torch.FloatTensor,
):
    """
    Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
    process from the learned model outputs (most often the predicted noise).

    Args:
        model_output (`torch.FloatTensor`):
            The direct output from learned diffusion model.
        timestep (`float`):
            The current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            A current instance of a sample created by the diffusion process.
        generator (`torch.Generator`, *optional*):
            A random number generator.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`.

    Returns:
        [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`:
            If return_dict is `True`, [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] is returned, otherwise a
            tuple is returned where the first element is the sample tensor.

    """
    t = timestep
    

    # 1. compute alphas, betas
    alpha_prod_t = noise_scheduler.alphas_cumprod.to(t.device)[t]
    beta_prod_t = 1 - alpha_prod_t.to(t.device)

    while len(alpha_prod_t.shape) < len(sample.shape):
        alpha_prod_t = alpha_prod_t.unsqueeze(-1)
        beta_prod_t = beta_prod_t.unsqueeze(-1)


    # 2. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
    if noise_scheduler.config.prediction_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    elif noise_scheduler.config.prediction_type == "sample":
        pred_original_sample = model_output
    elif noise_scheduler.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
    else:
        raise ValueError(
            f"prediction_type given as {noise_scheduler.config.prediction_type} must be one of `epsilon`, `sample` or"
            " `v_prediction`  for the DDPMScheduler."
        )
    return pred_original_sample

import re
def extract_numbers(filename):
    numbers = re.findall(r'\d+', filename)
    return tuple(map(int, numbers))  # 将提取出的数字转换为整数，并返回一个元组

class Mydataset(torch.utils.data.Dataset):
    def __init__(self, path_to_dataset,device):

        self.caption_dir=os.path.join(path_to_dataset,'caption')
        self.image_dir=os.path.join(path_to_dataset,'image')
        self.triplane_dir=os.path.join(path_to_dataset,'triplane')

        self.caption_name_list=os.listdir(self.caption_dir)
        self.caption_name_list=sorted(self.caption_name_list,key=extract_numbers)

        self.image_name_list=os.listdir(self.image_dir)
        self.image_name_list=sorted(self.image_name_list,key=extract_numbers)
        self.image_name_list=[name for name in self.image_name_list if 'img_0' in name]

        self.triplane_name_list=os.listdir(self.triplane_dir)
        self.triplane_name_list=sorted(self.triplane_name_list,key=extract_numbers)

        assert len(self.caption_name_list)==len(self.triplane_name_list), f"len(self.caption_name_list) shoud = len(self.triplane_name_list), got{len(self.caption_name_list)} and {len(self.triplane_name_list)}"

        self.caption_path_list=[os.path.join(self.caption_dir,name) for name in self.caption_name_list]
        self.image_path_list=[os.path.join(self.image_dir,name) for name in self.image_name_list]
        self.triplane_path_list=[os.path.join(self.triplane_dir,name) for name in self.triplane_name_list]

        self.caption_list=[open(path,'r').readline() for path in self.caption_path_list]
        self.converted_caption2feature=False

        self.device=device

    def tokenize_caption(self,tokenizer,clip):
        # tokenized_caption_list=[tokenizer(caption,return_tensors='pt',
        #                                   padding="max_length",
        #                                   max_length=tokenizer.model_max_length,
        #                                   truncation=True).input_ids 
        #                                   for caption in self.caption_list]
        # self.feature_list=[clip.get_text_features(text_input_ids.to(clip.device)) for text_input_ids in tokenized_caption_list]
        self.tokenizer=tokenizer
        self.clip=clip
        self.converted_caption2feature=True

    def __len__(self):
        return len(self.triplane_path_list)

    @torch.no_grad()
    def __getitem__(self, idx):
        assert self.converted_caption2feature, "Please run tokenize_caption first."
        image_path=self.image_path_list[idx]
        triplane_path=self.triplane_path_list[idx]
        
        # triplane is a .pt file with shape (c,h,w)
        triplane=torch.load(triplane_path,map_location='cpu').to(device=self.device,dtype=torch.float32)
        
        # caption_feature=self.feature_list[idx].to(self.device)
        caption_feature=self.clip.get_text_features(self.tokenizer(self.caption_list[idx],return_tensors='pt',
                                          padding="max_length",
                                          max_length=self.tokenizer.model_max_length,
                                          truncation=True).input_ids.to(self.clip.device))

        examples = {
            "caption_feature": caption_feature,
            "triplane": triplane
        }

        return examples


def generate_images():
    """Generate images using pretrained network_pkl pickle.

    Examples:

    \b
    # Generate an image using pre-trained FFHQ model.
    python gen_samples.py --outdir=output --trunc=0.7 --shapes=True\\
        --network_pkl=ffhq-rebalanced-128.pkl
    # python train_control3diff.py --use_ema
    """
    logger = get_logger(__name__, log_level="INFO")

    args = parse_args()
    network_pkl= args.network_pkl
    truncation_psi=args.truncation_psi
    truncation_cutoff=args.truncation_cutoff
    fov_deg=args.fov_deg

    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    device=accelerator.device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    if args.use_ema:
        logger.info("EMA is not supported in this code.")

    unet_id="runwayml/stable-diffusion-v1-5"
    unet=UNet2DConditionModel.from_pretrained(unet_id,
                                              subfolder='unet',
                                              torch_dtype=torch.float32).cuda()
    unet=expand_unet(unet,unet_id)
    unet.requires_grad_(False)

    # args.lora, args.controlnet, args.backbone should only have one True
    assert sum([args.lora, args.controlnet, args.backbone]) <= 1, "Only one of lora, controlnet, backbone can be True."


    if sum([args.lora, args.controlnet, args.backbone]) ==0:
        args.backbone=True
        logger.info("No args.lora, args.controlnet, args.backbone is True, set args.backbone=True.")

    if args.freeze_attentions:
        assert args.backbone, "args.backbone should be True when args.freeze_attentions is True."

    if args.lora:
        # Freeze the unet parameters before adding adapters
        for param in unet.parameters():
            param.requires_grad_(False)

        unet_lora_config = LoraConfig(
            r=4,
            lora_alpha=4,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )

        # Add adapter and make sure the trainable params are in float32.
        unet.add_adapter(unet_lora_config,adapter_name='default_1') #adapter_name is important because some bugs in lora's loading codes.

        lora_layers = list(filter(lambda p: p.requires_grad, unet.parameters()))

        for param in unet.conv_in.parameters():
            param.requires_grad_(True)

        for param in unet.conv_out.parameters():
            param.requires_grad_(True)

        for param in unet.parameters():
            # only upcast trainable parameters (LoRA) into fp32
            if param.requires_grad:
                param.data = param.data.to(torch.float32)

    if args.controlnet:
        controlnet = ControlNetModel.from_unet(unet)
        controlnet = controlnet.to(device=device, dtype=torch.float32)
        controlnet.train()
        controlnet.requires_grad_(True)
    
    if args.backbone:
        unet.requires_grad_(True)
        if args.freeze_attentions:
            for name, param in unet.named_parameters():
                if "attention" in name:
                    param.requires_grad_(False)
    
    unet.train()

    clip=CLIPModel.from_pretrained('openai/clip-vit-large-patch14').to(device)
    clip.requires_grad_(False)
    encoder=lambda x:clip.get_image_features(x)
                    
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_normalize_fn= lambda x:processor(images=(x*127.5+128).clamp(0,255).to(torch.uint8),return_tensors='pt').pixel_values.to(device) #假设输入是-1~1
    tokenizer=CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
    if args.verify_text:     
        with torch.no_grad():
            text_inputs = tokenizer(
                args.verify_text,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            # prompt_embeds = text_encoder(text_input_ids.to(text_encoder.device))[0] # (1,77,768)
            prompt_embeds = clip.get_text_features(text_input_ids.to(clip.device)) # (1,768)
            prompt_embeds=prompt_embeds/torch.norm(prompt_embeds,dim=-1,keepdim=True)
    
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            # SAVE
            if accelerator.is_main_process:
                for i, model in enumerate(models):
                    if isinstance(model, UNet2DConditionModel):
                        if args.backbone:
                            model.save_pretrained(os.path.join(output_dir, "unet"))
                    elif isinstance(model, ControlNetModel):
                        model.save_pretrained(os.path.join(output_dir, "controlnet"))
                    
                    weights.pop()


        def load_model_hook(models, input_dir):

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                if isinstance(model, UNet2DConditionModel):
                    if args.backbone:
                        load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                    else:
                        continue
                elif isinstance(model, ControlNetModel):
                    load_model = ControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")



    print('Loading networks from "%s"...' % network_pkl)

    with dnnlib.util.open_url(network_pkl) as f:
        loaded_pickle = legacy.load_network_pkl(f)
        # pdb.set_trace()
        G = loaded_pickle['G_ema'].to(device) # type: ignore
        if args.adv:
            D = loaded_pickle['D'].to(device) # type: ignore
            D = DualDiscriminator_Mine(*D.init_args, **D.init_kwargs).train().requires_grad_(False).to(device)
            misc.copy_params_and_buffers(loaded_pickle['D'], D, require_all=True)
            D.requires_grad_(True)
        del loaded_pickle

    # Specify reload_modules=True if you want code modifications to take effect; otherwise uses pickled code
    print("Reloading Modules!")
    G_new = TriPlaneGenerator_Modified(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
    misc.copy_params_and_buffers(G, G_new, require_all=True)
    G_new.neural_rendering_resolution = G.neural_rendering_resolution
    G_new.rendering_kwargs = G.rendering_kwargs
    G = G_new
    G.requires_grad_(False)

    cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0.2], device=device), radius=2.7, device=device)
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)

    noise_scheduler = diffusers.schedulers.DDIMScheduler.from_pretrained(unet_id, subfolder="scheduler")

    params=[]
    conv_params={'params':list(unet.conv_in.parameters())+list(unet.conv_out.parameters()),'lr':10*args.learning_rate}
    params.append(conv_params)
    if args.lora:
        params.append({'params':(lora_layers),'lr':args.learning_rate})

    if args.controlnet:
        params.append({'params':controlnet.parameters(),'lr':args.learning_rate})

    if args.backbone:
        # 获取conv_in和conv_out的所有参数ID
        exclude_params = {id(p) for p in list(unet.conv_in.parameters()) + list(unet.conv_out.parameters())}
        if args.freeze_attentions:
            # 获取所有注意力模块的参数ID
            for name, param in unet.named_parameters():
                if "attention" in name:
                    exclude_params.add(id(param))

        # 生成参数列表，排除特定的参数
        filtered_params = (p for p in unet.parameters() if id(p) not in exclude_params)

        # 将筛选后的参数添加到参数组中
        params.append({'params': filtered_params, 'lr': args.learning_rate})



   # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
   
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes if args.lr_warmup_steps is not None else None,
        # num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    if args.adv:
        optimizer_D = optimizer_cls(
            D.parameters(),
            lr=10*args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
        lr_scheduler_D = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer_D,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes if args.lr_warmup_steps is not None else None,
            # num_training_steps=args.max_train_steps * accelerator.num_processes,
        )

    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        init_kwargs={'wandb':{'mode':'offline'}} if args.wandb_offline else {}
        print('start init trackers')
        accelerator.init_trackers(args.tracker_project_name, tracker_config,init_kwargs)
    
    num_cross_attention_block=sum([1 for block in unet.down_blocks if getattr(block, "has_cross_attention", False)] +[1])


    unet, optimizer, lr_scheduler = accelerator.prepare(
        unet, optimizer, lr_scheduler
    )

    if args.controlnet:
        controlnet = accelerator.prepare(controlnet)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    lpips_fn=lpips.LPIPS(net='vgg').to(device)
    for param in lpips_fn.parameters():
        param.requires_grad=False

    epoch_size=1000
    first_epoch=0

    log_step_interval=args.log_step_interval

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(epoch_size / args.gradient_accumulation_steps)


    def collate_fn(examples):
        caption_feature = torch.stack([example["caption_feature"] for example in examples])
        triplane = torch.stack([example["triplane"] for example in examples])
        return {"caption_feature": caption_feature, "triplane": triplane}
    
    train_dataset=Mydataset(args.dataset_path,device)
    train_dataset.tokenize_caption(tokenizer,clip)
    
    if accelerator.num_processes >1:
        train_sampler=torch.utils.data.distributed.DistributedSampler(train_dataset)
        logger.info("Using distributed sampler.")

    else:
        train_sampler=None
        logger.info("Not using distributed sampler.")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=not accelerator.num_processes >1,
        sampler=train_sampler,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=0,
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            # debug_record1=unet.conv_in.weight.data.clone().detach()
            if not args.backbone: #如果训练整个网络,在上一步就完成了checkpoint的加载
                load_expanded_unet(accelerator.unwrap_model(unet), os.path.join(args.output_dir, path), device=device)
                # debug_record=lora_layers[0].data.clone().detach()
                if args.lora:
                    from peft import set_peft_model_state_dict
                    lora_state_dict, lora_network_alphas = \
                        StableDiffusionPipeline.lora_state_dict(
                            os.path.join(args.output_dir,path,'lora'),
                            weight_name='pytorch_lora_weights.safetensors'
                            )
                    keys = list(lora_state_dict.keys())
                    unet_keys = [k for k in keys if k.startswith('unet')]
                    lora_state_dict = {k.replace(f"unet.", ""): v for k, v in lora_state_dict.items() if k in unet_keys}           
                    set_peft_model_state_dict(accelerator.unwrap_model(unet), lora_state_dict, adapter_name='default_1') # does things
                    # debug_record2=lora_layers[0].data.clone().detach()
                    accelerator.unwrap_model(unet).load_attn_procs(
                        lora_state_dict, network_alphas=None, low_cpu_mem_usage=True, _pipeline=None
                    ) # does nothing
                    
                    # StableDiffusionPipeline.load_lora_into_unet(lora_state_dict,
                    #                                             lora_network_alphas,
                    #                                             accelerator.unwrap_model(unet))
                elif args.controlnet:
                    pass # controlnet is loaded in the accelerator.load_state

            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
            if args.adv and os.path.exists(os.path.join(args.output_dir, path, "discriminator")):
                logger.info(f"Loading discriminator from {os.path.join(args.output_dir, path, 'discriminator')}")
                input_dir=os.path.join(args.output_dir, path)
                load_model = DualDiscriminator_Mine.from_pretrained(os.path.join(input_dir,"discriminator"))
                D.register_to_config(**load_model.config)
                D.load_state_dict(load_model.state_dict())
                del load_model

                logger.info(f"Loading discriminator optimizer from {os.path.join(args.output_dir, path, 'discriminator')}")
                optimizer_name=f"optimizer_D_{global_step}.bin"
                input_optimizer_file = os.path.join(input_dir, "discriminator", optimizer_name)
                optimizer_state = torch.load(input_optimizer_file, map_location=device)
                optimizer_D.load_state_dict(optimizer_state)
                del optimizer_state

                logger.info(f"Loading discriminator lr_scheduler from {os.path.join(args.output_dir, path, 'discriminator')}")
                lr_scheduler_name=f"lr_scheduler_D_{global_step}.bin"
                input_scheduler_file = os.path.join(input_dir, "discriminator", lr_scheduler_name)
                scheduler_state=torch.load(input_scheduler_file, map_location=device)
                lr_scheduler_D.load_state_dict(scheduler_state)
                del scheduler_state
                del input_dir


    else:
        initial_global_step = 0
    global_step = initial_global_step

    # optimizer, lr_scheduler = accelerator.prepare(
    #     optimizer, lr_scheduler
    # )

    if accelerator.is_main_process:
        for file_idx in range(0,1000000):
            if not os.path.exists(os.path.join(args.output_dir,"train_control3diff_tmp_{}.py".format(str(file_idx)))):
                shutil.copy(__file__, os.path.join(args.output_dir,"train_control3diff_tmp_{}.py".format(str(file_idx))))
                break

    if not os.path.exists(os.path.join(args.output_dir,'stats_dict.json')):
        if os.path.exists('outputs/image_variation_finetune_freeze_attentions/stats_dict.json'):
            shutil.copy('outputs/image_variation_finetune_freeze_attentions/stats_dict.json',os.path.join(args.output_dir,'stats_dict.json'))
        else:
            if accelerator.is_main_process:
                total_sample=200000
                minibatch=64
                total_batch=total_sample//minibatch
                sample_count=0
                mean = torch.zeros([96, 64, 64], device=device,dtype=torch.float64)
                M2 = torch.zeros([96, 64, 64], device=device,dtype=torch.float64)
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
                        # interpolate to 64x64
                        planes = F.interpolate(planes, size=(64, 64), mode='bilinear', align_corners=False)


                        if not torch.isnan(planes).any():
                            sample_count += minibatch
                            delta = planes - mean
                            mean += delta.sum(dim=0) / sample_count
                            M2 += (1/sample_count)*((sample_count-minibatch)/(sample_count)*((delta**2).sum(dim=0))-M2)
                # 计算最终的方差 
                variance = M2 
                # 计算标准差
                std = torch.sqrt(variance) # (96,256,256)
                std = std.cpu().numpy().tolist()
                mean_list = mean.cpu().numpy().tolist()  # (96,256,256)
                stats_dict = {'mean': mean_list, 'std': std}
                with open(os.path.join(args.output_dir,'stats_dict.json'),'w') as f:
                    json.dump(stats_dict,f)
                    
    accelerator.wait_for_everyone()
    # read stats_dict
    with open(os.path.join(args.output_dir,'stats_dict.json'),'r') as f:
        stats_dict=json.load(f)
    mean_load=torch.tensor(stats_dict['mean'],device=device,dtype=weight_dtype).reshape(1,96,64,64)
    std_load=torch.tensor(stats_dict['std'],device=device,dtype=weight_dtype).reshape(1,96,64,64)

    

    if args.scaled:
        logger.info('using scaled')
        normalize_fn=lambda x: (x-mean_load)/(std_load*7.5)
        inv_normalize_fn=lambda x: x*std_load*7.5+mean_load
    else:
        normalize_fn=lambda x: (x-mean_load)/(std_load)
        inv_normalize_fn=lambda x: x*std_load+mean_load
    
    if mean_load.isnan().any():
        raise ValueError('mean has nan value')
    if std_load.isnan().any():
        raise ValueError('std has nan value')

    accelerator.wait_for_everyone()

    sanity_checked=False
    encoder_output_shape=None

    pitch_range = 0.25
    yaw_range = 0.35
    frames=50
    camera_params_val_list=[]
    for frame_idx in range(frames):
        cam2world_pose_val = LookAtPoseSampler.sample(3.14/2 + yaw_range * np.sin(2 * 3.14 * frame_idx / (frames)),
                                            3.14/2 -0.05 + pitch_range * np.cos(2 * 3.14 * frame_idx / (frames)),
                                            torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device), 
                                            radius=G.rendering_kwargs['avg_camera_radius'], device=device,batch_size=args.train_batch_size)
        camera_params_val = torch.cat([cam2world_pose_val.reshape(-1, 16), intrinsics.reshape(-1, 9).repeat(cam2world_pose_val.shape[0],1)], 1)
        camera_params_val_list.append(camera_params_val)

    G_step_warmup=0
    G_step_warmup_total=2500

    if args.prediction_type is not None:
        # set prediction_type of scheduler if defined
        noise_scheduler.register_to_config(prediction_type=args.prediction_type)

    for epoch in tqdm(range(first_epoch, args.num_train_epochs),desc='epoch:',disable=not accelerator.is_main_process):
        train_loss=0.0
        adv_step_interval=1
        miscs_list={}
        unet.train()
        for step, batch in enumerate(tqdm(train_dataloader,desc='step:',disable=not accelerator.is_main_process)):
            none_condition=False
            adv_update_D_this_step=global_step%adv_step_interval==0 
            G_step_warmup+=1
            encoder_states=batch['caption_feature'].detach()
            planes=batch['triplane']
            with torch.no_grad():
                z_generator=torch.randn(args.train_batch_size, G.z_dim, device=device)
                cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
                cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
                cam2world_pose=UniformCameraPoseSampler.sample(np.pi/2, np.pi/2 , horizontal_stddev=np.pi/4,vertical_stddev=np.pi/4,
                    lookat_position=torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device), 
                    radius=G.rendering_kwargs.get('avg_camera_radius', 2.7), device=device,batch_size=args.train_batch_size)
                conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device,batch_size=args.train_batch_size)
                camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9).repeat(cam2world_pose.shape[0],1)], 1)
                conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9).repeat(cam2world_pose.shape[0],1)], 1)

                # ws = G.mapping(z_generator, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
                # planes=G.get_planes(ws) # (batch_size,96,256,256)
                
                eg3doutput=G.render_from_planes(camera_params,planes)
                
                if args.multiview:
                    mv_number=2
                    cam2world_pose_mv=UniformCameraPoseSampler.sample(np.pi/2, np.pi/2 , horizontal_stddev=np.pi/4,vertical_stddev=np.pi/4,
                                    lookat_position=torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device), 
                                    radius=G.rendering_kwargs.get('avg_camera_radius', 2.7), device=device,batch_size=mv_number*args.train_batch_size)
                    camera_params_mv = torch.cat([cam2world_pose_mv.reshape(-1, 16), intrinsics.reshape(-1, 9).repeat(cam2world_pose_mv.shape[0],1)], 1)

                    planes_mv=planes.repeat(mv_number,1,1,1)
                    eg3doutput_mv=G.render_from_planes(camera_params_mv,planes_mv)
            
            planes=F.interpolate(planes,(64,64),mode='bilinear',align_corners=False)
            latents=normalize_fn(planes)

            noise=torch.randn_like(latents)
            bsz = planes.shape[0]
            assert bsz==args.train_batch_size
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=planes.device)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            if global_step%10==1 and encoder_output_shape is not None and not (global_step%log_step_interval==0 or not sanity_checked):
                encoder_states=torch.zeros(encoder_output_shape,device=device,dtype=noisy_latents.dtype)
                none_condition=True
            else:
                # encoder_states=encoder(clip_normalize_fn(eg3doutput['image'].detach())).unsqueeze(1) # image_embeds: (batch_size, 768)
                if encoder_output_shape is None:
                    encoder_output_shape=encoder_states.shape

            if args.controlnet:
                controlnet_image=(eg3doutput['image'].detach()).to(dtype=weight_dtype)

                down_block_res_samples, mid_block_res_sample = controlnet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_states,
                        controlnet_cond=controlnet_image,
                        return_dict=False,
                    )     

                unet_output=unet(noisy_latents,
                                timesteps,
                                encoder_hidden_states=encoder_states,
                                down_block_additional_residuals=[
                                    sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                                ],
                                mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                )
            else:
                unet_output=unet(noisy_latents,
                                timesteps,
                                encoder_hidden_states=encoder_states,
                )
        
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            elif noise_scheduler.config.prediction_type == "sample":
                target = latents

            snr = compute_snr(noise_scheduler, timesteps)
            mse_loss_weights=torch.sigmoid(torch.log(snr))
            losses={}
            loss_diff = F.mse_loss(unet_output.sample.float(), target.float(), reduction="none")
            loss_diff = loss_diff.mean(dim=list(range(1, len(loss_diff.shape)))) * mse_loss_weights
            loss_diff = loss_diff.mean() 

            losses['loss_diff']=loss_diff

            miscs={}


            if args.recon:
                denoised_output_recon=G.render_from_planes(camera_params,inv_normalize_fn(unet_output.sample))
                for k,v in eg3doutput.items():
                    losses['loss_{}_recon'.format(k)]=F.mse_loss(v,denoised_output_recon[k])
                    if k=='image' or k=='image_raw':
                        losses['loss_{}_lpips_recon'.format(k)]=lpips_fn(v,denoised_output_recon[k]).mean()
                if args.adv:
                    for k,v in losses.items():
                        if 'recon' in k:
                            losses[k]=v*max(1-G_step_warmup/G_step_warmup_total,0.01)

                    logits_fake_recon=D(denoised_output_recon,camera_params)

                    if adv_update_D_this_step:
                        eg3doutput_={k:v.detach().requires_grad_(True) for k,v in eg3doutput.items()}
                        logits_real_recon=D(eg3doutput_,camera_params)
                        miscs['D_real_accuracy_recon']=logits_real_recon.sign().mean().item()

                        loss_adv_D_real_recon=F.softplus(-logits_real_recon).mean()
                        losses['loss_adv_D_real_recon']=loss_adv_D_real_recon

                        miscs['D_fake_accuracy_recon']=-logits_fake_recon.sign().mean().item()

                        r1_grads_recon = torch.autograd.grad(outputs=[logits_real_recon.sum()], inputs=[eg3doutput_['image'], 
                                                                eg3doutput_['image_raw']], create_graph=True, only_inputs=True)
                        r1_grads_image_recon = r1_grads_recon[0]
                        r1_grads_image_raw_recon = r1_grads_recon[1]
                        r1_penalty_recon = r1_grads_image_recon.square().sum([1,2,3]) + r1_grads_image_raw_recon.square().sum([1,2,3])
                        loss_r1_recon = r1_penalty_recon.mean()
                        losses['loss_r1_recon']=loss_r1_recon

                    
                        loss_adv_D_fake_recon=F.softplus(logits_fake_recon).mean()
                        losses['loss_adv_D_fake_recon']=loss_adv_D_fake_recon

                    loss_adv_G_fake_recon=F.softplus(-logits_fake_recon).mean()
                    losses['loss_adv_G_fake_recon']=loss_adv_G_fake_recon*min(G_step_warmup/G_step_warmup_total,1.)*0.001
                    
                    

            if args.multiview:
                denoised_output_mv=G.render_from_planes(camera_params_mv,inv_normalize_fn((unet_output.sample).repeat(mv_number,1,1,1)))
                for k,v in eg3doutput_mv.items():
                    losses['loss_{}_mv'.format(k)]=F.mse_loss(v,denoised_output_mv[k])
                    if k=='image' or k=='image_raw':
                        losses['loss_{}_lpips_mv'.format(k)]=lpips_fn(v,denoised_output_mv[k]).mean()
                if args.adv:
                    for k,v in losses.items():
                        if 'mv' in k:
                            losses[k]=v*max(1-G_step_warmup/G_step_warmup_total,0.01)
                    # use fp32 for discriminator
                    logits_fake_mv=D(denoised_output_mv,camera_params_mv)
                    if adv_update_D_this_step:
                        eg3doutput_mv_={k:v.detach().requires_grad_(True) for k,v in eg3doutput_mv.items()}
                        logits_real_mv=D(eg3doutput_mv_,camera_params_mv)
                        miscs['D_real_accuracy_mv']=logits_real_mv.sign().mean().item()
                        loss_adv_D_real_mv=F.softplus(-logits_real_mv).mean()
                        losses['loss_adv_D_real_mv']=loss_adv_D_real_mv
                        r1_grads_mv = torch.autograd.grad(outputs=[logits_real_mv.sum()], 
                                        inputs=[eg3doutput_mv_['image'], eg3doutput_mv_['image_raw']], create_graph=True, only_inputs=True)
                        r1_grads_image_mv = r1_grads_mv[0]
                        r1_grads_image_raw_mv = r1_grads_mv[1]
                        r1_penalty_mv = r1_grads_image_mv.square().sum([1,2,3]) + r1_grads_image_raw_mv.square().sum([1,2,3])
                        loss_r1_mv = r1_penalty_mv.mean()
                        losses['loss_r1_mv']=loss_r1_mv

                        miscs['D_fake_accuracy_mv']=-logits_fake_mv.sign().mean().item()
                    
                        loss_adv_D_fake_mv=F.softplus(logits_fake_mv).mean()
                        losses['loss_adv_D_fake_mv']=loss_adv_D_fake_mv

                    loss_adv_G_fake_mv=F.softplus(-logits_fake_mv).mean()
                    losses['loss_adv_G_fake_mv']=loss_adv_G_fake_mv*min(G_step_warmup/G_step_warmup_total,1.)*0.001

            loss=0.
            for k,v in losses.items():
                if not 'adv_D' in k:
                    loss=loss+v

            if args.adv :
                D_loss=0.
                optimizer_D.zero_grad()
                if adv_update_D_this_step:
                    for k,v in losses.items():
                        if 'adv_D' in k:
                            D_loss=D_loss+v
                    D_loss.backward(retain_graph=True)
                    # 测试过，通信良好的时候这样的同步不怎么会花时间
                    params = [param for param in D.parameters() if param.numel() > 0 and param.grad is not None]
                    if len(params) > 0:
                        flat = torch.cat([param.grad.flatten() for param in params])
                        if accelerator.num_processes > 1:
                            torch.distributed.all_reduce(flat)
                            flat /= accelerator.num_processes
                        misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                        grads = flat.split([param.numel() for param in params])
                        for param, grad in zip(params, grads):
                            param.grad = grad.reshape(param.shape)
                    optimizer_D.step()
                    lr_scheduler_D.step()
               

            optimizer.zero_grad()
            avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
            train_loss += avg_loss.item() / args.gradient_accumulation_steps
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(sum([group['params'] for group in optimizer.param_groups if group['lr']>0],[]),     args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            
            for k,v in miscs.items():
                if not k in miscs_list:
                    miscs_list[k]=[]
                miscs_list[k].append(v)

            if accelerator.sync_gradients:
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                for k,v in losses.items():
                    accelerator.log({k: v.detach().item()}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0 or not sanity_checked:
                # if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)
                        
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)        

                        if not args.backbone:
                            unwrapped_unet = accelerator.unwrap_model(unet)
                            save_expanded_unet(unwrapped_unet, save_path)
                            if args.lora:
                                unet_lora_state_dict = get_peft_model_state_dict(unwrapped_unet,adapter_name='default_1')

                                lora_save_path=os.path.join(save_path,'lora')

                                diffusers.StableDiffusionPipeline.save_lora_weights(
                                    save_directory=lora_save_path,
                                    unet_lora_layers=unet_lora_state_dict,
                                    safe_serialization=True,
                                )
                            elif args.controlnet:
                                pass

                        logger.info(f"Saved state to {save_path}")

                        # 保存当前的py到checkpoint
                        shutil.copy(os.path.join(args.output_dir,"train_control3diff_tmp_{}.py".format(str(file_idx))), save_path)
                        # 保存当前的args到checkpoint
                        with open(os.path.join(save_path,'args.json'),'w') as f:
                            json.dump(vars(args),f)

                        if args.adv:
                            # save D
                            D.save_pretrained(os.path.join(save_path,"discriminator"))
                            # save optimizer_D
                            torch.save(optimizer_D.state_dict(), os.path.join(save_path,'discriminator',f"optimizer_D_{global_step}.bin"))
                            # save lr_scheduler_D
                            torch.save(lr_scheduler_D.state_dict(), os.path.join(save_path,'discriminator',f"lr_scheduler_D_{global_step}.bin"))
                        if not sanity_checked:
                            shutil.rmtree(save_path)


            if global_step%log_step_interval==0 or not sanity_checked:   
            # if global_step % args.checkpointing_steps == 0:
                if accelerator.is_main_process:
                    unet.eval()
                    with torch.no_grad():
                        pred_x0=pred_x0_fn(noise_scheduler,unet_output.sample, timesteps, noisy_latents)
                        latent2img_fn=lambda x,camera_params_=camera_params: (G.render_from_planes(camera_params_,inv_normalize_fn(x))['image'].permute(0, 2, 3, 1)*127.5+128).clamp(0,255).to(torch.uint8).cpu()
                        latent2img_fn=partial(latent2img_fn,camera_params_=camera_params[:4])
                        eg3doutput_=eg3doutput['image'][:4]
                        eg3doutput_=(eg3doutput_.permute(0, 2, 3, 1)*127.5+128).clamp(0,255).to(torch.uint8).cpu()
                        eg3doutput_noisy=latent2img_fn(noisy_latents[:4])
                        eg3doutput_denoised=latent2img_fn(pred_x0[:4])
                        img_concat=torch.cat([eg3doutput_, eg3doutput_noisy,eg3doutput_denoised],dim=2) # (batch_size,256,768,3)
                        print('img_concat shape:',img_concat.shape)

                        noise_scheduler.set_timesteps(args.num_inference_steps, device=device)
                        timesteps = noise_scheduler.timesteps
                        original_latents=latents.detach().clone()
                        latents=noise.detach().clone()
                        
                        unet_val=accelerator.unwrap_model(unet)
                        encoder_val=encoder   
                    
                        reverse_process_list,denoised_ddim=reverse_process(latent2img_fn,timesteps,noise_scheduler,noise[:4],unet_val,encoder_states[:4],camera_params[:4],
                                                            G,latent_guidance,lpips_fn,eg3doutput,inv_normalize_fn,reverse_type='normal',return_pred=True,return_latents=True)
                        eg3doutput_denoised_ema_multistep=reverse_process_list[-1]
                        img_concat=torch.cat([img_concat, eg3doutput_denoised_ema_multistep],dim=2)
                        reverse_process_list=torch.cat(reverse_process_list,dim=2).flatten(0,1)
                        accelerator.log({'reverse_process':wandb.Image(reverse_process_list.numpy())},step=global_step)
                        del reverse_process_list
                        
                        if True:
                            if False:                      
                                unet_val=accelerator.unwrap_model(unet)
                                encoder_val=encoder     

                                encoder_states=encoder_val(clip_normalize_fn(eg3doutput['image'].detach()))
                                # encoder_states=encoder_states/encoder_states.norm(dim=-1,keepdim=True)
                                unet_output=unet_val(noisy_latents,timesteps,encoder_states)     
                                eg3doutput_denoised_ema=latent2img_fn(unet_output.sample)
                                img_concat=torch.cat([img_concat, eg3doutput_denoised_ema],dim=2)  
                                
                                if not encoder_output_shape is None:
                                    encoder_states_none=[torch.zeros(encoder_output_shape[i],device=device,dtype=noisy_latents.dtype) for i in range(len(encoder_output_shape))]
                                    eg3doutput_denoised_none_condition=unet_val(noisy_latents,timesteps,encoder_states_none)     
                                    eg3doutput_denoised_none_condition=latent2img_fn(eg3doutput_denoised_none_condition.sample)
                                    img_concat=torch.cat([img_concat, eg3doutput_denoised_none_condition],dim=2)

                                noise_scheduler.set_timesteps(args.num_inference_steps, device=device)
                                timesteps = noise_scheduler.timesteps
                                original_latents=latents.detach().clone()
                                latents=noise.detach().clone()


                                forward_process_list=forward_process(latent2img_fn,timesteps,noise_scheduler,original_latents,noise)
                                forward_process_list=torch.cat(forward_process_list,dim=2).flatten(0,1)
                                # accelerator.log({'forward_process':wandb.Image(forward_process_list.numpy())},step=global_step)

                                reverse_process_list,denoised_ddim=reverse_process(latent2img_fn,timesteps,noise_scheduler,noise,unet_val,encoder_states,camera_params,
                                                                    G,latent_guidance,lpips_fn,eg3doutput,inv_normalize_fn,reverse_type='normal',return_pred=True,return_latents=True)
                                eg3doutput_denoised_ema_multistep=reverse_process_list[-1]
                                img_concat=torch.cat([img_concat, eg3doutput_denoised_ema_multistep],dim=2)
                                reverse_process_list=torch.cat(reverse_process_list,dim=2).flatten(0,1)
                                accelerator.log({'reverse_process':wandb.Image(reverse_process_list.numpy())},step=global_step)
                                del reverse_process_list

                                reverse_guidance_process_list,denoised_ddim_cfg=reverse_process(latent2img_fn,timesteps,noise_scheduler,noise,unet_val,encoder_states,camera_params,
                                                                            G,latent_guidance,lpips_fn,eg3doutput,inv_normalize_fn,reverse_type='cfg',return_pred=True,return_latents=True)
                                reverse_guidance_process_list=torch.cat(reverse_guidance_process_list,dim=2).flatten(0,1)
                                accelerator.log({'reverse_guidance_process_cfg':wandb.Image(reverse_guidance_process_list.numpy())},step=global_step)

                                # reverse_guidance_process_list,denoised_ddim_guided_ori=reverse_process(latent2img_fn,timesteps,noise_scheduler,noise,unet_val,encoder_states,camera_params,
                                #                                               G,latent_guidance,lpips_fn,eg3doutput,inv_normalize_fn,reverse_type='vgg_guidance_0',return_pred=True,return_latents=True)
                                # reverse_guidance_process_list=torch.cat(reverse_guidance_process_list,dim=2).flatten(0,1)
                                # accelerator.log({'reverse_guidance_process_original':wandb.Image(reverse_guidance_process_list.numpy())},step=global_step)
                                if args.verify_text:
                                    encoder_states_text=prompt_embeds.unsqueeze(0).unsqueeze(0).repeat(num_cross_attention_block,noise.shape[0],1,1)
                                    reverse_guidance_process_list_text,denoised_ddim_cfg_text=\
                                        reverse_process(latent2img_fn,timesteps,noise_scheduler,noise,unet_val,encoder_states_text,camera_params,
                                                        G,latent_guidance,lpips_fn,eg3doutput,inv_normalize_fn,reverse_type='normal',return_pred=True,
                                                        return_latents=True)
                                    reverse_guidance_process_list_text=torch.cat(reverse_guidance_process_list_text,dim=2).flatten(0,1)
                                    # accelerator.log({'reverse_guidance_process_normal_text':wandb.Image(reverse_guidance_process_list_text.numpy())},step=global_step)

                                    if args.additional_sample:
                                        reverse_process_list_text=[]
                                        denoised_ddim_text=[]
                                        for i in range(0,args.additional_sample,noise.shape[0]):
                                            noise_sample=torch.randn_like(noise)
                                            reverse_process_list_text_split,denoised_ddim_text_split=\
                                                    reverse_process(latent2img_fn,timesteps,noise_scheduler,noise_sample,unet_val,encoder_states_text,camera_params,
                                                                    G,latent_guidance,lpips_fn,eg3doutput,inv_normalize_fn,reverse_type='normal',return_pred=True,
                                                                    return_latents=True,i_end=40)
                                            denoised_ddim_text.append(denoised_ddim_text_split)
                                        # denoised_ddim_text shape: (additional_sample//batch_size, batch_size, 96, 256, 256)

                                        multiview_image_list=[]
                                        for camera_param_val in camera_params_val_list:
                                            multiview_image_list_per_frame=[]
                                            for denoised_ddim_text_split in denoised_ddim_text:
                                                denoised_img=latent2img_fn(denoised_ddim_text_split,camera_param_val).numpy()  # (batch_size,256,256,3)
                                                multiview_image_list_per_frame.append(denoised_img)
                                            multiview_image_list.append(np.concatenate(multiview_image_list_per_frame,axis=0)) # (additional_sample,256,256,3)

                                        # multiview_image_list: (frames,additional_sample,256,256,3)
                                        # change to square: (additional_sample,256,256,3) --> ((additional_sample**0.5)*256,256*(additional_sample**0.5),3)
                                        multiview_image_list=np.stack(multiview_image_list,axis=0) # (frames,additional_sample,256,256,3)

                                        sqrt_additional_sample=int(np.sqrt(args.additional_sample))
                                        assert sqrt_additional_sample*sqrt_additional_sample==args.additional_sample,'additional_sample must be square number'
                                        multiview_image_list=multiview_image_list.reshape(multiview_image_list.shape[0],sqrt_additional_sample,sqrt_additional_sample,
                                                                                        multiview_image_list.shape[2],multiview_image_list.shape[3],multiview_image_list.shape[4])
                                        multiview_image_list=multiview_image_list.swapaxes(2,3) # (frames,sqrt_additional_sample,256,sqrt_additional_sample,256,3)
                                        multiview_image_list=multiview_image_list.reshape(multiview_image_list.shape[0],sqrt_additional_sample*multiview_image_list.shape[2],
                                                                                            sqrt_additional_sample*multiview_image_list.shape[4],3) # (frames,sqrt_additional_sample*256,sqrt_additional_sample*256,3)
                                        
                                        original_prompt_with_line=args.verify_text.replace(' ','_')
                                        accelerator.log({f'reverse_process_normal_text_additional_sample_prompt_{original_prompt_with_line}':wandb.Image(multiview_image_list[0])},step=global_step)

                                        frame_height, frame_width = multiview_image_list.shape[1], multiview_image_list.shape[2]
                                        video_fps = 25.0

                                        # Define the codec and create VideoWriter object
                                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                                        
                                        out = cv2.VideoWriter(os.path.join(args.output_dir,
                                                                        'output_video_{}_additional_sample_prompt_{}.avi'.format(str(global_step),original_prompt_with_line)), 
                                                                        fourcc, video_fps, (frame_width, frame_height))

                                        # Iterate over each frame and write it to the video
                                        for i in range(multiview_image_list.shape[0]):
                                            frame = multiview_image_list[i]

                                            # OpenCV expects uint8 data in BGR format
                                            frame_bgr = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
                                            out.write(frame_bgr)

                                        # Release the VideoWriter object
                                        out.release()

                                    # encoder_states_text=prompt_embeds.unsqueeze(0).unsqueeze(0).repeat(num_cross_attention_block,noise.shape[0],1,1)
                                    # reverse_guidance_process_list_text,denoised_ddim_cfg_text=\
                                    #     reverse_process(latent2img_fn,timesteps,noise_scheduler,noise,unet_val,10*encoder_states_text,camera_params,
                                    #                     G,latent_guidance,lpips_fn,eg3doutput,inv_normalize_fn,reverse_type='normal',return_pred=True,
                                    #                     return_latents=True)
                                    # reverse_guidance_process_list_text=torch.cat(reverse_guidance_process_list_text,dim=2).flatten(0,1)
                                    # accelerator.log({'reverse_guidance_process_normal_text_scale_10':wandb.Image(reverse_guidance_process_list_text.numpy())},step=global_step)

                                if args.test_video:
                                    # reverse_guidance_process_list,reverse_guidance_process_g_cfg_langevin=reverse_process(latent2img_fn,timesteps,noise_scheduler,noise,unet_val,encoder_states,camera_params,
                                    #                                             G,latent_guidance,lpips_fn,eg3doutput,inv_normalize_fn,reverse_type='langevin_correct',return_pred=True,return_latents=True)
                                    # reverse_guidance_process_list=torch.cat(reverse_guidance_process_list,dim=2).flatten(0,1)
                                    # accelerator.log({'reverse_guidance_process_g_cfg_langevin':wandb.Image(reverse_guidance_process_list.numpy())},step=global_step)

                                    # reverse_guidance_process_list,reverse_guidance_process_cfg_langevin=reverse_process(latent2img_fn,timesteps,noise_scheduler,noise,unet_val,encoder_states,camera_params,
                                    #                                             G,latent_guidance,lpips_fn,eg3doutput,inv_normalize_fn,reverse_type='langevin_correct1',return_pred=True,return_latents=True)
                                    # reverse_guidance_process_list=torch.cat(reverse_guidance_process_list,dim=2).flatten(0,1)
                                    # accelerator.log({'reverse_guidance_process_cfg_langevin':wandb.Image(reverse_guidance_process_list.numpy())},step=global_step)

                                    # reverse_guidance_process_list,reverse_guidance_process_g_langevin=reverse_process(latent2img_fn,timesteps,noise_scheduler,noise,unet_val,encoder_states,camera_params,
                                    #                                             G,latent_guidance,lpips_fn,eg3doutput,inv_normalize_fn,reverse_type='langevin_correct2',return_pred=True,return_latents=True)
                                    # reverse_guidance_process_list=torch.cat(reverse_guidance_process_list,dim=2).flatten(0,1)
                                    # accelerator.log({'reverse_guidance_process_g_langevin.':wandb.Image(reverse_guidance_process_list.numpy())},step=global_step)
                                    if args.verify_text:
                                        pass

                                        # reverse_guidance_process_list_text,denoised_ddim_cfg_langevin_text=\
                                        #     reverse_process(latent2img_fn,timesteps,noise_scheduler,noise,unet_val,encoder_states_text,camera_params,
                                        #                     G,latent_guidance,lpips_fn,eg3doutput,inv_normalize_fn,reverse_type='langevin_correct1',return_pred=True,
                                        #                     return_latents=True)
                                        # reverse_guidance_process_list_text=torch.cat(reverse_guidance_process_list_text,dim=2).flatten(0,1)
                                        # accelerator.log({'reverse_guidance_process_cfg_langevin_text':wandb.Image(reverse_guidance_process_list_text.numpy())},step=global_step)

                                # reverse_guidance_process_list=reverse_process(latent2img_fn,timesteps,noise_scheduler,noise,unet_val,encoder_states,camera_params,G,latent_guidance,lpips_fn,eg3doutput,inv_normalize_fn,reverse_type='vgg_guidance_1')
                                # reverse_guidance_process_list=torch.cat(reverse_guidance_process_list,dim=2).flatten(0,1)
                                # accelerator.log({'reverse_guidance_process_sigmoid':wandb.Image(reverse_guidance_process_list.numpy())},step=global_step)                            

                                # reverse_guidance_process_list=reverse_process(latent2img_fn,timesteps,noise_scheduler,noise,unet_val,encoder_states,camera_params,G,latent_guidance,lpips_fn,eg3doutput,inv_normalize_fn,reverse_type='vgg_guidance_4')
                                # reverse_guidance_process_list=torch.cat(reverse_guidance_process_list,dim=2).flatten(0,1)
                                # accelerator.log({'reverse_guidance_process_sigmoid_small':wandb.Image(reverse_guidance_process_list.numpy())},step=global_step)                            

                                # reverse_guidance_process_list=reverse_process(latent2img_fn,timesteps,noise_scheduler,noise,unet_val,encoder_states,camera_params,G,latent_guidance,lpips_fn,eg3doutput,inv_normalize_fn,reverse_type='vgg_guidance_5')
                                # reverse_guidance_process_list=torch.cat(reverse_guidance_process_list,dim=2).flatten(0,1)
                                # accelerator.log({'reverse_guidance_process_original_small':wandb.Image(reverse_guidance_process_list.numpy())},step=global_step)  

                                # reverse_guidance_process_list=reverse_process(latent2img_fn,timesteps,noise_scheduler,noise,unet_val,encoder_states,camera_params,G,latent_guidance,lpips_fn,eg3doutput,inv_normalize_fn,reverse_type='vgg_guidance_6')
                                # reverse_guidance_process_list=torch.cat(reverse_guidance_process_list,dim=2).flatten(0,1)
                                # accelerator.log({'reverse_guidance_process_controled_stepsize_small':wandb.Image(reverse_guidance_process_list.numpy())},step=global_step)
                                
                                # reverse_guidance_process_list=reverse_process(latent2img_fn,timesteps,noise_scheduler,noise,unet_val,encoder_states,camera_params,G,latent_guidance,lpips_fn,eg3doutput,inv_normalize_fn,reverse_type='vgg_guidance_7')
                                # reverse_guidance_process_list=torch.cat(reverse_guidance_process_list,dim=2).flatten(0,1)
                                # accelerator.log({'reverse_guidance_process_controled_stepsize_large':wandb.Image(reverse_guidance_process_list.numpy())},step=global_step)  

                                # reverse_guidance_process_list=reverse_process(latent2img_fn,timesteps,noise_scheduler,noise,unet_val,encoder_states,camera_params,G,latent_guidance,lpips_fn,eg3doutput,inv_normalize_fn,reverse_type='vgg_guidance_8')
                                # reverse_guidance_process_list=torch.cat(reverse_guidance_process_list,dim=2).flatten(0,1)
                                # accelerator.log({'reverse_guidance_process_controled_rescale_0.5':wandb.Image(reverse_guidance_process_list.numpy())},step=global_step)  

                                # reverse_guidance_process_list=reverse_process(latent2img_fn,timesteps,noise_scheduler,noise,unet_val,encoder_states,camera_params,G,latent_guidance,lpips_fn,eg3doutput,inv_normalize_fn,reverse_type='vgg_guidance_9')
                                # reverse_guidance_process_list=torch.cat(reverse_guidance_process_list,dim=2).flatten(0,1)
                                # accelerator.log({'reverse_guidance_process_controled_rescale_0.8':wandb.Image(reverse_guidance_process_list.numpy())},step=global_step)  


                                encoder_states_first=[torch.cat([state[0:1]]*encoder_states[0].shape[0]) for state in encoder_states]
                                reverse_guidance_process_list=reverse_process(latent2img_fn,timesteps,noise_scheduler,noise,unet_val,encoder_states_first,camera_params,
                                                                            G,latent_guidance,lpips_fn,eg3doutput,inv_normalize_fn,reverse_type='normal')
                                reverse_guidance_process_list=torch.cat(reverse_guidance_process_list,dim=2).flatten(0,1)
                                accelerator.log({'reverse_guidance_process_change_noise':wandb.Image(reverse_guidance_process_list.numpy())},step=global_step)

                                noise_first=torch.cat([noise[0:1]]*noise.shape[0])
                                reverse_guidance_process_list=reverse_process(latent2img_fn,timesteps,noise_scheduler,noise_first,unet_val,encoder_states,camera_params,
                                                                            G,latent_guidance,lpips_fn,eg3doutput,inv_normalize_fn,reverse_type='normal')
                                reverse_guidance_process_list=torch.cat(reverse_guidance_process_list,dim=2).flatten(0,1)
                                accelerator.log({'reverse_guidance_process_change_encoder':wandb.Image(reverse_guidance_process_list.numpy())},step=global_step)

                                # reverse_guidance_process_list=reverse_process(latent2img_fn,timesteps,noise_scheduler,noise,unet_val,encoder_states,camera_params,G,latent_guidance,lpips_fn,eg3doutput,inv_normalize_fn,reverse_type='vgg_guidance_2')
                                # reverse_guidance_process_list=torch.cat(reverse_guidance_process_list,dim=2).flatten(0,1)
                                # accelerator.log({'reverse_guidance_process_const_small':wandb.Image(reverse_guidance_process_list.numpy())},step=global_step)   

                                # reverse_guidance_process_list=reverse_process(latent2img_fn,timesteps,noise_scheduler,noise,unet_val,encoder_states,camera_params,G,latent_guidance,lpips_fn,eg3doutput,inv_normalize_fn,reverse_type='vgg_guidance_3')
                                # reverse_guidance_process_list=torch.cat(reverse_guidance_process_list,dim=2).flatten(0,1)
                                # accelerator.log({'reverse_guidance_process_const_large':wandb.Image(reverse_guidance_process_list.numpy())},step=global_step)        
                                # del reverse_guidance_process_list

                                img_concat=img_concat.flatten(0,1) 


                                # # 从Original_latents开始加噪声，在不同的加噪声结果上去噪声
                                # choosen_timesteps=[(original_idx,timestep) for original_idx,timestep in enumerate(timesteps) if original_idx%10==0 or original_idx==len(timesteps)-1]
                                # image_list=[]
                                # for idx, (original_idx,timestep) in enumerate(choosen_timesteps):
                                #     noisy_latents=noise_scheduler.add_noise(original_latents, noise, timestep)
                                #     # image_list.append(((G.render_from_planes(camera_params,inv_normalize_fn(noisy_latents))['image']).permute(0, 2, 3, 1)*127.5+128).clamp(0,255).to(torch.uint8).cpu())
                                #     image_list.append(latent2img_fn(noisy_latents))
                                #     for i, t in enumerate(timesteps):
                                #         if i<original_idx:
                                #             continue
                                #         pred=unet_val(noisy_latents,t,encoder_states,return_dict=False)[0]
                                #         noisy_latents=noise_scheduler.step(pred, t, noisy_latents, return_dict=False)[0]
                                #     image_list.append(latent2img_fn(noisy_latents))
                                # image_list=torch.cat(image_list,dim=2).flatten(0,1)
                                # accelerator.log({'image_list':wandb.Image(image_list.numpy())},step=global_step)
                                # del image_list

                                # multiview_image_list=[]
                                # for camera_param_val in camera_params_val_list:
                                #     multiview_image_list.append(latent2img_fn(unet_output.sample,camera_param_val))

                                # multiview_image_list=torch.cat(multiview_image_list,dim=2).flatten(0,1)
                                # accelerator.log({'multiview_image_list':wandb.Image(multiview_image_list.numpy())},step=global_step)
                                # del multiview_image_list

    
                                # 首先，不同的encoder_states之间的相似度定义为里面每一个元素的相似度的算术平均
                                # 对于每一个元素，计算它和其他元素的相似度，然后取平均
                                cosine_similarity_val=torch.zeros((len(encoder_states),encoder_states[0].shape[0],encoder_states[0].shape[0]),device=device)
                                for i in range(len(encoder_states)):
                                    for j in range(0,len(encoder_states[i])):
                                        for k in range(j,len(encoder_states[i])):
                                            cosine_similarity_val[i,j,k]=F.cosine_similarity(encoder_states[i][j],encoder_states[i][k],dim=-1).mean()
                                        cosine_similarity_val[i,j,j]=1
                                cosine_similarity_val=cosine_similarity_val.mean(dim=0)
                                # save to txt as a matrix
                                cosine_similarity_val=cosine_similarity_val.cpu().numpy()
                                np.savetxt(os.path.join(args.output_dir,f'cosine_similarity_val_{global_step}.txt'),cosine_similarity_val)

                                # 可视化注意力图
                                # encoder_states_img=[encoder_state.reshape(-1,int(np.sqrt(encoder_state.shape[1])),int(np.sqrt(encoder_state.shape[1])),encoder_state.shape[-1]).permute(0,3,1,2) for encoder_state in encoder_states]
                                # # interpolate to 64*64
                                # encoder_states_img=[F.interpolate(encoder_state,(64,64)).mean(1,keepdim=True) for encoder_state in encoder_states_img]
                                # # normalize to 0-1
                                # encoder_states_img=[(encoder_state-encoder_state.min())/(encoder_state.max()-encoder_state.min()) for encoder_state in encoder_states_img]
                                # # encoder_states_img_noise_scheduler_attention=[(encoder_state-encoder_state.min())/(encoder_state.max()-encoder_state.min()) for encoder_state in encoder_states_img_noise_scheduler_attention]
                                # # concat
                                # encoder_states_img=torch.cat(encoder_states_img,dim=-1).permute(0,2,3,1).flatten(0,1).detach().cpu()
                                # accelerator.log({'encoder_states_img':wandb.Image(encoder_states_img.numpy())},step=global_step)

                                if args.test_video:
                                    multiview_image_list=[]
                                    for camera_param_val in camera_params_val_list:
                                        original_img=latent2img_fn(original_latents,camera_param_val)
                                        denoised_img=latent2img_fn(denoised_ddim,camera_param_val)
                                        denoised_cfg_img=latent2img_fn(denoised_ddim_cfg,camera_param_val)
                                        # denoised_guided_ori_img=latent2img_fn(denoised_ddim_guided_ori,camera_param_val)
                                        # denoised_guided_g_cfg_langevin_img=latent2img_fn(reverse_guidance_process_g_cfg_langevin,camera_param_val)
                                        # denoised_guided_cfg_langevin_img=latent2img_fn(reverse_guidance_process_cfg_langevin,camera_param_val)
                                        # denoised_guided_g_langevin_img=latent2img_fn(reverse_guidance_process_g_langevin,camera_param_val)
                                        denoised_cfg_img_text=latent2img_fn(denoised_ddim_cfg_text,camera_param_val)
                                        # denoised_cfg_langevin_img_text=latent2img_fn(denoised_ddim_cfg_langevin_text,camera_param_val)
                                        concat_img=torch.cat([original_img,denoised_img,denoised_cfg_img,
                                                            #   denoised_guided_g_cfg_langevin_img,denoised_guided_cfg_langevin_img,denoised_guided_g_langevin_img,
                                                                denoised_cfg_img_text,
                                                                # denoised_cfg_langevin_img_text,
                                                            ],dim=2).flatten(0,1).numpy() # in range 0-255, shape (batch_size*256,256*2,3)
                                        multiview_image_list.append(concat_img)
                                    # make video
                                    multiview_image_list=np.stack(multiview_image_list,axis=0) # (frames,batch_size*256,256*2,3)
                                    frame_height, frame_width = multiview_image_list.shape[1], multiview_image_list.shape[2]
                                    video_fps = 25.0

                                    # Define the codec and create VideoWriter object
                                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                                    
                                    out = cv2.VideoWriter(os.path.join(args.output_dir,'output_video_{}.avi'.format(str(global_step))), fourcc, video_fps, (frame_width, frame_height))

                                    # Iterate over each frame and write it to the video
                                    for i in range(multiview_image_list.shape[0]):
                                        frame = multiview_image_list[i]

                                        # OpenCV expects uint8 data in BGR format
                                        frame_bgr = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
                                        out.write(frame_bgr)

                                    # Release the VideoWriter object
                                    out.release()

                        # save to wandb
                        img_concat=img_concat.flatten(0,1) 
                        accelerator.log({'origin_noised_denoised_denoisedema_(denoised_none_condition)_denoisedemamultistep':wandb.Image(img_concat.cpu().numpy())},step=global_step)
                        del img_concat, eg3doutput_,eg3doutput_noisy,eg3doutput_denoised
                    unet.train()
            del losses,eg3doutput,planes,latents,noisy_latents,unet_output,loss
            torch.cuda.empty_cache()


            sanity_checked=True
            accelerator.wait_for_everyone()
        for k,v in miscs_list.items():
            v_mean=sum(v)/len(v)
            accelerator.log({k: v_mean}, step=global_step)



        accelerator.wait_for_everyone()



#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------


# accelerate launch --mixed_precision=fp16 train_t2i_sd_finetune.py --train_batch_size=4 --log_step_interval=2500 --checkpointing_steps=5000 --resume_from_checkpoint=latest --output_dir=outputs/t2i_finetune_freeze_attentions --wandb_offline  --prediction_type=epsilon --backbone --freeze_attentions
    
# accelerate launch --mixed_precision=fp16 train_t2i_sd_finetune.py --train_batch_size=2 --log_step_interval=2500 --checkpointing_steps=5000 --resume_from_checkpoint=latest --output=outputs/t2i_finetune_test --wandb_offline  --prediction_type=epsilon --dataset_path dataset_v1
    
# accelerate launch --mixed_precision=fp16 train_t2i_sd_finetune.py --train_batch_size=2 --log_step_interval=2500 --checkpointing_steps=5000 --resume_from_checkpoint=latest --output=outputs/t2i_finetune_test --wandb_offline  --prediction_type=epsilon --dataset_path dataset_v1 --freeze_attentions