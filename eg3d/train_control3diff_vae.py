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
from diffusers import UNet2DConditionModel_downsample
import shutil
import argparse
from accelerate.utils import ProjectConfiguration
from packaging import version
import torch.nn.functional as F
import math
import json
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers import AutoencoderKL
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

    @register_to_config
    def __init__(
        self,
        in_channels=3,
        down_block_types=("DownEncoderBlock2D",),
        block_out_channels=(64,),
        layers_per_block=2,
        norm_num_groups=32,
        act_fn="silu",
        down_sampling_ratio :int =1,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block
        self.down_sample_conv=nn.Conv2d(in_channels, in_channels, kernel_size=down_sampling_ratio, stride=down_sampling_ratio)

        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.mid_block = None
        self.down_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=self.layers_per_block,
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
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=None,
        )

        self.gradient_checkpointing = False

    def forward(self, x):
        sample = x
        sample=self.down_sample_conv(sample)
        sample = self.conv_in(sample)
        

        down_block_res_samples=tuple()

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            # down
            if is_torch_version(">=", "1.11.0"):
                for down_block in self.down_blocks:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(down_block), sample, use_reentrant=False
                    )
                    down_block_res_samples=down_block_res_samples+(sample,)
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block), sample, use_reentrant=False
                )
                down_block_res_samples=down_block_res_samples+(sample,)
            else:
                for down_block in self.down_blocks:
                    sample = torch.utils.checkpoint.checkpoint(create_custom_forward(down_block), sample)
                    down_block_res_samples=down_block_res_samples+(sample,)
                # middle
                sample = torch.utils.checkpoint.checkpoint(create_custom_forward(self.mid_block), sample)
                down_block_res_samples=down_block_res_samples+(sample,)

        else:
            # down
            for down_block in self.down_blocks:
                sample = down_block(sample)
                down_block_res_samples=down_block_res_samples+(sample,)
        
            # middle
            sample = self.mid_block(sample)
            down_block_res_samples=down_block_res_samples+(sample,)

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
    parser.add_argument("--num_train_epochs", type=int, default=100)
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
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
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
        default=None,
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
        '--network_pkl', help='Network pickle filename', required=False,default='/home1/jo_891/data1/eg3d/ffhq512-128.pkl'
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


    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args



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

              
    unet_config={'in_channels':128,
        'out_channels':128,
        'block_out_channels':(128, 256, 512, 512),
        'cross_attention_dim':(64, 128, 256, 512),
        'attention_head_dim':8,
        'down_sampling_ratio':1,
        'use_linear_projection':True
                                          }
    # unet 的输出维度是 (batch_size,64,16,16) (batch_size,128,8,8) (batch_size,256,4,4) (batch_size,512,2,2)

    unet=UNet2DConditionModel_downsample(**unet_config)
    unet.requires_grad_(True)

    if args.use_ema:
        ema_unet = EMAModel(unet.parameters(), model_cls=UNet2DConditionModel_downsample, model_config=unet.config)

    encoder_config={'in_channels':3,
                    'down_block_types':("DownEncoderBlock2D","DownEncoderBlock2D",
                                        "DownEncoderBlock2D","DownEncoderBlock2D"),
                    'block_out_channels':(64, 128, 256, 512),
                    'down_sampling_ratio':1,
                                }
    # encoder 的输出维度是 (batch_size,64,256,256) (batch_size,128,128,128) (batch_size,256,64,64) (batch_size,512,32,32)

    encoder=Encoder(**encoder_config)
    encoder.requires_grad_(True)


    if args.use_ema:
        ema_encoder = EMAModel(encoder.parameters(), model_cls=Encoder,model_config=encoder.config)
        

    vae_config={
        'in_channels':96,
        'out_channels':96,
        'down_block_types':("DownEncoderBlock2D","DownEncoderBlock2D","DownEncoderBlock2D"),
        'up_block_types':("UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D"),
        'block_out_channels':(192, 384, 384),
        'latent_channels':128,
        'sample_size':256,
        'scaling_factor':1.,
    }
    # vae 的输出维度是 (batch_size,latent_channels,32,32)

    vae=AutoencoderKL(**vae_config)
    vae.requires_grad_(True)

    unet_input_size=vae_config['sample_size']//8



    if args.use_ema:
        ema_vae = EMAModel(vae.parameters(), model_cls=AutoencoderKL,model_config=vae.config)

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_encoder.save_pretrained(os.path.join(output_dir, "encoder_ema"))
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))
                    ema_vae.save_pretrained(os.path.join(output_dir, "vae_ema"))

                for i, model in enumerate(models):
                    if isinstance(model, Encoder):
                        model.save_pretrained(os.path.join(output_dir, "encoder"))
                    elif isinstance(model, UNet2DConditionModel_downsample):
                        model.save_pretrained(os.path.join(output_dir, "unet"))
                    elif isinstance(model, AutoencoderKL):
                        model.save_pretrained(os.path.join(output_dir, "vae"))
                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel_downsample)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "encoder_ema"), Encoder)
                ema_encoder.load_state_dict(load_model.state_dict())
                ema_encoder.to(accelerator.device)
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "vae_ema"), AutoencoderKL)
                ema_vae.load_state_dict(load_model.state_dict())
                ema_vae.to(accelerator.device)

                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                if isinstance(model, Encoder):
                    load_model = Encoder.from_pretrained(input_dir, subfolder="encoder")
                elif isinstance(model, UNet2DConditionModel_downsample):
                    load_model = UNet2DConditionModel_downsample.from_pretrained(input_dir, subfolder="unet")
                elif isinstance(model, AutoencoderKL):
                    load_model = AutoencoderKL.from_pretrained(input_dir, subfolder="vae")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        vae.enable_gradient_checkpointing()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    print('Loading networks from "%s"...' % network_pkl)

    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    # Specify reload_modules=True if you want code modifications to take effect; otherwise uses pickled code
    print("Reloading Modules!")
    G_new = TriPlaneGenerator_Modified(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
    misc.copy_params_and_buffers(G, G_new, require_all=True)
    G_new.neural_rendering_resolution = G.neural_rendering_resolution
    G_new.rendering_kwargs = G.rendering_kwargs
    G = G_new


    cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0.2], device=device), radius=2.7, device=device)
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)
    

    noise_scheduler_config_dummpy={
        'num_train_timesteps': 1000,
        'beta_schedule': 'squaredcos_cap_v2',
        'prediction_type':'sample',
        'trained_betas':None,
    }
    if False:
        # 验证noise scaling策略是否正确
        noise_scheduler_dummy=diffusers.schedulers.DDPMScheduler(**noise_scheduler_config_dummpy)
        snr=1/((noise_scheduler_dummy.alphas_cumprod)**(-1)-1)
        logsnr=torch.log(snr)
        logsnr_shifted=logsnr+2*np.log(64/256)
        snr_shifted=torch.exp(logsnr_shifted)
        alphas_cumprod_shifted=1/(1+(1/snr_shifted))
        betas=torch.stack([1-alphas_cumprod_shifted[i+1]/alphas_cumprod_shifted[i] for i in range(len(alphas_cumprod_shifted)-1)]+[0.999*torch.ones_like(alphas_cumprod_shifted[-1])])
        betas=betas.cpu().numpy().tolist()
        noise_scheduler_config_dummpy['trained_betas']=betas
        noise_scheduler=diffusers.schedulers.DDPMScheduler(**noise_scheduler_config_dummpy)
        t=torch.linspace(0,1-0.001,1000)
        snr_calculated=torch.exp(-2*torch.log(torch.tan(t*np.pi/2))+2*np.log(64/256))
        alphas_cumprod_calculated=1/(1+snr_calculated)
        import matplotlib.pyplot as plt
        t=torch.linspace(0,1,1000)
        plt.plot(t.cpu().numpy(),(noise_scheduler_dummy.alphas_cumprod**0.5).cpu().numpy(),label='alphas_cumprod_sqrt_original')
        plt.plot(t.cpu().numpy(),(alphas_cumprod_shifted**0.5).cpu().numpy(),label='alphas_cumprod_sqrt_shifted')
        plt.plot(t.cpu().numpy(),(alphas_cumprod_calculated**0.5).cpu().numpy(),label='alphas_cumprod_sqrt_calculated')
        plt.plot(t.cpu().numpy(),(noise_scheduler.alphas_cumprod**0.5).cpu().numpy(),label='alphas_cumprod_sqrt_shifted_scheduler')
        plt.plot(t.cpu().numpy(),((1-noise_scheduler_dummy.alphas_cumprod)**0.5).cpu().numpy(),label='betas_sqrt')
        plt.plot(t.cpu().numpy(),((1-alphas_cumprod_shifted)**0.5).cpu().numpy(),label='betas_sqrt_shifted')
        plt.plot(t.cpu().numpy(),((1-alphas_cumprod_calculated)**0.5).cpu().numpy(),label='betas_sqrt_calculated')
        plt.plot(t.cpu().numpy(),((1-noise_scheduler.alphas_cumprod)**0.5).cpu().numpy(),label='betas_sqrt_shifted_scheduler')
        plt.legend()
        plt.savefig('schedule.png')
        # origin=torch.stack([1-noise_scheduler_dummy.alphas_cumprod[0]]+[1-noise_scheduler_dummy.alphas_cumprod[i+1]/noise_scheduler_dummy.alphas_cumprod[i] for i in range(len(alphas_cumprod_shifted)-1)])
        # torch.stack([1-noise_scheduler_dummy.alphas_cumprod[i+1]/noise_scheduler_dummy.alphas_cumprod[i] 
        #     for i in range(len(alphas_cumprod_shifted)-1)]+[torch.ones_like(alphas_cumprod_shifted[-1])])
        
        # snr_dummy=alphas_cumprod

    # noise_scheduler_dummy=diffusers.schedulers.DDIMScheduler(**noise_scheduler_config_dummpy)
    # snr=1/((noise_scheduler_dummy.alphas_cumprod)**(-1)-1)
    # logsnr=torch.log(snr)
    # logsnr_shifted=logsnr+2*np.log(64/256)
    # snr_shifted=torch.exp(logsnr_shifted)--log_step_interval
    # alphas_cumprod_shifted=1/(1+(1/snr_shifted))
    # betas=torch.stack([1-alphas_cumprod_shifted[i+1]/alphas_cumprod_shifted[i] for i in range(len(alphas_cumprod_shifted)-1)]+[noise_scheduler_dummy.betas[-1]])
    # betas=betas.cpu().numpy().tolist()
    # noise_scheduler_config_dummpy['trained_betas']=betas


    noise_scheduler=diffusers.schedulers.DDIMScheduler(**noise_scheduler_config_dummpy)


                   

    # params should be unet's parameters + encoder's parameters + vae's parameters
    # vae's lr is 10 times of unet's and encoder's
    unet_params={'params':unet.parameters(),'lr':args.learning_rate}
    encoder_params={'params':encoder.parameters(),'lr':args.learning_rate}
    vae_params={'params':vae.parameters(),'lr':10*args.learning_rate}
    params=[unet_params,encoder_params,vae_params]


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
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)
    unet, optimizer, lr_scheduler,encoder,vae = accelerator.prepare(
        unet, optimizer, lr_scheduler,encoder,vae
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)
        ema_encoder.to(accelerator.device)
        ema_vae.to(accelerator.device)


    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    epoch_size=1000
    first_epoch=0

    log_step_interval=args.log_step_interval

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(epoch_size / args.gradient_accumulation_steps)
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
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0
    global_step = initial_global_step


    if not os.path.exists(os.path.join(args.output_dir,'stats_dict.json')):
        if accelerator.is_main_process:
            total_batch=100000
            minibatch=64
            sample_count=0
            mean = torch.zeros([96, 256, 256], device=device,dtype=torch.float64)
            M2 = torch.zeros([96, 256, 256], device=device,dtype=torch.float64)
            with torch.no_grad():
                for idx in tqdm(range(total_batch//minibatch),desc='getting scale:'):
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
                    # planes=torch.tanh(planes).to(torch.float64)

                    if not torch.isnan(planes).any():
                        sample_count += minibatch
                        delta = planes - mean
                        mean += delta.sum(dim=0) / sample_count
                        M2 += (delta**2).sum(dim=0)
            # 计算最终的方差
            variance = M2 / (sample_count - 1)
            # 计算标准差
            std = torch.sqrt(variance).mean(dim=[1, 2]) # (96,)
            std = std.cpu().numpy().tolist()
            mean_list = mean.mean(dim=[1, 2]).cpu().numpy().tolist()  # (96,)
            stats_dict = {'mean': mean_list, 'std': std}
            with open(os.path.join(args.output_dir,'stats_dict.json'),'w') as f:
                json.dump(stats_dict,f)
                
    accelerator.wait_for_everyone()
    # read stats_dict
    with open(os.path.join(args.output_dir,'stats_dict.json'),'r') as f:
        stats_dict=json.load(f)
    mean=torch.tensor(stats_dict['mean'],device=device,dtype=weight_dtype).reshape(1,96,1,1)
    std=torch.tensor(stats_dict['std'],device=device,dtype=weight_dtype).reshape(1,96,1,1)
    normalize_fn=lambda x: (x-mean)/std
    inv_normalize_fn=lambda x: x*std+mean
    
    if mean.isnan().any():
        raise ValueError('mean has nan value')
    if std.isnan().any():
        raise ValueError('std has nan value')

    if False:
        if not os.path.exists(os.path.join(args.output_dir,'scale_factor_tanh.json')):
            if accelerator.is_main_process:
                total_batch=100000
                minibatch=64
                sample_count=0
                mean = torch.zeros([96, 256, 256], device=device,dtype=torch.float64)
                M2 = torch.zeros([96, 256, 256], device=device,dtype=torch.float64)
                with torch.no_grad():
                    for idx in tqdm(range(total_batch//minibatch),desc='getting scale:'):
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
                        planes= planes*scale_factor
                        planes= planes.to(torch.float64)
                        planes=torch.tanh(planes).to(torch.float64)

                        if not torch.isnan(planes).any():
                            sample_count += minibatch
                            delta = planes - mean
                            mean += delta.sum(dim=0) / sample_count
                            M2 += (delta**2).sum(dim=0)
                # 计算最终的方差
                variance = M2 / (sample_count - 1)
                # 计算标准差
                scale_factor_tanh = torch.sqrt(variance).mean(dim=[1, 2]) # (96,)
                scale_factor_tanh=1/scale_factor_tanh
                scale_factor_tanh = scale_factor_tanh.cpu().numpy().tolist()
                with open(os.path.join(args.output_dir,'scale_factor_tanh.json'),'w') as f:
                    json.dump(scale_factor_tanh,f)
                
    accelerator.wait_for_everyone()
    if False:
        # read scale_factor
        with open(os.path.join(args.output_dir,'scale_factor_tanh.json'),'r') as f:
            scale_factor_tanh=json.load(f)
        scale_factor_tanh=torch.tensor(scale_factor_tanh,device=device,dtype=weight_dtype).reshape(1,96,1,1)
        scale_factor_tanh=torch.ones_like(scale_factor_tanh)
        
        if scale_factor_tanh.isnan().any():
            raise ValueError('scale_factor_tanh has nan value')



    sanity_checked=False

    for epoch in tqdm(range(first_epoch, args.num_train_epochs),desc='epoch:',disable=not accelerator.is_main_process):
        unet_loss=0.0
        vae_kl_loss=0.0
        vae_recon_loss=0.0
        only_vae=epoch<10
        train_batch_size=args.train_batch_size if not only_vae else 8
        for step in tqdm(range(epoch_size),desc='step:',disable=not accelerator.is_main_process):
            with torch.no_grad():
                z_generator=torch.randn(train_batch_size, G.z_dim, device=device)
                cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
                cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
                cam2world_pose=UniformCameraPoseSampler.sample(np.pi/2, np.pi/2 , horizontal_stddev=np.pi/4,vertical_stddev=np.pi/4,
                    lookat_position=torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device), 
                    radius=G.rendering_kwargs.get('avg_camera_radius', 2.7), device=device,batch_size=train_batch_size)
                conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device,batch_size=train_batch_size)
                camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9).repeat(cam2world_pose.shape[0],1)], 1)
                conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9).repeat(cam2world_pose.shape[0],1)], 1)

                ws = G.mapping(z_generator, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
                planes=G.get_planes(ws) # (batch_size,96,256,256)
                eg3doutput=G.render_from_planes(camera_params,planes)
            
            # get input of vae
            planes_normalized=normalize_fn(planes)
            # encode with vae

            latents=accelerator.unwrap_model(vae).encode(planes_normalized).latent_dist
            latents_sample=latents.sample()
            if not only_vae:
                noise=torch.randn_like(latents_sample)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (train_batch_size,), device=planes.device)
                noisy_latents = noise_scheduler.add_noise(latents_sample, noise, timesteps)
                encoder_states=encoder(eg3doutput['image'].detach())

                encoder_states=[F.interpolate(encoder_state,scale_factor=unet_input_size/encoder_states[0].shape[-2],
                                            mode='bilinear',align_corners=False).flatten(-2,-1).permute(0,2,1) for encoder_state in encoder_states]

                unet_output=unet(noisy_latents,timesteps,encoder_states)

                if args.snr_gamma is None:
                    unet_loss = F.mse_loss(unet_output.sample.float(), latents_sample.float(), reduction="mean")
                else:
                    snr = compute_snr(noise_scheduler, timesteps)
                    if noise_scheduler.config.prediction_type == "v_prediction":
                        # Velocity objective requires that we add one to SNR values before we divide by them.
                        snr = snr + 1
                        mse_loss_weights = (
                            torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                        )
                    elif noise_scheduler.config.prediction_type == "sample":
                        mse_loss_weights = (
                            torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
                        )
                    elif noise_scheduler.config.prediction_type == "noise":
                        mse_loss_weights = (
                            torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                        )
                    else:
                        raise ValueError(f"Invalid prediction_type {noise_scheduler.config.prediction_type}")
                    unet_loss = F.mse_loss(unet_output.sample.float(), latents_sample.float(), reduction="none")
                    unet_loss = unet_loss.mean(dim=list(range(1, len(unet_loss.shape)))) * mse_loss_weights
                    unet_loss = unet_loss.mean()
            else:
                unet_loss=torch.tensor(0.0,device=device,dtype=latents_sample.dtype)

            latents_mode=latents.mode()
            planes_recon=accelerator.unwrap_model(vae).decode(latents_mode).sample
            vae_kl_loss=latents.kl().mean()
            vae_recon_loss=F.mse_loss(planes_recon,planes_normalized,reduction='mean')
            vae_loss=vae_recon_loss+vae_kl_loss

            loss=unet_loss+vae_loss*10.

            avg_unet_loss = accelerator.gather(unet_loss.repeat(train_batch_size)).mean()
            unet_loss += avg_unet_loss.item() / args.gradient_accumulation_steps
            avg_vae_kl_loss = accelerator.gather(vae_kl_loss.repeat(train_batch_size)).mean()
            vae_kl_loss += avg_vae_kl_loss.item() / args.gradient_accumulation_steps
            avg_vae_recon_loss = accelerator.gather(vae_recon_loss.repeat(train_batch_size)).mean()
            vae_recon_loss += avg_vae_recon_loss.item() / args.gradient_accumulation_steps


            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(list(unet.parameters())+list(encoder.parameters())+list(vae.parameters()), 
                                            args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                    ema_encoder.step(encoder.parameters())
                    ema_vae.step(vae.parameters())
                global_step += 1
                accelerator.log({"unet_loss": unet_loss}, step=global_step)
                accelerator.log({"vae_kl_loss": vae_kl_loss}, step=global_step)
                accelerator.log({"vae_recon_loss": vae_recon_loss}, step=global_step)
                unet_loss = 0.0
                vae_kl_loss = 0.0
                vae_recon_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
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
                        logger.info(f"Saved state to {save_path}")


            if global_step%log_step_interval==0:
                if accelerator.is_main_process:
                    with torch.no_grad():
                        image_names=[]
                        original_render=eg3doutput['image']
                        recon_latent_render=G.render_from_planes(camera_params,
                                            inv_normalize_fn(planes_recon))['image']
                        original_render=(original_render.permute(0, 2, 3, 1)*127.5+128).clamp(0,255).to(torch.uint8)
                        recon_latent_render=(recon_latent_render.permute(0, 2, 3, 1)*127.5+128).clamp(0,255).to(torch.uint8)
                        image_names.append('original')
                        image_names.append('recon')
                        if not only_vae:
                            noisy_latents_render=G.render_from_planes(camera_params,inv_normalize_fn(accelerator.unwrap_model(vae).decode(noisy_latents).sample))['image']
                            denoised_latent_render=G.render_from_planes(camera_params,
                                                                    inv_normalize_fn(accelerator.unwrap_model(vae).decode(unet_output.sample).sample))['image']
                            noisy_latents_render=(noisy_latents_render.permute(0, 2, 3, 1)*127.5+128).clamp(0,255).to(torch.uint8)
                            denoised_latent_render=(denoised_latent_render.permute(0, 2, 3, 1)*127.5+128).clamp(0,255).to(torch.uint8)
                            img_concat=torch.cat([original_render,recon_latent_render,noisy_latents_render,denoised_latent_render],dim=2) # (batch_size,256,768,3)

                            image_names.append('noisy_latents')
                            image_names.append('denoised_latent')
                            if args.use_ema:
                                ema_unet.store(unet.parameters())
                                ema_unet.copy_to(unet.parameters())
                                ema_encoder.store(encoder.parameters())
                                ema_encoder.copy_to(encoder.parameters())
                                ema_vae.store(vae.parameters())
                                ema_vae.copy_to(vae.parameters())
                                encoder_states=encoder(eg3doutput['image'].detach())
                                encoder_states=[F.interpolate(encoder_state,scale_factor=unet_input_size/encoder_states[0].shape[-2],
                                                mode='bilinear',align_corners=False).flatten(-2,-1).permute(0,2,1) for encoder_state in encoder_states]
                                unet_output=unet(noisy_latents,timesteps,encoder_states)     
                                eg3doutput_denoised_ema=G.render_from_planes(camera_params,
                                                            inv_normalize_fn(accelerator.unwrap_model(vae).decode(unet_output.sample).sample))['image'] 
                                
                                eg3doutput_denoised_ema=(eg3doutput_denoised_ema.permute(0, 2, 3, 1)*127.5+128).clamp(0,255).to(torch.uint8)
                                img_concat=torch.cat([img_concat, eg3doutput_denoised_ema],dim=2)  
                                image_names.append('denoised_latent_ema')
                                
                                noise_scheduler.set_timesteps(args.num_inference_steps, device=device)
                                timesteps = noise_scheduler.timesteps
                                original_latents=latents_mode.detach().clone()
                                latents=noise.detach().clone()
                                reverse_process_list=[]
                                forward_process_list=[]
                                reverse_process_list.append(((G.render_from_planes(camera_params,
                                        inv_normalize_fn(accelerator.unwrap_model(vae).decode(latents).sample))['image']).permute(0, 2, 3, 1)*127.5+128).clamp(0,255).to(torch.uint8).cpu())
                                
                                for i, t in enumerate(timesteps):
                                    if i%3==0 or i==len(timesteps)-1:
                                        forward_process_list.append(((G.render_from_planes(camera_params,inv_normalize_fn(accelerator.unwrap_model(vae).decode(noise_scheduler.add_noise(original_latents, noise, t)).sample))['image']).permute(0, 2, 3, 1)*127.5+128).clamp(0,255).to(torch.uint8).cpu())
                                    pred=unet(latents,t,encoder_states,return_dict=False)[0] 
                                    latents=noise_scheduler.step(pred, t, latents, return_dict=False)[0]
                                    if i%3==0 or i==len(timesteps)-1:
                                        reverse_process_list.append(((G.render_from_planes(camera_params,
                                                                            inv_normalize_fn(accelerator.unwrap_model(vae).decode(latents).sample))['image']).permute(0, 2, 3, 1)*127.5+128).clamp(0,255).to(torch.uint8).cpu())
                                reverse_process_list=torch.cat(reverse_process_list,dim=2).flatten(0,1)
                                forward_process_list=torch.cat(forward_process_list,dim=2).flatten(0,1)
                                accelerator.log({'reverse_process':wandb.Image(reverse_process_list.numpy())},step=global_step)
                                accelerator.log({'forward_process':wandb.Image(forward_process_list.numpy())},step=global_step)

                                eg3doutput_denoised_ema_multistep=((G.render_from_planes(camera_params,
                                                                        inv_normalize_fn(accelerator.unwrap_model(vae).decode(latents).sample))['image']).permute(0, 2, 3, 1)*127.5+128).clamp(0,255).to(torch.uint8)
                                img_concat=torch.cat([img_concat, eg3doutput_denoised_ema_multistep],dim=2)
                                image_names.append('denoised_latent_ema_multistep')

                                img_concat=img_concat[:16].flatten(0,1)

                                ema_unet.restore(unet.parameters())
                                ema_encoder.restore(encoder.parameters()) 
                                ema_vae.restore(vae.parameters())
                        else:
                            img_concat=torch.cat([original_render,recon_latent_render],dim=2) # (batch_size,256,768,3)
                            img_concat=img_concat[:16].flatten(0,1)

                        # convert image_names to str with sep '--'
                        image_names='--'.join(image_names)
                        # save to wandb
                        accelerator.log({'Images':wandb.Image(img_concat.cpu().numpy())},step=global_step)
                        # accelerator.log({'Image_names':image_names},step=global_step)
                        logger.info(f"Image_names: {image_names}")
        accelerator.wait_for_everyone()



#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
