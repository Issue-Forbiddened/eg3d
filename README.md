# Diffusion Models Meet Faces
## Unofficial Implementation of Control3diff
Paper:http://arxiv.org/abs/2304.06700

### 3D Inversion
See ./eg3d/train_control3diff.py for details.

[![3D Inversion](https://img.youtube.com/vi/sIEEeOhxUKE/hqdefault.jpg)](https://youtube.com/shorts/sIEEeOhxUKE?feature=share)

### Text-to-3D
See ./eg3d/train_control3diff_clip.py for details.

<!-- insert https://youtube.com/shorts/k079vpNctvI?feature=share -->
[![Text-to-3D](https://img.youtube.com/vi/k079vpNctvI/hqdefault.jpg)](https://youtube.com/shorts/k079vpNctvI?feature=share)

### Credits
EG3D: https://github.com/NVlabs/eg3d for its 3D GAN and checkpoints.

## SD and Triplanes
In progress.

### Installation
```bash
cd eg3d
conda env create -f environment.yml
conda activate eg3d
```

```bash
pip install git+https://github.com/huggingface/diffusers@main # Essential for using lora
```

Install all other dependencies if required.

### Download Pretrained Models
```bash
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/research/eg3d/versions/1/zip -O eg3d_1.zip
```

Unzip it and place anywhere you like. Use ffhqrebalanced512-128.pkl as the network_pkl.

### Training
See ./eg3d/train_image_variation_finetune.py for details.

```bash
accelerate launch --mixed_precision=fp16 train_image_variation_finetune.py --train_batch_size=8 --log_step_interval=2500 --checkpointing_steps=5000 --resume_from_checkpoint=latest --output=image_variation_finetune --wandb_offline  --prediction_type=epsilon --network_pkl=PATH_TO_EG3D_CHECKPOINT
```
