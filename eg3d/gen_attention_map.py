from diffusers import UNet2DConditionModel,AutoencoderKL,StableDiffusionPipeline
import torch
from PIL import Image
import numpy as np

# use stabilityai/stable-diffusion-2-1-base
model = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-1-base",
                                             subfolder='unet',
                                             torch_dtype=torch.float32).cuda()
model.eval()

vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1",
                                    subfolder='vae',
                                    torch_dtype=torch.float32).cuda()

# load image
img = Image.open("/home1/jo_891/data1/eg3d/eg3d/WechatIMG463.jpg").convert('RGB').resize((512,512))
img = torch.tensor(np.array(img).transpose(2,0,1)).unsqueeze(0).cuda().float()/255.0 # [1,3,512,512] in range [0,1]
img=img*2-1 # [1,3,512,512] in range [-1,1]

# encode image with vae
z = vae.encode(img).latent_dist.sample() # [1,4,64,64]



pass





