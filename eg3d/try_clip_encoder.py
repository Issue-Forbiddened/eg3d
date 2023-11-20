from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import StableDiffusionImageVariationPipeline
from PIL import Image
from torchvision import transforms

device = "cuda:0"
clip=CLIPVisionModelWithProjection.from_pretrained("lambdalabs/sd-image-variations-diffusers",subfolder='image_encoder')
sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
  "lambdalabs/sd-image-variations-diffusers",
  revision="v2.0",
  )
sd_pipe = sd_pipe.to(device)

im = Image.open("/home1/jo_891/data1/eg3d/eg3d/WechatIMG463.jpg").convert("RGB")
tform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(
        (224, 224),
        interpolation=transforms.InterpolationMode.BICUBIC,
        antialias=False,
        ),
    transforms.Normalize(
      [0.48145466, 0.4578275, 0.40821073],
      [0.26862954, 0.26130258, 0.27577711]),
])

# image_embeddings = sd_pipe._encode_image(im, device, 1, False)
inp = tform(im).to(device).unsqueeze(0)

out = sd_pipe(inp, guidance_scale=3)
pass