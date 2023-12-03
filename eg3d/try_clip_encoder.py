from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection,CLIPTokenizer,CLIPTextModel,CLIPModel,AutoProcessor,CLIPFeatureExtractor
from diffusers import StableDiffusionImageVariationPipeline
from PIL import Image
from torchvision import transforms
import torch
device = "cuda:0"
sd_clip_image_encoder=CLIPVisionModelWithProjection.from_pretrained("lambdalabs/sd-image-variations-diffusers",subfolder='image_encoder').to(device)
# sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
#   "lambdalabs/sd-image-variations-diffusers",
#   revision="v2.0",
#   ).to(device)

clip_tokenizer=CLIPTokenizer.from_pretrained('CompVis/stable-diffusion-v1-4',subfolder='tokenizer') #就是原本clip里的,已确认
text_encoder=CLIPTextModel.from_pretrained('CompVis/stable-diffusion-v1-4',subfolder='text_encoder').to(device) #就是原本clip里的
original_clip=CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)

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

prompt='a man with glasses'
with torch.no_grad():
  text_inputs = clip_tokenizer(
      prompt,
      padding="max_length",
      max_length=clip_tokenizer.model_max_length,
      truncation=True,
      return_tensors="pt",
  )
  text_input_ids = text_inputs.input_ids
  prompt_embeds = text_encoder(text_input_ids.to(text_encoder.device))[0] # (1,77,768)


processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
image_inputs  = processor(images=im, return_tensors="pt")

diff_img_embed=sd_clip_image_encoder(image_inputs.pixel_values.to(sd_clip_image_encoder.device))[0]
original_clip_embed=original_clip.get_image_features(image_inputs.pixel_values.to(original_clip.device))[0]

original_clip_text_embed=original_clip.get_text_features(text_input_ids.to(original_clip.device))[0]

# compute the cosine similarity between the two img embeds
cos_sim=torch.cosine_similarity(diff_img_embed,original_clip_embed,dim=-1) # 发现是1


# image_embeddings = sd_pipe._encode_image(im, device, 1, False)
inp = tform(im).to(device).unsqueeze(0)

# out = sd_pipe(inp, guidance_scale=3)
pass