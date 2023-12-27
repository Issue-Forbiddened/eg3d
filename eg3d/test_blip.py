from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", load_in_8bit=False, device_map={"": 0}, torch_dtype=torch.float16
)  # doctest: +IGNORE_RESULT


image = Image.open("/home1/jo_891/data1/eg3d/eg3d/WechatIMG463.jpg")


# inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)

# generated_ids = model.generate(**inputs)
# generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
# print(generated_text)

# prompt = "Question: how do you describe this image to a portraiter? Answer:" #a man with glasses and a tie on
# prompt = "Question: how do you describe the facial features of the face in this image in detail for a portrait artist ? Answer:" #the face is a man with a beard and glasses
# prompt= "Question: 1.what race is this person? 2.Does it wear glasses 3. Give me three most obvious features to describe this face  Answer:" # 1. White 2. Yes 3. Eyebrows
# prompt="Question: how do you describe this image to a portraiter? Answer:"
inputs = processor(images=image, text=prompt, return_tensors="pt").to(device="cuda", dtype=torch.float16)

generated_ids = model.generate(**inputs)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)

pass