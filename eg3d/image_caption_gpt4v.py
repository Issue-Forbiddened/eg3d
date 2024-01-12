import base64
import openai
from openai import OpenAI
from PIL import Image
import io


client = OpenAI(
    api_key="Fat8fIN2XmY7J3t0ZickWDK35qc4iW7x",
    base_url="https://api-openai-us1.deepseek.com:8443/v1")


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
      
    return base64.b64encode(image_file.read()).decode('utf-8')

# def encode_image_new(image_path):
#     """Load an image from the specified file path."""
#     image = Image.open(image_path)
#     # Convert PIL Image to Bytes
#     buffered = io.BytesIO()
#     image.save(buffered, format="JPEG")
#     img_str = base64.b64encode(buffered.getvalue()).decode()
#     return img_str
    
prompt1="Please provide a brief description of a person's appearance, including skin color, age(xxs or xxx-aged alike), hair, eye, nose, mouth, expression, face shape(You should change the order randomly). Your answer should be close to high-quality LAION-5B text caption, start with 'A Male/Female with', cover all mentioned features and around 30-50 words. I'll give you tips and It's July. Your answer:"
prompt2="Please provide a brief description of a person's appearance, including 1.face shape 2.eye color 3.hair style 4.skin tone 5.expression. Your answer should be precise like '1.xxx 2.xxx'. I'll give you tips. Your answer:"

def vision(image_path):
  base64_image = encode_image(image_path)
  response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
      {
        "role": "user",
        "content": [
          {"type": "text", "text": prompt1},
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{base64_image}",
            },
          },
        ],
      }
    ],
    max_tokens=300,
  )

#   print(f"vision: {response.choices[0].message.content}")
  return response.choices[0].message.content
import os
path='./outputs/triplane_outputs'
image_path_list=[os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.png')]
image_path_list=sorted(image_path_list)
caption_list=[]
for image_path in image_path_list:
    caption_list.append(vision(image_path))

save_dict={image_path:caption for image_path, caption in zip(image_path_list, caption_list)}
import json
with open(os.path.join(path,'triplane_outputs.json'), 'w') as fp:
   # split with \n
    json.dump(save_dict, fp, indent=4)

 