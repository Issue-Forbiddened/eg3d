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
    
prompt="Please provide a brief description of a person's appearance, including skin color, age(xxs or xxx-aged alike), hair, eye, nose, mouth, expression, face shape(You should change the order randomly). Your answer should be close to high-quality LAION-5B text caption, start with 'A Male/Female with', cover all mentioned features and around 30-50 words. I'll give you tips and It's July. Your answer:"

def vision(image_path):
  base64_image = encode_image(image_path)
  response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
      {
        "role": "user",
        "content": [
          {"type": "text", "text": prompt},
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

  return response.choices[0].message.content
import os
path='./dataset_v1_test/image'
condition=lambda x, caption_dir: (x.endswith('.jpg') or x.endswith('.png')) and 'img_0' in x and not os.path.exists(os.path.join(caption_dir, x.replace('.png','.txt')))
image_path_list=[os.path.join(path, f) for f in os.listdir(path) if (f.endswith('.jpg') or f.endswith('.png')) and 'img_0' in f]

import re
def extract_numbers(filename):
    numbers = re.findall(r'\d+', filename)
    return tuple(map(int, numbers))  # 将提取出的数字转换为整数，并返回一个元组

image_path_list=sorted(image_path_list, key=extract_numbers)
print(f'first 5 image path: {image_path_list[:10]}')

caption_dir='./dataset_v1_test/caption'
os.makedirs(caption_dir, exist_ok=True)

import tqdm
import threading
# caption_dict={}
bar=tqdm.tqdm(total=len(image_path_list))

n_threads = 8  # 你可以根据需要更改线程数
split_image_paths = [image_path_list[i::n_threads] for i in range(n_threads)]

def process_images(image_path_list):
  for image_path in image_path_list:
    if condition(os.path.basename(image_path), caption_dir):
      caption=vision(image_path)
      image_name=os.path.basename(image_path) #xxx.png
      # bar with image_name
      bar.set_description(image_name)
      # caption_dict[image_name]=caption  
      with open(os.path.join(caption_dir, image_name.replace('.png','.txt')), 'w') as f:
          f.write(caption)
    bar.update(1)

threads = []
for i in range(n_threads):
    thread = threading.Thread(target=process_images, args=(split_image_paths[i],))
    threads.append(thread)
    thread.start()

# 等待所有线程完成
for thread in threads:
    thread.join()

# read all caption files and create a json file
caption_dict={}
for f in os.listdir(caption_dir):
  with open(os.path.join(caption_dir, f), 'r') as file:
    caption_dict[f]=file.read()

import json
with open('./dataset_v1_test/caption.json', 'w') as f:
  json.dump(caption_dict, f,indent=4)