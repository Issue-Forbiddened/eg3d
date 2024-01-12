from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
from PIL import Image
model_id = "stabilityai/stable-diffusion-2-1-base"

scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# prompt_dict={}
# prompt_dict['prompt_1'] = "In this image, there is a smiling woman with medium-length brown hair that gently frames her face. She has a joyous expression, with her eyes almost squinting as she smiles widely, showcasing her teeth. The woman's face shape appears to be oval, and she is wearing a pair of dark-framed sunglasses that slightly obscure her eyes while adding to her cheerful demeanor. The background is indistinct, allowing the focus to remain on her engaging smile and casual appearance. The overall impression is of a genuine moment of happiness."
# prompt_dict['prompt_1_shortened']='A person with an oval face and medium-length brown hair, smiling joyously with eyes squinting and teeth visible. She wears dark-framed sunglasses, adding to her cheerful demeanor, against an indistinct background, highlighting her genuine happiness.'
# prompt_dict['prompt_2']='The individual in the image has a friendly and approachable demeanor. He has short, neatly combed gray hair with hints of the original dark color near the roots, suggesting a mature age. His eyes are a light color, possibly blue or gray, and they are set behind a pair of rectangular-framed glasses with a subtle metallic finish. The glasses accentuate his steady gaze.  His nose is well-proportioned to his face, straight with a rounded tip, and his mouth is set in a gentle, closed-mouth smile that conveys a sense of warmth and geniality. His skin appears smooth with a healthy complexion, and there are visible laugh lines at the corners of his eyes and around his mouth, indicating a disposition to smiling.  The overall shape of his face is somewhat oval, with a prominent chin that adds a sense of firmness to his visage. He is wearing what looks like a professional outfit a smart shirt paired with a suit jacket indicating a formal or business setting. The overall impression is of a congenial and self-assured individual who carries a sense of quiet confidence.'
# prompt_dict['prompt_2_shortened']='A person with short, neatly combed gray hair showing dark roots, light blue or gray eyes behind rectangular metallic glasses, and a gentle smile. His oval face has laugh lines, indicating warmth and geniality, complemented by a smart shirt and suit jacket, suggesting a professional demeanor.'
# prompt_dict['prompt_3']='The person in the image appears to be a middle-aged woman with a friendly expression. She has medium-length blonde hair with bangs swept to the side and soft layers framing her face. Her eyes are visible behind a pair of rectangular glasses with a dark rim, which complements her sparkling dangle earrings. She has a soft, round face with a gentle smile, full lips with a subtle lipstick, and a straight nose. Her warm demeanor is underlined by her slight, congenial smile.'
# prompt_dict['prompt_3_shortened']='A person with medium-length blonde hair, side-swept bangs, and a friendly expression. She has round, soft facial features, full lips with subtle lipstick, and a gentle smile. Her eyes, visible behind rectangular dark-rimmed glasses, are complemented by sparkling dangle earrings, enhancing her warm, congenial demeanor.'
# prompt_dict['prompt_4']='The person in the image has a round face with a warm, medium skin tone. They have dark brown eyes that convey a friendly and content expression, complemented by the slight creases suggesting a gentle smile. Their hair is very short, almost cropped, revealing a bit of grey amidst the darker tones, suggesting middle age. They have a neatly trimmed mustache connecting to a goatee, both with specks of grey similar to the hair on their head. They wear a simple, collarless pink shirt that has a casual feel. The person exudes a relaxed and approachable demeanor.'
# prompt_dict['prompt_4_shortened']='A person with a round face, warm medium skin tone, and dark brown eyes, displaying a friendly, content expression with a gentle smile. They have very short, almost cropped hair with hints of grey, suggesting middle age, and a neatly trimmed mustache and goatee with specks of grey. Dressed in a simple, collarless pink shirt, they exude a relaxed and approachable demeanor.'

prompt_dict1={
    "prompt_0": "In this image, there is a smiling woman with medium-length brown hair that gently frames her face. She has a joyous expression, with her eyes almost squinting as she smiles widely, showcasing her teeth. The woman's face shape appears to be oval, and she is wearing a pair of dark-framed sunglasses that slightly obscure her eyes while adding to her cheerful demeanor. The background is indistinct, allowing the focus to remain on her engaging smile and casual appearance. The overall impression is of a genuine moment of happiness.",
    "prompt_1": "This is an image of a person with a friendly and approachable expression. The person has short, dark brown hair with a slight wave and a receding hairline. They possess a pair of bright, engaging hazel eyes set below slightly arched eyebrows. The individual's nose is of medium size with a straight profile. A content smile graces their lips, revealing a set of well-aligned white teeth. Their face shape is oval with visible signs of a stubble, suggesting a clean-shaven look on most days. Overall, the person appears to exude a sense of warmth and geniality.",
    "prompt_2": "In the image is a smiling woman with a warm countenance. She has light brown hair that shows graceful signs of greying, styled just past shoulder length with a gentle wave. Her eyes, framed by rectangular eyeglasses with dark rims, are a soft hue that conveys friendliness. Her nose is of medium size, well-proportionate to her rounded and amiable face. Her mouth curves into a gentle smile, revealing teeth that radiate positivity. Her overall expression is one of approachability and kindness, with subtle laugh lines that suggest a joyful disposition. She is wearing a purple top that adds a touch of vibrancy to her appearance. The background is nondescript, keeping the focus entirely on her.",
    "prompt_3": "The person in the image has a cheerful expression, with a friendly smile showcasing teeth. This person has medium-length light brown hair with bangs that frame a round face. The eyes are obscured by a pair of rectangular, dark-rimmed sunglasses, which adds a touch of mystery to the overall appearance. The nose is of average size with an undefined bridge, and the mouth has full lips. The background suggests an outdoor setting, likely to be a natural environment.",
    "prompt_4": "The individual in the image appears to be a woman with medium-length straight brown hair, slightly parted to one side. Her eyes are a warm hazel, beneath subtly defined eyebrows. Her nose is well-proportioned, and her mouth is set in a friendly smile, revealing a set of straight white teeth. The woman has a radiant expression with a hint of crow's feet which suggests joy or contentment. Her face is round, and her skin appears smooth with a healthy, sun-kissed complexion. The background is indistinct, allowing her features to be the main focus of the image.",
    "prompt_5": "The individual in the image has a friendly and approachable demeanor. He has short, neatly combed gray hair with hints of the original dark color near the roots, suggesting a mature age. His eyes are a light color, possibly blue or gray, and they are set behind a pair of rectangular-framed glasses with a subtle metallic finish. The glasses accentuate his steady gaze. His nose is well-proportioned to his face, straight with a rounded tip, and his mouth is set in a gentle, closed-mouth smile that conveys a sense of warmth and geniality. His skin appears smooth with a healthy complexion, and there are visible laugh lines at the corners of his eyes and around his mouth, indicating a disposition to smiling. The overall shape of his face is somewhat oval, with a prominent chin that adds a sense of firmness to his visage. He is wearing what looks like a professional outfit—a smart shirt paired with a suit jacket—indicating a formal or business setting. The overall impression is of a congenial and self-assured individual who carries a sense of quiet confidence.",
    "prompt_6": "The person in the image appears to be a middle-aged woman with a friendly expression. She has medium-length blonde hair with bangs swept to the side and soft layers framing her face. Her eyes are visible behind a pair of rectangular glasses with a dark rim, which complements her sparkling dangle earrings. She has a soft, round face with a gentle smile, full lips with a subtle lipstick, and a straight nose. Her warm demeanor is underlined by her slight, congenial smile.",
    "prompt_7": "The person in the image has short, light brown hair with a natural wave. Their eyes are hazel, surrounded by gentle lines that are pronounced when smiling. The person has a rounded nose and full lips set in a soft, approachable expression. The face shape is round with a fair complexion. The person appears to be content and relaxed, with a hint of a smile suggesting friendliness.",
    "prompt_8": "The individual in the image has a warm, friendly expression characterized by a gentle smile. This person has short, cropped hair with flecks of gray indicating middle age, and his facial hair includes a mustache and goatee that are neatly trimmed. His eyes are dark brown, set under slightly arching brows that convey a sense of contentment or mild amusement. The nose is straight with a rounded tip, and the mouth features full lips that curve upwards into a welcoming smile. The face shape is oval, with the cheeks appearing slightly flushed, giving him a healthy complexion. The skin tone is light brown, and the background suggests an indoor environment with a wooden structure. Wearing a pinkish casual collar shirt, the person seems approachable and at ease.",
    "prompt_9": "The person in the image has a warm complexion and presents a soft, amiable expression. They have voluminous, curly brown hair that frames their face, which has a rounded shape with gentle contours. The eyes are light brown, providing a friendly and open gaze. Their nose is prominent with a round tip and complements their full, well-shaped lips that curve into a slight smile. The person's facial features, combined with their relaxed demeanor, convey a sense of approachability and easy-going character."
}

# prompt_dict2={
#     "prompt_0": "A person with medium-length brown hair, an oval face, and a joyous expression, smiling widely with dark-framed sunglasses, in a casual setting.",
#     "prompt_1": "A person with short, dark brown hair, hazel eyes, and a friendly expression, smiling with well-aligned teeth and a clean-shaven look.",
#     "prompt_2": "A person with light brown hair showing signs of greying, friendly eyes behind rectangular glasses, and a gentle smile, wearing a purple top.",
#     "prompt_3": "A person with medium-length light brown hair, a round face, and a cheerful expression, wearing rectangular sunglasses in an outdoor setting.",
#     "prompt_4": "A person with medium-length straight brown hair, warm hazel eyes, a friendly smile, and a sun-kissed complexion, in an indistinct background.",
#     "prompt_5": "A person with short gray hair, light-colored eyes behind rectangular glasses, and a gentle smile, dressed in a professional outfit.",
#     "prompt_6": "A person with medium-length blonde hair, a soft round face, and a friendly expression, wearing rectangular glasses and sparkling earrings.",
#     "prompt_7": "A person with short, light brown hair, hazel eyes, a rounded nose, and a soft, approachable expression, appearing content and relaxed.",
#     "prompt_8": "A person with short, cropped hair with gray flecks, a gentle smile, dark brown eyes, and wearing a pinkish casual shirt, appearing warm and friendly.",
#     "prompt_9": "A person with curly brown hair, a warm complexion, light brown eyes, and a soft expression, looking friendly and easy-going."
# }

prompt_dict2={
    "prompt_0": "A person with medium-length brown hair and an oval face, wearing dark-framed sunglasses, smiling joyously with teeth visible. Background is indistinct, focusing on her casual, happy demeanor.",
    "prompt_1": "A person with short, wavy dark brown hair and a receding hairline. Features bright hazel eyes, a medium-sized straight nose, and a content smile with white teeth. Face shape is oval with a stubble.",
    "prompt_2": "A smiling woman with light brown, gently greying hair, past shoulder length. Wearing rectangular eyeglasses, she has a rounded face, medium-sized nose, and a gentle smile. Wearing a purple top.",
    "prompt_3": "A person with a cheerful expression, medium-length light brown hair with bangs, and round face. Eyes obscured by dark-rimmed sunglasses, and a mouth with full lips. Background suggests an outdoor setting.",
    "prompt_4": "A woman with medium-length straight brown hair and warm hazel eyes. Features a well-proportioned nose and a friendly smile with straight white teeth. Face is round with a sun-kissed complexion.",
    "prompt_5": "A person with short, neatly combed gray hair, light-colored eyes behind rectangular glasses, and a gentle smile. Face shape is oval with laugh lines, wearing a smart shirt and suit jacket.",
    "prompt_6": "A middle-aged woman with medium-length blonde hair and bangs. Eyes visible behind rectangular glasses, and a gentle smile with full lips. Wearing sparkling dangle earrings and subtle lipstick.",
    "prompt_7": "A person with short, light brown wavy hair and hazel eyes. Face shape is round with a fair complexion, and features a rounded nose and full lips set in a soft, approachable expression.",
    "prompt_8": "A person with short, gray-flecked cropped hair and a mustache-goatee combo. Dark brown eyes, straight nose, and full lips. Wearing a pinkish casual collar shirt, with an oval face and healthy complexion.",
    "prompt_9": "A person with voluminous, curly brown hair and light brown eyes. Rounded face shape with a prominent nose and full lips set in a slight smile. Features a warm complexion and a friendly demeanor."
}

pos_prompt_endfix='extremely realisitic, 8k, real person'
neg_prompt_endfix='ugly, bad anatomy, blurry, pixelated obscure, unnatural colors, poor lighting, dull, and unclear, cropped, lowres, low quality, artifacts, duplicate, morbid, mutilated, poorly drawn face, deformed, dehydrated, bad proportions'

prompt_dict={key:value for key, value in prompt_dict1.items()}

# append _shortened in prompt_dict2' keys
for key, value in prompt_dict2.items():
    prompt_dict[key+'_shortened']=value

for key, value in prompt_dict.items():
    prompt_dict[key]=value+'.'+pos_prompt_endfix

import os
outdir='t2i_outputs'
os.makedirs(outdir, exist_ok=True)

def generate_images(prompt_dict, num_images_per_prompt=1):
    """
    Generate images from the prompt using the model.
    """
    keys=[key for key in prompt_dict.keys() if 'shortened' not in key]

    for key in keys:
        prompt=prompt_dict[key]

        # Generate images
        image = pipe(prompt, num_images_per_prompt=num_images_per_prompt,negative_prompt=neg_prompt_endfix).images

        # Concatenate images horizontally
        concatenated_image = Image.new('RGB', (image[0].width * num_images_per_prompt, 2*image[0].height))
        for i in range(num_images_per_prompt):
            concatenated_image.paste(image[i], (image[0].width * i, 0))

        # Generated images with shortend prompt
        prompt=prompt_dict[key+'_shortened']
        image = pipe(prompt, num_images_per_prompt=num_images_per_prompt,negative_prompt=neg_prompt_endfix).images

        # Concatenate images horizontally
        for i in range(num_images_per_prompt):
            concatenated_image.paste(image[i], (image[0].width * i, image[0].height))

        path=os.path.join(outdir, key+".png")
        concatenated_image.save(path)

image=generate_images(prompt_dict, num_images_per_prompt=10)


