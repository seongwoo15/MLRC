import torch
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset
from PIL import Image
import requests
from io import BytesIO


model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


dataset = load_dataset("svhn", 'full_numbers')


def load_image(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    return img


example = dataset['train'][0]
print(example)
exit()
image = load_image(example['image']) 

text = ["a photo of a number"] 
inputs = processor(text=text, images=image, return_tensors="pt", padding=True)


outputs = model(**inputs)


logits_per_image = outputs.logits_per_image
logits_per_text = outputs.logits_per_text

print(logits_per_image, logits_per_text)
