from PIL import Image
import numpy as np
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, Lambda

# Load CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load MNIST dataset
dataset = load_dataset("mnist")

# Transformation pipeline
transform = Compose([
    Lambda(lambda image: Image.fromarray(np.array(image, dtype=np.uint8))),
    Resize((224, 224)),  # Resize to fit CLIP input dimensions
    ToTensor(),  # Converts PIL Image to tensor
    Lambda(lambda tensor: tensor.repeat(3, 1, 1) if tensor.size(0) == 1 else tensor)  # Ensure RGB format
])

def preprocess(examples):
    images = [transform(image) for image in examples["image"]]
    return {"image": images, "label": examples["label"]}

dataset = dataset.with_transform(preprocess)
train_dataset = dataset["train"]

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Digit descriptions
digit_descriptions = [
    "a photo of the number zero", "a photo of the number one",
    "a photo of the number two", "a photo of the number three",
    "a photo of the number four", "a photo of the number five",
    "a photo of the number six", "a photo of the number seven",
    "a photo of the number eight", "a photo of the number nine"
]

# Use torch.optim.AdamW
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Loss function
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
num_epochs = 3
model.train()
for epoch in range(num_epochs):
    for batch in train_loader:
        images = batch["image"]
        labels = batch["label"]
        texts = [digit_descriptions[label] for label in labels]

        # Process images and texts
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True)

        # Forward pass
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        logits_per_text = outputs.logits_per_text

        # Symmetrize the loss
        loss = (loss_fn(logits_per_image, labels) + loss_fn(logits_per_text, labels)) / 2

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save the fine-tuned model
torch.save(model.state_dict(), 'clip_vit_mnist_finetuned.pth')
