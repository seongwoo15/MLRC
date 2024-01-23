import os
from PIL import Image
from datasets import load_dataset
# Load the dataset
dataset = load_dataset("Yura32000/eurosat")
# Define the root directory for the splits
root_dir = "EuroSAT_splits"
os.makedirs(root_dir, exist_ok=True)
# Define the subdirectories for train, test, and valid
splits = ['train', 'test', 'valid']
label_names = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
label_counters = {}
for split in splits:
    split_dir = os.path.join(root_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    # Iterate over each entry in the dataset split
    for entry in dataset[split]:
        image, label = entry['image'], entry['label']
        label_name = label_names[label]
        label_dir = os.path.join(split_dir, label_name)
        # Create a directory for each label if it doesn't exist
        os.makedirs(label_dir, exist_ok=True)
        # Keep track of the number of images for each label
        if label_name not in label_counters:
            label_counters[label_name] = 0
        label_counters[label_name] += 1
        # Define image filename with label and index
        image_filename = f"{label_name}_{label_counters[label_name]}.png"
        image_path = os.path.join(label_dir, image_filename)
        # Save the image
        # Note: Convert `image` to a PIL Image if it's not already
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)  # Convert only if it's not already a PIL Image
        image.save(image_path)