import os
import random
import shutil

# Paths
source_images = 'frames_with_detections'
destination = 'dataset'

# Create directories
for subset in ['train', 'val']:
    os.makedirs(os.path.join(destination, 'images', subset), exist_ok=True)
    os.makedirs(os.path.join(destination, 'labels', subset), exist_ok=True)

# Get all frame files and shuffle
image_files = [f for f in os.listdir(source_images) if f.endswith('.jpg')]
random.shuffle(image_files)

# Split into 80% train and 20% validation
split_index = int(0.8 * len(image_files))
train_files = image_files[:split_index]
val_files = image_files[split_index:]

# Copy images and labels to respective directories
for files, subset in [(train_files, 'train'), (val_files, 'val')]:
    for file in files:
        shutil.copy(os.path.join(source_images, file), os.path.join(destination, 'images', subset, file))

        # Copy the corresponding label file
        label_file = file.replace('.jpg', '.txt')
        shutil.copy(os.path.join(source_images, label_file), os.path.join(destination, 'labels', subset, label_file))

print("Dataset split into training and validation sets.")