#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Define paths
base_frames_dir = r'C:\projectslr\dataset_sl\Train and Test model'
output_train_folder = r'C:\projectslr\dataset_sl\train_final_augmented'
output_test_folder = r'C:\projectslr\dataset_sl\test_final_augmented'

# Create output directories if they don't exist
os.makedirs(output_train_folder, exist_ok=True)
os.makedirs(output_test_folder, exist_ok=True)

# Initialize ImageDataGenerator for augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

def normalize_image(image):
    """Normalize the image."""
    return image / 255.0  # Normalize pixel values to [0, 1]

def augment_image(image, output_path, prefix, count=3):
    """Augment a single image and save augmented versions."""
    image = image.reshape((1,) + image.shape)
    for i, _ in enumerate(datagen.flow(image, batch_size=1, save_to_dir=output_path, save_prefix=prefix, save_format='jpeg')):
        if i >= count:
            break

def process_category(category, folder_path, output_train_path, output_test_path):
    """Process a single category."""
    train_subfolder = os.path.join(folder_path, category, 'train')
    test_subfolder = os.path.join(folder_path, category, 'test')

    output_train_subfolder = os.path.join(output_train_path, category)
    output_test_subfolder = os.path.join(output_test_path, category)

    os.makedirs(output_train_subfolder, exist_ok=True)
    os.makedirs(output_test_subfolder, exist_ok=True)

    # Process training frames
    if os.path.exists(train_subfolder):
        process_frames_in_subfolder(train_subfolder, output_train_subfolder)

    # Process testing frames
    if os.path.exists(test_subfolder):
        process_frames_in_subfolder(test_subfolder, output_test_subfolder)

def process_frames_in_subfolder(subfolder_path, output_subfolder):
    """Process all frames in a subfolder."""
    for frame_folder in os.listdir(subfolder_path):
        frame_folder_path = os.path.join(subfolder_path, frame_folder)
        if os.path.isdir(frame_folder_path):
            for frame_name in os.listdir(frame_folder_path):
                frame_path = os.path.join(frame_folder_path, frame_name)
                image = cv2.imread(frame_path)
                if image is not None:
                    normalized_image = normalize_image(image)
                    augment_image(normalized_image, output_subfolder, frame_folder, count=3)

def process_frames_in_parallel(folder_path, output_train_path, output_test_path):
    """Process all categories in parallel using ThreadPoolExecutor."""
    categories = sorted([f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))])

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_category, category, folder_path, output_train_path, output_test_path)
            for category in categories
        ]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing categories", unit="category"):
            pass

if __name__ == "__main__":
    print("Processing training and testing frames...")
    process_frames_in_parallel(base_frames_dir, output_train_folder, output_test_folder)
    print("Augmented frames saved successfully!")

