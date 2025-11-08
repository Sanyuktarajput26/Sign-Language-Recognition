#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import cv2
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Paths and parameters
train_data_path = r"C:\projectslr\dataset_sl\train_final_augmented"
test_data_path = r"C:\projectslr\dataset_sl\test_final_augmented"
img_size = (224, 224)  # Required input size for VGG16
sequence_length = 10  # Adjust based on your sequences

# Limit to the first 300 class folders for both train and test sets
train_classes = sorted(os.listdir(train_data_path))[:300]
test_classes = sorted(os.listdir(test_data_path))[:300]

# Load VGG16 model pre-trained on ImageNet without the fully connected layers
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

# Function to load and preprocess images
def load_and_preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, img_size)
        img = preprocess_input(img)
    return img

# Function to extract features for a dataset and save them incrementally
def extract_and_save_features(data_path, classes, output_dir, sequence_length=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    class_map = {cls: idx for idx, cls in enumerate(classes)}
    all_features = []
    all_labels = []

    for cls in classes:
        cls_path = os.path.join(data_path, cls)
        if not os.path.isdir(cls_path):
            print(f"Skipping non-directory: {cls_path}")
            continue

        frames = sorted(os.listdir(cls_path))[:sequence_length]  # Use only the first N frames
        sequence = []

        # Load and preprocess images for the sequence
        for frame in frames:
            img_path = os.path.join(cls_path, frame)
            img = load_and_preprocess_image(img_path)
            if img is not None:
                sequence.append(img)

        if len(sequence) == sequence_length:
            # Convert sequence to numpy array and expand dimensions for batch processing
            sequence = np.array(sequence)
            sequence_features = feature_extractor.predict(sequence)
            all_features.append(sequence_features)
            all_labels.append(class_map[cls])

    # Convert to numpy arrays
    all_features = np.array(all_features)
    all_labels = np.array(all_labels)

    # Save the features and labels to disk
    feature_file = os.path.join(output_dir, f"{len(classes)}_features.npy")
    label_file = os.path.join(output_dir, f"{len(classes)}_labels.npy")

    np.save(feature_file, all_features)
    np.save(label_file, all_labels)

    print(f"Features and labels for {len(classes)} classes saved to {output_dir}")

# Define output directories for train and test
train_output_dir = r"C:\projectslr\dataset_sl\train_features"
test_output_dir = r"C:\projectslr\dataset_sl\test_features"

# Extract and save features for train and test sets
extract_and_save_features(train_data_path, train_classes, train_output_dir)
extract_and_save_features(test_data_path, test_classes, test_output_dir)

print("Feature extraction and saving completed.")

