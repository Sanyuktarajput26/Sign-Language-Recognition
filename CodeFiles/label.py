#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Define paths
base_frames_dir = r'C:\projectslr\dataset_sl\cropped_hands'
output_csv_file = r'C:\projectslr\dataset_sl\frame_labels.csv'
label_encoding_file = r'C:\projectslr\dataset_sl\label_encoding.csv'

# Create a label encoder instance
label_encoder = LabelEncoder()

# Step 1: Traverse and label each parent folder
parent_folders = [f for f in os.listdir(base_frames_dir) if os.path.isdir(os.path.join(base_frames_dir, f))]
labels = label_encoder.fit_transform(parent_folders)

# Step 2: Create a dictionary for parent folder labels
folder_label_mapping = dict(zip(parent_folders, labels))

# Initialize an empty list to save results
results = []

# Traverse each parent folder
for parent_folder, label in folder_label_mapping.items():
    parent_folder_path = os.path.join(base_frames_dir, parent_folder)

    # Traverse each subfolder within the parent folder
    subfolders = [f for f in os.listdir(parent_folder_path) if os.path.isdir(os.path.join(parent_folder_path, f))]

    # Label frames in each subfolder
    for subfolder in subfolders:
        subfolder_path = os.path.join(parent_folder_path, subfolder)
        frames = os.listdir(subfolder_path)

        # Label each frame according to the parent folder label
        for frame in frames:
            # Append frame information to results
            results.append({
                "frame_label": f"{label}_{frame}",  # e.g., "0_frame1.jpg"
                "folder_name": parent_folder,
                "subfolder": subfolder,
                "frame_path": f"{parent_folder}/{subfolder}/{frame}"
            })

# Step 3: Save the frame label information in a CSV file
df = pd.DataFrame(results)
df.to_csv(output_csv_file, index=False)

# Step 4: Save the label encoding dictionary in a separate CSV file
label_df = pd.DataFrame(list(folder_label_mapping.items()), columns=["folder_name", "label"])
label_df.to_csv(label_encoding_file, index=False)

print("Labels and frame paths have been saved successfully to the CSV file!")
print("Label encoding has been saved successfully to a separate CSV file!")

