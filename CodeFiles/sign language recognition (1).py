#%%
pip install kagglehub

#%%
import kagglehub

# Download latest version
path = kagglehub.dataset_download("waseemnagahhenes/sign-language-dataset-wlasl-videos")

print("Path to dataset files:", path)
#%%
import os

# Define the path to your dataset
dataset_path = r"C:\Users\sanay\.cache\kagglehub\datasets\waseemnagahhenes\sign-language-dataset-wlasl-videos\versions\1\dataset\SL"


# List all the folders (representing sign labels)
folders = os.listdir(dataset_path)

# Dictionary to collect missing or corrupt data
issues = {}

for folder in folders:
    folder_path = os.path.join(dataset_path, folder)
    
    # Check if the folder contains video files
    if not os.path.isdir(folder_path):
        issues[folder] = "Not a valid directory"
        continue
    
    # List all files in the folder
    files = os.listdir(folder_path)
    
    # Check if the folder is empty or contains non-video files
    if not files:
        issues[folder] = "Empty folder"
    else:
        for file in files:
            if not file.endswith(('.mp4', '.avi', '.mkv')):
                issues.setdefault(folder, []).append(f"Invalid file: {file}")

# Report issues if any
if issues:
    print("Annotation issues found:")
    for folder, problem in issues.items():
        print(f"Folder '{folder}': {problem}")
else:
    print("All annotations and video files are present and correct.")
#%%
import os

# Correct dataset path
dataset_path = r"C:\Users\sanay\.cache\kagglehub\datasets\waseemnagahhenes\sign-language-dataset-wlasl-videos\versions\1\dataset\SL"

# Check if the path exists
if os.path.exists(dataset_path):
    # Count the number of folders and files within the dataset directory
    folder_count = len(os.listdir(dataset_path))
    print(f"Total count: {folder_count}")
else:
    print("Directory does not exist. Please check the path.")
#%%
import os

# Correct dataset path
dataset_path = r"C:\Users\sanay\.cache\kagglehub\datasets\waseemnagahhenes\sign-language-dataset-wlasl-videos\versions\1\dataset\SL"

# Check if the path exists
if os.path.exists(dataset_path):
    folder_video_count = {}
    
    # Iterate over each folder in the dataset directory
    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        
        # Only count if it's a directory
        if os.path.isdir(folder_path):
            video_count = len([file for file in os.listdir(folder_path) if file.endswith(('.mp4', '.avi', '.mkv'))])
            folder_video_count[folder] = video_count
    
    # Print folder names and video counts
    for folder, count in folder_video_count.items():
        print(f"Folder: {folder}, Video count: {count}")
else:
    print("Directory does not exist. Please check the path.")
#%%
print(folders)
print(video_count)
#%%
import os
import matplotlib.pyplot as plt
from collections import Counter

# Correct dataset path
dataset_path = r"C:\Users\sanay\.cache\kagglehub\datasets\waseemnagahhenes\sign-language-dataset-wlasl-videos\versions\1\dataset\SL"

# Check if the path exists
if os.path.exists(dataset_path):
    folder_video_count = []
    
    # Iterate over each folder in the dataset directory
    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        
        # Only count if it's a directory
        if os.path.isdir(folder_path):
            video_count = len([file for file in os.listdir(folder_path) if file.endswith(('.mp4', '.avi', '.mkv'))])
            folder_video_count.append(video_count)

    # Count how many folders have the same number of videos
    video_count_distribution = Counter(folder_video_count)
    
    # Sort the data for better visualization
    video_counts, folder_counts = zip(*sorted(video_count_distribution.items()))

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.bar(video_counts, folder_counts, color='skyblue')
    plt.xlabel('Number of Videos in a Folder')
    plt.ylabel('Number of Folders')
    plt.title('Distribution of Folders by Number of Videos')
    plt.tight_layout()
    plt.show()

else:
    print("Directory does not exist. Please check the path.")
#%%
import cv2
import os
from collections import Counter

def resize_video(video_path, output_path, size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec for writing the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = size
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, size)
        out.write(resized_frame)
    
    cap.release()
    out.release()

def process_all_videos(input_folder, output_folder, size=(224, 224)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    video_format_counter = Counter()  # To count video formats
    
    # Walk through each subfolder in the input directory
    for subdir, _, files in os.walk(input_folder):
        # Create the corresponding subfolder in the output directory
        relative_subdir = os.path.relpath(subdir, input_folder)
        output_subdir = os.path.join(output_folder, relative_subdir)
        
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)
        
        video_files = [file for file in files if file.endswith(('.mp4', '.avi', '.mkv'))]
        total_videos = len(video_files)

        for idx, video_file in enumerate(video_files):
            video_path = os.path.join(subdir, video_file)
            output_path = os.path.join(output_subdir, f"resized_{video_file}")
            
            # Track the format of the video
            video_extension = os.path.splitext(video_file)[1].lower()
            video_format_counter[video_extension] += 1

            # Resize and save the video
            resize_video(video_path, output_path, size)
            
            # Progress reporting per subfolder
            print(f"Processed {idx + 1}/{total_videos} in folder {relative_subdir}")

        # Save after completing all videos in a subfolder
        print(f"Completed processing folder: {relative_subdir}")
        print(f"Saved resized videos to: {output_subdir}")

    # Print the final count of video formats
    print("\nVideo Format Summary:")
    for format, count in video_format_counter.items():
        print(f"Format: {format}, Count: {count}")

# Example usage
input_folder = r"C:\Users\sanay\.cache\kagglehub\datasets\waseemnagahhenes\sign-language-dataset-wlasl-videos\versions\1\dataset\SL"
output_folder = r"C:\projectslr\dataset_sl\resized"
process_all_videos(input_folder, output_folder)
#%%
import cv2
import os
from collections import Counter

def resize_and_convert_to_grayscale(video_path, output_path, size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec for writing the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = size
    
    # Create the VideoWriter with the same size but grayscale (hence single-channel)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Resize the grayscale frame
        resized_frame = cv2.resize(gray_frame, size, interpolation=cv2.INTER_LINEAR)
        # Write the resized grayscale frame to the output video
        out.write(resized_frame)
    
    cap.release()
    out.release()

def process_all_videos(input_folder, output_folder, size=(224, 224)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    video_format_counter = Counter()  # To count video formats
    
    # Walk through each subfolder in the input directory
    for subdir, _, files in os.walk(input_folder):
        # Create the corresponding subfolder in the output directory
        relative_subdir = os.path.relpath(subdir, input_folder)
        output_subdir = os.path.join(output_folder, relative_subdir)
        
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)
        
        video_files = [file for file in files if file.endswith(('.mp4', '.avi', '.mkv'))]
        total_videos = len(video_files)

        for idx, video_file in enumerate(video_files):
            video_path = os.path.join(subdir, video_file)
            output_video_path = os.path.join(output_subdir, f"resized_grayscale_{video_file}")
            
            # Track the format of the video
            video_extension = os.path.splitext(video_file)[1].lower()
            video_format_counter[video_extension] += 1

            # Resize and convert to grayscale
            resize_and_convert_to_grayscale(video_path, output_video_path, size)
            
            # Progress reporting per subfolder
            print(f"Processed {idx + 1}/{total_videos} in folder {relative_subdir}")

        # Save after completing all videos in a subfolder
        print(f"Completed processing folder: {relative_subdir}")
        print(f"Saved resized grayscale videos to: {output_subdir}")

    # Print the final count of video formats
    print("\nVideo Format Summary:")
    for format, count in video_format_counter.items():
        print(f"Format: {format}, Count: {count}")

# Example usage
input_folder = r"C:/projectslr/dataset_sl/resized"
output_folder = r"C:/projectslr/dataset_sl/gs"
process_all_videos(input_folder, output_folder)
#%%

#%%
import cv2
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Paths for input and output directories
dataset_path = r"C:\projectslr\dataset_sl\gs"
output_path = r"C:\projectslr\dataset_sl\frame_extraction"
os.makedirs(output_path, exist_ok=True)

# Frame extraction frequency (e.g., every frame if set to 1, every 2nd frame if set to 2, etc.)
frame_skip = 1  # Use a smaller value like 1 or 2 for more frames

def extract_frames_from_video(video_path, video_output_folder, frame_skip=1):
    """Extract frames from a single video and save them in the video output folder."""
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    success, frame = cap.read()

    while success:
        if frame_count % frame_skip == 0:
            frame = cv2.resize(frame, (224, 224))  # Resize frame for consistency
            frame_filename = f"frame{frame_count:04d}.jpg"
            frame_output_path = os.path.join(video_output_folder, frame_filename)
            cv2.imwrite(frame_output_path, frame)

        frame_count += 1
        success, frame = cap.read()

    cap.release()
    return f"Finished processing {os.path.basename(video_path)}"

def process_videos_in_subfolders(label_folder, output_label_folder):
    """Finds and processes videos within subfolders, creating a single folder for each video's frames."""
    for root, _, files in os.walk(label_folder):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(root, file)
                
                # Use the label folder name as the subfolder name for all frames from each video
                video_id_folder = os.path.basename(root)
                video_output_folder = os.path.join(output_label_folder, video_id_folder)
                os.makedirs(video_output_folder, exist_ok=True)
                
                extract_frames_from_video(video_path, video_output_folder, frame_skip)

# Main script to iterate through label folders and subfolders
for label in os.listdir(dataset_path):
    label_folder = os.path.join(dataset_path, label)
    if os.path.isdir(label_folder):
        output_label_folder = os.path.join(output_path, label)
        os.makedirs(output_label_folder, exist_ok=True)
        process_videos_in_subfolders(label_folder, output_label_folder)

print("Frame extraction complete!")

#%%

#%%
import cv2
import mediapipe as mp
import os
import numpy as np

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True,  # Use static mode for individual frames
                       max_num_hands=2,
                       min_detection_confidence=0.7)

# Initialize MediaPipe Drawing module
mp_drawing = mp.solutions.drawing_utils

# Define input and output folder paths
input_folder = r"C:\projectslr\dataset_sl\frame_extraction"
output_folder = r"C:\projectslr\dataset_sl\cropped_hands"
os.makedirs(output_folder, exist_ok=True)

def process_image(image_path, frames_folder, frame_file):
    """Processes an individual frame image for hand detection and cropping."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read {image_path}")
        return False

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * image.shape[1])
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * image.shape[1])
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * image.shape[0])
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * image.shape[0])

            # Add padding to the bounding box
            padding = 20
            x_min = max(x_min - padding, 0)
            y_min = max(y_min - padding, 0)
            x_max = min(x_max + padding, image.shape[1])
            y_max = min(y_max + padding, image.shape[0])

            # Crop the hand region and resize
            cropped_hand = image[y_min:y_max, x_min:x_max]
            cropped_hand_resized = cv2.resize(cropped_hand, (224, 224))

            # Save cropped image to designated folder
            output_path = os.path.join(frames_folder, f"{os.path.splitext(frame_file)[0]}_cropped.jpg")
            cv2.imwrite(output_path, cropped_hand_resized)
            print(f"Saved cropped hand image: {output_path}")
            return True
    return False

# Process frames in each label's subfolder
for label in os.listdir(input_folder):
    label_folder = os.path.join(input_folder, label)
    if os.path.isdir(label_folder):
        label_output_folder = os.path.join(output_folder, label)
        os.makedirs(label_output_folder, exist_ok=True)

        for video_id in os.listdir(label_folder):
            video_id_folder = os.path.join(label_folder, video_id)
            if os.path.isdir(video_id_folder):
                output_id_folder = os.path.join(label_output_folder, video_id)
                os.makedirs(output_id_folder, exist_ok=True)

                for frame_file in os.listdir(video_id_folder):
                    frame_path = os.path.join(video_id_folder, frame_file)
                    if frame_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        hand_detected = process_image(frame_path, output_id_folder, frame_file)

                # Remove empty output folder
                if not os.listdir(output_id_folder):
                    os.rmdir(output_id_folder)
                    print(f"Removed empty folder: {output_id_folder}")

print("Hand detection and cropping complete!")

#%%

#%%
import cv2
import mediapipe as mp
import os
import numpy as np

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True,  # Use static mode for individual frames
                       max_num_hands=2,
                       min_detection_confidence=0.7)

# Initialize MediaPipe Drawing module
mp_drawing = mp.solutions.drawing_utils

# Define input and output folder paths
input_folder = r"C:\projectslr\dataset_sl\frame_extraction"
output_folder = r"C:\projectslr\dataset_sl\cropped_hands"
os.makedirs(output_folder, exist_ok=True)

def process_image(image_path, frames_folder, frame_file):
    """Processes an individual frame image for hand detection, drawing, and cropping."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read {image_path}")
        return False

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the original image
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                                      mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2))

            # Calculate bounding box around the hand
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * image.shape[1])
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * image.shape[1])
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * image.shape[0])
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * image.shape[0])

            # Add padding to the bounding box
            padding = 20
            x_min = max(x_min - padding, 0)
            y_min = max(y_min - padding, 0)
            x_max = min(x_max + padding, image.shape[1])
            y_max = min(y_max + padding, image.shape[0])

            # Crop the hand region and resize
            cropped_hand = image[y_min:y_max, x_min:x_max]
            cropped_hand_resized = cv2.resize(cropped_hand, (224, 224))

            # Save cropped image to designated folder
            output_path = os.path.join(frames_folder, f"{os.path.splitext(frame_file)[0]}_cropped.jpg")
            cv2.imwrite(output_path, cropped_hand_resized)
            print(f"Saved cropped hand image: {output_path}")
            return True
    return False

# Process frames in each label's subfolder
for label in os.listdir(input_folder):
    label_folder = os.path.join(input_folder, label)
    if os.path.isdir(label_folder):
        label_output_folder = os.path.join(output_folder, label)
        os.makedirs(label_output_folder, exist_ok=True)

        for video_id in os.listdir(label_folder):
            video_id_folder = os.path.join(label_folder, video_id)
            if os.path.isdir(video_id_folder):
                output_id_folder = os.path.join(label_output_folder, video_id)
                os.makedirs(output_id_folder, exist_ok=True)

                for frame_file in os.listdir(video_id_folder):
                    frame_path = os.path.join(video_id_folder, frame_file)
                    if frame_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        hand_detected = process_image(frame_path, output_id_folder, frame_file)

                # Remove empty output folder
                if not os.listdir(output_id_folder):
                    os.rmdir(output_id_folder)
                    print(f"Removed empty folder: {output_id_folder}")

print("Hand detection and cropping complete!")


#%%

#%%
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

#%%

#%%
import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Define paths
base_frames_dir = r'C:\projectslr\dataset_sl\cropped_hands'
output_dir = r'C:\projectslr\dataset_sl\Train and Test model'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

def copy_folder(src, dest):
    """Use shutil to copy directories in a cross-platform way."""
    try:
        shutil.copytree(src, dest)
        print(f"Copied {src} to {dest}")
    except FileExistsError:
        print(f"Skipping {src} (already exists)")
    except Exception as e:
        print(f"Error copying {src} to {dest}: {e}")

def split_folders(main_folder):
    """Split subfolders into 80% train and 20% test."""
    subfolders = [f for f in os.listdir(main_folder)
                  if os.path.isdir(os.path.join(main_folder, f))]
    random.shuffle(subfolders)
    split_index = int(0.8 * len(subfolders))
    return subfolders[:split_index], subfolders[split_index:]

def process_single_folder(main_folder):
    """Process a single folder by splitting it into train and test sets."""
    main_folder_path = os.path.join(base_frames_dir, main_folder)
    
    # Create corresponding output folder in augmented frames
    output_main_folder = os.path.join(output_dir, main_folder)
    train_folder = os.path.join(output_main_folder, 'train')
    test_folder = os.path.join(output_main_folder, 'test')

    # Create train and test directories if they don't exist
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Split subfolders into train and test
    train_subfolders, test_subfolders = split_folders(main_folder_path)

    # Copy train subfolders
    for subfolder in train_subfolders:
        src = os.path.join(main_folder_path, subfolder)
        dest = os.path.join(train_folder, subfolder)

        if not os.path.exists(dest):  # Avoid re-copying
            copy_folder(src, dest)
        else:
            print(f"Skipping {subfolder} (already exists in train)")

    # Copy test subfolders
    for subfolder in test_subfolders:
        src = os.path.join(main_folder_path, subfolder)
        dest = os.path.join(test_folder, subfolder)

        if not os.path.exists(dest):  # Avoid re-copying
            copy_folder(src, dest)
        else:
            print(f"Skipping {subfolder} (already exists in test)")

def process_folders():
    """Process all folders using multithreading for faster execution."""
    all_folders = [f for f in os.listdir(base_frames_dir)
                   if os.path.isdir(os.path.join(base_frames_dir, f))]

    # Use ThreadPoolExecutor to parallelize the processing
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Track progress with tqdm
        list(tqdm(executor.map(process_single_folder, all_folders), 
                  total=len(all_folders), desc="Processing folders"))

if __name__ == "__main__":
    process_folders()
    print("Folders split and copied successfully!")

#%%

#%%
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

#%%

#%%
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

# Limit to the first 5 class folders for both train and test sets
train_classes = sorted(os.listdir(train_data_path))[:50]
test_classes = sorted(os.listdir(test_data_path))[:50]

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

# Function to extract features for a dataset
def extract_features(data_path, classes):
    features = []
    labels = []
    class_map = {cls: idx for idx, cls in enumerate(classes)}

    for cls in classes:
        cls_path = os.path.join(data_path, cls)
        if not os.path.isdir(cls_path):
            print(f"Skipping non-directory: {cls_path}")
            continue

        frames = sorted(os.listdir(cls_path))[:sequence_length]  # Use only the first N frames

        # Load and preprocess images
        sequence = []
        for frame in frames:
            img_path = os.path.join(cls_path, frame)
            img = load_and_preprocess_image(img_path)
            if img is not None:
                sequence.append(img)

        if len(sequence) == sequence_length:
            # Convert sequence to numpy array and expand dimensions for batch processing
            sequence = np.array(sequence)
            sequence_features = feature_extractor.predict(sequence)
            features.append(sequence_features)
            labels.append(class_map[cls])

    features = np.array(features)
    labels = np.array(labels)
    return features, labels

# Extract features for train and test sets
train_features, train_labels = extract_features(train_data_path, train_classes)
test_features, test_labels = extract_features(test_data_path, test_classes)

# Convert labels to categorical
num_classes = len(train_classes)  # Assumes the number of classes is the same in both sets
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)

# Split the data into train/test if needed
X_train, X_test, y_train, y_test = train_test_split(train_features, train_labels, test_size=0.3, random_state=42)

print("Feature extraction completed.")
print("Train feature shape:", X_train.shape)
print("Test feature shape:", X_test.shape)

#%%

#%%
import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, Bidirectional, LSTM, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Paths and parameters
data_path = r"C:\projectslr\dataset_sl\train_final_augmented"  # Modify with correct folder if needed
img_size = (64, 64)  # Resize each image to 64x64 pixels
sequence_length = 10  # Number of frames per sequence
num_folders = 50  # Only use the first 50 folders

# Load the first 50 gesture class folders
classes = sorted(os.listdir(data_path))[:num_folders]
num_classes = len(classes)
class_map = {cls: idx for idx, cls in enumerate(classes)}

# Initialize data lists
sequences = []
labels = []

# Load data
for cls in classes:
    cls_path = os.path.join(data_path, cls)
    sequences_in_class = sorted(os.listdir(cls_path))

    for seq_folder in sequences_in_class:
        seq_path = os.path.join(cls_path, seq_folder)
        frames = sorted(os.listdir(seq_path))[:sequence_length]  # Limit to the first N frames

        sequence = []
        for frame in frames:
            img_path = os.path.join(seq_path, frame)
            img = cv2.imread(img_path)
            img = cv2.resize(img, img_size)
            sequence.append(img)

        if len(sequence) == sequence_length:
            sequences.append(sequence)
            labels.append(class_map[cls])

# Convert to numpy arrays
sequences = np.array(sequences)
labels = np.array(labels)

# Split data into train/test
X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.3, random_state=42)
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Model architecture
model = Sequential([
    TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=(sequence_length, img_size[0], img_size[1], 3)),
    TimeDistributed(MaxPooling2D(2, 2)),
    TimeDistributed(Conv2D(64, (3, 3), activation='relu')),
    TimeDistributed(MaxPooling2D(2, 2)),
    TimeDistributed(Flatten()),
    Bidirectional(LSTM(64)),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=16)

# Evaluate the model
train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

print(f"Training accuracy: {train_accuracy:.2f}")
print(f"Testing accuracy: {test_accuracy:.2f}")

#%%
