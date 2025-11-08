#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

# Define paths
train_base_path = r"C:\projectslr\dataset_sl\train_final_augmented"
test_base_path = r"C:\projectslr\dataset_sl\test_final_augmented"
output_folder = r"C:\projectslr\dataset_sl\sign lr"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Parameters
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 10
MAX_CLASSES = 10

# Data Augmentation and Preprocessing
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

# Get limited classes
all_classes = sorted(os.listdir(train_base_path))[:MAX_CLASSES]

# Load Training Data
train_data = datagen.flow_from_directory(
    train_base_path,
    target_size=IMG_SIZE,
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=all_classes,
    subset='training'
)

# Load Validation Data
val_data = datagen.flow_from_directory(
    train_base_path,
    target_size=IMG_SIZE,
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=all_classes,
    subset='validation'
)

# Load Testing Data
test_data = datagen.flow_from_directory(
    test_base_path,
    target_size=IMG_SIZE,
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=all_classes
)

# Define CNN Model
def create_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Instantiate Model
input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)
num_classes = len(all_classes)
model = create_cnn_model(input_shape, num_classes)
model.summary()

# Compile the Model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Define Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

model_checkpoint = ModelCheckpoint(
    filepath=os.path.join(output_folder, 'final_trained_model.keras'),  # Updated to .keras
    save_best_only=True,
    monitor='val_loss',
    verbose=1
)

# Train the Model
print("\nTraining the model...")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[early_stopping, model_checkpoint],
    verbose=1
)

# Save the model in HDF5 format as well, if needed
model.save(os.path.join(output_folder, 'final_trained_model.h5'))

# Evaluate the Model on Test Data
print("\nEvaluating the model on test data...")
test_loss, test_accuracy = model.evaluate(test_data)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Plot Training and Validation Metrics
def plot_metrics(history, output_folder):
    # Accuracy Plot
    plt.figure()
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_folder, 'accuracy_plot.png'))
    plt.show()

    # Loss Plot
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_folder, 'loss_plot.png'))
    plt.show()

plot_metrics(history, output_folder)

print(f"\nTraining complete. Model and metrics saved in '{output_folder}'.")

