pip install streamlit
import streamlit as st
import tensorflow as tf
import cv2
import numpy as np

@st.cache(allow_output_mutation=True)
def load_model():
    return tf.keras.models.load_model('sign_language_recognition_model.h5')

model = load_model()
gesture_labels = ['a', 'a lot', 'abdomen', 'able']

st.title("Sign Language Recognition")
uploaded_video = st.file_uploader("D:\Dataset\archive\dataset\SL\a\01610.mp4", type=["mp4", "avi"])

if uploaded_video:
    st.video(uploaded_video)
    # Process video and extract frames
    frames = extract_frames(uploaded_video, "temp_frames")
    preprocessed_frames = [preprocess_frame(frame) for frame in frames]
    prediction = model.predict(np.expand_dims(preprocessed_frames, axis=0))
    predicted_label = gesture_labels[np.argmax(prediction)]
    st.write(f"Predicted Gesture: {predicted_label}")
