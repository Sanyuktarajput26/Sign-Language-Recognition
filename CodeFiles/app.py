#%%writefile app.py

import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import tempfile
import os

# CSS styling function to set background image, text colors, and font sizes
def set_background_image(image_url, text_color="#FFFFFF", font_size="18px"):
    st.markdown(
        f"""
        <style>
            .stApp {{
                background-image: url("{image_url}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
                color: {text_color};
                font-family: 'Arial', sans-serif;
                font-size: {font_size};
            }}
            h1, h2, h3 {{
                color: {text_color};
            }}
            .button {{
                background-color: #007BFF;
                color: black;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                cursor: pointer;
                transition: background-color 0.3s;
            }}
            .button:hover {{
                background-color: #0056b3;
            }}
            .file-upload {{
                border: 2px dashed #007BFF;
                border-radius: 10px;
                padding: 20px;
                text-align: center;
                margin: 20px 0;
            }}
            .footer {{
                color: {text_color};
                text-align: center;
                margin-top: 50px;
                font-size: 0.9em;
                font-style: italic;
            }}
            .sidebar .block-container {{
                color: #0a0a0a;
                font-size: 18px;
                font-weight: bold;
            }}
            .sidebar .sidebar-content {{
                color: #0a0a0a;
                font-weight: bold;
            }}
        </style>
        """, unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_sign_language_model():
    model_path = "D:/Infosys Springboard/final_model.keras"
    model = load_model(model_path)
    return model

# Video processing and prediction function
def predict_sign_language(video_path, model):
    cap = cv2.VideoCapture(video_path)
    predictions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess the frame for the model (resize, normalize, etc.)
        frame = cv2.resize(frame, (224, 224))
        frame = frame / 255.0
        frame = np.expand_dims(frame, axis=0)

        # Make prediction
        prediction = model.predict(frame)
        predicted_class = np.argmax(prediction, axis=1)
        predictions.append(predicted_class)

    cap.release()
    
    # Simple logic to return the most frequent prediction
    final_prediction = max(set(predictions), key=predictions.count)
    return final_prediction

# Login function
def login():
    set_background_image("https://img.freepik.com/premium-photo/blurred-lights-abstract-background_3248-3681.jpg?semt=ais_hybrid", text_color="#000000")
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        st.session_state["logged_in"] = True
        st.success("Logged in successfully!", icon="✅")
        st.markdown('<style>.st-alert p { color: #39FF14; }</style>', unsafe_allow_html=True)

# Sidebar navigation function
def sidebar_navigation():
    st.sidebar.title("Navigation")
    return st.sidebar.radio("Go to:", ["Welcome", "Sign Language Recognition", "About Us"])

# Sign Language Recognition Page
def sign_language_recognition_page():
    set_background_image("https://img.freepik.com/free-vector/futuristic-background-design_23-2148503793.jpg", text_color="#5fc2e3", font_size="18px")
    st.title("Sign Language Recognition")
    st.write("Upload a video of sign language to get the recognized text.")
     # Instructions with larger, styled text
    st.subheader("Instructions:")
    st.write("""
        1. Ensure the video file is clear and only contains the signing gesture.
        2. Upload the video in MP4, AVI, or MOV format.
        3. Click the **Upload** button to start the recognition process.
        4. Once the processing is complete, the app will display the recognized text interpretation.
    """)
    
    uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name

        st.video(video_path)
        
        model = load_sign_language_model()
        prediction = predict_sign_language(video_path, model)
        
        st.write("Recognized Word:")
        st.subheader(str(prediction))

        # Clean up
        os.remove(video_path)

    st.markdown('<div class="footer">Developed by Sayan Chakraborty</div>', unsafe_allow_html=True)

# Other Pages (Welcome, About Us)
def welcome_page():
    set_background_image("https://t3.ftcdn.net/jpg/06/92/00/12/360_F_692001223_Y36tREumbw858v6QZSftO9HRaqVGl6X9.jpg", text_color="#144d04", font_size="20px")
    st.title("Welcome to the Sign Language Recognition App")
    st.write("""
        This Sign Language Recognition App is dedicated to breaking communication barriers for individuals with hearing and speech impairments.
        Through this innovative application, users can upload videos of sign language, and our advanced machine learning algorithms 
        translate these signs into comprehensible text. We believe in creating a more accessible and inclusive society for everyone, 
        where communication is not hindered by physical limitations.

        The app is designed with user-friendly features and state-of-the-art technology to recognize gestures accurately. Our mission 
        is to provide a seamless, empowering experience, enabling people to express themselves and connect with others effortlessly.
        
        Whether you’re a family member, a friend, or an educator, this app can help bridge the gap and foster deeper understanding and 
        stronger relationships.
    """)
    st.markdown('<div class="footer">Developed by Sayan Chakraborty</div>', unsafe_allow_html=True)

def about_us_page():
    set_background_image("https://e1.pxfuel.com/desktop-wallpaper/30/550/desktop-wallpaper-light-color-backgrounds-for-websites-light-backgrounds.jpg", text_color="#0c186b", font_size="20px")
    st.title("About Us")
    st.write("""
        This Sign Language Recognition App is a groundbreaking tool aimed at promoting inclusivity and accessibility in communication.
        People with hearing or speech impairments face significant communication challenges, especially with those who are unfamiliar 
        with sign language. Our app bridges this communication gap by using cutting-edge technology to interpret sign language gestures 
        and convert them into readable text.

        **Our Vision:**  
        We envision a world where everyone, regardless of physical ability, can communicate freely and effectively. We are committed 
        to fostering an inclusive society by empowering individuals and making essential communication tools available to all.

        **Our Mission:**  
        To develop intuitive, high-quality solutions that make everyday interactions easier for individuals with communication challenges. 
        By harnessing the power of machine learning, our app recognizes the nuances of sign language and delivers accurate text interpretations.

        **Conclusion:**  
        The Sign Language Recognition App is designed to empower people with hearing or speech impairments, enabling seamless communication 
        with those unfamiliar with sign language. This app represents a step towards a more inclusive society, where everyone has a voice, 
        irrespective of any physical limitations.
    """)
    st.markdown('<div class="footer">Developed by Sayan Chakraborty</div>', unsafe_allow_html=True)

# Main function to control app flow
def main():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        login()
    else:
        selected_page = sidebar_navigation()
        if selected_page == "Welcome":
            welcome_page()
        elif selected_page == "Sign Language Recognition":
            sign_language_recognition_page()
        elif selected_page == "About Us":
            about_us_page()

if __name__ == "__main__":
    main()
