## Sign Language Recognition App

The Sign Language Recognition web App is a machine learning based tool designed to recognize and translate the gestures into text. The app uses deep learning models trained on video data to identify specific hand gestures and predict the corresponding sign language word or text.

## Motivation
Communication barriers faced by hearing impaired individuals often limit their interaction in daily life. This project is motivated by the need to build an intelligent system that can automatically recognize sign language gestures and convert them into text, thereby making communication more inclusive and accessible with the help of AI and computer vision.


### Features
#### 1. **Video Preprocessing**
- **Standardization**: Resizes input videos to a consistent dimension for uniformity.
- **Frame Rate Adjustment**: Normalizes video frame rates to 30 FPS.
- **Dataset Splitting**: Randomly divides videos into training, validation, and testing datasets for effective evaluation.

#### 2. **Feature Extraction**
- **Hand Gesture Analysis**: Extracts 60 distinct features from hand movements (e.g., angles, distances) using the **MediaPipe** library.
- **Data Export**: Saves extracted features into CSV files for:
  - Training Dataset
  - Validation Dataset
  - Testing Dataset

#### 3. **Model Training**
- **BiLSTM Architecture**: A Bidirectional LSTM model trained on extracted features for sequential pattern recognition.

#### 4. **Deployment**
- **Streamlit Web Application**: The model is deployed as an interactive web app using **Streamlit**, making it accessible and easy to use.

---

### Installation and Usage

#### Prerequisites
- **Python latest Version**: 
- **Required Libraries**:
  - mediapipe
  - tensorflow
  - numpy
  - pandas
  - streamlit
