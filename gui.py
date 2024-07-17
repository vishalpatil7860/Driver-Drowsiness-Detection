import streamlit as st
import cv2
import dlib
from scipy.spatial import distance
import numpy as np
import time
import simpleaudio as sa
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the pre-trained facial landmark predictor and eye state model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Function to safely load the model
def load_keras_model(model_path):
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the eye state model
model = load_keras_model('model.h5')

# Function to extract eye regions and preprocess for the model
def extract_eye_region(frame, eye_points):
    x1, y1 = np.min(eye_points, axis=0)
    x2, y2 = np.max(eye_points, axis=0)
    eye_region = frame[y1:y2, x1:x2]
    eye_region = cv2.resize(eye_region, (24, 24))
    eye_region = eye_region.astype("float") / 255.0
    eye_region = img_to_array(eye_region)
    eye_region = np.expand_dims(eye_region, axis=0)
    return eye_region

# Function to detect drowsiness using the neural network model
def detect_drowsiness(frame, gray, start_time, drowsy_threshold=3):
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        left_eye_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        right_eye_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
        
        left_eye_region = extract_eye_region(frame, left_eye_points)
        right_eye_region = extract_eye_region(frame, right_eye_points)
        
        left_eye_prediction = model.predict(left_eye_region)
        right_eye_prediction = model.predict(right_eye_region)
        
        left_eye_state = np.argmax(left_eye_prediction)
        right_eye_state = np.argmax(right_eye_prediction)
        
        if left_eye_state == 0 and right_eye_state == 0:  # Assuming 0 is the label for closed eyes
            if start_time is None:
                start_time = time.time()
            else:
                elapsed_time = time.time() - start_time
                if elapsed_time >= drowsy_threshold:
                    return True, start_time
        else:
            start_time = None
    return False, start_time

# Function to play buzzer sound
def play_buzzer():
    wave_obj = sa.WaveObject.from_wave_file('buzzer.wav')
    play_obj = wave_obj.play()
    play_obj.wait_done()

# Function to capture video and process frames
def capture_video():
    cap = cv2.VideoCapture(0)
    frame_window = st.image([])
    stop_button = st.button("Stop", key="stop")
    start_time = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to capture video")
            break

        frame = cv2.flip(frame, 1)  # Flip the frame horizontally
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        drowsy, start_time = detect_drowsiness(frame, gray, start_time)
        if drowsy:
            cv2.putText(frame, "Drowsiness Detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            play_buzzer()

        # Display the video frame with the drowsiness detection result
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame_rgb)

        if stop_button:
            break

    cap.release()
    cv2.destroyAllWindows()

# Streamlit main function
def main():
    st.set_page_config(
        page_title="Driver Drowsiness Detection",
        page_icon=":car:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    st.title("Driver Drowsiness Detection System")
    st.sidebar.title("Control Panel")
    st.sidebar.markdown("## Instructions")
    st.sidebar.markdown("""
        1. Click on **Start Detection** to begin monitoring.
        2. If drowsiness is detected, an alert will be triggered.
        3. Click on **Stop** to end the monitoring.
    """)
    
    st.sidebar.markdown("## Actions")
    if st.sidebar.button('Start Detection', key="start"):
        capture_video()

if __name__ == '__main__':
    main()
