#!/usr/bin/env python3
# fast_prediction_server.py - Optimized standalone prediction server

from flask import Flask, Response
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import time

app = Flask(__name__)

# Initialize MediaPipe with optimized settings
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Define actions (must match your training)
ACTIONS = np.array([
    'hello', 'me', 'yes', 'help', 'please', 'thank you', 'what',
    'eat food', 'more', 'learn', 'sign', 'done', 'name', 'i love you', 'you'
])

def mediapipe_detection(image, model):
    """Optimized MediaPipe detection"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_essential_landmarks(image, results):
    """Draw only essential landmarks for sign language"""
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
        )
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )
    return image

def extract_keypoints(results):
    """Ensure this matches your model's expected feature count"""
    # Original version with all landmarks (1662 features)
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

@app.route('/')
def video_feed():
    """Video streaming route with predictions"""
    # Load model
    model = load_model('models/action.h5')
    
    # Configuration
    sequence = []
    threshold = 0.95
    current_prediction = ""
    last_update_time = time.time()
    min_display_duration = 1.0
    
    # Initialize camera with reduced resolution
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    def generate():
        nonlocal sequence, current_prediction, last_update_time
        
        with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3
        ) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every 3rd frame for better performance
                if int(time.time() * 10) % 3 == 0:
                    image, results = mediapipe_detection(frame, holistic)
                    image = draw_essential_landmarks(image, results)
                    keypoints = extract_keypoints(results)
                    
                    sequence.append(keypoints)
                    sequence = sequence[-20:]  # Reduced sequence length
                    
                    # Reset prediction if display time elapsed
                    if current_prediction and (time.time() - last_update_time) > min_display_duration:
                        current_prediction = ""
                    
                    # Make prediction when buffer is full
                    if len(sequence) == 20:
                        res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                        predicted_idx = np.argmax(res)
                        confidence = res[predicted_idx]
                        
                        if confidence > threshold:
                            current_prediction = ACTIONS[predicted_idx]
                            last_update_time = time.time()
                    
                    # Add prediction to frame
                    if current_prediction:
                        cv2.rectangle(image, (0,0), (640, 60), (245,117,16), -1)
                        cv2.putText(image, current_prediction, 
                                  (10, 40), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, 
                                  (255,255,255), 2, cv2.LINE_AA)
                else:
                    image = frame
                
                # Convert to JPEG
                ret, buffer = cv2.imencode('.jpg', image)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(port=5001, threaded=True)