from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os
from utils.helper import mediapipe_detection, draw_styled_landmarks, extract_keypoints

app = Flask(__name__)

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Load model
model = tf.keras.models.load_model('models/action.h5')

# Actions array (must match the model's training classes)
actions = np.array([
    'hello', 'me', 'yes', 'help', 'please', 'thank you', 'what', 
    'eat food', 'more', 'learn', 'sign', 'done', 'name', 'i love you', 'you'
])

# Global variables
sequence = []
prediction = None
threshold = 0.95

def generate_frames():
    """Generate camera frames with predictions"""
    global sequence, prediction
    
    # Initialize webcam
    camera = cv2.VideoCapture(0)
    
    # Set up holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            success, frame = camera.read()
            if not success:
                break
            
            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            
            # Draw landmarks
            draw_styled_landmarks(image, results)
            
            # Extract keypoints
            keypoints = extract_keypoints(results)
            
            # Update sequence
            sequence.append(keypoints)
            sequence = sequence[-30:]  # Keep only last 30 frames
            
            # Make prediction when we have enough frames
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                predicted_idx = np.argmax(res)
                confidence = res[predicted_idx]
                
                if confidence > threshold:
                    prediction = {
                        'label': actions[predicted_idx],
                        'confidence': float(confidence)
                    }
                else:
                    prediction = None
            
            # Add prediction to frame if available
            if prediction:
                cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image, f"{prediction['label']} ({prediction['confidence']:.2f})", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Convert to jpg for streaming
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
        camera.release()

@app.route('/')
def index():
    """Home page route"""
    return render_template('index.html')

@app.route('/prediction')
def prediction_page():
    """Live prediction page route"""
    return render_template('prediction.html')

@app.route('/learn')
def learn_page():
    """Learn signs page route"""
    # Dictionary of signs with YouTube links
    signs = {
    'hello': 'https://www.youtube.com/embed/MbggypBmHGw',
    'me': 'https://www.youtube.com/embed/E3dGBJ8wXu4',
    'yes': 'https://www.youtube.com/embed/0X8RoDuhCt0',
    'help': 'https://www.youtube.com/embed/bN8tjcN9qIY',
    'please': 'https://www.youtube.com/embed/K0mD_RcESHI',
    'thank you': 'https://www.youtube.com/embed/wM7vu5SqB0I',
    'what': 'https://www.youtube.com/embed/Gm0CzJ0ao-E',
    'eat food': 'https://www.youtube.com/embed/kI7EQIAm-tE',
    'more': 'https://www.youtube.com/embed/wJXgkXBCrgY',
    'learn': 'https://www.youtube.com/embed/yTMzjMI0K8U',
    'sign': 'https://www.youtube.com/embed/zRLit07QViE',
    'done': 'https://www.youtube.com/embed/Fc8NUTJtpYk',
    'name': 'https://www.youtube.com/embed/IB3p-Xp12ws',
    'i love you': 'https://www.youtube.com/embed/jzJjdvTF10A',
    'you': 'https://www.youtube.com/embed/3x9pPcMbDBY?start=30'
}

    return render_template('learn.html', signs=signs)

@app.route('/about')
def about_page():
    """About page route"""
    return render_template('about.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_prediction')
def get_prediction():
    """Return current prediction as JSON"""
    global prediction
    if prediction:
        return jsonify(prediction)
    return jsonify({"label": "", "confidence": 0.0})

if __name__ == '__main__':
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Check if model exists, otherwise show warning
    if not os.path.exists('models/action.h5'):
        print("WARNING: Model file 'models/action.h5' not found.")
        print("Please place your trained model in the models directory.")
    
    app.run(debug=True)