import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, TimeDistributed, Conv1D
from tensorflow.keras.layers import Input, Attention, GlobalAveragePooling1D, Concatenate, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, classification_report
from scipy import stats
import random
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import imgaug.augmenters as iaa

# 1. Setup MediaPipe Holistic model
mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

def mediapipe_detection(image, model):
    """
    Process the image through MediaPipe model and return results
    """
    # Convert color format from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    # Make prediction using the model
    results = model.process(image)
    image.flags.writeable = True
    # Convert back to BGR for OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    """
    Draw all the landmarks without styling
    """
    if results.face_landmarks:
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def draw_styled_landmarks(image, results):
    """
    Draw landmarks with styling for better visibility
    """
    # Draw face connections
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image, 
            results.face_landmarks, 
            mp_holistic.FACEMESH_TESSELATION,
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
        )
    
    # Draw pose connections
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, 
            results.pose_landmarks, 
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
        )
    
    # Draw left hand connections
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, 
            results.left_hand_landmarks, 
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
        )
    
    # Draw right hand connections
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, 
            results.right_hand_landmarks, 
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

def extract_keypoints(results):
    """
    Extract keypoints from MediaPipe results with more focus on hands
    """
    # Extract pose landmarks (more weight to upper body)
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    
    # Extract face landmarks (reduced dimensionality - select key points only)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark[0:138:4]]).flatten() if results.face_landmarks else np.zeros(35*3)
    
    # Extract left hand landmarks
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    
    # Extract right hand landmarks
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    # Concatenate all keypoints with greater weight to hands
    return np.concatenate([pose, face, lh * 2, rh * 2])  # Scale hand features for greater importance

def keypoint_augmentation(keypoints, max_offset=0.02):
    """
    Apply small random offsets to keypoints for data augmentation
    """
    # Add small random noise to keypoints
    noise = np.random.uniform(-max_offset, max_offset, keypoints.shape)
    augmented = keypoints + noise
    return augmented

def augment_sequence(sequence):
    """
    Apply augmentation to a full sequence
    """
    # Apply consistent temporal augmentation to sequence
    augmented_seq = []
    # Random offset for the entire sequence
    offset = np.random.uniform(-0.01, 0.01, sequence[0].shape)
    
    for frame in sequence:
        # Add jitter that's consistent across the sequence
        aug_frame = frame + offset
        # Add small per-frame jitter
        aug_frame += np.random.uniform(-0.005, 0.005, frame.shape)
        augmented_seq.append(aug_frame)
    
    return augmented_seq

def prob_viz(res, actions, input_frame, colors):
    """
    Visualize the prediction probabilities
    """
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60+num*40), (int(prob*100), 90+num*40), colors[num % len(colors)], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame

# 2. Create data collection and model training functions
def test_detection():
    """
    Test the MediaPipe detection
    """
    cap = cv2.VideoCapture(0)
    
    # Set MediaPipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture video")
                break
            
            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            
            # Draw landmarks
            draw_styled_landmarks(image, results)
            
            # Show to screen
            cv2.imshow('OpenCV Feed', image)
            
            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
def collect_data(actions, no_sequences=30, sequence_length=30, start_folder=0):
    """
    Collect and save keypoint data for model training
    """
    # Path for exported data, numpy arrays
    DATA_PATH = os.path.join('MP_Data')
    
    # Create directories
    for action in actions:
        # Check if directory exists
        action_dir = os.path.join(DATA_PATH, action)
        if not os.path.exists(action_dir):
            os.makedirs(action_dir)
            
        # Get maximum folder number or start at 0
        try:
            dirmax = np.max(np.array(os.listdir(action_dir)).astype(int)) if os.listdir(action_dir) else -1
        except:
            dirmax = -1
            
        # Create sequence folders
        for sequence in range(start_folder, start_folder + no_sequences):
            seq_dir = os.path.join(action_dir, str(sequence))
            if not os.path.exists(seq_dir):
                os.makedirs(seq_dir)
    
    # Capture video feed
    cap = cv2.VideoCapture(0)
    
    # Set MediaPipe model
    with mp_holistic.Holistic(
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5,
        model_complexity=2  # Higher model complexity for better accuracy
    ) as holistic:
        # Loop through actions
        for action in actions:
            # Loop through sequences
            for sequence in range(start_folder, start_folder + no_sequences):
                # Loop through video length (sequence length)
                for frame_num in range(sequence_length):
                    # Read feed
                    ret, frame = cap.read()
                    if not ret:
                        print("Failed to capture video")
                        break
                    
                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)
                    
                    # Draw landmarks
                    draw_styled_landmarks(image, results)
                    
                    # Display collection progress
                    if frame_num == 0:
                        cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15, 12),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', image)
                        cv2.waitKey(2000)  # Wait for 2 seconds to prepare
                    else:
                        cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15, 12),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', image)
                    
                    # Export keypoints
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)
                    
                    # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        return
        
        cap.release()
        cv2.destroyAllWindows()

def preprocess_data(actions, sequence_length=30, augmentation=True, balance_classes=True):
    """
    Preprocess data for training with augmentation and class balancing
    """
    # Path for exported data, numpy arrays
    DATA_PATH = os.path.join('MP_Data')
    
    # Map labels to numbers
    label_map = {label: num for num, label in enumerate(actions)}
    
    # Collect sequences and labels
    sequences, labels = [], []
    
    # Count samples per class for balancing
    class_counts = {action: 0 for action in actions}
    
    for action in actions:
        action_dir = os.path.join(DATA_PATH, action)
        if not os.path.exists(action_dir):
            print(f"Directory {action_dir} does not exist")
            continue
            
        for sequence in os.listdir(action_dir):
            sequence_dir = os.path.join(action_dir, sequence)
            if os.path.isdir(sequence_dir):
                window = []
                for frame_num in range(sequence_length):
                    frame_path = os.path.join(sequence_dir, f"{frame_num}.npy")
                    if os.path.exists(frame_path):
                        res = np.load(frame_path)
                        window.append(res)
                
                if len(window) == sequence_length:  # Only add complete sequences
                    sequences.append(window)
                    labels.append(label_map[action])
                    class_counts[action] += 1
    
    print("Class distribution before augmentation:")
    for action, count in class_counts.items():
        print(f"{action}: {count} sequences")
    
    # Apply data augmentation if enabled
    if augmentation:
        # Find max count for balancing
        if balance_classes:
            max_count = max(class_counts.values())
            
            # Generate augmented data for underrepresented classes
            augmented_sequences = []
            augmented_labels = []
            
            for action in actions:
                action_sequences = [sequences[i] for i in range(len(sequences)) if labels[i] == label_map[action]]
                current_count = class_counts[action]
                
                # Calculate how many augmented samples we need
                needed = max_count - current_count
                
                if needed > 0:
                    print(f"Augmenting {action} with {needed} new sequences")
                    
                    # Generate new augmented sequences
                    for _ in range(needed):
                        # Randomly select a sequence to augment
                        seq_to_augment = random.choice(action_sequences)
                        # Apply augmentation
                        aug_seq = augment_sequence(seq_to_augment)
                        
                        augmented_sequences.append(aug_seq)
                        augmented_labels.append(label_map[action])
            
            # Add augmented data
            sequences.extend(augmented_sequences)
            labels.extend(augmented_labels)
    
    # Convert to numpy arrays and one-hot encode labels
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, stratify=y)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def build_advanced_model(input_shape, num_classes):
    """
    Build an advanced model with CNN features, BiLSTM, and attention mechanism
    """
    # Input layer
    inputs = Input(shape=input_shape)
    
    # 1D CNN for feature extraction
    x = TimeDistributed(Conv1D(64, 3, activation='relu', padding='same'))(inputs)
    x = TimeDistributed(Conv1D(128, 3, activation='relu', padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(GlobalAveragePooling1D())(x)
    
    # Bidirectional LSTM layers
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(x)
    x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2))(x)
    
    # Attention mechanism
    attention = Attention()([x, x])
    
    # Concatenate attention output with LSTM output
    x = Concatenate()([x, attention])
    
    # Final LSTM and Dense layers
    x = Bidirectional(LSTM(32, return_sequences=False))(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model with a lower learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(
        optimizer=optimizer, 
        loss='categorical_crossentropy', 
        metrics=['categorical_accuracy']
    )
    
    return model

def build_and_train_model(X_train, y_train, X_test, y_test, actions, epochs=200):
    """
    Build and train advanced model with callbacks
    """
    # Create log directory
    log_dir = os.path.join('Logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create model directory
    model_dir = os.path.join('Models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Define callbacks
    tb_callback = TensorBoard(log_dir=log_dir)
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    
    # Model checkpoint to save best model
    model_checkpoint = ModelCheckpoint(
        os.path.join(model_dir, 'best_model.h5'),
        monitor='val_categorical_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    # Reduce learning rate when learning plateaus
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=0.00001,
        verbose=1
    )
    
    # Calculate class weights to handle class imbalance
    y_integers = np.argmax(y_train, axis=1)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_integers),
        y=y_integers
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    # Build advanced model
    model = build_advanced_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        num_classes=actions.shape[0]
    )
    
    # Print model summary
    model.summary()
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=16,
        callbacks=[tb_callback, early_stopping, model_checkpoint, reduce_lr],
        class_weight=class_weight_dict,
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    # Save model
    model.save('action_advanced.h5')
    
    return model, history

def evaluate_model(model, X_test, y_test, actions):
    """
    Evaluate model performance with detailed metrics
    """
    # Make predictions
    yhat = model.predict(X_test)
    
    # Convert predictions to class indices
    ytrue = np.argmax(y_test, axis=1).tolist()
    yhat = np.argmax(yhat, axis=1).tolist()
    
    # Compute confusion matrix and accuracy
    conf_matrix = multilabel_confusion_matrix(ytrue, yhat)
    acc = accuracy_score(ytrue, yhat)
    
    # Generate classification report
    class_report = classification_report(
        ytrue, 
        yhat, 
        target_names=actions,
        output_dict=True
    )
    
    # Convert report to dataframe for better visualization
    report_df = pd.DataFrame(class_report).transpose()
    
    print("Model Accuracy:", acc)
    print("\nClassification Report:")
    print(pd.DataFrame(class_report).transpose())
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    for i, (cm, action) in enumerate(zip(conf_matrix, actions)):
        plt.subplot(5, 5 if len(actions) > 10 else 3, i+1)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix: {action}')
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('True')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    plt.close()
    
    return conf_matrix, acc, report_df

def detect_action_sequence(model, actions, threshold=0.7, sequence_length=30):
    """
    Improved real-time prediction with sequence-based detection and feedback
    """
    # Colors for visualization (extended for more classes)
    colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245), 
              (255, 0, 0), (0, 255, 0), (0, 0, 255),
              (255, 255, 0), (255, 0, 255), (0, 255, 255),
              (128, 0, 0), (0, 128, 0), (0, 0, 128),
              (128, 128, 0), (128, 0, 128), (0, 128, 128),
              (192, 192, 192), (128, 128, 128), (64, 64, 64),
              (255, 128, 0), (255, 0, 128), (128, 255, 0),
              (0, 255, 128), (0, 128, 255), (128, 0, 255),
              (255, 255, 255)]
    
    # Detection variables
    sequence = []
    sentence = []
    predictions = []
    confidence_threshold = threshold
    
    cap = cv2.VideoCapture(0)
    
    # Set MediaPipe model
    with mp_holistic.Holistic(
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5,
        model_complexity=2  # Higher complexity for better accuracy
    ) as holistic:
        while cap.isOpened():
            # Read feed
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture video")
                break
            
            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            
            # Draw landmarks
            draw_styled_landmarks(image, results)
            
            # Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-sequence_length:]  # Keep only last N frames
            
            if len(sequence) == sequence_length:
                # Make prediction
                input_data = np.expand_dims(sequence, axis=0)
                res = model.predict(input_data)[0]
                predicted_action = actions[np.argmax(res)]
                confidence = res[np.argmax(res)]
                
                # Add prediction with confidence check
                if confidence > confidence_threshold:
                    predictions.append(np.argmax(res))
                
                # Visualization logic - require consistent predictions
                if len(predictions) > 10:
                    # Check if we have consistent predictions (mode of last N predictions)
                    mode_pred, count = stats.mode(predictions[-10:])
                    
                    # If prediction is consistent and confident
                    if count[0] > 8:  # At least 8 out of 10 predictions are the same
                        # Check if this is a new detection
                        if len(sentence) == 0 or actions[mode_pred[0]] != sentence[-1]:
                            sentence.append(actions[mode_pred[0]])
                            # Provide visual feedback for new detection
                            cv2.rectangle(image, (0, 0), (640, 40), (0, 255, 0), -1)
                
                # Limit sentence length
                if len(sentence) > 5:
                    sentence = sentence[-5:]  # Keep only last 5 predictions
                
                # Visualize probabilities
                image = prob_viz(res, actions, image, colors)
            
            # Display detected signs
            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Display confidence threshold
            cv2.putText(image, f"Threshold: {confidence_threshold:.2f}", (500, 450),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Instructions for adjusting threshold
            cv2.putText(image, "Press + to increase threshold", (10, 450),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, "Press - to decrease threshold", (10, 470),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Show to screen
            cv2.imshow('Sign Language Recognition', image)
            
            # Key handling
            key = cv2.waitKey(10) & 0xFF
            
            # Break gracefully with q
            if key == ord('q'):
                break
            
            # Adjust threshold with + and -
            elif key == ord('+') or key == ord('='):
                confidence_threshold = min(0.95, confidence_threshold + 0.05)
                print(f"Threshold increased to {confidence_threshold:.2f}")
            
            elif key == ord('-') or key == ord('_'):
                confidence_threshold = max(0.5, confidence_threshold - 0.05)
                print(f"Threshold decreased to {confidence_threshold:.2f}")
        
        cap.release()
        cv2.destroyAllWindows()


def realtime_prediction(model, actions):
    """
    Make real-time predictions with voting and temporal smoothing
    """
    # Start detection sequence
    detect_action_sequence(model, actions)

# 3. Main execution function
def main():
    # Extended list of sign language actions
    actions = np.array([
        'hello', 'thanks', 'iloveyou', 'please', 'sorry',
        'yes', 'no', 'help', 'water', 'food',
        'bathroom', 'stop', 'go', 'good', 'bad',
        'name', 'home', 'work', 'school', 'family',
        'friend', 'time', 'what', 'where', 'how'
    ])
    
    # Ask user what they want to do
    print("Advanced Sign Language Recognition System")
    print("1. Test MediaPipe detection")
    print("2. Collect training data")
    print("3. Train model")
    print("4. Run real-time detection")
    print("5. Evaluate model performance")
    choice = input("Enter choice (1-5): ")
    
    if choice == '1':
        # Test detection
        test_detection()

    elif choice == '2':
        # Collect data
        no_sequences = int(input("Enter number of sequences to collect per action (default 30): ") or 30)
        start_folder = int(input("Enter starting folder number (default 0): ") or 0)
        
        # Select actions to collect data for
        print("Available actions:")
        for i, action in enumerate(actions):
            print(f"{i+1}. {action}")
        
        action_indices = input("Enter action numbers to collect data for (comma-separated, or 'all'): ")
        
        if action_indices.lower() == 'all':
            selected_actions = actions
        else:
            # Parse selected action indices
            try:
                indices = [int(idx.strip()) for idx in action_indices.split(',')]
                selected_actions = actions[np.array(indices) - 1]
            except:
                print("Invalid input. Collecting data for all actions.")
                selected_actions = actions
        
        print(f"Collecting data for: {', '.join(selected_actions)}")
        collect_data(selected_actions, no_sequences, 30, start_folder)
    
    elif choice == '3':
        # Train model
        print("Preprocessing data...")
        X_train, X_test, y_train, y_test = preprocess_data(
            actions, 
            sequence_length=30, 
            augmentation=True, 
            balance_classes=True
        )
        
        epochs = int(input("Enter number of epochs (default 200): ") or 200)
        
        print("Building and training model...")
        model, history = build_and_train_model(X_train, y_train, X_test, y_test, actions, epochs)
        
# Evaluate model
        print("Evaluating model...")
        conf_matrix, acc, report = evaluate_model(model, X_test, y_test, actions)
        print(f"Model accuracy: {acc}")
    
    elif choice == '4':
        # Load model
        model_path = input("Enter model path (default: action_advanced.h5): ") or "action_advanced.h5"
        
        try:
            # Load model with custom objects for attention layer
            model = load_model(model_path)
            # Run real-time detection
            realtime_prediction(model, actions)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Train the model first or provide a valid model path.")
    
    elif choice == '5':
        # Load model for evaluation
        model_path = input("Enter model path (default: action_advanced.h5): ") or "action_advanced.h5"
        
        try:
            # Load model
            model = load_model(model_path)
            
            # Preprocess data for evaluation
            print("Preprocessing data for evaluation...")
            X_train, X_test, y_train, y_test = preprocess_data(
                actions, 
                sequence_length=30, 
                augmentation=False,  # No augmentation for evaluation
                balance_classes=False
            )
            
            # Evaluate model
            print("Evaluating model...")
            conf_matrix, acc, report = evaluate_model(model, X_test, y_test, actions)
            
            # Print detailed per-class metrics
            print("\nDetailed per-class metrics:")
            for action in actions:
                precision = report.loc[action, 'precision']
                recall = report.loc[action, 'recall']
                f1 = report.loc[action, 'f1-score']
                support = report.loc[action, 'support']
                print(f"{action}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, Samples={int(support)}")
        
        except Exception as e:
            print(f"Error during evaluation: {e}")
    
    else:
        print("Invalid choice")

def predict_single_sequence(model, actions, sequence_data):
    """
    Make prediction on a single pre-recorded sequence
    Useful for debugging and model validation
    """
    # Ensure sequence is the right shape
    if len(sequence_data) != 30:  # Assuming 30 frames
        print(f"Warning: Expected 30 frames, got {len(sequence_data)}. Padding or truncating.")
        # Pad or truncate to 30 frames
        if len(sequence_data) < 30:
            # Pad with the last frame repeated
            last_frame = sequence_data[-1]
            while len(sequence_data) < 30:
                sequence_data.append(last_frame)
        else:
            # Truncate to 30 frames
            sequence_data = sequence_data[:30]
    
    # Prepare input for model
    input_data = np.expand_dims(np.array(sequence_data), axis=0)
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Get top 3 predictions
    top_3_indices = prediction.argsort()[-3:][::-1]
    top_3_actions = [(actions[i], prediction[i]) for i in top_3_indices]
    
    # Print results
    print("Top predictions:")
    for action, confidence in top_3_actions:
        print(f"{action}: {confidence:.4f}")
    
    return top_3_actions

def benchmark_model(model, X_test, y_test, actions):
    """
    Benchmark model performance including speed metrics
    """
    # Measure inference time
    start_time = time.time()
    predictions = model.predict(X_test)
    end_time = time.time()
    
    # Calculate inference time
    total_time = end_time - start_time
    average_time = total_time / len(X_test)
    fps = 1.0 / average_time
    
    # Calculate accuracy and other metrics
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=actions)
    
    # Print results
    print(f"Model Performance Benchmark:")
    print(f"Total inference time: {total_time:.2f} seconds")
    print(f"Average inference time per sequence: {average_time*1000:.2f} ms")
    print(f"Effective FPS: {fps:.2f}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    # Calculate confusion matrix and visualize
    cm = multilabel_confusion_matrix(y_true, y_pred)
    
    # Return metrics as dictionary
    metrics = {
        'accuracy': accuracy,
        'inference_time': average_time,
        'fps': fps,
        'confusion_matrix': cm
    }
    
    return metrics

def generate_augmented_data(actions, augmentation_factor=2):
    """
    Generate additional augmented data from existing dataset
    """
    # Path for exported data
    DATA_PATH = os.path.join('MP_Data')
    AUG_DATA_PATH = os.path.join('MP_Data_Augmented')
    
    # Create augmented data directory
    if not os.path.exists(AUG_DATA_PATH):
        os.makedirs(AUG_DATA_PATH)
    
    # Process each action
    for action in actions:
        action_dir = os.path.join(DATA_PATH, action)
        aug_action_dir = os.path.join(AUG_DATA_PATH, action)
        
        if not os.path.exists(action_dir):
            print(f"Directory {action_dir} does not exist")
            continue
        
        if not os.path.exists(aug_action_dir):
            os.makedirs(aug_action_dir)
        
        # Get existing sequences
        sequences = [d for d in os.listdir(action_dir) if os.path.isdir(os.path.join(action_dir, d))]
        
        # Generate augmented sequences
        aug_sequence_idx = len(sequences)
        
        for sequence in sequences:
            sequence_dir = os.path.join(action_dir, sequence)
            
            # Load sequence data
            frames = []
            for frame_num in range(30):  # Assuming 30 frames
                frame_path = os.path.join(sequence_dir, f"{frame_num}.npy")
                if os.path.exists(frame_path):
                    keypoints = np.load(frame_path)
                    frames.append(keypoints)
            
            if len(frames) != 30:
                print(f"Skipping incomplete sequence {sequence_dir}")
                continue
            
            # Generate augmented sequences
            for _ in range(augmentation_factor):
                aug_sequence = augment_sequence(frames)
                
                # Save augmented sequence
                aug_seq_dir = os.path.join(aug_action_dir, str(aug_sequence_idx))
                if not os.path.exists(aug_seq_dir):
                    os.makedirs(aug_seq_dir)
                
                for frame_num, frame_data in enumerate(aug_sequence):
                    np.save(os.path.join(aug_seq_dir, str(frame_num)), frame_data)
                
                aug_sequence_idx += 1
        
        print(f"Generated {augmentation_factor} augmented sequences for action '{action}'")
    
    print("Augmentation complete")

if __name__ == "__main__":
    main()