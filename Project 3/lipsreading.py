#!/usr/bin/env python3
"""
Lipreading to Text Converter
A complete system for converting lip movements to text using computer vision and deep learning.
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import dlib
import os
import pickle
from collections import deque
import threading
import queue
import time
from sklearn.preprocessing import LabelEncoder
import mediapipe as mp

class LipExtractor:
    """Extract lip region and features from video frames"""
    
    def __init__(self):
        # Initialize face detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # Lip landmark indices for MediaPipe
        self.lip_indices = [
            # Outer lip
            61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318,
            # Inner lip  
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324
        ]
        
    def extract_lip_region(self, frame):
        """Extract lip region from frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
            
        landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]
        
        # Get lip landmarks
        lip_points = []
        for idx in self.lip_indices:
            x = int(landmarks.landmark[idx].x * w)
            y = int(landmarks.landmark[idx].y * h)
            lip_points.append([x, y])
        
        lip_points = np.array(lip_points)
        
        # Create bounding box around lips
        x_min, y_min = np.min(lip_points, axis=0)
        x_max, y_max = np.max(lip_points, axis=0)
        
        # Add padding
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        # Extract lip region
        lip_region = frame[y_min:y_max, x_min:x_max]
        
        if lip_region.size == 0:
            return None
            
        # Resize to standard size
        lip_region = cv2.resize(lip_region, (64, 64))
        
        return lip_region, lip_points
    
    def extract_lip_features(self, lip_region):
        """Extract features from lip region"""
        if lip_region is None:
            return np.zeros((64, 64, 1))
            
        # Convert to grayscale
        gray = cv2.cvtColor(lip_region, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Normalize
        gray = gray.astype(np.float32) / 255.0
        
        return np.expand_dims(gray, axis=-1)

class LipreadingModel:
    """Neural network model for lipreading"""
    
    def __init__(self, sequence_length=30, vocab_size=1000):
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.model = None
        self.label_encoder = LabelEncoder()
        
    def build_model(self):
        """Build the lipreading neural network"""
        # Input for lip image sequences
        video_input = keras.Input(shape=(self.sequence_length, 64, 64, 1), name='video_input')
        
        # 3D CNN for spatial-temporal feature extraction
        x = layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu'))(video_input)
        x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
        x = layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation='relu'))(x)
        x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
        x = layers.TimeDistributed(layers.Conv2D(128, (3, 3), activation='relu'))(x)
        x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
        
        # LSTM for sequence modeling
        x = layers.LSTM(256, return_sequences=True, dropout=0.2)(x)
        x = layers.LSTM(128, dropout=0.2)(x)
        
        # Dense layers for classification
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        
        # Output layer for character prediction
        output = layers.Dense(self.vocab_size, activation='softmax', name='character_output')(x)
        
        self.model = keras.Model(inputs=video_input, outputs=output)
        
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train_model(self, X_train, y_train, X_val=None, y_val=None, epochs=50):
        """Train the lipreading model"""
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=16,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict_sequence(self, sequence):
        """Predict text from lip movement sequence"""
        if len(sequence.shape) == 4:
            sequence = np.expand_dims(sequence, axis=0)
            
        predictions = self.model.predict(sequence, verbose=0)
        predicted_chars = np.argmax(predictions, axis=-1)
        
        return predicted_chars

class LipreadingSystem:
    """Complete lipreading system"""
    
    def __init__(self, model_path=None):
        self.lip_extractor = LipExtractor()
        self.model = LipreadingModel()
        self.sequence_buffer = deque(maxlen=30)
        self.vocab = self._create_vocabulary()
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocab)}
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.model.build_model()
    
    def _create_vocabulary(self):
        """Create character vocabulary"""
        # Basic English alphabet + space + common punctuation
        vocab = list(' abcdefghijklmnopqrstuvwxyz.,!?')
        vocab.append('<UNK>')  # Unknown character
        vocab.append('<PAD>')  # Padding character
        return vocab
    
    def process_video_file(self, video_path):
        """Process video file and extract lip sequences"""
        cap = cv2.VideoCapture(video_path)
        sequences = []
        current_sequence = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            lip_data = self.lip_extractor.extract_lip_region(frame)
            if lip_data is not None:
                lip_region, _ = lip_data
                lip_features = self.lip_extractor.extract_lip_features(lip_region)
                current_sequence.append(lip_features)
                
                # Create sequences of fixed length
                if len(current_sequence) == 30:
                    sequences.append(np.array(current_sequence))
                    current_sequence = current_sequence[15:]  # Overlap sequences
        
        cap.release()
        return np.array(sequences) if sequences else None
    
    def predict_from_video(self, video_path):
        """Predict text from video file"""
        sequences = self.process_video_file(video_path)
        if sequences is None:
            return "No lip movements detected"
        
        predictions = []
        for sequence in sequences:
            pred_chars = self.model.predict_sequence(sequence)
            text = ''.join([self.idx_to_char.get(idx, '') for idx in pred_chars[0]])
            predictions.append(text)
        
        # Combine predictions and clean up
        full_text = ' '.join(predictions)
        return self._clean_text(full_text)
    
    def real_time_prediction(self):
        """Real-time lipreading from webcam"""
        cap = cv2.VideoCapture(0)
        
        print("Starting real-time lipreading. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract lip region
            lip_data = self.lip_extractor.extract_lip_region(frame)
            if lip_data is not None:
                lip_region, lip_points = lip_data
                lip_features = self.lip_extractor.extract_lip_features(lip_region)
                
                # Add to sequence buffer
                self.sequence_buffer.append(lip_features)
                
                # Display lip region
                cv2.imshow('Lips', cv2.resize(lip_region, (200, 200)))
                
                # Predict when buffer is full
                if len(self.sequence_buffer) == 30:
                    sequence = np.array(list(self.sequence_buffer))
                    pred_chars = self.model.predict_sequence(sequence)
                    text = ''.join([self.idx_to_char.get(idx, '') for idx in pred_chars[0]])
                    print(f"Predicted: {self._clean_text(text)}")
            
            # Display original frame
            cv2.imshow('Lipreading', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def _clean_text(self, text):
        """Clean and format predicted text"""
        # Remove padding and unknown tokens
        text = text.replace('<PAD>', '').replace('<UNK>', '')
        
        # Remove excessive spaces
        text = ' '.join(text.split())
        
        return text.strip()
    
    def save_model(self, path):
        """Save the trained model"""
        self.model.model.save(path)
        
        # Save vocabulary
        with open(f"{path}_vocab.pkl", 'wb') as f:
            pickle.dump({
                'vocab': self.vocab,
                'char_to_idx': self.char_to_idx,
                'idx_to_char': self.idx_to_char
            }, f)
    
    def load_model(self, path):
        """Load a trained model"""
        self.model.model = keras.models.load_model(path)
        
        # Load vocabulary
        vocab_path = f"{path}_vocab.pkl"
        if os.path.exists(vocab_path):
            with open(vocab_path, 'rb') as f:
                vocab_data = pickle.load(f)
                self.vocab = vocab_data['vocab']
                self.char_to_idx = vocab_data['char_to_idx']
                self.idx_to_char = vocab_data['idx_to_char']

def create_synthetic_data(num_samples=1000):
    """Create synthetic training data for demonstration"""
    print("Creating synthetic training data...")
    
    # Generate random lip movement sequences
    X = np.random.random((num_samples, 30, 64, 64, 1))
    
    # Generate random character labels (for demonstration)
    y = np.random.randint(0, 29, (num_samples,))
    
    return X, y

def main():
    """Main function to demonstrate the lipreading system"""
    print("Lipreading System Initialization...")
    
    # Initialize the system
    system = LipreadingSystem()
    
    # Create synthetic data for demonstration
    print("Generating synthetic training data...")
    X_train, y_train = create_synthetic_data(1000)
    X_val, y_val = create_synthetic_data(200)
    
    # Train the model (uncomment to train)
    print("Training model...")
    # system.model.train_model(X_train, y_train, X_val, y_val, epochs=5)
    
    print("\nLipreading System Ready!")
    print("\nAvailable options:")
    print("1. Real-time lipreading from webcam")
    print("2. Process video file")
    print("3. Train model with custom data")
    
    while True:
        choice = input("\nEnter your choice (1-3) or 'q' to quit: ").strip()
        
        if choice == '1':
            try:
                system.real_time_prediction()
            except KeyboardInterrupt:
                print("\nReal-time prediction stopped.")
        
        elif choice == '2':
            video_path = input("Enter video file path: ").strip()
            if os.path.exists(video_path):
                result = system.predict_from_video(video_path)
                print(f"Predicted text: {result}")
            else:
                print("Video file not found!")
        
        elif choice == '3':
            print("Training with synthetic data...")
            system.model.train_model(X_train, y_train, X_val, y_val, epochs=5)
            
            save_path = input("Enter path to save model (optional): ").strip()
            if save_path:
                system.save_model(save_path)
                print(f"Model saved to {save_path}")
        
        elif choice.lower() == 'q':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()