"""
EEG Model Training Script
Trains a CNN model for EEG classification
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, 
                                    Flatten, Dense, Dropout)
from sklearn.model_selection import train_test_split
from eeg_preprocessing import load_eeg_data, preprocess_eeg

def create_eeg_model(input_shape, num_classes):
    """Create 2D CNN model for EEG classification"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

def main():
    # Load and preprocess data
    print("Loading EEG data...")
    X, y = load_eeg_data('data/raw/eegmmidb/S001/S001R01.edf')
    X = preprocess_eeg(X)
    
    # Reshape for CNN (samples, height, width, channels)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # Create and train model
    print("Training model...")
    model = create_eeg_model(X_train.shape[1:], len(np.unique(y)))
    
    history = model.fit(X_train, y_train,
                       epochs=20,
                       batch_size=32,
                       validation_split=0.2)
    
    # Save model
    model.save('models/eeg_model.h5')
    print("Model saved to models/eeg_model.h5")
    
    # Evaluate
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()