import numpy as np
from tensorflow.keras.models import load_model
from ecg_preprocessing import load_ecg_data, preprocess_data
from utils.visualize import plot_training_history

def train_and_save_model():
    # Load and preprocess data
    signals, labels = load_ecg_data(['100', '101', '102'])
    X_train, X_test, y_train, y_test = preprocess_data(signals, labels)
    
    # Model architecture (similar to previous but more organized)
    model = Sequential([
        Conv1D(32, 5, activation='relu', input_shape=(200, 1)),
        MaxPooling1D(2),
        Conv1D(64, 3, activation='relu'),
        MaxPooling1D(2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    # Train model
    history = model.fit(X_train, y_train, 
                       epochs=15, 
                       validation_split=0.2)
    
    # Save model
    model.save('models/ecg_model.h5')
    
    # Visualize results
    plot_training_history(history)
    
    return model

if __name__ == "__main__":
    train_and_save_model()