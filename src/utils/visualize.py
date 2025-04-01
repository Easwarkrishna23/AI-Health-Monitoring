import matplotlib.pyplot as plt

def plot_training_history(history):
    """Plot training and validation accuracy/loss"""
    plt.figure(figsize=(12, 4))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('reports/figures/training_history.png')
    plt.close()

def plot_ecg_beat(signal, label):
    """Plot single ECG beat with annotation"""
    plt.figure(figsize=(10, 4))
    plt.plot(signal)
    plt.title(f"ECG Beat - {'Normal' if label == 0 else 'Abnormal'}")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.savefig(f'reports/figures/ecg_beat_{label}.png')
    plt.close()