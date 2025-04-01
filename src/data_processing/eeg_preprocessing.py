import mne
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_eeg_data(file_path):
    """Load EEG data from EDF file"""
    raw = mne.io.read_raw_edf(file_path, preload=True)
    raw.filter(1, 40)  # Bandpass filter
    
    # Extract events if available
    events, _ = mne.events_from_annotations(raw)
    
    # Create epochs
    epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.5,
                       baseline=(-0.2, 0), preload=True)
    
    return epochs.get_data(), events[:, -1]  # Returns data and labels

def preprocess_eeg(eeg_data):
    """Normalize and prepare EEG data"""
    # Reshape: (epochs, channels, time) -> (epochs, time, channels)
    eeg_data = np.transpose(eeg_data, (0, 2, 1))
    
    # Normalize each channel separately
    scaler = StandardScaler()
    n_epochs, n_times, n_channels = eeg_data.shape
    eeg_data = scaler.fit_transform(
        eeg_data.reshape(-1, n_channels)).reshape(n_epochs, n_times, n_channels)
    
    return eeg_data