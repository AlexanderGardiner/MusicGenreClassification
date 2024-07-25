import os
from pathlib import Path
import numpy as np
import librosa
from scipy.signal import convolve
import soundfile as sf

def convolve_audio(file_path, impulse_path):
    try:
        print(file_path)
        # Load the audio files
        audio1, sr1 = librosa.load(file_path, sr=None)
        audio2, sr2 = librosa.load(impulse_path, sr=None)
        audio2_resampled = librosa.resample(audio2, orig_sr=sr2, target_sr=sr1)
        # Convolve the two audio signals
        result = convolve(audio1, audio2_resampled)

        impulse_path = Path(impulse_path)
        last_folder = impulse_path.name
        # Save the result to a new WAV file
        output_path = replace_first_folder(file_path, "convolved_GTZAN_"+last_folder[:-4])
        # Create the output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        sf.write(output_path, result, sr1)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def replace_first_folder(filepath, new_folder):
    # Split the path into components
    parts = filepath.split(os.sep)
    
    # Check if there are enough parts to replace the folder
    if len(parts) < 2:
        return filepath  # Not enough parts to replace
    
    # Replace the first folder with the new folder
    parts[0] = new_folder
    
    # Reassemble the path
    new_path = os.sep.join(parts)
    
    return new_path

def process_folder(parent_folder, impulse_path):
    for category in os.listdir(parent_folder):
        category_path = os.path.join(parent_folder, category)
        if os.path.isdir(category_path):
            for file in os.listdir(category_path):
                if file.endswith('.wav'):
                    file_path = os.path.join(category_path, file)
                    convolve_audio(file_path, impulse_path)

if __name__ == '__main__':
    # Specify the parent folder containing subfolders of audio files
    parent_folder = './GTZAN/genres_original'
    process_folder(parent_folder, "real_rirs_isotropic_noises/air_type1_air_phone_stairway1_hfrp.wav")
