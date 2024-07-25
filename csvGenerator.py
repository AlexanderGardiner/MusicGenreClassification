import os
import librosa
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

def extract_features(file_path):
    try:
        print(file_path)
        y, sr = librosa.load(file_path, sr=None)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        harmony = librosa.effects.harmonic(y)
        perceptr = librosa.effects.percussive(y)
        tempo = librosa.beat.tempo(y=y, sr=sr)[0]
        
        features = {
            'filename': os.path.basename(file_path),
            'length': len(y),
            'chroma_stft_mean': np.mean(chroma_stft),
            'chroma_stft_var': np.var(chroma_stft),
            'rms_mean': np.mean(rms),
            'rms_var': np.var(rms),
            'spectral_centroid_mean': np.mean(spectral_centroid),
            'spectral_centroid_var': np.var(spectral_centroid),
            'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
            'spectral_bandwidth_var': np.var(spectral_bandwidth),
            'rolloff_mean': np.mean(rolloff),
            'rolloff_var': np.var(rolloff),
            'zero_crossing_rate_mean': np.mean(zero_crossing_rate),
            'zero_crossing_rate_var': np.var(zero_crossing_rate),
            'harmony_mean': np.mean(harmony),
            'harmony_var': np.var(harmony),
            'perceptr_mean': np.mean(perceptr),
            'perceptr_var': np.var(perceptr),
            'tempo': tempo
        }
        
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for i in range(1, 21):
            features[f'mfcc{i}_mean'] = np.mean(mfccs[i-1])
            features[f'mfcc{i}_var'] = np.var(mfccs[i-1])
        
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_file(file_path, label):
    features = extract_features(file_path)
    if features is not None:
        features['label'] = label
    return features

def process_folder(parent_folder):
    data = []
    with ProcessPoolExecutor() as executor:
        futures = []
        for category in os.listdir(parent_folder):
            category_path = os.path.join(parent_folder, category)
            if os.path.isdir(category_path):
                for file in os.listdir(category_path):
                    if file.endswith('.wav'):
                        file_path = os.path.join(category_path, file)
                        futures.append(executor.submit(process_file, file_path, category))
        
        for future in futures:
            result = future.result()
            if result is not None:
                data.append(result)
    
    df = pd.DataFrame(data)
    df.to_csv(parent_folder+'.csv', index=False)

if __name__ == '__main__':
    # Specify the parent folder containing subfolders of audio files
    parent_folder = './convolved_GTZAN_air_type1_air_binaural_aula_carolina_1_3_0_3'
    process_folder(parent_folder)
