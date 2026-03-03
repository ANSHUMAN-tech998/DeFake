# Audio Resampling Script
# This script resamples audio files from the ASVspoof2019 dataset to 16kHz and saves them in a new directory.

import os
import librosa
import soundfile as sf
from config import DATASETS

def resample_all_audio():
    # Construct paths
    audio_path = os.path.join(DATASETS["audio"], "ASVspoof2019_LA_train", "flac")
    output_path = os.path.join("processed_data", "resampled_audio")
    os.makedirs(output_path, exist_ok=True)

    # Get every .flac file in the directory
    files = [f for f in os.listdir(audio_path) if f.endswith('.flac')]
    total_files = len(files)
    
    print(f"🚀 Starting mass resampling of {total_files} files to 16kHz...")

    for index, file in enumerate(files):
        input_file = os.path.join(audio_path, file)
        output_file = os.path.join(output_path, file.replace('.flac', '.wav'))
        
        # Skip if already processed (useful if script is interrupted)
        if os.path.exists(output_file):
            continue

        # Load and Resample
        # librosa.load with sr=16000 automatically handles the math
        y, _ = librosa.load(input_file, sr=16000)
        
        # Save as standard .wav for Wav2Vec 2.0 compatibility
        sf.write(output_file, y, 16000)
        
        # Show progress every 50 files so you know it hasn't frozen
        if (index + 1) % 50 == 0:
            print(f"Progress: {index + 1}/{total_files} files completed...")
    
    print(f"✅ COMPLETED! All audio saved to: {output_path}")

if __name__ == "__main__":
    resample_all_audio()