import os
import torch

# 1. Project Root
# This automatically finds the folder where this script is saved
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Dataset Paths (Update these names if your folders are different)
DATASETS = {
    "audio": os.path.join(BASE_DIR, "datasets", "asvspoof-2019-la-subset"),
    "audio_protocols": os.path.join(BASE_DIR, "datasets", "asvspoof-2019-la-subset", "ASVspoof2019_LA_cm_protocols"),
    "visual": os.path.join(BASE_DIR, "datasets", "deepfake-detection-challenge"),
    "visual_train": os.path.join(BASE_DIR, "datasets", "deepfake-detection-challenge", "train_sample_videos"),
    "biological": os.path.join(BASE_DIR, "datasets", "ubfc-2")
}

# 3. Hardware Configuration
# This ensures your RTX 4050 is always prioritized
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VRAM_LIMIT_GB = 6  # Reminder for your RTX 4050 capacity

# 4. Preprocessing Constants
SAMPLE_RATE_AUDIO = 16000  # Required for Wav2Vec 2.0
IMG_SIZE = (224, 224)  # Standard for EfficientViT
HEART_RATE_RANGE = (42, 240)  # Frequency range for rPPG (0.7 - 4.0 Hz)
