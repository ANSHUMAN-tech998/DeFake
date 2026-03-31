import os
import torch

# 1. Project Root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. NEW: Partitioned Dataset Paths (Faculty Requirement)
# These now point to your 60-40 split folders
TRAIN_ROOT = os.path.join(BASE_DIR, "processed_data", "Training_Set")
TEST_ROOT = os.path.join(BASE_DIR, "processed_data", "Testing_Set")

# Specific paths for your branches
DATA_PATHS = {
    "train_frames": os.path.join(TRAIN_ROOT, "frames"),
    "train_crops": os.path.join(TRAIN_ROOT, "face_crops"),
    "test_frames": os.path.join(TEST_ROOT, "frames"),
    "test_crops": os.path.join(TEST_ROOT, "face_crops"),
}

# Keep original raw datasets for reference
DATASETS = {
    "audio": os.path.join(BASE_DIR, "datasets", "asvspoof-2019-la-subset"),
    "visual": os.path.join(BASE_DIR, "datasets", "deepfake-detection-challenge"),
    "biological": os.path.join(BASE_DIR, "datasets", "ubfc-2")
}

# 3. Hardware Configuration (Optimized for RTX 4050)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VRAM_LIMIT_GB = 6  

# 4. Preprocessing Constants
SAMPLE_RATE_AUDIO = 16000  # Required for Wav2Vec 2.0
IMG_SIZE = (224, 224)      # Standard for EfficientViT
HEART_RATE_RANGE = (42, 240) # Frequency range for rPPG
