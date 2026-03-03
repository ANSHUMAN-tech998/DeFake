# DeFake: Multimodal Deepfake Detection


## 🚀 Setup Instructions

### 1. Project Initialization
```bash
# Create the environment
python -m venv venv

# Activate it (Windows)
.\venv\Scripts\activate

# Install all required libraries
pip install -r requirements.txt


Videos → visual_extraction.py → Folders of Raw Frames.

Raw Frames → face_alignment.py → Clean 224x224 Face Images.

Face Images → signal_extraction.py → CSV Pulse Data.

Audio Files → audio_resampling.py → 16kHz Sound Files.



Mini Project 2/
├── datasets/
│   ├── asvspoof-2019-la-subset/ (Unzip ASVspoof here)
│   ├── deepfake-detection-challenge/ (Unzip DFDC sample here)
│   └── ubfc-2/ (Unzip UBFC-rPPG here)
├── config.py
└── ... (your scripts)