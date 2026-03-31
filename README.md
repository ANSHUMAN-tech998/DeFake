# DeFake Pro: Multimodal Forensic Analysis Lab

**DeFake Pro** is a state-of-the-art deepfake detection system built for the **KJSIT-IET INTECH 2K26** competition. It utilizes a "Triple-Threat" architecture to cross-verify digital media across Visual, Audio, and Biological domains.

## 🚀 Key Features
* **Multimodal Fusion:** Synchronized analysis of 1,408 unique forensic features.
* **Biological Liveness:** Uses rPPG (Remote Photoplethysmography) to detect human heartbeats.
* **High Accuracy:** Achieved **98.61% accuracy** on test datasets through staged training.
* **Adaptive Routing:** Automatically handles Video, Photo, or Audio-only uploads.

## 🛠️ Technical Architecture
The system employs three parallel expert backbones:
1.  **Visual Branch (EfficientViT-B0):** Extracts 1,000 spatial features to detect pixel-level GAN artifacts.
2.  **Audio Branch (Wav2Vec 2.0):** Extracts 256 spectral features to identify synthetic voice clones.
3.  **Biological Branch (1D-CNN):** Extracts 152 temporal features to verify natural human pulse rhythms.



## 📈 Training Strategy
Due to the high-fidelity nature of our 192-sample dataset, we implemented:
* **Heavy Stochastic Augmentation:** Virtually expanded 120 training samples into thousands of unique views using Albumentations.
* **Staged Optimization:** A two-phase training approach (Fusion Head stabilization followed by Full Backbone unlocking) using the **Adam Optimizer** and **ReduceLROnPlateau** scheduler.



## 💻 Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/defake-pro.git](https://github.com/yourusername/defake-pro.git)
   cd defake-pro


## 🚀 Setup Instructions

### 1. Project Initialization
```bash
# Create the environment
python -m venv venv

# Activate it (Windows)
.\venv\Scripts\activate

# Install all required libraries
pip install -r requirements.txt

streamlit run app.py


MINI PROJECT 2/
├── datasets/                 # Raw source datasets (DFDC, ASVspoof, UBFC)
├── processed_data/           # Cleaned frames, aligned faces, and 16kHz audio
├── venv/                     # Virtual environment dependencies
├── .gitignore                # Exclusion list for large weights and caches
│
├── 🧠 Core Architecture
│   ├── main_model.py         # The "Triple-Threat" Fusion Model (1408-D)
│   ├── audio_branch.py       # Wav2Vec 2.0 feature extraction logic
│   ├── biological_branch.py   # 1D-CNN rPPG liveness detection logic
│   └── visual_branch.py      # EfficientViT-B0 spatial analysis logic
│
├── ⚙️ Preprocessing & Alignment
│   ├── face_alignment.py     # MTCNN/MediaPipe facial normalization
│   ├── audio_resampling.py   # Standardizing audio to 16kHz Mono
│   ├── signal_extraction.py  # POS Algorithm for biological pulse mapping
│   ├── visual_extraction.py  # Automated frame-by-frame dissection
│   └── *_prep.py / *_prep.py # Specialized cleaning scripts for each modality
│
├── 📈 Training & Optimization
│   ├── train.py              # Main entry point for Staged Training
│   ├── train_efficient.py    # Fine-tuning the visual backbone
│   ├── dataset_loader.py     # Custom PyTorch DataLoader for multimodal triplets
│   ├── config.py             # Hyperparameters (LR=10^-4, Dropout=0.4, etc.)
│   └── master_mapping.csv    # Central registry for train/test labels
│
├── 🛡️ Model Weights (.pth)
│   ├── defake_best_model.pth # The 98.61% accuracy production weights
│   ├── defake_efficientvit_v1.pth
│   └── defake_model_epoch_1-10.pth # Checkpoints from the training journey
│
└── 🚀 Deployment
    ├── app.py                # Streamlit Forensic Dashboard UI
    ├── demo.py               # Lightweight CLI inference script
    └── requirements.txt      # Comprehensive dependency list