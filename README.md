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


## 📊 Datasets & Sources
To ensure forensic accuracy, this project leverages three industry-standard benchmarks for multimodal training and validation.

### 1. Deepfake Detection Challenge (DFDC)
Modality: Visual

Primary Use Case: Identifying spatial artifacts, GAN-generated faces, and facial manipulation.

Direct Link: https://www.kaggle.com/c/deepfake-detection-challenge/data

Authors: Brian Dolhansky, Joanna Bitton, Ben Pflaum, Jiliang Lu, Russ Howes, Menglin Wang, Cristian Canton Ferrer.

Reference: The Deepfake Detection Challenge (DFDC) Preview Dataset, arXiv:1910.08854, 2019.

### 2. ASVspoof 2019 (LA Subset)
Modality: Audio

Primary Use Case: Detection of synthetic speech, voice cloning, and logical access attacks.

Direct Link: https://www.asvspoof.org/

Authors: Andreas Nautsch, Jose Patino, Natalia Tomashenko, Junichi Yamagishi, Massimiliano Todisco, Nicholas Evans, Jean-François Bonastre.

Reference: ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech, Interspeech, 2019.

### 3. UBFC-rPPG
Modality: Biological

Primary Use Case: Establishing ground-truth for heartbeat verification and remote physiological liveness.

Direct Link: https://sites.google.com/view/ybenezeth/ubfcrppg

Authors: Serge Bobbia, Richard Macwan, Yannick Benezeth, Alamin Mansouri, Julien Dubois.

Reference: Unsupervised skin-based video-based heart rate estimation, ACM Trans. Multimedia Comput. Commun. Appl., 2019.


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

@article{dolhansky2020dfdc,
  title={The Deepfake Detection Challenge (DFDC) Dataset},
  author={Dolhansky, Brian and others},
  journal={arXiv preprint arXiv:2006.07397},
  year={2020}
}

@inproceedings{wang2019asvspoof,
  title={ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech},
  author={Wang, Xin and others},
  booktitle={Interspeech},
  year={2019}
}

@article{bobbia2019ubfc,
  title={Unsupervised skin-based video-based heart rate estimation},
  author={Bobbia, S. and others},
  journal={ACM Trans. Multimedia Comput. Commun. Appl.},
  year={2019}
}
