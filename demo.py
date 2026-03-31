import torch
import os
import random
import sys
from main_model import DeFakeFusionModel
from dataset_loader import DeFakeDataset

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "defake_efficientvit_v1.pth"
DATA_ROOT = "processed_data"

def run_prediction_demo():
    print("🎬 --- DeFake: Multimodal Deepfake Detection System ---")
    print(f"🚀 Initializing inference on: {DEVICE}")
    
    # 1. Load the Integrated Model
    try:
        model = DeFakeFusionModel().to(DEVICE)
        
        if not os.path.exists(MODEL_PATH):
            print(f"❌ ERROR: Trained model weights not found at {MODEL_PATH}")
            print("👉 Please ensure you have run 'train_efficient.py' first.")
            return

        # FIXED: map_location correctly points to your RTX 4050
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval() 
        print(f"✅ Neural Network Weights Loaded Successfully.")
    except Exception as e:
        print(f"❌ Initialization Failed: {e}")
        return

    # 2. Initialize Data Loader
    try:
        dataset = DeFakeDataset(root_dir=DATA_ROOT)
    except Exception as e:
        print(f"❌ Dataset Error: {e}")
        return
    
    # Pick a random sample from your 125 samples for the demo
    sample_idx = random.randint(0, len(dataset) - 1)
    img, aud, bio, label = dataset[sample_idx]
    
    # 3. Prepare Tensors (Add batch dimension and move to GPU)
    img_t = img.unsqueeze(0).to(DEVICE)
    aud_t = aud.unsqueeze(0).to(DEVICE)
    bio_t = bio.unsqueeze(0).to(DEVICE)

    print(f"\n📡 ANALYZING SAMPLE #{sample_idx}...")
    
    
    with torch.no_grad():
        # The Forward Pass: Fusing Visual, Audio, and Bio Features
        prediction_prob = model(img_t, aud_t, bio_t).item()

    # 4. Result Logic (Logistic Regression Threshold)
    is_fake = prediction_prob > 0.5
    verdict = "🔴 FAKE (Deepfake Detected)" if is_fake else "🟢 REAL (Authentic)"
    
    # Confidence calculation
    confidence = prediction_prob if is_fake else (1 - prediction_prob)

    # 5. FINAL OUTPUT
    print(f"Result:         {verdict}")
    print(f"AI Score:       {prediction_prob:.4f}")
    print(f"Confidence:     {confidence * 100:.2f}%")
    print(f"Ground Truth:   {'Fake' if label == 1 else 'Real'}")
    
    # Accuracy check for presentation feedback
    if (is_fake and label == 1) or (not is_fake and label == 0):
        print("📊 Verification: MATCHED (AI successfully identified the sample)")
    else:
        print("📊 Verification: MISMATCHED (AI failed to identify correctly)")
    
    
    print("💡 Technical Note: This score is a fusion of EfficientViT spatial analysis,")
    print("   Wav2Vec 2.0 frequency fingerprints, and Biological liveness signals.")

if __name__ == "__main__":
    run_prediction_demo()