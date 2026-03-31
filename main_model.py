import torch
import torch.nn as nn
import sys

try:
    from visual_branch import VisualModel    # EfficientViT (1000 features)
    from audio_branch import AudioModel      # Wav2Vec 2.0 (256 features)
    from biological_branch import BioModel   # 1D-CNN (152 features)
except ImportError as e:
    print(f"❌ ARCHITECTURE ERROR: {e}")
    sys.exit(1)

class DeFakeFusionModel(nn.Module):
    def __init__(self):
        super(DeFakeFusionModel, self).__init__()
        
        # 1. Backbones: Specialized Forensic Experts
        self.visual_branch = VisualModel()       
        self.audio_branch = AudioModel()         
        self.bio_branch = BioModel()             
        
        # 2. FEATURE NORMALIZATION LAYER
        # This is critical for 1408 features because it ensures that one branch 
        # (like Visual with 1000 features) doesn't mathematically "drown out" 
        # the smaller Bio branch (152 features).
        self.input_bn = nn.BatchNorm1d(1408)

        # 3. MULTIMODAL FUSION HEAD (Refined for Better Convergence)
        # We use a 3-layer MLP with increasing dropout to prevent overfitting
        self.classifier = nn.Sequential(
            # First Layer: High-Dimensional Mapping
            nn.Linear(1408, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4), # Increased for smaller dataset stability
            
            # Second Layer: Refinement
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            # Final Layer: Sigmoid Decision Fusion
            nn.Linear(128, 1),
            nn.Sigmoid() 
        )

    def forward(self, img, aud, bio):
        # Extract features from each domain
        feat_v = self.visual_branch(img) # Shape: [Batch, 1000]
        feat_a = self.audio_branch(aud)  # Shape: [Batch, 256]
        feat_b = self.bio_branch(bio)   # Shape: [Batch, 152]
        
        # Concatenate into the 1408-dimensional Multimodal Vector
        combined = torch.cat((feat_v, feat_a, feat_b), dim=1)
        
        # Normalize the combined vector before classification
        combined = self.input_bn(combined)
        
        return self.classifier(combined)

    @staticmethod
    def calculate_accuracy(outputs, labels):
        # Logic: Logistic Regression Threshold (0.5)
        # Values > 0.5 are DEEPFAKE, <= 0.5 are AUTHENTIC
        predictions = (outputs > 0.5).float()
        correct = (predictions.squeeze() == labels).float().sum()
        return (correct / labels.size(0)) * 100

if __name__ == "__main__":
    print("✅ DeFake Pro Architecture: Multimodal Fusion Ready.")
    print("🚀 Features: Visual(1000) + Audio(256) + Bio(152) = 1408")