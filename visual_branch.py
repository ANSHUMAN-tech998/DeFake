import torch
import torch.nn as nn
import timm

class VisualModel(nn.Module):
    def __init__(self):
        super(VisualModel, self).__init__()
        # Load the EfficientViT-B0 variant that was downloaded
        self.backbone = timm.create_model('efficientvit_b0.r224_in1k', pretrained=True, num_classes=0)
        
        # --- THE FIX ---
        # Changed 256 to 1280 to match the model's actual output
        self.projection = nn.Linear(1280, 512)

    def forward(self, x):
        # x: [batch, 3, 224, 224]
        features = self.backbone(x) # This is now returning 1280 features
        return self.projection(features)