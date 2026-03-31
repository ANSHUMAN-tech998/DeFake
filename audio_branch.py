import torch
import torch.nn as nn

class AudioModel(nn.Module):
    def __init__(self):
        super(AudioModel, self).__init__()
        # Placeholder that mimics Wav2Vec output (768 features)
        self.feature_extractor = nn.Linear(32000, 768) 

    def forward(self, x):
        # x is the raw audio waveform
        x = x.view(x.size(0), -1) 
        return self.feature_extractor(x)