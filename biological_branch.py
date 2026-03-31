import torch.nn as nn

class BioModel(nn.Module):
    def __init__(self):
        super(BioModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 75, 128) # 75 is 150 frames / 2 (pooling)
        )

    def forward(self, x):
        # x shape: (Batch, 150) -> (Batch, 1, 150)
        return self.conv_layers(x.unsqueeze(1))