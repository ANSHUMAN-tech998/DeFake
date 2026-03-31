import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_loader import DeFakeDataset
from main_model import DeFakeFusionModel

# 1. TUNED HYPERPARAMETERS FOR RTX 4050
BATCH_SIZE = 4       # EfficientViT is heavier; keep batch size low
EPOCHS = 20          # Your requested 20-epoch sprint
LEARNING_RATE = 2e-5 # Low LR to preserve pre-trained EfficientViT weights
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    print(f"🚀 Training EfficientViT Fusion Model on: {DEVICE}")
    model = DeFakeFusionModel().to(DEVICE)
    dataset = DeFakeDataset(root_dir="processed_data")
    # Add 'drop_last=True'
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    criterion = nn.BCELoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        for i, (images, audio, bio, labels) in enumerate(train_loader):
            images, audio, bio, labels = images.to(DEVICE), audio.to(DEVICE), bio.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images, audio, bio).squeeze()
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}], Loss: {loss.item():.4f}")

        # Save the best weights for your demo
        avg_loss = epoch_loss / len(train_loader)
        torch.save(model.state_dict(), f"defake_efficientvit_v1.pth")
        print(f"✅ Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    train()