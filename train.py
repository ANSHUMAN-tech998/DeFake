import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_loader import DeFakeDataset
from main_model import DeFakeFusionModel
from config import TRAIN_ROOT, TEST_ROOT, DEVICE

# --- HYPERPARAMETERS ---
BATCH_SIZE = 8
EPOCHS = 100
INITIAL_LR = 1e-4  # Higher LR for the classifier phase
FINETUNE_LR = 1e-6 # Ultra-low LR for backbone fine-tuning
UNFREEZE_EPOCH = 30 # Switch to deep learning at this stage

def train():
    print(f"🚀 Starting High-Accuracy Run (Fine-Tuning Mode)")
    
    # Use is_training=True for augmentation on 120 samples, False for 72 test samples
    train_ds = DeFakeDataset(root_dir=TRAIN_ROOT, mapping_csv="train_mapping.csv", is_training=True)
    test_ds = DeFakeDataset(root_dir=TEST_ROOT, mapping_csv="test_mapping.csv", is_training=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    model = DeFakeFusionModel().to(DEVICE)
    
    # --- PHASE 1: FREEZE BACKBONES ---
    # We only train the Fusion Head (Classifier) initially
    print("🔒 Backbones Frozen: Training Fusion Head...")
    for param in model.visual_branch.parameters(): param.requires_grad = False
    for param in model.audio_branch.parameters(): param.requires_grad = False
    
    criterion = nn.BCELoss() 
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=INITIAL_LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)

    best_acc = 0.0

    for epoch in range(EPOCHS):
        # --- PHASE 2: UNFREEZE FOR DEEP FINE-TUNING ---
        if epoch == UNFREEZE_EPOCH:
            print("🔓 UNFREEZING BACKBONES: Commencing Deep Forensic Analysis...")
            for param in model.parameters():
                param.requires_grad = True
            
            # Re-initialize optimizer with ALL parameters and lower LR
            optimizer = optim.Adam(model.parameters(), lr=FINETUNE_LR)
            # Update scheduler to match new optimizer
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)

        model.train()
        train_acc = 0
        for images, audio, bio, labels in train_loader:
            images, audio, bio, labels = images.to(DEVICE), audio.to(DEVICE), bio.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images, audio, bio).squeeze() 
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_acc += model.calculate_accuracy(outputs, labels).item()

        # Validation Loop
        model.eval()
        test_acc = 0
        with torch.no_grad():
            for images, audio, bio, labels in test_loader:
                images, audio, bio, labels = images.to(DEVICE), audio.to(DEVICE), bio.to(DEVICE), labels.to(DEVICE)
                outputs = model(images, audio, bio).squeeze()
                test_acc += model.calculate_accuracy(outputs, labels).item()

        avg_train_acc = train_acc / len(train_loader)
        avg_test_acc = test_acc / len(test_loader)
        
        scheduler.step(avg_test_acc)

        mode = "Classifier" if epoch < UNFREEZE_EPOCH else "Deep Fine-Tune"
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Mode: {mode} | Train: {avg_train_acc:.2f}% | TEST: {avg_test_acc:.2f}%")

        if avg_test_acc > best_acc:
            best_acc = avg_test_acc
            torch.save(model.state_dict(), "defake_best_model.pth")
            print(f"✨ NEW BEST: {best_acc:.2f}% - Model Weights Updated!")

if __name__ == "__main__":
    train()