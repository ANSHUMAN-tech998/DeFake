import os
import cv2
import torch
import pandas as pd
import torchaudio
from torch.utils.data import Dataset
from torchvision import transforms
from config import IMG_SIZE

class DeFakeDataset(Dataset):
    def __init__(self, root_dir, mapping_csv, is_training=True):
        self.root_dir = os.path.abspath(root_dir)
        self.is_training = is_training
        
        if not os.path.exists(mapping_csv):
            raise FileNotFoundError(f"❌ Mapping file not found: {mapping_csv}")
            
        self.samples = pd.read_csv(mapping_csv)
        print(f"📊 Loader: Linked {len(self.samples)} samples from {self.root_dir}")
        self.debug_count = 0 

        # --- HEAVY AUGMENTATION PIPELINE (The "Accuracy Booster") ---
        if self.is_training:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                # Spatial Variation: Forces model to find artifacts anywhere in the frame
                transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)), 
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                
                # Photometric Variation: Handles different lighting conditions
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
                transforms.RandomGrayscale(p=0.1),
                
                # Tensor Conversion
                transforms.ToTensor(),
                
                # Noise Injection: Simulates compression/sensor noise
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
                
                # Standardization: Essential for EfficientViT nodes
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            # Clean Validation: No random changes for testing
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        attempts = 0
        while attempts < len(self.samples):
            try:
                row = self.samples.iloc[idx]
                v_id = str(row['video_id']).strip()
                
                # --- 2. SMART VISUAL PATHING ---
                face_crops_root = os.path.join(self.root_dir, "face_crops")
                video_folder = os.path.join(face_crops_root, v_id)

                if self.debug_count < 5:
                    print(f"🔍 DEBUG: Attempting to load ID: {v_id}")
                    self.debug_count += 1

                if not os.path.exists(video_folder):
                    raise FileNotFoundError(f"Missing folder: {v_id}")

                # Strict Format Filtering
                all_files = os.listdir(video_folder)
                images = [f for f in all_files if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

                if not images:
                    raise FileNotFoundError(f"No valid images in {v_id}")

                img_path = os.path.join(video_folder, images[0])
                image = cv2.imread(img_path)
                if image is None:
                    raise ValueError(f"OpenCV read error: {img_path}")
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Apply Heavy Augmentation
                img_t = self.transform(image)

                # --- 3. AUDIO PATHING ---
                aud_path = os.path.join(self.root_dir, "resampled_audio", str(row['audio_file']))
                waveform, _ = torchaudio.load(aud_path)
                
                target_length = 32000
                if waveform.shape[1] < target_length:
                    waveform = torch.nn.functional.pad(waveform, (0, target_length - waveform.shape[1]))
                else:
                    waveform = waveform[:, :target_length]

                # --- 4. BIO PATHING ---
                sig_path = os.path.join(self.root_dir, "signals", str(row['signal_file']))
                sig_df = pd.read_csv(sig_path, header=None)
                sig_t = torch.tensor(sig_df.iloc[:, 0].values).float()
                
                if len(sig_t) < 150:
                    sig_t = torch.nn.functional.pad(sig_t, (0, 150 - len(sig_t)))
                else:
                    sig_t = sig_t[:150]

                return img_t, waveform, sig_t, torch.tensor(row['label']).float()

            except Exception as e:
                if attempts < 5:
                    print(f"⚠️ Skipping index {idx} due to: {str(e)}")
                
                idx = (idx + 1) % len(self.samples)
                attempts += 1
        
        raise RuntimeError("❌ ALL SAMPLES FAILED. Please check the paths printed above.")

