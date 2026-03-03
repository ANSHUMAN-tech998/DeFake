import torch
import cv2
import librosa

print("--- DEFAKE-LIVE SYSTEM CHECK ---")
# Check GPU
if torch.cuda.is_available():
    print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
    print(f"✅ VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("❌ GPU NOT DETECTED - Check CUDA installation")

# Check Libraries
print(f"✅ OpenCV Version: {cv2.__version__}")
print(f"✅ Librosa Version: {librosa.__version__}")