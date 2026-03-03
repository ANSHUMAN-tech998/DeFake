import cv2
import os
import numpy as np
import pandas as pd

def extract_green_signal():
    # Paths from your Week 2 setup
    input_root = os.path.join("processed_data", "face_crops")
    output_root = os.path.join("processed_data", "signals")
    
    # Safety Check: Ensure the crops exist
    if not os.path.exists(input_root):
        print(f"❌ Error: {input_root} not found. Please run face_alignment.py first!")
        return

    os.makedirs(output_root, exist_ok=True)

    # Get all video folders (e.g., 'aagfhgt.mp4')
    video_folders = [f for f in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, f))]
    
    print(f"🚀 Starting Signal Extraction for {len(video_folders)} videos...")

    for folder in video_folders:
        folder_path = os.path.join(input_root, folder)
        green_values = []
        
        # Sort files (frame_0.jpg, frame_1.jpg...) to maintain time order
        frames = sorted(os.listdir(folder_path), key=lambda x: int(x.split('_')[1].split('.')[0]))
        
        for frame_file in frames:
            img = cv2.imread(os.path.join(folder_path, frame_file))
            if img is not None:
                # OpenCV uses BGR format. Index 1 is the Green Channel.
                # We calculate the average intensity of green pixels in the face crop
                avg_green = np.mean(img[:, :, 1]) 
                green_values.append(avg_green)
        
        # Save as a CSV for the Biological Lead to analyze
        if green_values:
            df = pd.DataFrame(green_values, columns=['green_mean'])
            output_file = os.path.join(output_root, f"{folder}_signal.csv")
            df.to_csv(output_file, index=False)
            print(f"✅ Extracted signal for: {folder}")

    print(f"\n✨ All signals saved to: {output_root}")

if __name__ == "__main__":
    extract_green_signal()