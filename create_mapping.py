import os
import pandas as pd
from config import TRAIN_ROOT, TEST_ROOT

def create_partition_csv(root_dir, csv_name):
    data = []
    crops_dir = os.path.join(root_dir, "face_crops")
    
    if not os.path.exists(crops_dir):
        print(f"⚠️ Error: {crops_dir} not found!")
        return

    # Scan each video folder
    for v_id in os.listdir(crops_dir):
        # LOGIC: If ID starts with 'subject', it's REAL (0). 
        # Otherwise, we assume it's a DEEPFAKE (1).
        label = 0 if v_id.startswith("subject") else 1
        
        # Match filenames
        audio_file = f"{v_id}.wav"
        signal_file = f"{v_id}_signal.csv"
        
        data.append({
            "video_id": v_id,
            "audio_file": audio_file,
            "signal_file": signal_file,
            "label": label
        })

    df = pd.DataFrame(data)
    df.to_csv(csv_name, index=False)
    print(f"✅ Created {csv_name} with {len(df)} samples.")

if __name__ == "__main__":
    # Create the two required files for your 60-40 split
    create_partition_csv(TRAIN_ROOT, "train_mapping.csv")
    create_partition_csv(TEST_ROOT, "test_mapping.csv")