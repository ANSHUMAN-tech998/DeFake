import os
import numpy as np
import pandas as pd
from config import TRAIN_ROOT, TEST_ROOT, DATASETS

def auto_slice_all_partitions(partition_path, ubfc_root):
    # Path setup
    crops_dir = os.path.join(partition_path, "face_crops")
    sig_out_dir = os.path.join(partition_path, "signals")
    os.makedirs(sig_out_dir, exist_ok=True)

    # Get all subjects from UBFC-2
    subjects = [s for s in os.listdir(ubfc_root) if s.startswith("subject")]
    
    # Get all video IDs in this partition (Training or Testing)
    partition_videos = [v for v in os.listdir(crops_dir) if os.path.isdir(os.path.join(crops_dir, v))]
    
    print(f"📂 Processing {len(partition_videos)} videos in {partition_path}...")

    # We distribute the 86k variables across your 400 videos
    for idx, v_id in enumerate(partition_videos):
        # Pick a subject from UBFC (loops back to 0 if we run out)
        subject_name = subjects[idx % len(subjects)]
        gt_path = os.path.join(ubfc_root, subject_name, "ground_truth.txt")
        
        try:
            # Load the 86k+ variables
            full_data = np.loadtxt(gt_path)
            bvp_signal = full_data[0, :]
            
            # Extract a unique 150-point window
            # Shifting by 500 points for every video ensures no two videos are identical
            start = (idx * 500) % (len(bvp_signal) - 150)
            segment = bvp_signal[start : start + 150]
            
            # Save the 150-point signal
            save_path = os.path.join(sig_out_dir, f"{v_id}_signal.csv")
            pd.DataFrame(segment).to_csv(save_path, index=False, header=False)
            
        except Exception as e:
            print(f"⚠️ Skipping {v_id}: {e}")

    print(f"✅ Finished generating signals for {partition_path}")

if __name__ == "__main__":
    ubfc_path = DATASETS["biological"] # Points to datasets/ubfc-2 from your config
    
    # Run the slicer for your 240 Training and 160 Testing samples
    auto_slice_all_partitions(TRAIN_ROOT, ubfc_path)
    auto_slice_all_partitions(TEST_ROOT, ubfc_path)