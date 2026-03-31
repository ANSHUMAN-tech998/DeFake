import os
import pandas as pd
import shutil # Added for robust folder deletion
from config import TRAIN_ROOT, TEST_ROOT

def clean_and_remap(partition_root, csv_name):
    crops_dir = os.path.join(partition_root, "face_crops")
    
    if not os.path.exists(csv_name):
        print(f"⚠️ {csv_name} not found, skipping.")
        return

    df = pd.read_csv(csv_name)
    valid_ids = []
    removed_count = 0

    print(f"🧹 Cleaning {csv_name}...")

    for v_id in df['video_id']:
        folder_path = os.path.join(crops_dir, str(v_id))
        
        # Logic: If folder exists AND has at least one valid image
        if os.path.exists(folder_path):
            images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            
            if len(images) > 0:
                valid_ids.append(v_id)
            else:
                # Use rmtree to force delete even if there are hidden non-image files
                try:
                    shutil.rmtree(folder_path)
                except Exception as e:
                    print(f"⚠️ Could not delete {v_id}: {e}")
                removed_count += 1
        else:
            # If folder is totally missing, it's definitely not a valid sample
            removed_count += 1

    # Filter the dataframe to only include folders that actually have images
    clean_df = df[df['video_id'].isin(valid_ids)]
    clean_df.to_csv(csv_name, index=False)
    
    print(f"✅ Cleaned {csv_name}!")
    print(f"📊 Removed {removed_count} broken samples.")
    print(f"📊 New total: {len(clean_df)} valid multimodal triplets.\n")

if __name__ == "__main__":
    clean_and_remap(TRAIN_ROOT, "train_mapping.csv")
    clean_and_remap(TEST_ROOT, "test_mapping.csv")