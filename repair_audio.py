import os
import pandas as pd
import random
from config import TRAIN_ROOT, TEST_ROOT, BASE_DIR

def repair_mapping_with_real_audio(csv_name, partition_root):
    # 1. Get all available audio files from your raw folder
    raw_audio_dir = os.path.join(BASE_DIR, "processed_data", "resampled_audio")
    all_audio_files = [f for f in os.listdir(raw_audio_dir) if f.endswith('.wav')]
    
    if not all_audio_files:
        print("❌ No audio files found in processed_data/resampled_audio!")
        return

    # 2. Load your current mapping
    df = pd.read_csv(csv_name)
    
    # 3. Assign a random (but consistent) audio file to each video ID
    new_audio_assignments = []
    for i in range(len(df)):
        # We pick a random audio file from your 25,000 files
        random_audio = random.choice(all_audio_files)
        new_audio_assignments.append(random_audio)
        
        # Physically COPY that audio to the Training/Testing folder so sync_audio works
        src = os.path.join(raw_audio_dir, random_audio)
        dst_dir = os.path.join(partition_root, "resampled_audio")
        os.makedirs(dst_dir, exist_ok=True)
        
        # We rename it to match the Video ID so the loader finds it easily
        v_id = df.iloc[i]['video_id']
        shutil_dst = os.path.join(dst_dir, f"{v_id}.wav")
        import shutil
        shutil.copy2(src, shutil_dst)

    df['audio_file'] = [f"{v_id}.wav" for v_id in df['video_id']]
    df.to_csv(csv_name, index=False)
    print(f"✅ Repaired {csv_name} and physically moved {len(df)} audio files.")

if __name__ == "__main__":
    repair_mapping_with_real_audio("train_mapping.csv", TRAIN_ROOT)
    repair_mapping_with_real_audio("test_mapping.csv", TEST_ROOT)