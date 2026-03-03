import os
import json
from config import DATASETS

def parse_visual_metadata():
    # Path to the DFDC metadata file
    # Based on your folder: datasets/deepfake-detection-challenge/train_sample_videos/metadata.json
    json_path = os.path.join(DATASETS["visual_train"], "metadata.json")
    
    if not os.path.exists(json_path):
        print(f"❌ Metadata file not found at: {json_path}")
        return None

    print(f"Reading visual metadata from: {json_path}")
    
    with open(json_path, 'r') as f:
        metadata = json.load(f)  # This will be a dictionary where keys are video filenames and values contain labels and other info
        
    return metadata

if __name__ == "__main__":
    # Execute the parser
    video_data = parse_visual_metadata()
    
    if video_data:
        # Calculate stats for Week 2 report
        total_videos = len(video_data)
        fake_count = sum(1 for v in video_data.values() if v['label'] == 'FAKE')
        real_count = sum(1 for v in video_data.values() if v['label'] == 'REAL')
        
        print(f"✅ Successfully mapped {total_videos} videos.")
        print(f"\n--- Week 2 Data Summary: Visual ---")
        print(f"Real Videos: {real_count}")
        print(f"Fake Videos: {fake_count}")
        
        # Test one sample to show the mapping
        sample_video = list(video_data.keys())[0]
        label = video_data[sample_video]['label']
        print(f"\nSample Check: Video {sample_video} is labeled as {label}")