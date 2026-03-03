import cv2
import os
from config import DATASETS, IMG_SIZE
from visual_prep import parse_visual_metadata

def extract_frames(video_filename, num_frames=5):
    video_path = os.path.join(DATASETS["visual_train"], video_filename)
    output_folder = os.path.join("processed_data", "frames", video_filename.split('.')[0])
    
    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate interval to pick frames spread across the video
    interval = total_frames // num_frames
    
    count = 0
    extracted = 0
    
    while cap.isOpened() and extracted < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        if count % interval == 0:
            # Resize for EfficientViT standards
            frame_resized = cv2.resize(frame, IMG_SIZE)
            frame_name = f"frame_{extracted}.jpg"
            cv2.imwrite(os.path.join(output_folder, frame_name), frame_resized)
            extracted += 1
        count += 1
        
    cap.release()
    return output_folder

if __name__ == "__main__":
    metadata = parse_visual_metadata()
    if metadata:
        # Loop through EVERY video in your metadata
        for video_filename in list(metadata.keys()):
            print(f"Processing: {video_filename}")
            # You can change 'num_frames' to 10 or 20 if needed
            extract_frames(video_filename, num_frames=10) 
            
        print(f"✅ All videos processed!")