import cv2
import os
import urllib.request
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from config import IMG_SIZE

# --- AUTO-DOWNLOADER SECTION ---
MODEL_PATH = 'face_landmarker.task'
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print(f"📥 Downloading face_landmarker.task (AI Brain)... please wait.")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("✅ Download Complete!")

# --- SETUP DETECTOR ---
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False, 
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

def process_video_folder(input_folder, output_folder):
    """
    Scans a folder of frames and ensures at least one face crop is saved.
    """
    frame_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png'))]
    frame_files.sort() # Ensure we start from the beginning of the video

    success = False
    
    # FALLBACK LOGIC: Try frames until we find a face
    for frame_file in frame_files:
        input_path = os.path.join(input_folder, frame_file)
        
        # MediaPipe processing
        mp_image = mp.Image.create_from_file(input_path)
        detection_result = detector.detect(mp_image)

        if detection_result.face_landmarks:
            img_cv2 = cv2.imread(input_path)
            if img_cv2 is None: continue
            
            h, w, _ = img_cv2.shape
            landmarks = detection_result.face_landmarks[0]
            
            # Extract coordinates
            all_x = [l.x * w for l in landmarks]
            all_y = [l.y * h for l in landmarks]
            
            x_min, x_max = int(min(all_x)), int(max(all_x))
            y_min, y_max = int(min(all_y)), int(max(all_y))

            # Padding for rPPG (forehead/cheeks)
            pad_w = int((x_max - x_min) * 0.25)
            pad_h = int((y_max - y_min) * 0.25)
            
            y1, y2 = max(0, y_min-pad_h), min(h, y_max+pad_h)
            x1, x2 = max(0, x_min-pad_w), min(w, x_max+pad_w)
            
            face_crop = img_cv2[y1:y2, x1:x2]
            
            if face_crop.size > 0:
                face_final = cv2.resize(face_crop, IMG_SIZE)
                output_path = os.path.join(output_folder, frame_file)
                cv2.imwrite(output_path, face_final)
                success = True
                # Once we find a good face for this video, we can move to the next video
                # (You can remove 'break' if you want to crop EVERY frame)
                break 

    return success

if __name__ == "__main__":
    from config import TRAIN_ROOT, TEST_ROOT
    
    # We will process both your Training and Testing partitions
    partitions = [TRAIN_ROOT, TEST_ROOT]
    
    print("🚀 Starting Resilient Face Alignment on Splits...")
    
    for part in partitions:
        # In your split, the frames are inside the 'face_crops' folder 
        # but we need to check if they are actually there or in a 'frames' folder
        input_root = os.path.join(part, "face_crops") 
        
        if not os.path.exists(input_root):
            print(f"⚠️ Warning: {input_root} not found. Skipping partition.")
            continue

        print(f"📂 Processing partition: {part}")
        folders = [f for f in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, f))]
        
        total_fixed = 0
        for video_folder in folders:
            folder_path = os.path.join(input_root, video_folder)
            
            # Check if the folder is empty
            if not any(f.lower().endswith(('.jpg', '.png')) for f in os.listdir(folder_path)):
                print(f"🔍 Empty folder found: {video_folder}. Attempting recovery...")
                
                # IMPORTANT: Since you probably deleted the raw 'frames' folder, 
                # we have to extract them from the original .mp4 / .avi video again.
                # Do you still have the original videos?