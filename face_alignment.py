import cv2
import os
import urllib.request
import mediapipe as mp
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
    # CHANGE 'face' TO 'facial' HERE:
    output_facial_transformation_matrixes=False, 
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

def crop_to_face(image_path, output_path):
    # MediaPipe Tasks requires its own Image format
    mp_image = mp.Image.create_from_file(image_path)
    detection_result = detector.detect(mp_image)

    if detection_result.face_landmarks:
        img_cv2 = cv2.imread(image_path)
        h, w, _ = img_cv2.shape
        
        # Get all landmark points for this face
        landmarks = detection_result.face_landmarks[0]
        all_x = [l.x * w for l in landmarks]
        all_y = [l.y * h for l in landmarks]
        
        # Create the bounding box
        x_min, x_max = int(min(all_x)), int(max(all_x))
        y_min, y_max = int(min(all_y)), int(max(all_y))

        # Add 20% padding to ensure we capture the whole face for rPPG
        pad_w = int((x_max - x_min) * 0.2)
        pad_h = int((y_max - y_min) * 0.2)
        
        face_crop = img_cv2[max(0, y_min-pad_h):min(h, y_max+pad_h), 
                            max(0, x_min-pad_w):min(w, x_max+pad_w)]
        
        if face_crop.size > 0:
            face_final = cv2.resize(face_crop, IMG_SIZE)
            cv2.imwrite(output_path, face_final)
            return True
    return False

if __name__ == "__main__":
    input_root = os.path.join("processed_data", "frames")
    output_root = os.path.join("processed_data", "face_crops")
    
    if not os.path.exists(input_root):
        print(f"❌ Error: {input_root} not found. Run visual_extraction.py first!")
    else:
        print("🚀 Starting MediaPipe Face Alignment on Python 3.13...")
        # ... (rest of your folder looping logic)
        folders = os.listdir(input_root)
        
        for video_folder in folders:
            folder_path = os.path.join(input_root, video_folder)
            save_folder = os.path.join(output_root, video_folder)
            os.makedirs(save_folder, exist_ok=True)
            
            print(f"Aligning faces for: {video_folder}")
            for frame_file in os.listdir(folder_path):
                input_path = os.path.join(folder_path, frame_file)
                output_path = os.path.join(save_folder, frame_file)
                crop_to_face(input_path, output_path)
                
        print(f"✅ Alignment complete! Clean crops saved in: {output_root}")