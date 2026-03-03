import os
import matplotlib.pyplot as plt
from config import DATASETS

def verify_biological_data():
    # Path to the UBFC-2 dataset
    bio_path = DATASETS["biological"]
    subjects = ["subject3", "subject4", "subject5"]
    
    report = {}

    for subject in subjects:
        subject_dir = os.path.join(bio_path, subject)
        video_file = os.path.join(subject_dir, "vid.avi")
        gt_file = os.path.join(subject_dir, "ground_truth.txt")
        
        # Check if files exist
        if os.path.exists(video_file) and os.path.exists(gt_file):
            # Load first 5 values of ground truth to verify format
            with open(gt_file, 'r') as f:
                # Format is usually: Trace (PPG), HR (BPM), Time
                first_lines = [f.readline().strip() for _ in range(5)]
            
            report[subject] = {
                "status": "✅ Ready",
                "sample_gt": first_lines
            }
        else:
            report[subject] = {"status": "❌ Missing Files"}
            
    return report

if __name__ == "__main__":
    results = verify_biological_data()
    
    print("--- Week 2 Data Summary: Biological (rPPG) ---")
    for sub, info in results.items():
        print(f"\n{sub}: {info['status']}")
        if "sample_gt" in info:
            print(f"Sample Pulse Data (Trace, HR, Time): {info['sample_gt']}")

    print("\n💡 Tip: The green channel average extraction logic will go here next.")