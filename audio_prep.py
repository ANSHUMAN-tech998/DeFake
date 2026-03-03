import os
from config import DATASETS

def parse_audio_labels():
    # Path to the training protocol file
    # Based on your folder structure: datasets/asvspoof-2019-la-subset/ASVspoof2019_LA_cm_protocols/
    protocol_path = os.path.join(DATASETS["audio_protocols"], "ASVspoof2019.LA.cm.train.trn.txt")
    
    label_map = {}
    
    # Check if the protocol file exists before trying to open it
    if not os.path.exists(protocol_path):
        print(f"❌ Protocol file not found at: {protocol_path}")
        return None

    print(f"Reading labels from: {protocol_path}")
    
    with open(protocol_path, 'r') as f:
        for line in f:
            # Protocol format: Speaker_ID, File_ID, System_ID, Key_Label
            # Example: LA_0079 LA_T_0000607 - - bonafide
            parts = line.strip().split()
            if len(parts) >= 5:
                file_id = parts[1]      # The filename without .flac
                status = parts[-1]       # 'bonafide' (Real) or 'spoof' (Fake)
                label_map[file_id] = status
                
    return label_map

if __name__ == "__main__":
    # Execute the parser
    labels = parse_audio_labels()
    
    if labels:
        print(f"✅ Successfully mapped {len(labels)} audio files.")
        
        # Calculate dataset balance for Week 9 report
        bonafide_count = list(labels.values()).count('bonafide')
        spoof_count = list(labels.values()).count('spoof')
        
        print(f"\n--- Week 2 Data Summary: Audio ---")
        print(f"Real (Bonafide): {bonafide_count}")
        print(f"Fake (Spoof): {spoof_count}")
        
        # Test one sample to ensure it's correct
        sample_key = list(labels.keys())[0]
        print(f"\nSample Check: File {sample_key} is labeled as {labels[sample_key]}")