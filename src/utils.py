import os
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

def analyze_dataset(data_dir):
    """Scans dataset directory to report class counts and check for unreadable images."""
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} not found! Please check your dataset path.")
        return None
        
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    stats = []
    
    print("\n--- DATASET EDA SUMMARY ---")
    for cls in class_names:
        cls_path = os.path.join(data_dir, cls)
        files = os.listdir(cls_path)
        count = len(files)
        corrupted = 0
        
        for file in files:
            img_path = os.path.join(cls_path, file)
            img = cv2.imread(img_path)
            if img is None:
                corrupted += 1
                
        stats.append({'Class': cls, 'Total Images': count, 'Corrupted': corrupted})
    
    df = pd.DataFrame(stats)
    print(df.to_string(index=False))
    return df

def evaluate_pipeline(y_true, y_pred, class_names):
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\n=== EVALUATION REPORT ===")
    print(report)
    return report