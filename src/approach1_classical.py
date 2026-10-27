import os
import cv2
import numpy as np
import joblib
from sklearn.svm import SVC

def extract_hsv_features(image_path, bins=(8, 12, 3)):
    """Extracts 3D Color Histogram in HSV space."""
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def train_classical_model(data_dir):
    X, y = [], []
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    label_map = {name: i for i, name in enumerate(class_names)}
    
    print("\n[Approach 1] Extracting HSV feature vectors from images...")
    for cls in class_names:
        cls_path = os.path.join(data_dir, cls)
        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)
            features = extract_hsv_features(img_path)
            if features is not None:
                X.append(features)
                y.append(label_map[cls])
                
    X, y = np.array(X), np.array(y)
    
    print(f"[Approach 1] Training SVM Classifier on {len(X)} samples...")
    clf = SVC(kernel='rbf', C=1.0, class_weight='balanced')
    clf.fit(X, y)
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "models/classical_svm.pkl")
    print("[Approach 1] Model successfully saved to models/classical_svm.pkl")
    return clf, class_names, X, y