import sys
import torch
import torchvision.transforms as transforms
import joblib
from PIL import Image
from src.approach1_classical import extract_hsv_features
from src.approach2_dl import build_mobilenet

def predict_single_image(image_path, class_names):
    print(f"\n=== RUNNING DEMO PREDICTION FOR: {image_path} ===")
    
    # 1. Classical Model Prediction
    try:
        svm_clf = joblib.load("models/classical_svm.pkl")
        hsv_feat = extract_hsv_features(image_path)
        if hsv_feat is not None:
            class_pred_id = svm_clf.predict([hsv_feat])[0]
            print(f"Approach 1 (HSV + SVM) Prediction         : {class_names[class_pred_id]}")
    except Exception as e:
        print(f"Approach 1 prediction skipped: {e}")
    
    # 2. Deep Learning Model Prediction
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_mobilenet(len(class_names))
        model.load_state_dict(torch.load("models/mobilenet_avocado.pth", map_location=device))
        model.to(device)
        model.eval()
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img = Image.open(image_path).convert('RGB')
        tensor_img = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(tensor_img)
            probs = torch.softmax(outputs, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_idx].item() * 100
            
        print(f"Approach 2 (MobileNetV2) Prediction       : {class_names[pred_idx]} ({confidence:.2f}% confidence)")
    except Exception as e:
        print(f"Approach 2 prediction skipped: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_img = sys.argv[1]
        demo_classes = ['blackspot', 'canker', 'fresh', 'grenning']
        predict_single_image(test_img, demo_classes)
    else:
        print("Usage: python src/predict.py <path_to_image.jpg>")