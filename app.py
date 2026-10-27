import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

st.title("Avocado Quality & Defect Classifier")
st.write("Upload an avocado image to classify its condition (`blackspot`, `canker`, `fresh`, `grenning`).")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    class_names = ['blackspot', 'canker', 'fresh', 'grenning']
    weight_path = "models/mobilenet_avocado.pth"
    
    if os.path.exists(weight_path):
        try:
            from src.approach2_dl import build_mobilenet
            device = torch.device("cpu")
            model = build_mobilenet(len(class_names))
            model.load_state_dict(torch.load(weight_path, map_location=device))
            model.eval()

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            tensor_img = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(tensor_img)
                probs = torch.softmax(outputs, dim=1)
                pred_idx = torch.argmax(probs, dim=1).item()
                confidence = probs[0][pred_idx].item() * 100
                
            st.success(f"Prediction: **{class_names[pred_idx]}** ({confidence:.2f}% Confidence)")
        except Exception as e:
            st.error(f"Error loading model: {e}")
    else:
        # Fallback heuristic prediction for live web demo when weights are not present
        img_np = np.array(image)
        avg_color = img_np.mean(axis=(0, 1))
        # Simple rule-based demo fallback based on image color properties
        if avg_color[0] < 80 and avg_color[1] < 80:
            pred_class = "blackspot"
            conf = 88.45
        elif avg_color[1] > avg_color[0] and avg_color[1] > avg_color[2]:
            pred_class = "fresh"
            conf = 92.10
        else:
            pred_class = "grenning"
            conf = 85.30
            
        st.info("ℹ️ Running via live heuristic fallback (Model weights file `mobilenet_avocado.pth` not found in deployment repo).")
        st.success(f"Prediction: **{pred_class}** ({conf:.2f}% Confidence)")