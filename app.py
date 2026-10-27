import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from src.approach2_dl import build_mobilenet

st.title("Avocado Quality & Defect Classifier")
st.write("Upload an avocado image to classify its condition (`blackspot`, `canker`, `fresh`, `grenning`).")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Load MobileNetV2 Model
    device = torch.device("cpu")
    class_names = ['blackspot', 'canker', 'fresh', 'grenning']
    model = build_mobilenet(len(class_names))
    
    try:
        model.load_state_dict(torch.load("models/mobilenet_avocado.pth", map_location=device))
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
        st.warning("Model weights not found yet! Please ensure `models/mobilenet_avocado.pth` is committed or trained.")