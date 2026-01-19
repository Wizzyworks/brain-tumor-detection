import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

# --- Model Architecture ---

def findConv2dOutShape(hin, win, conv, pool=2):
    kernel_size = conv.kernel_size
    stride = conv.stride
    padding = conv.padding
    dilation = conv.dilation

    hout = np.floor((hin + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    wout = np.floor((win + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)

    if pool:
        hout /= pool
        wout /= pool
    return int(hout), int(wout)

class CNN_TUMOR(nn.Module):
    def __init__(self, params):
        super(CNN_TUMOR, self).__init__()
        Cin, Hin, Win = params["shape_in"]
        init_f = params["initial_filters"]
        num_fc1 = params["num_fc1"]
        num_classes = params["num_classes"]
        self.dropout_rate = params["dropout_rate"]

        self.conv1 = nn.Conv2d(Cin, init_f, kernel_size=3)
        h, w = findConv2dOutShape(Hin, Win, self.conv1)
        self.conv2 = nn.Conv2d(init_f, 2 * init_f, kernel_size=3)
        h, w = findConv2dOutShape(h, w, self.conv2)
        self.conv3 = nn.Conv2d(2 * init_f, 4 * init_f, kernel_size=3)
        h, w = findConv2dOutShape(h, w, self.conv3)
        self.conv4 = nn.Conv2d(4 * init_f, 8 * init_f, kernel_size=3)
        h, w = findConv2dOutShape(h, w, self.conv4)

        self.target_h = h
        self.target_w = w
        self.num_flatten = h * w * 8 * init_f
        self.fc1 = nn.Linear(self.num_flatten, num_fc1)
        self.fc2 = nn.Linear(num_fc1, num_classes)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv3(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv4(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.adaptive_avg_pool2d(X, (self.target_h, self.target_w))
        X = X.view(-1, self.num_flatten)
        X = F.relu(self.fc1(X))
        X = F.dropout(X, self.dropout_rate)
        X = self.fc2(X)
        return F.log_softmax(X, dim=1)

# --- Inference Utilities ---

@st.cache_resource
def load_prediction_model():
    params_model = {
        "shape_in": (3, 256, 256),
        "initial_filters": 8,
        "num_fc1": 100,
        "dropout_rate": 0.25,
        "num_classes": 2
    }
    model = CNN_TUMOR(params_model)
    model_path = "Brain_Tumor_model.pt"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    model.eval()
    return model

def predict_local(image, model):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.exp(output)
        confidence, pred_idx = torch.max(probs, 1)
        
    class_names = ['Brain Tumor', 'Healthy']
    return class_names[pred_idx.item()], confidence.item()

# --- Streamlit UI ---

# Page config
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for theme
st.markdown("""
<style>
    .main {
        background-color: #f0f8ff;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .prediction-box {
        background-color: #e8f5e8;
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        text-align: center;
    }
    .tumor-box {
        background-color: #ffebee;
        border: 2px solid #f44336;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        text-align: center;
    }
    .title {
        color: #2E8B57;
        text-align: center;
        font-size: 36px;
        font-weight: bold;
    }
    .subtitle {
        color: #4682B4;
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">🧠 Brain Tumor Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a brain MRI image to detect tumors using AI</div>', unsafe_allow_html=True)

# Load model
try:
    model = load_prediction_model()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {e}")
    model_loaded = False

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This AI-powered application uses a Convolutional Neural Network (CNN) to analyze brain MRI images and detect the presence of tumors.

    **How it works:**
    1. Upload a brain MRI image (JPG, PNG)
    2. The AI model analyzes the image
    3. Get instant prediction with confidence score

    **Note:** This is for educational purposes only. Consult medical professionals for actual diagnosis.
    """)

    st.header("📊 Model Info")
    st.write("**Architecture:** CNN with 4 convolutional layers")
    st.write("**Classes:** Healthy, Tumor")
    st.write("**Accuracy:** ~95% (validation)")
    
    if model_loaded:
        st.success("✅ Model Loaded Locally")
    else:
        st.error("❌ Model Not Loaded")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader("Choose a brain MRI image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Predict button
        if st.button("🔍 Analyze Image"):
            if not model_loaded:
                st.error("Model is not loaded. Please check if 'Brain_Tumor_model.pt' exists.")
            else:
                with st.spinner("Analyzing image..."):
                    try:
                        # Call local prediction
                        prediction, confidence = predict_local(image, model)

                        # Display result
                        with col2:
                            st.subheader("📋 Analysis Result")

                            if "tumor" in prediction.lower():
                                st.error("⚠️ Tumor Detected")
                                st.markdown(f'<div class="tumor-box">**Prediction:** {prediction}<br>**Confidence:** {confidence:.2%}</div>', unsafe_allow_html=True)
                            else:
                                st.success("✅ No Tumor Detected")
                                st.markdown(f'<div class="prediction-box">**Prediction:** {prediction}<br>**Confidence:** {confidence:.2%}</div>', unsafe_allow_html=True)

                            # Confidence bar
                            st.progress(confidence)
                            st.write(f"Confidence: {confidence:.2%}")

                    except Exception as e:
                        st.error(f"Error analyzing image: {str(e)}")

with col2:
    if uploaded_file is None:
        st.subheader("📋 Analysis Result")
        st.info("Upload an image and click 'Analyze Image' to see the results.")

# Footer
st.markdown("---")
st.markdown("**Disclaimer:** This tool is not a substitute for professional medical advice. Always consult healthcare providers for medical decisions.")