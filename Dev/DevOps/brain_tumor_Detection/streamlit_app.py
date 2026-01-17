import streamlit as st
import requests
from PIL import Image

# Page config
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API endpoint
API_URL = "http://localhost:8001/predict"

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

    st.header("🔗 API Status")
    try:
        response = requests.get("http://localhost:8001/")
        if response.status_code == 200:
            st.success("✅ API Server Running")
        else:
            st.error("❌ API Server Not Responding")
    except Exception as e:
        st.error(f"❌ Cannot Connect to API Server: {str(e)}")
        st.info("Make sure to run: `uvicorn app:app --host 127.0.0.1 --port 8001 --reload`")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader("Choose a brain MRI image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Predict button
        if st.button("🔍 Analyze Image"):
            with st.spinner("Analyzing image..."):
                try:
                    # Prepare file for API
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}

                    # Call API
                    response = requests.post(API_URL, files=files)

                    if response.status_code == 200:
                        result = response.json()
                        prediction = result["prediction"]
                        confidence = result["confidence"]

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

                    else:
                        st.error(f"API Error: {response.status_code} - {response.text}")

                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API server. Please start the server with: `uvicorn app:app --host 127.0.0.1 --port 8001 --reload`")
                except Exception as e:
                    st.error(f"Error analyzing image: {str(e)}")

with col2:
    if uploaded_file is None:
        st.subheader("📋 Analysis Result")
        st.info("Upload an image and click 'Analyze Image' to see the results.")

# Footer
st.markdown("---")
st.markdown("**Disclaimer:** This tool is not a substitute for professional medical advice. Always consult healthcare providers for medical decisions.")