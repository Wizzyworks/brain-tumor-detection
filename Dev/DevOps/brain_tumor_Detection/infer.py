
import numpy as np
import onnxruntime as ort
from PIL import Image
import os

# Global ONNX Session
onnx_model_path = "model.onnx"
device = "cpu"
ort_session = None

def load_model():
    global ort_session
    print(f"Loading ONNX model from {onnx_model_path}...")
    try:
        if not os.path.exists(onnx_model_path):
            raise FileNotFoundError(f"{onnx_model_path} not found.")
            
        ort_session = ort.InferenceSession(onnx_model_path)
        print("ONNX model loaded successfully.")
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        ort_session = None
        raise e

# Labels
class_names = ['Brain Tumor', 'Healthy']

def preprocess_image(image_path):
    """
    Preprocess image using PIL and Numpy to match Torchvision transforms.
    Resize(256,256) -> ToTensor() -> Normalize(mean, std)
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((256, 256), Image.BILINEAR)
    
    # Convert to numpy and normalize to [0, 1]
    img_np = np.array(img).astype(np.float32) / 255.0
    
    # Normalize (Standard ImageNet mean/std)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    # (H, W, C) -> Normalize
    img_np = (img_np - mean) / std
    
    # (H, W, C) -> (C, H, W)
    img_np = img_np.transpose(2, 0, 1)
    
    # Add batch dimension -> (1, C, H, W)
    img_np = np.expand_dims(img_np, axis=0)
    
    return img_np

def predict(image_path):
    if ort_session is None:
        raise RuntimeError("Model not loaded.")

    try:
        # 1. Preprocess
        input_tensor = preprocess_image(image_path)
        
        # 2. Prepare Input
        ort_inputs = {ort_session.get_inputs()[0].name: input_tensor}
        
        # 3. Inference
        ort_outs = ort_session.run(None, ort_inputs)
        
        # 4. Post-process
        # Output is log_softmax -> exp -> probs
        log_probs = ort_outs[0]
        probs = np.exp(log_probs)
        
        # Get class and confidence
        pred_idx = np.argmax(probs, axis=1)[0]
        confidence = probs[0][pred_idx]
        
        return class_names[pred_idx], float(confidence)
        
    except Exception as e:
        print(f"Inference failed: {e}")
        raise e

if __name__ == "__main__":
    # Test run
    try:
        load_model()
        test_image = "images/tumor/test3.jpg"
        if os.path.exists(test_image):
            label, conf = predict(test_image)
            print(f"Prediction: {label}, Confidence: {conf:.2f}")
        else:
            print(f"Test image {test_image} not found. Skipped test.")
    except Exception as e:
        print(f"Test failed: {e}")