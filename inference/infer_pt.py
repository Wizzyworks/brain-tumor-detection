
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from model import CNN_TUMOR

# Model configuration and paths
model_path = "Brain_Tumor_model.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

# Labels
class_names = ['Brain Tumor', 'Healthy']

def load_model():
    """
    Initializes the model architecture and loads weights from the .pt file.
    """
    global model
    print(f"Loading PyTorch model from {model_path} on {device}...")
    
    params_model = {
        "shape_in": (3, 256, 256),
        "initial_filters": 8,
        "num_fc1": 100,
        "dropout_rate": 0.25,
        "num_classes": 2
    }
    
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_path} not found.")
            
        model = CNN_TUMOR(params_model)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print("PyTorch model loaded successfully.")
    except Exception as e:
        print(f"Error loading PyTorch model: {e}")
        model = None
        raise e

def preprocess_image(image_path):
    """
    Preprocess image using Torchvision transforms.
    """
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return img_tensor.to(device)

def predict(image_path):
    """
    Performs inference on a single image.
    """
    if model is None:
        raise RuntimeError("Model not loaded.")

    try:
        # 1. Preprocess
        input_tensor = preprocess_image(image_path)
        
        # 2. Inference
        with torch.no_grad():
            output = model(input_tensor)
            
            # Post-process (Output is log_softmax -> exp -> probs)
            probs = torch.exp(output)
            confidence, pred_idx = torch.max(probs, 1)
        
        return class_names[pred_idx.item()], float(confidence.item())
        
    except Exception as e:
        print(f"Inference failed: {e}")
        raise e

if __name__ == "__main__":
    # Test run
    import argparse
    parser = argparse.ArgumentParser(description="Inference using PyTorch model")
    parser.add_argument("--image", type=str, default="images/tumor/test3.jpg", help="Path to brain MRI image")
    args = parser.parse_args()

    try:
        load_model()
        if os.path.exists(args.image):
            label, conf = predict(args.image)
            print(f"\n--- Prediction Results ---")
            print(f"Image: {args.image}")
            print(f"Prediction: {label}")
            print(f"Confidence: {conf:.2%}")
        else:
            print(f"Error: Image '{args.image}' not found.")
    except Exception as e:
        print(f"Process failed: {e}")
