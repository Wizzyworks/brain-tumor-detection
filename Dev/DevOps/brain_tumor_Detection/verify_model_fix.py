
import torch
import torch.nn as nn
from model import CNN_TUMOR

def test_model():
    print("Testing model with dynamic shapes...")
    
    # Standard params (simulating what might be in the real app)
    params = {
        "shape_in": (3, 64, 64), # Example shape based on 12544 flatten size/64 channels -> 14x14 feature map? 
        # Wait, the user said: Expected: 64 × 14 × 14 = 12544
        # So input must have been creating 14x14 output from convs.
        # Let's use a standard shape.
        "initial_filters": 8,
        "num_fc1": 100,
        "num_classes": 2,
        "dropout_rate": 0.25
    }
    
    try:
        model = CNN_TUMOR(params)
        model.eval()
        print("Model initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return

    # Test Case 1: Standard expected shape 
    # (We don't know exact training shape, but let's try 64x64 or 128x128)
    # If flatten is 12544 = 64 * 14 * 14. 
    # With 4 max pools (div by 16 total), input roughly 14*16 = 224?
    # Let's try a few shapes.
    
    shapes_to_test = [
        (1, 3, 224, 224),
        (1, 3, 256, 256), # Larger
        (1, 3, 200, 200)  # Smaller/Weird
    ]

    for shape in shapes_to_test:
        dummy_input = torch.randn(*shape)
        try:
            output = model(dummy_input)
            print(f"PASS: Input shape {shape} -> Output shape {output.shape}")
        except RuntimeError as e:
            print(f"FAIL: Input shape {shape} caused error: {e}")
            if "size mismatch" in str(e) or "shape" in str(e):
                print(" -> This is likely the error we were fixing!")
        except Exception as e:
             print(f"FAIL: Input shape {shape} caused unexpected error: {e}")

if __name__ == "__main__":
    test_model()
