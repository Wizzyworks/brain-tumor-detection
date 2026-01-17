import torch
from model import CNN_TUMOR
params_model={
        "shape_in": (3,256,256), 
        "initial_filters": 8,    
        "num_fc1": 100,
        "dropout_rate": 0.25,
        "num_classes": 2}

# 1️⃣ Recreate the model
model = CNN_TUMOR(params_model)

# 2️⃣ Load weights
state_dict = torch.load("Brain_Tumor_model.pt", map_location="cpu")
model.load_state_dict(state_dict)

# 3️⃣ Switch to inference mode
model.eval()

# 4️⃣ Dummy input
dummy_input = torch.randn(1, 3, 256, 256)

# 5️⃣ Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=18,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size", 2: "height", 3: "width"},
        "output": {0: "batch_size"}
    }
)

print("✅ ONNX exported from .pt weights")
