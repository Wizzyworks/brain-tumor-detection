# Brain Tumor Detection Project – Progress & Current Issues

## 1. Project Goal (Big Picture)
The long-term objective of this project is **to learn end-to-end deployment of an ML model using Docker and Kubernetes**.

This project is intentionally being used as a **learning vehicle**, not just a model showcase.

Final target pipeline:
- PyTorch model
- FastAPI inference service
- Docker containerization
- Kubernetes deployment (local → cloud)

---

## 2. What Has Been Completed So Far

### 2.1 Model Development (PyTorch)
- Implemented a custom CNN (`CNN_TUMOR`) for brain tumor detection
- Training completed
- Model saved in `.pt` / `.pth` format

Learnings:
- Manual CNN architecture design
- Understanding conv → pool → flatten → FC flow

---

### 2.2 Inference Script (`infer.py`)
- Created a separate inference script
- Device handling (CPU/GPU)
- Model loading logic added

Issues faced and resolved:
- `FileNotFoundError` → incorrect model path
- PyTorch 2.6+ `weights_only=True` default behavior
- Misunderstanding between:
  - `state_dict` vs full model object

Key realization:
- Some checkpoints store **entire model objects**, not just `state_dict`

---

### 2.3 FastAPI Service
- FastAPI app already written
- Inference endpoint implemented
- Model loading at startup

Status:
- API logic works
- Ready to be containerized

---

### 2.4 Dockerization
- Dockerfile created
- FastAPI app containerized successfully
- Docker image built (`brain-tumor-detection`)

Learnings:
- Exposing ports correctly
- Understanding container networking

Issues faced:
- Port binding error (`port already allocated`)
- Learned how host ↔ container ports work

---

## 3. ONNX Conversion Journey

### 3.1 Motivation for ONNX
Planned reasons:
- Faster inference using ONNX Runtime
- Framework-agnostic model format
- Better production readiness
- Clean separation of training vs inference

---

### 3.2 PyTorch → ONNX Exploration
Questions explored:
- Can PyTorch models be converted directly to ONNX? → YES
- Can conversion happen after saving `.pt`? → YES
- Can conversion happen inside Docker? → YES

Key understanding:
- ONNX export uses a **dummy input**
- Shape mismatches are caught strictly

---

### 3.3 ONNX Conversion Error (Current Blocker)

**Runtime Error:**
```
shape '[-1, 12544]' is invalid for input of size 9216
```

Meaning:
- Model expects flattened feature size = **12544**
- Actual tensor size during ONNX export = **9216**

---

## 4. Root Cause Analysis (Critical Learning)

### 4.1 Flatten Logic in Model
```python
self.num_flatten = h * w * 8 * init_f
self.fc1 = nn.Linear(self.num_flatten, num_fc1)
```

Assumption:
- Spatial dimensions (`h`, `w`) are manually calculated

Reality:
- Actual output feature map after conv + pooling layers is **smaller**
- Pooling layers reduced spatial size more than expected

Mismatch:
- Expected: `64 × 14 × 14 = 12544`
- Actual: `64 × 12 × 12 = 9216`

---

### 4.2 Why ONNX Fails but PyTorch Training Worked
- PyTorch training tolerated the shape assumption
- ONNX export performs strict graph validation
- ONNX exposed a **design flaw** in the CNN

Key realization:
> ONNX errors are often architecture bugs, not ONNX problems

---

## 5. Explored (but Rejected) Fix

### Idea:
Change ONNX dummy input shape to force output = 12544

Decision:
- Technically possible
- Architecturally wrong

Why rejected:
- Fixes symptom, not root cause
- Breaks for different input sizes
- Unsafe for production and Kubernetes deployment

---

## 6. Correct Direction Identified

### Industry-Grade Fix Options
1. **Adaptive Pooling (Best Practice)**
   - `nn.AdaptiveAvgPool2d((X, X))`
   - Makes model input-size agnostic

2. Dynamic flatten inference
   - Infer feature size during forward pass

3. Avoid hard-coded spatial assumptions

---

## 7. Kubernetes Learning Context

Important realization:
- Kubernetes does NOT require ONNX
- Kubernetes deploys containers, not models

ONNX is optional but valuable for:
- Performance
- Portability
- Production realism

---

## 8. Current Status Summary

### ✅ Done
- Model trained
- FastAPI inference built
- Docker image built

### ❌ Blocked
- ONNX conversion failing due to CNN architecture flaw

### 🔜 Next Logical Steps
1. Fix CNN architecture properly
2. Re-export PyTorch → ONNX cleanly
3. Use ONNX Runtime in FastAPI (optional)
4. Deploy container to Kubernetes

---

## 9. Key Learning So Far

- Deployment exposes architectural weaknesses
- ONNX is a validator, not just a converter
- Docker solves packaging
- Kubernetes solves orchestration
- Clean model design is non-negotiable for production

---

## 10. Mindset Shift Achieved

From:
> "Just make it work"

To:
> "Make it correct, portable, and deployable"

This project has transitioned from **ML-only thinking** to **production system thinking**.

