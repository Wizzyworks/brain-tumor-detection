from fastapi import FastAPI, File, UploadFile, HTTPException
import io
import tempfile
import os
from contextlib import asynccontextmanager
from infer import predict, load_model

# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model on startup
    try:
        print("Startup: Loading model...")
        load_model()
    except Exception as e:
        print(f"Startup Error: Failed to load model: {e}")
        # In production, you might want to force exit here depending on policy
    yield
    # Clean up resources if needed on shutdown
    print("Shutdown: App is closing.")

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health_check():
    """Health check endpoint for Kubernetes/Docker probes."""
    # Build a simple check. If predict raises 'Model not loaded', we are unhealthy.
    # Alternatively, we could expose a variable from infer.py, but trying a dry check or just returning 200 is common.
    # Let's return 200 if app is running. Ideally, we check state.
    from infer import ort_session
    if ort_session is None:
         raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "service": "brain-tumor-detection"}

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Save uploaded file to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(await file.read())
            temp_path = temp_file.name

        # Call predict function from infer.py
        label, confidence = predict(temp_path)

        # Clean up temp file
        os.unlink(temp_path)

        return {"prediction": label, "confidence": confidence}
    except Exception as e:
        # Handle specific model not loaded error with 503
        if "Model not loaded" in str(e):
             raise HTTPException(status_code=503, detail="Model is not ready")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Brain Tumor Detection API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
