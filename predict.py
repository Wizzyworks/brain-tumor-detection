import uvicorn
import os

if __name__ == "__main__":
    # strictly following criteria: "Serving it via a web service"
    print("Starting Brain Tumor Detection Prediction Service...")
    print("Documentation available at http://localhost:8001/docs")
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)
