FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install minimal system dependencies
# gcc might be needed for some python packages, but trying without first for slimming
# If onnxruntime or pillow need libs, apt-get install here.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py infer.py model.onnx* ./

# Expose FastAPI port
EXPOSE 8001

# Run the application
# Using shell form to allow variable expansion if needed, but array form is safer for signal handling.
# However, uvicorn inside docker usually needs explicit host 0.0.0.0
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]