# Brain Tumor Detection

A machine learning project for detecting brain tumors using deep learning and FastAPI.

## Project Overview

This project implements a brain tumor detection system using a PyTorch-based deep learning model, optimized with ONNX for production deployment. The system includes both a REST API (FastAPI) and a user-friendly web interface (Streamlit).

## Features

- 🧠 Brain tumor detection using deep learning
- 🚀 FastAPI backend for production-ready inference
- 🎨 Streamlit web interface for easy interaction
- 🐳 Docker containerization for easy deployment
- ☸️ Kubernetes deployment configurations
- ⚡ ONNX model optimization for faster inference

## Project Structure

```
brain_tumor_Detection/
├── app.py                  # FastAPI application
├── streamlit_app.py        # Streamlit web interface
├── model.py                # Model architecture
├── infer.py                # Inference script
├── train.py                # Training script
├── onnx_convert.py         # ONNX conversion utility
├── Dockerfile              # Docker configuration
├── requirements.txt        # Python dependencies
├── k8s/                    # Kubernetes deployment files
└── test_app.py             # API tests
```

## Technologies Used

- **Deep Learning**: PyTorch
- **API Framework**: FastAPI
- **Web Interface**: Streamlit
- **Model Optimization**: ONNX Runtime
- **Containerization**: Docker
- **Orchestration**: Kubernetes
- **Python Version**: 3.11+

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Docker (optional, for containerized deployment)
- Kubernetes (optional, for K8s deployment)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/brain_tumor_Detection.git
cd brain_tumor_Detection
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

#### FastAPI Server
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

#### Streamlit Interface
```bash
streamlit run streamlit_app.py
```

### Docker Deployment

Build and run the Docker container:
```bash
docker build -t brain-tumor-detection .
docker run -p 8000:8000 brain-tumor-detection
```

### Kubernetes Deployment

Deploy to Kubernetes cluster:
```bash
kubectl apply -f k8s/
```

## API Endpoints

- `GET /health` - Health check endpoint
- `POST /predict` - Predict brain tumor from image

## Model Information

The model is a custom deep learning architecture trained on brain MRI images. It has been optimized using ONNX for efficient inference in production environments.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Brain tumor dataset providers
- PyTorch and ONNX communities
- FastAPI and Streamlit frameworks

## Contact

For questions or feedback, please open an issue on GitHub.
