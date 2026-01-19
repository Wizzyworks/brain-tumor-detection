# Brain Tumor Detection

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.10.x-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95-green)
![Docker](https://img.shields.io/badge/docker-ready-blue)

## 📌 Problem Description

Brain tumors are a serious health condition requiring early and accurate detection. Manual diagnosis from MRI scans is time-consuming and prone to human error.

**This project offers a deep learning-based solution** to automatically detect and classify brain tumors from MRI images. By deploying this model as a responsive API, we enable medical professionals to get instant "second opinions" on scans.

---

## 📦 Data Setup (Crucial Step)

**⚠️ IMPORTANT: The dataset is NOT included in this repository.**

1.  **Download** the dataset from Kaggle:
    👉 [**Brain Tumor Dataset (Kaggle)**](https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset)
    *(You will need a free Kaggle account)*

2.  **Unzip** the downloaded file.

3.  **Place** the content into the `data/` folder in this project.
    Your folder structure must look exactly like this:
    ```
    brain_tumor_Detection/
    └── data/
        └── brain_tumor_dataset/
            ├── Brain Tumor/
            └── Healthy/
    ```

---

## 🛠️ Installation & Setup

### 1. Check Python Version
This project requires **Python 3.10.x**. It is important to use this exact version to avoid errors.

*   Open your terminal (Command Prompt or PowerShell).
*   Run:
    ```bash
    python --version
    ```
*   If it says `Python 3.10.x` → YOU ARE GOOD! ✅
*   If it says `Python 3.11`, `3.12`, or `3.9` → Please install Python 3.10 from [python.org](https://www.python.org/downloads/).

### 2. Create a Virtual Environment (The "Box")
We will create a "box" (virtual environment) to keep all our project libraries inside, so they don't mess up your computer.

**For Windows (PowerShell or Command Prompt):**
```bash
# 1. Create the environment named '.venv'
python -m venv .venv

# 2. Activate it
# You must see (.venv) appear at the start of your line after running this!
.\.venv\Scripts\activate
```

**For Mac / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Libraries
Now that your "box" is active, let's fill it with the tools we need.

```bash
# 1. Update the installer (pip) just in case
python -m pip install --upgrade pip

# 2. Install all the project tools
pip install -r requirements.txt
```

---

## 🧠 Architecture Logic: Why this Model?

Our architectural choices are directly driven by the data analysis (see `notebook.ipynb`):

1.  **4 Convolutional Layers (16 → 128 filters)**
    *   *Reasoning:* EDA showed tumors have complex, non-uniform textures. A shallow network detects edges, but we need depth to capture the "mass" and "boundary" features of a tumor.
2.  **Adaptive Pooling (`AdaptiveAvgPool2d`)**
    *   *Reasoning:* MRI machines produce scans of varying resolutions. This layer forces *any* input size into a fixed feature vector, making the model robust to different image dimensions without aggressive warping.
3.  **Dropout (0.25)**
    *   *Reasoning:* The dataset is relatively small (< 5000 images). Deep networks easily memorize small datasets. Dropout forces the network to learn robust features by randomly disabling neurons.
4.  **LogSoftmax + NLLLoss**
    *   *Reasoning:* We use this combination for numerical stability over standard Softmax, ensuring gradients don't vanish during training on medical imagery.

---

## 🚀 Usage

### Option 1: Run Locally (Easiest)
Once you have installed everything (above), you can run the prediction server.

1.  **Start the Server:**
    ```bash
    python predict.py
    ```
2.  **Test It:**
    *   Open your browser to: `http://localhost:8001/docs`
    *   Click "POST /predict" -> "Try it out" -> Upload an MRI image -> "Execute".

### Option 2: Run with Docker
If you have Docker installed, you don't need to install Python or libraries!

```bash
# 1. Build the container
docker build -t brain-tumor-detection .

# 2. Run it
docker run -p 8001:8001 brain-tumor-detection
```

---

## ☁️ Kubernetes Deployment

We provide files to deploy this to a Kubernetes cluster (like Kind or Minikube).

**1. Apply Configuration:**
```bash
kubectl apply -f k8s/
```

**2. Access the Service:**
We have set up a `NodePort` service which exposes the app specifically on port **30001**.
*   **URL:** `http://localhost:30001/docs`

---

## 📊 Exploration & Training

*   **Exploratory Data Analysis (EDA)**: See **[`notebook.ipynb`](notebook.ipynb)** for charts and analysis of the MRI images.
*   **Training Script**: See **[`train.py`](train.py)** if you want to re-train the model yourself.

---

## 📂 Project Structure

| File | Description |
|------|-------------|
| `notebook.ipynb` | **EDA**: Visual analysis of the dataset. |
| `predict.py` | **Start Server**: The command to run the App. |
| `train.py` | **Training**: Script to train the model. |
| `app.py` | **Backend**: The code that handles API requests. |
| `k8s/` | **Kubernetes**: Deployment and Service files. |
| `requirements.txt` | **Libraries**: List of tools used. |
