# QuantumHealthAI v2 Enhanced

> **Hybrid Quantum-Classical Disease Risk Prediction System**
> Predicts risk of **Diabetes**, **Cardiovascular Disease (CVD)**, and **Chronic Kidney Disease (CKD)** from patient lab reports using a fusion of Random Forest, XGBoost, and a Variational Quantum Classifier (VQC).

---

## Authors

| Name | University | Email |
|---|---|---|
| Marthand Bhargav Jalasutram | VIT-AP University | bhargav.22bce8928@vitapstudent.ac.in |
| Harshith Gude | VIT-AP University | — |
| Aakash Gonuguntla | VIT-AP University | — |

---

## Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Backend Setup (Python)](#2-backend-setup-python)
  - [3. Frontend Setup (Node.js)](#3-frontend-setup-nodejs)
  - [4. MongoDB Setup](#4-mongodb-setup)
  - [5. Environment Variables](#5-environment-variables)
- [Training the Models](#training-the-models)
- [Running the Application](#running-the-application)
- [API Reference](#api-reference)
- [Running Tests](#running-tests)
- [Docker (Optional)](#docker-optional)
- [Project Structure](#project-structure)
- [Research Paper](#research-paper)

---

## Project Overview

QuantumHealthAI is a clinical decision support system that:

1. Accepts a patient PDF medical report via upload
2. Extracts lab values using OCR (Tesseract / OCRmyPDF)
3. Runs them through a **hybrid quantum-classical ML pipeline**
4. Returns a structured risk score with plain-language explanations

### Key Technical Contributions

| Component | Technology | Role |
|---|---|---|
| Classical ML | Random Forest + XGBoost | High-accuracy baseline predictions |
| Quantum ML | PennyLane VQC (6-qubit) | Quantum-enhanced feature encoding |
| Fusion | AUC-weighted hybrid | Adaptive combination of all three models |
| Explainability | SHAP TreeExplainer | Top-3 feature attributions per prediction |
| Clinical Safety | Rule Engine | Hard threshold overrides (glucose, creatinine, BP) |

---

## System Architecture

```
Patient uploads PDF
        │
        ▼
   OCR Extraction (Tesseract / OCRmyPDF)
        │
        ▼
   Data Governance Module
   (validate ranges, unit conversion, IQR clipping)
        │
        ▼
   Feature Mapper (disease-specific subsets: 6/6/5 features)
        │
        ├─────────────────────────────────┐
        ▼                                 ▼
   Random Forest + XGBoost           PCA → VQC (6 qubits)
        │                                 │
        └──────────────┬──────────────────┘
                       ▼
              AUC-Weighted Hybrid Fusion
                       │
                       ▼
              Clinical Rule Engine
                       │
                       ▼
              SHAP Explainability Layer
                       │
                       ▼
              Structured JSON Response
```

---

## Prerequisites

Make sure the following are installed on your system before proceeding:

| Tool | Version | Download |
|---|---|---|
| Python | 3.11+ | https://www.python.org/downloads/ |
| Node.js | 18+ | https://nodejs.org/ |
| MongoDB | 6.0+ | https://www.mongodb.com/try/download/community |
| Git | any | https://git-scm.com/ |
| Tesseract OCR | 5.x | https://github.com/UB-Mannheim/tesseract/wiki *(Windows)* |

**Optional (for GPU-accelerated XGBoost training):**
- CUDA Toolkit 11.8+ and a compatible NVIDIA GPU

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/marthand-2004/QunatumHealth_AI.git
cd QunatumHealth_AI
```

---

### 2. Backend Setup (Python)

#### Create and activate a virtual environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

#### Install Python dependencies

```bash
pip install -r requirements.txt
```

> **Note:** PennyLane's `lightning.qubit` backend requires a C++ compiler. On Windows, install [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) if the install fails.

#### Install Tesseract OCR

**Windows:** Download and install from https://github.com/UB-Mannheim/tesseract/wiki  
Then add Tesseract to your PATH, or set the path in `.env`:
```
TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
```

**Ubuntu / Debian:**
```bash
sudo apt-get install tesseract-ocr ocrmypdf
```

**macOS:**
```bash
brew install tesseract ocrmypdf
```

---

### 3. Frontend Setup (Node.js)

```bash
cd frontend
npm install
cd ..
```

---

### 4. MongoDB Setup

Start MongoDB locally (default port 27017):

**Windows:** Start the MongoDB service from Services, or run:
```bash
mongod --dbpath C:\data\db
```

**macOS / Linux:**
```bash
mongod --dbpath /data/db
```

The application will automatically create the required database and collections on first run.

---

### 5. Environment Variables

Copy the example environment file and fill in your values:

```bash
cp .env.example .env
```

Edit `.env` with the following required values:

```env
# MongoDB
MONGODB_URL=mongodb://localhost:27017
DATABASE_NAME=quantumhealthai

# JWT Authentication
SECRET_KEY=your-secret-key-change-this-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# File Encryption
ENCRYPTION_KEY=your-32-byte-fernet-key

# Tesseract (Windows only — skip on Linux/macOS)
TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe

# Optional: OpenAI for LLM explanations
OPENAI_API_KEY=your-openai-key
```

To generate a secure `ENCRYPTION_KEY`, run:
```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

To generate a secure `SECRET_KEY`, run:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

---

## Training the Models

The training pipeline consists of 6 sequential scripts. Run them in order from the project root:

```bash
# Step 1: Preprocess raw datasets
python training/scripts/01_preprocess_datasets.py

# Step 2: Train classical models (Random Forest + XGBoost)
python training/scripts/02_train_classical_models_v2.py

# Step 3: Train Variational Quantum Classifier
python training/scripts/03_train_quantum_vqc_v2.py

# Step 4: Compute AUC-weighted hybrid fusion weights
python training/scripts/04_hybrid_fusion_v2.py

# Step 5: Generate evaluation plots and metrics
python training/scripts/05_evaluate_v2.py

# Step 6: Run robustness tests under Gaussian noise
python training/scripts/06_robustness_test.py
```

> **Pre-trained models are already included** in `data/models/`. You can skip training and go straight to running the application.

Training outputs are saved to:
- `data/models/` — RF, XGBoost, VQC weights, PCA, scalers
- `training/models/` — thresholds, fusion weights, HPO params
- `docs/figures/` — evaluation plots

---

## Running the Application

### Start the Backend API

```bash
python app.py
```

The FastAPI backend will start at: **http://localhost:8000**  
Interactive API docs: **http://localhost:8000/docs**

### Start the Frontend

In a separate terminal:

```bash
cd frontend
npm run dev
```

The React frontend will start at: **http://localhost:5173**

### Default Login Credentials

On first run, create an account via the Register page, or use the seeded admin account:

| Role | Email | Password |
|---|---|---|
| Admin | admin@quantumhealthai.com | admin123 |

---

## API Reference

### Authentication

```
POST /auth/register    — Create a new account
POST /auth/login       — Get JWT access token
```

### Prediction

```
POST /predict/v2       — Run hybrid quantum-classical prediction
POST /predict/what-if  — Simulate lifestyle change impact
```

**Example prediction request:**
```json
{
  "glucose": 126,
  "hba1c": 6.5,
  "bmi": 28.5,
  "age": 45,
  "systolic_bp": 135,
  "diastolic_bp": 88,
  "cholesterol": 210,
  "smoking": 0,
  "creatinine": 1.1,
  "hemoglobin": 13.5
}
```

**Example response:**
```json
{
  "diabetes": {
    "risk_score": 62.4,
    "risk_level": "High",
    "shap_features": [
      {"feature": "glucose", "value": 0.182, "direction": "increases risk"},
      {"feature": "hba1c",   "value": 0.091, "direction": "increases risk"},
      {"feature": "bmi",     "value": 0.085, "direction": "increases risk"}
    ],
    "triggered_rules": ["glucose > 126 mg/dL"],
    "robustness_score": 0.979
  }
}
```

### Documents

```
POST /documents/upload  — Upload and OCR a PDF medical report
GET  /documents/        — List uploaded documents
```

---

## Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_e2e_prediction_v2.py -v

# Run with coverage report
pytest --cov=backend --cov-report=term-missing
```

---

## Docker (Optional)

If you prefer Docker, build and run the full stack:

```bash
# Build and start all services
docker-compose up --build

# Run in background
docker-compose up -d --build
```

Services started:
- `backend` — FastAPI on port 8000
- `frontend` — React on port 5173
- `mongodb` — MongoDB on port 27017

---

## Project Structure

```
QunatumHealth_AI/
├── app.py                          # FastAPI application entry point
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variable template
├── pytest.ini                      # Test configuration
│
├── backend/
│   ├── core/
│   │   ├── config.py               # App configuration
│   │   ├── database.py             # MongoDB connection
│   │   ├── deps.py                 # FastAPI dependencies
│   │   └── security.py             # JWT + password hashing
│   ├── models/                     # Pydantic data models
│   ├── routers/                    # API route handlers
│   │   ├── predict_v2.py           # Hybrid prediction endpoint
│   │   └── ...
│   └── services/
│       ├── classical_ml_v2.py      # RF + XGBoost inference
│       ├── quantum_engine_v2.py    # VQC inference (PennyLane)
│       ├── hybrid_fusion.py        # AUC-weighted fusion
│       ├── data_governance.py      # Input validation + preprocessing
│       ├── feature_mapper.py       # Disease-specific feature selection
│       ├── clinical_rule_engine.py # Hard threshold overrides
│       ├── xai_service_v2.py       # SHAP explanations
│       ├── what_if_analyzer.py     # Lifestyle simulation
│       └── ocr_service.py          # PDF text extraction
│
├── frontend/
│   ├── src/
│   │   ├── pages/                  # React page components
│   │   ├── components/             # Reusable UI components
│   │   ├── api/                    # API client functions
│   │   └── context/                # Auth context
│   └── package.json
│
├── training/
│   ├── datasets/                   # Raw CSV datasets
│   ├── scripts/                    # Training pipeline (01–06)
│   └── models/                     # Saved scalers, thresholds
│
├── data/
│   ├── models/                     # Trained model artifacts (.joblib, .npy)
│   └── uploads/                    # Encrypted uploaded files
│
├── tests/                          # pytest test suite
│
└── docs/
    ├── PROJECT_OVERVIEW.md         # Full project documentation
    ├── research_paper.tex          # IEEE-format LaTeX research paper
    └── project-figures/            # Generated figures for the paper
```

---

## Research Paper

A full IEEE-format research paper is included at `docs/research_paper.tex`.

To compile it:
1. Install [MiKTeX](https://miktex.org/download) or upload to [Overleaf](https://overleaf.com)
2. Compile with `pdflatex` twice:
   ```bash
   cd docs
   pdflatex research_paper.tex
   pdflatex research_paper.tex
   ```

Figures are in `docs/project-figures/`.

---

## License

This project is developed for academic purposes at VIT-AP University.

---

*QuantumHealthAI v2 Enhanced — VIT-AP University, 2026*
