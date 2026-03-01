# CardioCare-AI — Backend API

> **Production-ready Flask ML API for cardiovascular disease risk prediction.**

[![Flask](https://img.shields.io/badge/Flask-3.0-blue?style=flat-square&logo=flask)](https://flask.palletsprojects.com/)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow?style=flat-square&logo=python)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4.2-orange?style=flat-square&logo=scikitlearn)](https://scikit-learn.org/)
[![Deployed on Render](https://img.shields.io/badge/Deployed-Render-success?style=flat-square)](https://render.com)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [API Endpoints](#-api-endpoints)
- [ML Model](#-ml-model)
- [Setup & Installation](#-setup--installation)
- [Environment Variables](#-environment-variables)
- [Deployment (Render)](#-deployment-render)
- [Logging](#-logging)
- [Rate Limits & Security](#-rate-limits--security)

---

## 🧠 Overview

The CardioCare-AI backend is a **Flask REST API** that serves a trained **Gradient Boosting Machine (GBM)** model to predict cardiovascular disease risk. It handles:

- Input validation & blood pressure clamping
- Feature scaling via `StandardScaler`
- 3-tier risk classification (Low / Moderate / High)
- Personalized health tip generation
- Prediction persistence in SQLite (or PostgreSQL)
- Audit logging, rate limiting, and security headers

---

## ✨ Features

- 🔮 **Real-time predictions** — sub-20ms response times
- ✅ **Strict input validation** — field-level range checks + cross-field validation (e.g. systolic > diastolic)
- 📊 **Model metrics endpoint** — live ROC curve, confusion matrix, classification report, correlation matrix
- 🧾 **Audit trail** — every prediction, error, and system event is logged to `AuditLog`
- 🔄 **Hot model reload** — `POST /api/model/reload` swaps the model without restarting
- 📈 **System monitoring** — CPU, memory, disk, uptime via `/api/system`
- 🛡️ **Security** — Flask-Talisman (security headers), Flask-Limiter, Flask-CORS
- 🗜️ **Compression** — Flask-Compress for gzip response encoding
- 🪵 **Rotating logs** — 3 separate log files, each capped at 1 MB with 10 backups

---

## 📁 Project Structure

```
backend/
├── app.py                  # Main Flask app — all routes & business logic
├── config.py               # Centralized configuration (paths, secrets, rules)
├── evaluate_and_report.py  # Offline model evaluation & report generation
├── verify_db.py            # Database verification utility
├── model.pkl               # Trained GBM model (joblib)
├── scaler.pkl              # StandardScaler (joblib)
├── requirements.txt        # Python dependencies
├── Procfile                # Gunicorn start command for Render/Heroku
├── .env                    # Environment variables (not committed)
├── instance/
│   └── cardio.db           # SQLite database (auto-created on first run)
├── logs/
│   ├── app.log             # General application events
│   ├── error.log           # Error-level events only
│   └── prediction.log      # Per-prediction audit log
└── Weekly_Task/
    └── cardio_train.csv    # Training dataset (required for live metrics)
```

---

## 📡 API Endpoints

Base URL: `http://localhost:5000` (local) | `https://cardiocare-backend.onrender.com` (production)

| Method | Endpoint | Rate Limit | Description |
|---|---|---|---|
| GET | `/` | — | API info & endpoint listing |
| GET | `/api/health` | — | Model status & uptime |
| POST | `/api/predict` | 30/min | Run a risk prediction |
| GET | `/api/history` | — | Paginated prediction history |
| GET | `/api/history/<id>` | — | Single prediction detail |
| DELETE | `/api/history/<id>` | — | Delete a prediction record |
| GET | `/api/stats` | — | Prediction stats & daily trends |
| GET | `/api/metrics` | — | ROC curve, confusion matrix, classification report |
| GET | `/api/system` | — | CPU, memory, disk, uptime |
| POST | `/api/explain` | — | Feature importance for a given input |
| POST | `/api/model/reload` | — | Hot-reload model & scaler from disk |

---

### `POST /api/predict` — Request Body

| Field | Type | Range / Values | Description |
|---|---|---|---|
| `age` | int | 18 – 100 | Age in years |
| `gender` | int | 1 (Female), 2 (Male) | Biological sex |
| `height` | int | 120 – 250 | Height in cm |
| `weight` | float | 30 – 300 | Weight in kg |
| `ap_hi` | int | 60 – 250 | Systolic blood pressure (mmHg) |
| `ap_lo` | int | 30 – 200 | Diastolic blood pressure (mmHg) |
| `cholesterol` | int | 1 / 2 / 3 | Normal / Above Normal / Well Above Normal |
| `gluc` | int | 1 / 2 / 3 | Normal / Above Normal / Well Above Normal |
| `smoke` | int | 0 / 1 | No / Yes |
| `alco` | int | 0 / 1 | No / Yes |
| `active` | int | 0 / 1 | No / Yes |

### `POST /api/predict` — Example

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 50, "gender": 1, "height": 165, "weight": 72,
    "ap_hi": 140, "ap_lo": 90, "cholesterol": 2, "gluc": 1,
    "smoke": 0, "alco": 0, "active": 1
  }'
```

```json
{
  "status": "success",
  "result": "High Risk",
  "risk_level": "Moderate Risk",
  "probability": 61.45,
  "bmi": 26.4,
  "bmi_category": "Overweight",
  "tips": [
    { "icon": "💓", "text": "Blood pressure is elevated. Monitor dietary sodium and consult a doctor." },
    { "icon": "⚖️", "text": "Your BMI is 26.4 (Overweight). Weight management can improve heart health." }
  ],
  "feature_importance": [
    { "feature": "ap_hi", "importance": 0.312 },
    { "feature": "age",   "importance": 0.218 }
  ],
  "model_version": "v1.0",
  "response_time_ms": 14.2
}
```

---

## 🤖 ML Model

| Property | Value |
|---|---|
| **Algorithm** | Gradient Boosting Classifier (GBM) |
| **Dataset** | `cardio_train.csv` — 70,000 patient records |
| **Test Split** | 30% stratified |
| **Accuracy** | ~73% |
| **AUC-ROC** | ~0.80 |
| **Serialization** | `joblib` (model.pkl + scaler.pkl) |
| **Version** | v1.0 |

### Risk Thresholds

| Level | Condition |
|---|---|
| 🟢 Low Risk | probability < 40% |
| 🟡 Moderate Risk | 40% ≤ probability < 65% |
| 🔴 High Risk | probability ≥ 65% |

---

## 🚀 Setup & Installation

```bash
# 1. Clone and enter the backend directory
git clone https://github.com/your-username/CardioCare-AI.git
cd CardioCare-AI/backend

# 2. Create a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables (see below)
# Create a .env file in this directory

# 5. Run the server
python app.py
```

The API will be available at `http://localhost:5000`.

---

## 🔐 Environment Variables

Create a `.env` file inside the `backend/` directory:

```env
# Required
SECRET_KEY=your_secure_random_secret_key

# Optional — defaults to SQLite if not set
DATABASE_URL=sqlite:///instance/cardio.db

# Comma-separated list of allowed frontend origins
ALLOWED_ORIGINS=http://localhost:3000,https://your-frontend.vercel.app
```

**Generate a secret key:**
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

---

## ☁️ Deployment (Render)

1. Push this `backend/` folder (or the full repo) to GitHub
2. Create a new **Web Service** on [Render](https://render.com)
3. Configure:
   | Setting | Value |
   |---|---|
   | **Runtime** | Python 3 |
   | **Build Command** | `pip install -r requirements.txt` |
   | **Start Command** | `gunicorn app:app` |
4. Add environment variables: `SECRET_KEY`, `DATABASE_URL`, `ALLOWED_ORIGINS`
5. Deploy — Render will auto-deploy on every push to `main`

---

## 🪵 Logging

Three rotating log files are written to `logs/` (auto-created):

| File | Content |
|---|---|
| `app.log` | General request & event info |
| `error.log` | Error-level messages only |
| `prediction.log` | Per-prediction audit entries (IP, age, gender, result, probability, response time) |

Each file is capped at **1 MB** with up to **10 backups**.

---

## 🛡️ Rate Limits & Security

| Concern | Implementation |
|---|---|
| **CORS** | `Flask-CORS` — only `ALLOWED_ORIGINS` can access `/api/*` |
| **Security Headers** | `Flask-Talisman` (X-Frame-Options, X-Content-Type, etc.) |
| **Rate Limiting** | `Flask-Limiter` — 100 req/min globally, **30 req/min** on `/api/predict` |
| **Input Validation** | Field-level type casting, min/max range checks, cross-field validation |
| **BP Clamping** | `ap_hi` clamped to 70–220, `ap_lo` clamped to 40–140 before model inference |

---

> ⚠️ **Disclaimer:** This API is for educational and research purposes only. It is not a substitute for professional medical advice.
