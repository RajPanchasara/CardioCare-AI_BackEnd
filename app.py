"""
HeartSense Backend — Production-Ready ML API
=============================================
Cardiovascular Disease Prediction API with:
- Input validation & preprocessing
- Model management & versioning
- Audit logging (app / error / prediction)
- Rate limiting & security
- System monitoring & metrics
- Feature explainability
"""

import json
import time
import traceback
import logging
import os
import threading
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from functools import wraps

import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc
)
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify, g
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_talisman import Talisman
from flask_compress import Compress
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from config import Config

# ────────────────────────────────────────────
# App Factory
# ────────────────────────────────────────────
app = Flask(__name__)
app.config.from_object(Config)

db = SQLAlchemy(app)
Compress(app)
Talisman(app, content_security_policy=None, force_https=False, force_https_permanent=False)
CORS(app, resources={r"/api/*": {"origins": Config.ALLOWED_ORIGINS}})

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=[Config.RATE_LIMIT_DEFAULT],
    storage_uri="memory://",
)

APP_START_TIME = time.time()

# ────────────────────────────────────────────
# Logging Setup (3 files: app, error, prediction)
# ────────────────────────────────────────────
os.makedirs(Config.LOG_DIR, exist_ok=True)

_log_fmt = logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
)


def _make_handler(path, level=logging.INFO):
    h = RotatingFileHandler(path, maxBytes=Config.LOG_MAX_BYTES, backupCount=Config.LOG_BACKUP_COUNT)
    h.setFormatter(_log_fmt)
    h.setLevel(level)
    return h


# App logger (general events)
app.logger.addHandler(_make_handler(Config.APP_LOG))
app.logger.setLevel(logging.INFO)

# Error logger
error_logger = logging.getLogger('error')
error_logger.addHandler(_make_handler(Config.ERROR_LOG, logging.ERROR))
error_logger.setLevel(logging.ERROR)

# Prediction logger
prediction_logger = logging.getLogger('prediction')
prediction_logger.addHandler(_make_handler(Config.PREDICTION_LOG))
prediction_logger.setLevel(logging.INFO)

app.logger.info('HeartSense Backend Startup — version %s', Config.MODEL_VERSION)

# ────────────────────────────────────────────
# Database Models
# ────────────────────────────────────────────


class Prediction(db.Model):
    """Stores every prediction made by the system."""
    id = db.Column(db.Integer, primary_key=True)
    input_data = db.Column(db.Text)          # Raw JSON input
    age = db.Column(db.Integer)
    gender = db.Column(db.String(10))
    height = db.Column(db.Integer)
    weight = db.Column(db.Float)
    ap_hi = db.Column(db.Integer)
    ap_lo = db.Column(db.Integer)
    cholesterol = db.Column(db.String(20))
    gluc = db.Column(db.String(20))
    smoke = db.Column(db.String(5))
    alco = db.Column(db.String(5))
    active = db.Column(db.String(5))
    prediction_result = db.Column(db.String(20))
    risk_level = db.Column(db.String(20))
    probability = db.Column(db.Float)
    model_version = db.Column(db.String(20))
    response_time_ms = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)


class AuditLog(db.Model):
    """Tracks all significant system events."""
    id = db.Column(db.Integer, primary_key=True)
    action = db.Column(db.String(50))        # PREDICTION, ERROR, STARTUP, MODEL_RELOAD
    details = db.Column(db.Text)             # JSON blob
    ip_address = db.Column(db.String(50))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)


class RequestMetric(db.Model):
    """Stores per-request performance data."""
    id = db.Column(db.Integer, primary_key=True)
    endpoint = db.Column(db.String(100))
    method = db.Column(db.String(10))
    status_code = db.Column(db.Integer)
    response_time_ms = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)


with app.app_context():
    db.create_all()
    # Log startup audit
    startup_entry = AuditLog(
        action="STARTUP",
        details=json.dumps({"model_version": Config.MODEL_VERSION}),
        ip_address="server",
    )
    db.session.add(startup_entry)
    db.session.commit()

# ────────────────────────────────────────────
# Model Loading
# ────────────────────────────────────────────
model = None
scaler = None


def load_model():
    """Load model and scaler from disk. Returns (model, scaler) tuple."""
    global model, scaler
    try:
        model = joblib.load(Config.MODEL_PATH)
        scaler = joblib.load(Config.SCALER_PATH)
        app.logger.info('Model loaded successfully — %s', Config.MODEL_VERSION)
        return True
    except Exception as e:
        error_logger.error('Failed to load model: %s\n%s', str(e), traceback.format_exc())
        model = None
        scaler = None
        return False


load_model()


# ────────────────────────────────────────────
# Model Metrics Cache (computed once at startup)
# ────────────────────────────────────────────
MODEL_METRICS_CACHE = {}


def compute_model_metrics():
    """Compute classification report, confusion matrix, ROC, and correlation at startup."""
    global MODEL_METRICS_CACHE

    csv_path = os.path.join(Config.BASE_DIR, 'Weekly_Task', 'cardio_train.csv')
    source = 'computed'

    try:
        if model is None or scaler is None:
            raise RuntimeError('Model not loaded')

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f'Training CSV not found at {csv_path}')

        # Load training data
        df = pd.read_csv(csv_path, sep=';')

        # Preprocess: convert age from days to years if needed
        if df['age'].mean() > 200:  # likely in days
            df['age'] = (df['age'] / 365.25).astype(int)

        # Features and target
        feature_cols = Config.FEATURE_NAMES
        target_col = 'cardio'

        X = df[feature_cols].values
        y = df[target_col].values

        # Train/test split (same as training)
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Scale
        X_test_scaled = scaler.transform(X_test)

        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = float(auc(fpr, tpr))

        # Downsample ROC for JSON (max 100 points)
        step = max(1, len(fpr) // 100)
        fpr_list = [round(float(f), 4) for f in fpr[::step]]
        tpr_list = [round(float(t), 4) for t in tpr[::step]]

        # Feature correlation matrix
        corr = df[feature_cols].corr()
        corr_matrix = [[round(float(v), 2) for v in row] for row in corr.values]

        MODEL_METRICS_CACHE = {
            'source': source,
            'model': {
                'type': 'Gradient Boosting Classifier',
                'accuracy': round(report.get('accuracy', 0), 4),
                'auc': round(roc_auc, 4),
                'dataset_size': len(df),
                'test_size': len(y_test),
                'features': feature_cols,
                'version': Config.MODEL_VERSION,
            },
            'classification_report': {
                '0': {
                    'precision': round(report['0']['precision'], 2),
                    'recall': round(report['0']['recall'], 2),
                    'f1': round(report['0']['f1-score'], 2),
                    'support': int(report['0']['support']),
                },
                '1': {
                    'precision': round(report['1']['precision'], 2),
                    'recall': round(report['1']['recall'], 2),
                    'f1': round(report['1']['f1-score'], 2),
                    'support': int(report['1']['support']),
                },
                'macro_avg': {
                    'precision': round(report['macro avg']['precision'], 2),
                    'recall': round(report['macro avg']['recall'], 2),
                    'f1': round(report['macro avg']['f1-score'], 2),
                    'support': int(report['macro avg']['support']),
                },
                'weighted_avg': {
                    'precision': round(report['weighted avg']['precision'], 2),
                    'recall': round(report['weighted avg']['recall'], 2),
                    'f1': round(report['weighted avg']['f1-score'], 2),
                    'support': int(report['weighted avg']['support']),
                },
            },
            'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]],
            'roc_curve': {
                'fpr': fpr_list,
                'tpr': tpr_list,
                'auc': round(roc_auc, 4),
            },
            'correlation_matrix': {
                'features': feature_cols,
                'matrix': corr_matrix,
            },
        }
        app.logger.info('Model metrics computed successfully (source: %s, AUC: %.4f)', source, roc_auc)

    except Exception as e:
        app.logger.warning('Could not compute metrics from data: %s — using static fallback', str(e))
        MODEL_METRICS_CACHE = {
            'source': 'static',
            'model': {
                'type': 'Gradient Boosting Classifier',
                'accuracy': 0.73,
                'auc': 0.80,
                'dataset_size': 70000,
                'test_size': 21000,
                'features': Config.FEATURE_NAMES,
                'version': Config.MODEL_VERSION,
            },
            'classification_report': {
                '0': {'precision': 0.71, 'recall': 0.81, 'f1': 0.76, 'support': 10353},
                '1': {'precision': 0.77, 'recall': 0.65, 'f1': 0.70, 'support': 9914},
                'macro_avg': {'precision': 0.74, 'recall': 0.73, 'f1': 0.73, 'support': 20267},
                'weighted_avg': {'precision': 0.74, 'recall': 0.73, 'f1': 0.73, 'support': 20267},
            },
            'confusion_matrix': [[8180, 2173], [3185, 6729]],
            'roc_curve': {
                'fpr': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                'tpr': [0.0, 0.45, 0.62, 0.72, 0.78, 0.83, 0.87, 0.91, 0.95, 0.98, 1.0],
                'auc': 0.80,
            },
            'correlation_matrix': {
                'features': Config.FEATURE_NAMES,
                'matrix': [],
            },
        }


threading.Thread(target=compute_model_metrics, daemon=True).start()

# ────────────────────────────────────────────
# Request Timing Middleware
# ────────────────────────────────────────────


@app.before_request
def _start_timer():
    g.start_time = time.time()


@app.after_request
def _log_request(response):
    if hasattr(g, 'start_time'):
        elapsed_ms = (time.time() - g.start_time) * 1000
        # Store metric
        try:
            metric = RequestMetric(
                endpoint=request.path,
                method=request.method,
                status_code=response.status_code,
                response_time_ms=round(elapsed_ms, 2),
            )
            db.session.add(metric)
            db.session.commit()
        except Exception:
            db.session.rollback()

        app.logger.info('%s %s %s — %.1fms',
                        request.method, request.path, response.status_code, elapsed_ms)
    return response


# ────────────────────────────────────────────
# Validation Helpers
# ────────────────────────────────────────────


def validate_input(data):
    """
    Validate prediction input against Config.VALIDATION_RULES.
    Returns (cleaned_data_dict, errors_dict).
    """
    errors = {}
    cleaned = {}

    if not data:
        return None, {"_general": "Request body is empty or not valid JSON"}

    for field, rules in Config.VALIDATION_RULES.items():
        raw = data.get(field)

        # Missing field
        if raw is None or raw == '':
            errors[field] = f"Required field '{field}' is missing"
            continue

        # Type casting
        try:
            value = rules['type'](raw)
        except (ValueError, TypeError):
            errors[field] = f"'{field}' must be a valid {rules['type'].__name__}"
            continue

        # Range check
        if value < rules['min'] or value > rules['max']:
            errors[field] = f"'{field}' must be between {rules['min']} and {rules['max']}, got {value}"
            continue

        cleaned[field] = value

    # Cross-field validation: ap_hi must be > ap_lo
    if 'ap_hi' in cleaned and 'ap_lo' in cleaned:
        if cleaned['ap_hi'] <= cleaned['ap_lo']:
            errors['ap_hi'] = f"Systolic pressure ({cleaned['ap_hi']}) must be greater than diastolic ({cleaned['ap_lo']})"

    return cleaned, errors


def clamp_bp(values):
    """Clamp blood pressure values to medically reasonable ranges for the model."""
    bp_clamp = Config.BP_CLAMP
    values['ap_hi'] = max(bp_clamp['ap_hi']['min'], min(bp_clamp['ap_hi']['max'], values['ap_hi']))
    values['ap_lo'] = max(bp_clamp['ap_lo']['min'], min(bp_clamp['ap_lo']['max'], values['ap_lo']))
    return values


def classify_risk(probability):
    """Return risk level string based on probability thresholds."""
    if probability < Config.RISK_THRESHOLDS['low']:
        return "Low Risk"
    elif probability < Config.RISK_THRESHOLDS['moderate']:
        return "Moderate Risk"
    else:
        return "High Risk"


def generate_tips(values, bmi, bmi_category):
    """Generate personalized health tips based on input values."""
    tips = []
    if values['smoke'] == 1:
        tips.append({"icon": "🚭", "text": "Smoking significantly increases heart disease risk. Consider a cessation program."})
    if values['alco'] == 1:
        tips.append({"icon": "🍷", "text": "Limit alcohol consumption to moderate levels to reduce heart strain."})
    if values['active'] == 0:
        tips.append({"icon": "🏃", "text": "Aim for at least 30 minutes of moderate activity daily."})
    if bmi >= 25:
        tips.append({"icon": "⚖️", "text": f"Your BMI is {bmi} ({bmi_category}). Weight management can improve heart health."})
    if values['ap_hi'] > 130 or values['ap_lo'] > 85:
        tips.append({"icon": "💓", "text": "Blood pressure is elevated. Monitor dietary sodium and consult a doctor."})
    if values['cholesterol'] > 1:
        tips.append({"icon": "🍔", "text": "Cholesterol levels are high. Consider a diet low in saturated fats."})
    if values['gluc'] > 1:
        tips.append({"icon": "🍬", "text": "Glucose levels are above normal. Monitor sugar intake and consider regular checkups."})
    if values['age'] >= 60:
        tips.append({"icon": "🧓", "text": "Age is a significant risk factor. Regular cardiovascular screenings are recommended."})
    if len(tips) == 0:
        tips.append({"icon": "✅", "text": "Great job! Your reported lifestyle habits align with heart health."})
    return tips


def get_feature_importance():
    """Extract feature importances from the model if available."""
    if model is None:
        return None

    importances = None
    # Tree-based models (RandomForest, GradientBoosting, DecisionTree, etc.)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    # Linear models (LogisticRegression, SVM with linear kernel)
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
    else:
        return None

    feature_names = Config.FEATURE_NAMES
    importance_dict = {}
    for name, imp in zip(feature_names, importances):
        importance_dict[name] = round(float(imp), 4)

    # Sort descending
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    return [{"feature": f, "importance": v} for f, v in sorted_features]


# ────────────────────────────────────────────
# API Endpoints
# ────────────────────────────────────────────


@app.route('/')
def home():
    return jsonify({
        "status": "online",
        "message": "HeartSense — Cardiovascular Disease Prediction API",
        "version": Config.MODEL_VERSION,
        "endpoints": {
            "predict": "POST /api/predict",
            "health": "GET /api/health",
            "history": "GET /api/history",
            "history_detail": "GET /api/history/<id>",
            "stats": "GET /api/stats",
            "metrics": "GET /api/metrics",
            "system": "GET /api/system",
            "explain": "POST /api/explain",
            "model_reload": "POST /api/model/reload",
        }
    })


# ── Health ──────────────────────────────────

@app.route('/api/health', methods=['GET'])
def api_health():
    return jsonify({
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "model_version": Config.MODEL_VERSION,
        "uptime_seconds": round(time.time() - APP_START_TIME, 1),
    })


# ── Predict ─────────────────────────────────

@app.route('/api/predict', methods=['POST'])
@limiter.limit("30/minute")
def api_predict():
    req_start = time.time()

    # --- Model guard ---
    if model is None or scaler is None:
        error_logger.error('Prediction attempted but model is not loaded')
        return jsonify({
            "status": "error",
            "error": "Model not available",
            "details": "The prediction model failed to load. Contact the administrator.",
            "code": 503
        }), 503

    try:
        data = request.get_json(silent=True)

        # --- Validate ---
        cleaned, errors = validate_input(data)
        if errors:
            app.logger.warning('Validation failed: %s', json.dumps(errors))
            return jsonify({
                "status": "error",
                "error": "Invalid input",
                "details": errors,
                "code": 400
            }), 400

        # --- Preprocess ---
        clamped = clamp_bp(cleaned.copy())

        # BMI calculation
        height_m = cleaned['height'] / 100
        bmi = round(cleaned['weight'] / (height_m ** 2), 1)
        bmi_category = (
            "Underweight" if bmi < 18.5 else
            "Normal Weight" if bmi < 25 else
            "Overweight" if bmi < 30 else
            "Obese"
        )

        # Build feature vector in exact training order
        features = np.array([[
            clamped['age'], clamped['gender'], clamped['height'], clamped['weight'],
            clamped['ap_hi'], clamped['ap_lo'], clamped['cholesterol'], clamped['gluc'],
            clamped['smoke'], clamped['alco'], clamped['active']
        ]])

        # Scale
        features_scaled = scaler.transform(features)

        # --- Predict ---
        prediction = int(model.predict(features_scaled)[0])
        probability = float(model.predict_proba(features_scaled)[0][1])
        probability_pct = round(probability * 100, 2)

        # Risk classification (3-tier)
        risk_level = classify_risk(probability)

        # Result text for backward compatibility
        result_text = "High Risk" if prediction == 1 else "Low Risk"

        # Health tips
        tips = generate_tips(cleaned, bmi, bmi_category)

        # Feature importance
        feature_imp = get_feature_importance()

        # Response time
        response_time_ms = round((time.time() - req_start) * 1000, 2)

        # --- String mappings ---
        gender_str = "Male" if cleaned['gender'] == 2 else "Female"
        smoke_str = "Yes" if cleaned['smoke'] == 1 else "No"
        alco_str = "Yes" if cleaned['alco'] == 1 else "No"
        active_str = "Yes" if cleaned['active'] == 1 else "No"
        chol_map = {1: 'Normal', 2: 'Above Normal', 3: 'Well Above Normal'}
        gluc_map = {1: 'Normal', 2: 'Above Normal', 3: 'Well Above Normal'}
        chol_str = chol_map.get(cleaned['cholesterol'], 'Unknown')
        gluc_str = gluc_map.get(cleaned['gluc'], 'Unknown')

        # --- Save to DB ---
        new_entry = Prediction(
            input_data=json.dumps(cleaned),
            age=cleaned['age'],
            gender=gender_str,
            height=cleaned['height'],
            weight=cleaned['weight'],
            ap_hi=cleaned['ap_hi'],
            ap_lo=cleaned['ap_lo'],
            cholesterol=chol_str,
            gluc=gluc_str,
            smoke=smoke_str,
            alco=alco_str,
            active=active_str,
            prediction_result=result_text,
            risk_level=risk_level,
            probability=probability_pct,
            model_version=Config.MODEL_VERSION,
            response_time_ms=response_time_ms,
        )
        db.session.add(new_entry)

        # Audit log
        audit = AuditLog(
            action="PREDICTION",
            details=json.dumps({
                "input": cleaned,
                "result": result_text,
                "risk_level": risk_level,
                "probability": probability_pct,
                "response_time_ms": response_time_ms,
            }),
            ip_address=request.remote_addr,
        )
        db.session.add(audit)
        db.session.commit()

        # Prediction log
        prediction_logger.info(
            'PREDICTION | IP=%s | age=%d gender=%s result=%s risk=%s prob=%.2f%% time=%.1fms',
            request.remote_addr, cleaned['age'], gender_str,
            result_text, risk_level, probability_pct, response_time_ms,
        )

        return jsonify({
            "status": "success",
            "result": result_text,
            "risk_level": risk_level,
            "probability": probability_pct,
            "bmi": bmi,
            "bmi_category": bmi_category,
            "tips": tips,
            "feature_importance": feature_imp,
            "model_version": Config.MODEL_VERSION,
            "response_time_ms": response_time_ms,
        })

    except Exception as e:
        db.session.rollback()
        error_logger.error('Prediction error: %s\n%s', str(e), traceback.format_exc())

        # Audit log the error
        try:
            audit = AuditLog(
                action="ERROR",
                details=json.dumps({"error": str(e), "traceback": traceback.format_exc()}),
                ip_address=request.remote_addr,
            )
            db.session.add(audit)
            db.session.commit()
        except Exception:
            db.session.rollback()

        return jsonify({
            "status": "error",
            "error": "Prediction failed",
            "details": "An internal error occurred. Please try again or contact support.",
            "code": 500
        }), 500


# ── History ─────────────────────────────────

@app.route('/api/history', methods=['GET'])
def api_history():
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    per_page = min(per_page, 100)  # Cap at 100

    pagination = Prediction.query.order_by(Prediction.timestamp.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )

    history_data = []
    for p in pagination.items:
        history_data.append({
            "id": p.id,
            "date": p.timestamp.strftime("%Y-%m-%d %H:%M"),
            "age": p.age,
            "gender": p.gender,
            "result": p.prediction_result,
            "risk_level": p.risk_level,
            "probability": p.probability,
            "model_version": p.model_version,
        })

    return jsonify({
        "history": history_data,
        "pagination": {
            "page": pagination.page,
            "per_page": pagination.per_page,
            "total": pagination.total,
            "pages": pagination.pages,
            "has_next": pagination.has_next,
            "has_prev": pagination.has_prev,
        }
    })


@app.route('/api/history/<int:pred_id>', methods=['GET'])
def api_history_detail(pred_id):
    p = db.session.get(Prediction, pred_id)
    if not p:
        return jsonify({"status": "error", "error": "Prediction not found", "code": 404}), 404

    return jsonify({
        "id": p.id,
        "input_data": json.loads(p.input_data) if p.input_data else None,
        "date": p.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "age": p.age,
        "gender": p.gender,
        "height": p.height,
        "weight": p.weight,
        "ap_hi": p.ap_hi,
        "ap_lo": p.ap_lo,
        "cholesterol": p.cholesterol,
        "gluc": p.gluc,
        "smoke": p.smoke,
        "alco": p.alco,
        "active": p.active,
        "result": p.prediction_result,
        "risk_level": p.risk_level,
        "probability": p.probability,
        "model_version": p.model_version,
        "response_time_ms": p.response_time_ms,
    })


@app.route('/api/history/<int:pred_id>', methods=['DELETE'])
def api_history_delete(pred_id):
    p = db.session.get(Prediction, pred_id)
    if not p:
        return jsonify({"status": "error", "error": "Prediction not found", "code": 404}), 404

    db.session.delete(p)

    audit = AuditLog(
        action="DELETE_PREDICTION",
        details=json.dumps({"prediction_id": pred_id}),
        ip_address=request.remote_addr,
    )
    db.session.add(audit)
    db.session.commit()

    app.logger.info('Prediction %d deleted by %s', pred_id, request.remote_addr)
    return jsonify({"status": "success", "message": f"Prediction {pred_id} deleted"})


# ── Stats ───────────────────────────────────

@app.route('/api/stats', methods=['GET'])
def api_stats():
    total = Prediction.query.count()
    high_risk = Prediction.query.filter(Prediction.prediction_result == "High Risk").count()
    low_risk = Prediction.query.filter(Prediction.prediction_result == "Low Risk").count()
    moderate_risk = Prediction.query.filter(Prediction.risk_level == "Moderate Risk").count()

    avg_prob = db.session.query(db.func.avg(Prediction.probability)).scalar() or 0

    # Total visits: count GET requests (excluding API endpoints and static files)
    total_visits = RequestMetric.query.filter(
        RequestMetric.method == 'GET',
        ~RequestMetric.endpoint.startswith('/api/'),
        RequestMetric.status_code < 400,
    ).count()

    # Predictions per day (last 7 days)
    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    daily_counts = (
        db.session.query(
            db.func.date(Prediction.timestamp).label('date'),
            db.func.count(Prediction.id).label('count')
        )
        .filter(Prediction.timestamp >= seven_days_ago)
        .group_by(db.func.date(Prediction.timestamp))
        .order_by(db.func.date(Prediction.timestamp))
        .all()
    )

    return jsonify({
        "total_visits": total_visits,
        "total_predictions": total,
        "high_risk_count": high_risk,
        "moderate_risk_count": moderate_risk,
        "low_risk_count": low_risk,
        "average_probability": round(float(avg_prob), 2),
        "predictions_per_day": [
            {"date": str(row.date), "count": row.count}
            for row in daily_counts
        ],
        "model_version": Config.MODEL_VERSION,
    })


# ── Metrics ─────────────────────────────────

@app.route('/api/metrics', methods=['GET'])
def api_metrics():
    total_requests = RequestMetric.query.count()
    avg_response = db.session.query(db.func.avg(RequestMetric.response_time_ms)).scalar() or 0

    error_count = RequestMetric.query.filter(RequestMetric.status_code >= 400).count()
    error_rate = round((error_count / total_requests * 100), 2) if total_requests > 0 else 0

    total_predictions = Prediction.query.count()
    avg_pred_time = (
        db.session.query(db.func.avg(Prediction.response_time_ms))
        .filter(Prediction.response_time_ms.isnot(None))
        .scalar() or 0
    )

    return jsonify({
        "total_requests": total_requests,
        "total_predictions": total_predictions,
        "avg_response_time_ms": round(float(avg_response), 2),
        "avg_prediction_time_ms": round(float(avg_pred_time), 2),
        "error_count": error_count,
        "error_rate_pct": error_rate,
        "uptime_seconds": round(time.time() - APP_START_TIME, 1),
    })


# ── System ──────────────────────────────────

@app.route('/api/system', methods=['GET'])
def api_system():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()

    return jsonify({
        "cpu_usage_pct": psutil.cpu_percent(interval=0.1),
        "memory_usage_mb": round(mem_info.rss / (1024 * 1024), 1),
        "model_loaded": model is not None,
        "model_version": Config.MODEL_VERSION,
        "uptime_seconds": round(time.time() - APP_START_TIME, 1),
        "python_pid": os.getpid(),
        "disk_usage_pct": psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:\\').percent,
    })


# ── Explain ─────────────────────────────────

@app.route('/api/explain', methods=['POST'])
@limiter.limit("20/minute")
def api_explain():
    if model is None:
        return jsonify({"status": "error", "error": "Model not loaded", "code": 503}), 503

    data = request.get_json(silent=True)
    cleaned, errors = validate_input(data)
    if errors:
        return jsonify({"status": "error", "error": "Invalid input", "details": errors, "code": 400}), 400

    # Feature importance from the model
    feature_imp = get_feature_importance()
    if feature_imp is None:
        return jsonify({
            "status": "error",
            "error": "Explainability not supported for this model type",
            "code": 501
        }), 501

    # Also run prediction to give context
    clamped = clamp_bp(cleaned.copy())
    features = np.array([[
        clamped['age'], clamped['gender'], clamped['height'], clamped['weight'],
        clamped['ap_hi'], clamped['ap_lo'], clamped['cholesterol'], clamped['gluc'],
        clamped['smoke'], clamped['alco'], clamped['active']
    ]])
    features_scaled = scaler.transform(features)
    probability = float(model.predict_proba(features_scaled)[0][1])
    risk_level = classify_risk(probability)

    # Per-feature contribution (input * importance)
    contributions = []
    for i, name in enumerate(Config.FEATURE_NAMES):
        imp = next((f['importance'] for f in feature_imp if f['feature'] == name), 0)
        contributions.append({
            "feature": name,
            "value": cleaned[name],
            "importance": imp,
            "contribution_score": round(float(features_scaled[0][i]) * imp, 4),
        })
    contributions.sort(key=lambda x: abs(x['contribution_score']), reverse=True)

    return jsonify({
        "status": "success",
        "risk_level": risk_level,
        "probability_pct": round(probability * 100, 2),
        "feature_importance": feature_imp,
        "feature_contributions": contributions,
        "model_version": Config.MODEL_VERSION,
    })


# ── Model Reload ────────────────────────────

@app.route('/api/model/reload', methods=['POST'])
@limiter.limit("5/minute")
def api_model_reload():
    app.logger.info('Model reload requested by %s', request.remote_addr)
    success = load_model()

    audit = AuditLog(
        action="MODEL_RELOAD",
        details=json.dumps({"success": success, "version": Config.MODEL_VERSION}),
        ip_address=request.remote_addr,
    )
    db.session.add(audit)
    db.session.commit()

    if success:
        return jsonify({
            "status": "success",
            "message": "Model reloaded successfully",
            "model_version": Config.MODEL_VERSION,
        })
    else:
        return jsonify({
            "status": "error",
            "error": "Model reload failed — check logs",
            "code": 500
        }), 500


# ── Model Metrics (Cached) ──────────────────

@app.route('/api/model-metrics', methods=['GET'])
def api_model_metrics():
    if not MODEL_METRICS_CACHE:
        return jsonify({
            "status": "computing",
            "message": "Model metrics are being computed. Please retry in a few seconds.",
        }), 202
    return jsonify(MODEL_METRICS_CACHE)


# ── Error Handlers ──────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({"status": "error", "error": "Endpoint not found", "code": 404}), 404


@app.errorhandler(429)
def rate_limited(e):
    return jsonify({"status": "error", "error": "Rate limit exceeded. Please slow down.", "code": 429}), 429


@app.errorhandler(500)
def internal_error(e):
    db.session.rollback()
    error_logger.error('Internal server error: %s', str(e))
    return jsonify({"status": "error", "error": "Internal server error", "code": 500}), 500


# ────────────────────────────────────────────
# Main
# ────────────────────────────────────────────
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
