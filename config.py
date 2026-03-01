"""
HeartSense Backend Configuration
Centralizes all paths, secrets, and settings.
"""
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    """Base configuration."""
    BASE_DIR = BASE_DIR
    _secret = os.getenv('SECRET_KEY')
    if not _secret:
        raise RuntimeError(
            "SECRET_KEY environment variable is not set. "
            "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
        )
    SECRET_KEY = _secret

    # Database
    SQLALCHEMY_DATABASE_URI = os.getenv(
        'DATABASE_URL',
        f'sqlite:///{os.path.join(BASE_DIR, "instance", "cardio.db")}'
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Model
    MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
    SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')
    MODEL_VERSION = "v1.0"

    # Logging
    LOG_DIR = os.path.join(BASE_DIR, 'logs')
    APP_LOG = os.path.join(LOG_DIR, 'app.log')
    ERROR_LOG = os.path.join(LOG_DIR, 'error.log')
    PREDICTION_LOG = os.path.join(LOG_DIR, 'prediction.log')
    LOG_MAX_BYTES = 1_048_576  # 1 MB
    LOG_BACKUP_COUNT = 10

    # Security
    ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', 'http://localhost:3001').split(',')
    RATE_LIMIT_DEFAULT = "100/minute"

    # Feature names (must match training order exactly)
    FEATURE_NAMES = [
        'age', 'gender', 'height', 'weight',
        'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
        'smoke', 'alco', 'active'
    ]

    # Validation ranges
    VALIDATION_RULES = {
        'age':         {'type': int,   'min': 18,  'max': 100},
        'gender':      {'type': int,   'min': 1,   'max': 2},
        'height':      {'type': int,   'min': 120, 'max': 250},
        'weight':      {'type': float, 'min': 30,  'max': 300},
        'ap_hi':       {'type': int,   'min': 60,  'max': 250},
        'ap_lo':       {'type': int,   'min': 30,  'max': 200},
        'cholesterol': {'type': int,   'min': 1,   'max': 3},
        'gluc':        {'type': int,   'min': 1,   'max': 3},
        'smoke':       {'type': int,   'min': 0,   'max': 1},
        'alco':        {'type': int,   'min': 0,   'max': 1},
        'active':      {'type': int,   'min': 0,   'max': 1},
    }

    # Blood pressure clamping for model input (outlier protection)
    BP_CLAMP = {
        'ap_hi': {'min': 70, 'max': 220},
        'ap_lo': {'min': 40, 'max': 140},
    }

    # Risk classification thresholds
    RISK_THRESHOLDS = {
        'low':      0.40,   # probability < 40%
        'moderate': 0.65,   # 40% <= probability < 65%
        # >= 65% is High Risk
    }
