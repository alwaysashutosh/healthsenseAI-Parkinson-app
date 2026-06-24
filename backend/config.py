"""Flask application configuration."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-change-me-in-production-0123456789")
    JWT_SECRET = os.environ.get("JWT_SECRET", SECRET_KEY)
    JWT_EXP_HOURS = int(os.environ.get("JWT_EXP_HOURS", "24"))

    # SQLite database (report-specified persistence layer)
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'parkinsons.db')}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # ML artifacts produced by ml/train_knn.py
    ARTIFACTS_DIR = os.environ.get(
        "ARTIFACTS_DIR", os.path.join(PROJECT_ROOT, "ml", "artifacts")
    )
    ML_DIR = os.path.join(PROJECT_ROOT, "ml")

    # Upload handling
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB
    ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
