"""Flask application factory — Parkinson's healthcare platform (RBAC).

React frontend -> Flask REST API (auth + RBAC: patient/doctor/admin) ->
KNN inference + SQLite (users, patients, doctors, predictions).

Run (dev):   python backend/app.py
Run (prod):  gunicorn "app:create_app()"   (from the backend/ directory)
"""
import os

from flask import Flask, jsonify
from flask_cors import CORS

from config import Config
from extensions import db


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    CORS(app, resources={r"/api/*": {"origins": "*"}})  # tighten origins in prod
    db.init_app(app)

    from auth import auth_bp
    from routes import api_bp
    from patient_routes import patient_bp
    from doctor_routes import doctor_bp
    from admin_routes import admin_bp
    from hospitals import hospitals_bp
    for bp in (auth_bp, api_bp, patient_bp, doctor_bp, admin_bp, hospitals_bp):
        app.register_blueprint(bp)

    with app.app_context():
        import models  # noqa: F401  ensure models are registered
        db.create_all()
        _seed_admin(app)

    @app.get("/")
    def index():
        return jsonify({
            "service": "Parkinson's Detection Healthcare Platform",
            "framework": "Flask",
            "roles": ["patient", "doctor", "admin"],
        })

    return app


def _seed_admin(app):
    """Create a default admin on first run (admins are not self-registerable)."""
    from models import User, ROLE_ADMIN
    username = os.environ.get("ADMIN_USERNAME", "admin")
    password = os.environ.get("ADMIN_PASSWORD", "admin12345")
    if User.query.filter_by(role=ROLE_ADMIN).first():
        return
    admin = User(username=username, name="Administrator", role=ROLE_ADMIN)
    admin.set_password(password)
    db.session.add(admin)
    db.session.commit()
    app.logger.warning("Seeded default admin '%s' — change ADMIN_PASSWORD in production.", username)


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
