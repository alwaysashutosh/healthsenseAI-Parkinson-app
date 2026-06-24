"""Authentication + RBAC: role-based register / login (JWT) and access decorators."""
import datetime
from functools import wraps

import jwt
from flask import Blueprint, request, jsonify, current_app, g

from extensions import db
from models import User, Patient, Doctor, ROLE_PATIENT, ROLE_DOCTOR, ROLE_ADMIN

auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")


# --------------------------------------------------------------------------- #
# Token helpers
# --------------------------------------------------------------------------- #
def make_token(user):
    payload = {
        "sub": str(user.id),
        "role": user.role,
        "exp": datetime.datetime.utcnow()
        + datetime.timedelta(hours=current_app.config["JWT_EXP_HOURS"]),
        "iat": datetime.datetime.utcnow(),
    }
    return jwt.encode(payload, current_app.config["JWT_SECRET"], algorithm="HS256")


def _extract_token():
    header = request.headers.get("Authorization", "")
    return header[7:] if header.startswith("Bearer ") else None


def _user_from_token(token):
    try:
        payload = jwt.decode(token, current_app.config["JWT_SECRET"], algorithms=["HS256"])
        return db.session.get(User, int(payload["sub"]))
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Decorators
# --------------------------------------------------------------------------- #
def token_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        user = _user_from_token(_extract_token() or "")
        if user is None:
            return jsonify({"error": "Authentication required"}), 401
        g.current_user = user
        return f(*args, **kwargs)
    return wrapper


def role_required(*roles):
    """Require an authenticated user whose role is in `roles`."""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            user = _user_from_token(_extract_token() or "")
            if user is None:
                return jsonify({"error": "Authentication required"}), 401
            if user.role not in roles:
                return jsonify({"error": "Forbidden: insufficient role"}), 403
            # Doctors must be admin-approved before acting.
            if user.role == ROLE_DOCTOR and user.doctor and not user.doctor.verified:
                return jsonify({"error": "Your doctor account is awaiting admin approval"}), 403
            g.current_user = user
            return f(*args, **kwargs)
        return wrapper
    return decorator


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@auth_bp.post("/register")
def register():
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    role = (data.get("role") or ROLE_PATIENT).strip().lower()

    if len(username) < 3 or len(password) < 6:
        return jsonify({"error": "Username (>=3) and password (>=6) required"}), 400
    if role not in (ROLE_PATIENT, ROLE_DOCTOR):
        return jsonify({"error": "Role must be 'patient' or 'doctor'"}), 400
    if User.query.filter_by(username=username).first():
        return jsonify({"error": "Username already taken"}), 409

    user = User(
        username=username,
        name=(data.get("name") or "").strip() or username,
        email=(data.get("email") or "").strip() or None,
        role=role,
    )
    user.set_password(password)
    db.session.add(user)
    db.session.flush()  # assign user.id

    if role == ROLE_PATIENT:
        db.session.add(Patient(
            user_id=user.id,
            age=_to_int(data.get("age")),
            gender=data.get("gender"),
            phone=data.get("phone"),
            address=data.get("address"),
            medical_history=data.get("medical_history"),
        ))
    else:  # doctor — created unverified, awaits admin approval
        db.session.add(Doctor(
            user_id=user.id,
            registration_number=data.get("registration_number"),
            specialization=data.get("specialization"),
            hospital=data.get("hospital"),
            experience=_to_int(data.get("experience")),
            verified=False,
        ))

    db.session.commit()
    return jsonify({"token": make_token(user), "user": user.to_dict()}), 201


@auth_bp.post("/login")
def login():
    data = request.get_json(silent=True) or {}
    user = User.query.filter_by(username=(data.get("username") or "").strip()).first()
    if user is None or not user.check_password(data.get("password") or ""):
        return jsonify({"error": "Invalid credentials"}), 401
    return jsonify({"token": make_token(user), "user": user.to_dict()})


@auth_bp.get("/me")
@token_required
def me():
    return jsonify({"user": g.current_user.to_dict()})


def _to_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None
