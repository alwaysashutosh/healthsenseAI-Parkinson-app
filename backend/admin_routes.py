"""Admin-scoped API: approve doctors, manage users, view platform analytics."""
from flask import Blueprint, jsonify

from extensions import db
from models import User, Patient, Doctor, PredictionHistory, Appointment, ROLE_ADMIN
from auth import role_required

admin_bp = Blueprint("admin", __name__, url_prefix="/api/admin")


@admin_bp.get("/doctors")
@role_required(ROLE_ADMIN)
def doctors():
    rows = Doctor.query.all()
    return jsonify({"doctors": [
        {
            "doctor_id": d.id,
            "name": d.user.name if d.user else None,
            "username": d.user.username if d.user else None,
            "registration_number": d.registration_number,
            "specialization": d.specialization,
            "hospital": d.hospital,
            "experience": d.experience,
            "verified": d.verified,
        }
        for d in rows
    ]})


@admin_bp.post("/doctors/<int:doctor_id>/approve")
@role_required(ROLE_ADMIN)
def approve(doctor_id):
    d = db.session.get(Doctor, doctor_id)
    if d is None:
        return jsonify({"error": "Doctor not found"}), 404
    d.verified = True
    db.session.commit()
    return jsonify({"ok": True, "doctor_id": d.id, "verified": True})


@admin_bp.post("/doctors/<int:doctor_id>/revoke")
@role_required(ROLE_ADMIN)
def revoke(doctor_id):
    d = db.session.get(Doctor, doctor_id)
    if d is None:
        return jsonify({"error": "Doctor not found"}), 404
    d.verified = False
    db.session.commit()
    return jsonify({"ok": True, "doctor_id": d.id, "verified": False})


@admin_bp.get("/users")
@role_required(ROLE_ADMIN)
def users():
    rows = User.query.order_by(User.created_at.desc()).all()
    return jsonify({"users": [u.to_dict() for u in rows]})


@admin_bp.get("/stats")
@role_required(ROLE_ADMIN)
def stats():
    return jsonify({
        "patients": Patient.query.count(),
        "doctors": Doctor.query.count(),
        "doctors_pending": Doctor.query.filter_by(verified=False).count(),
        "predictions": PredictionHistory.query.count(),
        "parkinsons_flagged": PredictionHistory.query.filter_by(label="Parkinson's").count(),
        "appointments": Appointment.query.count(),
    })
