"""Patient-scoped API: profile, voice prediction, own reports."""
import os
import tempfile

from flask import Blueprint, request, jsonify, g, current_app
from werkzeug.utils import secure_filename

from extensions import db
from models import (
    PredictionHistory, Doctor, Appointment, ROLE_PATIENT,
    APPT_PENDING, APPT_COMPLETED, APPT_CANCELLED,
)
from auth import role_required
from ml_service import ml_service

patient_bp = Blueprint("patient", __name__, url_prefix="/api/patient")


def _profile():
    return g.current_user.patient


@patient_bp.get("/profile")
@role_required(ROLE_PATIENT)
def get_profile():
    return jsonify({"user": g.current_user.to_dict()})


@patient_bp.put("/profile")
@role_required(ROLE_PATIENT)
def update_profile():
    data = request.get_json(silent=True) or {}
    p = _profile()
    if "name" in data:
        g.current_user.name = data["name"]
    for field in ("age", "gender", "phone", "address", "medical_history"):
        if field in data:
            setattr(p, field, _to_int(data[field]) if field == "age" else data[field])
    db.session.commit()
    return jsonify({"user": g.current_user.to_dict()})


def _allowed(filename):
    return os.path.splitext(filename)[1].lower() in current_app.config["ALLOWED_EXTENSIONS"]


@patient_bp.post("/predict")
@role_required(ROLE_PATIENT)
def predict():
    if "file" not in request.files or not request.files["file"].filename:
        return jsonify({"error": "No audio file provided (field 'file')"}), 400
    f = request.files["file"]
    if not _allowed(f.filename):
        return jsonify({"error": "Unsupported file type"}), 415
    if not ml_service.ready:
        return jsonify({"error": "Model not available."}), 503

    filename = secure_filename(f.filename)
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=os.path.splitext(filename)[1] or ".wav")
        os.close(fd)
        f.save(tmp_path)
        result = ml_service.predict(tmp_path)
    except Exception as e:
        current_app.logger.exception("Prediction failed")
        return jsonify({"error": f"Could not process audio: {e}"}), 422
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    record = PredictionHistory(
        patient_id=_profile().id,
        source=request.form.get("source", "record"),
        filename=filename,
        label=result["label"],
        pd_probability=result["pd_probability"],
        healthy_probability=result["healthy_probability"],
    )
    db.session.add(record)
    db.session.commit()

    result["report_id"] = record.id
    result["disclaimer"] = (
        "Screening aid only, not a medical diagnosis. Feature extraction from "
        "recorded audio is approximate relative to the UCI training distribution."
    )
    return jsonify(result)


@patient_bp.get("/reports")
@role_required(ROLE_PATIENT)
def reports():
    rows = (
        PredictionHistory.query
        .filter_by(patient_id=_profile().id)
        .order_by(PredictionHistory.created_at.desc())
        .all()
    )
    return jsonify({"reports": [r.to_dict() for r in rows]})


# --------------------------------------------------------------------------- #
# Appointments
# --------------------------------------------------------------------------- #
@patient_bp.get("/doctors")
@role_required(ROLE_PATIENT)
def verified_doctors():
    """Verified doctors a patient can book with."""
    rows = Doctor.query.filter_by(verified=True).all()
    return jsonify({"doctors": [
        {
            "doctor_id": d.id,
            "name": d.user.name if d.user else None,
            "specialization": d.specialization,
            "hospital": d.hospital,
            "experience": d.experience,
        }
        for d in rows
    ]})


@patient_bp.post("/appointments")
@role_required(ROLE_PATIENT)
def book_appointment():
    data = request.get_json(silent=True) or {}
    doctor_id = data.get("doctor_id")
    doctor = db.session.get(Doctor, doctor_id) if doctor_id else None
    if doctor is None or not doctor.verified:
        return jsonify({"error": "Select a valid, approved doctor"}), 400
    if not data.get("date") or not data.get("time"):
        return jsonify({"error": "Date and time are required"}), 400

    appt = Appointment(
        patient_id=_profile().id,
        doctor_id=doctor.id,
        date=data["date"],
        time=data["time"],
        reason=data.get("reason"),
        status=APPT_PENDING,
    )
    db.session.add(appt)
    db.session.commit()
    return jsonify({"appointment": appt.to_dict()}), 201


@patient_bp.get("/appointments")
@role_required(ROLE_PATIENT)
def my_appointments():
    rows = (
        Appointment.query
        .filter_by(patient_id=_profile().id)
        .order_by(Appointment.date.desc(), Appointment.time.desc())
        .all()
    )
    return jsonify({"appointments": [a.to_dict() for a in rows]})


@patient_bp.post("/appointments/<int:appt_id>/cancel")
@role_required(ROLE_PATIENT)
def cancel_appointment(appt_id):
    appt = db.session.get(Appointment, appt_id)
    if appt is None or appt.patient_id != _profile().id:
        return jsonify({"error": "Appointment not found"}), 404
    if appt.status == APPT_COMPLETED:
        return jsonify({"error": "Completed appointments cannot be cancelled"}), 400
    appt.status = APPT_CANCELLED
    db.session.commit()
    return jsonify({"appointment": appt.to_dict()})


def _to_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None
