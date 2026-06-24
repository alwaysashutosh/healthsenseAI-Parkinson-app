"""Doctor-scoped API: view patients, their reports, add diagnosis notes.

Doctors must be admin-approved (verified) — enforced by role_required(ROLE_DOCTOR).
"""
from flask import Blueprint, request, jsonify, g

from extensions import db
from models import Patient, PredictionHistory, Appointment, ROLE_DOCTOR, APPT_STATUSES
from auth import role_required

doctor_bp = Blueprint("doctor", __name__, url_prefix="/api/doctor")


@doctor_bp.get("/patients")
@role_required(ROLE_DOCTOR)
def patients():
    rows = Patient.query.all()
    out = []
    for p in rows:
        latest = (
            PredictionHistory.query
            .filter_by(patient_id=p.id)
            .order_by(PredictionHistory.created_at.desc())
            .first()
        )
        out.append({
            "patient_id": p.id,
            "name": p.user.name if p.user else None,
            "age": p.age,
            "gender": p.gender,
            "report_count": len(p.predictions),
            "latest_label": latest.label if latest else None,
            "latest_pd_probability": latest.pd_probability if latest else None,
        })
    return jsonify({"patients": out})


@doctor_bp.get("/patients/<int:patient_id>/reports")
@role_required(ROLE_DOCTOR)
def patient_reports(patient_id):
    p = db.session.get(Patient, patient_id)
    if p is None:
        return jsonify({"error": "Patient not found"}), 404
    rows = (
        PredictionHistory.query
        .filter_by(patient_id=patient_id)
        .order_by(PredictionHistory.created_at.desc())
        .all()
    )
    return jsonify({
        "patient": {
            "name": p.user.name if p.user else None,
            "age": p.age, "gender": p.gender, "phone": p.phone,
            "medical_history": p.medical_history,
        },
        "reports": [r.to_dict() for r in rows],
    })


@doctor_bp.post("/reports/<int:report_id>/notes")
@role_required(ROLE_DOCTOR)
def add_notes(report_id):
    report = db.session.get(PredictionHistory, report_id)
    if report is None:
        return jsonify({"error": "Report not found"}), 404
    data = request.get_json(silent=True) or {}
    report.doctor_notes = data.get("notes", "")
    report.reviewed_by = g.current_user.doctor.id
    db.session.commit()
    return jsonify({"report": report.to_dict()})


@doctor_bp.get("/appointments")
@role_required(ROLE_DOCTOR)
def appointments():
    rows = (
        Appointment.query
        .filter_by(doctor_id=g.current_user.doctor.id)
        .order_by(Appointment.date.desc(), Appointment.time.desc())
        .all()
    )
    return jsonify({"appointments": [a.to_dict() for a in rows]})


@doctor_bp.post("/appointments/<int:appt_id>/status")
@role_required(ROLE_DOCTOR)
def update_status(appt_id):
    appt = db.session.get(Appointment, appt_id)
    if appt is None or appt.doctor_id != g.current_user.doctor.id:
        return jsonify({"error": "Appointment not found"}), 404
    status = (request.get_json(silent=True) or {}).get("status")
    if status not in APPT_STATUSES:
        return jsonify({"error": f"Status must be one of {sorted(APPT_STATUSES)}"}), 400
    appt.status = status
    db.session.commit()
    return jsonify({"appointment": appt.to_dict()})
