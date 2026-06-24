"""SQLAlchemy models (SQLite): RBAC users + role profiles + predictions.

Roles: PATIENT, DOCTOR, ADMIN. A User holds credentials + role; role-specific
attributes live in the Patient / Doctor profile tables linked 1:1 to the user.
"""
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

from extensions import db

ROLE_PATIENT = "patient"
ROLE_DOCTOR = "doctor"
ROLE_ADMIN = "admin"
ROLES = {ROLE_PATIENT, ROLE_DOCTOR, ROLE_ADMIN}


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120))
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(255), unique=True, nullable=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(16), nullable=False, default=ROLE_PATIENT, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    patient = db.relationship("Patient", backref="user", uselist=False, cascade="all, delete-orphan")
    doctor = db.relationship("Doctor", backref="user", uselist=False, cascade="all, delete-orphan")

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def to_dict(self):
        data = {
            "id": self.id,
            "name": self.name,
            "username": self.username,
            "email": self.email,
            "role": self.role,
        }
        if self.role == ROLE_PATIENT and self.patient:
            data["profile"] = self.patient.to_dict()
        elif self.role == ROLE_DOCTOR and self.doctor:
            data["profile"] = self.doctor.to_dict()
        return data


class Patient(db.Model):
    __tablename__ = "patients"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), unique=True, nullable=False)
    age = db.Column(db.Integer)
    gender = db.Column(db.String(16))
    phone = db.Column(db.String(32))
    address = db.Column(db.String(255))
    medical_history = db.Column(db.Text)

    predictions = db.relationship("PredictionHistory", backref="patient", lazy=True)

    def to_dict(self):
        return {
            "id": self.id,
            "age": self.age,
            "gender": self.gender,
            "phone": self.phone,
            "address": self.address,
            "medical_history": self.medical_history,
        }


class Doctor(db.Model):
    __tablename__ = "doctors"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), unique=True, nullable=False)
    registration_number = db.Column(db.String(64))
    specialization = db.Column(db.String(120))
    hospital = db.Column(db.String(255))
    experience = db.Column(db.Integer)         # years
    verified = db.Column(db.Boolean, default=False)  # admin approval gate

    def to_dict(self):
        return {
            "id": self.id,
            "registration_number": self.registration_number,
            "specialization": self.specialization,
            "hospital": self.hospital,
            "experience": self.experience,
            "verified": self.verified,
        }


APPT_PENDING = "Pending"
APPT_APPROVED = "Approved"
APPT_COMPLETED = "Completed"
APPT_CANCELLED = "Cancelled"
APPT_STATUSES = {APPT_PENDING, APPT_APPROVED, APPT_COMPLETED, APPT_CANCELLED}


class Appointment(db.Model):
    __tablename__ = "appointments"

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey("patients.id"), nullable=False, index=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey("doctors.id"), nullable=False, index=True)
    date = db.Column(db.String(10))   # YYYY-MM-DD
    time = db.Column(db.String(5))    # HH:MM
    reason = db.Column(db.Text)
    status = db.Column(db.String(16), default=APPT_PENDING)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    patient = db.relationship("Patient", backref="appointments")
    doctor = db.relationship("Doctor", backref="appointments")

    def to_dict(self):
        return {
            "id": self.id,
            "patient_id": self.patient_id,
            "doctor_id": self.doctor_id,
            "date": self.date,
            "time": self.time,
            "reason": self.reason,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "patient_name": self.patient.user.name if self.patient and self.patient.user else None,
            "doctor_name": self.doctor.user.name if self.doctor and self.doctor.user else None,
            "doctor_specialization": self.doctor.specialization if self.doctor else None,
        }


class PredictionHistory(db.Model):
    __tablename__ = "prediction_history"

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey("patients.id"), nullable=True, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    source = db.Column(db.String(16), default="record")   # 'record' | 'upload'
    filename = db.Column(db.String(255))
    label = db.Column(db.String(32))                       # 'Healthy' | "Parkinson's"
    pd_probability = db.Column(db.Float)
    healthy_probability = db.Column(db.Float)
    # Doctor review (foundation for the doctor workflow)
    doctor_notes = db.Column(db.Text)
    reviewed_by = db.Column(db.Integer, db.ForeignKey("doctors.id"), nullable=True)

    def to_dict(self, with_patient=False):
        data = {
            "id": self.id,
            "patient_id": self.patient_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "source": self.source,
            "filename": self.filename,
            "label": self.label,
            "pd_probability": self.pd_probability,
            "healthy_probability": self.healthy_probability,
            "doctor_notes": self.doctor_notes,
            "reviewed": self.reviewed_by is not None,
        }
        if with_patient and self.patient and self.patient.user:
            data["patient_name"] = self.patient.user.name
            data["patient_age"] = self.patient.age
            data["patient_gender"] = self.patient.gender
        return data
