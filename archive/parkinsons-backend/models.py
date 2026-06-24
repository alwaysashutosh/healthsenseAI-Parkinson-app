from flask_sqlalchemy import SQLAlchemy
from app import app
import time

db = SQLAlchemy(app)


class file_results(db.Model):
    time = db.Column(db.Integer, primary_key=True)
    ip = db.Column(db.String(16), primary_key=True)
    filename = db.Column(db.String(100))
    Presult = db.Column(db.Integer)

    def __init__(self, filename, ip, result):
        self.time = int(time.time())
        self.ip = ip
        self.filename = filename
        self.Presult = result


