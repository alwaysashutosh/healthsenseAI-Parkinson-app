from flask import Flask
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ['DATABASE_URL']


from routes import *
import models

if __name__ == '__main__':
    app.run()