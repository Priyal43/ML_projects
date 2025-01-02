from flask import Flask
from app.routes import main

def create_app():
    app = Flask(__name__)
    app.secret_key = "supersecretkey"  # Replace with a unique, non-guessable string
    app.register_blueprint(main)
    return app
