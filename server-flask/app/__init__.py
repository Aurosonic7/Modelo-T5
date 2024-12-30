# app/__init__.py

from flask import Flask
from .config import load_config
from .routes import register_routes

def create_app():
    app = Flask(__name__)
    load_config(app)

    # Añade esta línea para que Flask no "escape" caracteres en JSON
    app.config['JSON_AS_ASCII'] = False
    
    # Registrar rutas
    register_routes(app)
    
    return app