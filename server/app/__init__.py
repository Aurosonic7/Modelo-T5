from flask import Flask
from dotenv import load_dotenv

def create_app():
    # Cargar variables de entorno
    load_dotenv()

    # Crear instancia de Flask
    app = Flask(__name__)
    app.config.from_object('server.config.DevelopmentConfig')

    # Registrar rutas
    from .routes import example_routes
    app.register_blueprint(example_routes.bp)

    # Integración del Blueprint model_routes
    from server.app.routes.model_routes import bp as model_routes
    app.register_blueprint(model_routes)

    return app