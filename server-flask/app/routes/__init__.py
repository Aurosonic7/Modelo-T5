# app/routes/__init__.py

from .t5_small_model_routes import t5_small_model_bp
from .t5_base_model_routes import t5_base_model_bp
from .t5_large_model_routes import t5_large_model_bp
from .flan_t5_model_routes import flan_t5_model_bp
from .pegasus_model_routes import pegasus_model_bp
from .bart_model_routes import bart_model_bp
from .mt5_model_routes import mt5_model_bp
from .opus_mt_fr_en_model_routes import opus_mt_fr_en_bp

# NUEVO
from .opus_mt_en_es_model_routes import opus_mt_en_es_bp

def register_routes(app):
    app.register_blueprint(t5_small_model_bp, url_prefix="/api/t5-small")
    app.register_blueprint(t5_base_model_bp, url_prefix="/api/t5-base")
    app.register_blueprint(t5_large_model_bp, url_prefix="/api/t5-large")
    app.register_blueprint(flan_t5_model_bp, url_prefix="/api/flan-t5")
    app.register_blueprint(pegasus_model_bp, url_prefix="/api/pegasus")
    app.register_blueprint(bart_model_bp, url_prefix="/api/bart")
    app.register_blueprint(mt5_model_bp, url_prefix="/api/mt5")
    app.register_blueprint(opus_mt_fr_en_bp, url_prefix="/api/opus-fr-en")

    # Registro para EN->ES
    app.register_blueprint(opus_mt_en_es_bp, url_prefix="/api/opus-en-es")