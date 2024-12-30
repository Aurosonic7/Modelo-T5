# app/routes/opus_mt_en_es_model_routes.py

from flask import Blueprint, request, jsonify
from app.utils.model_loader import get_model, AVAILABLE_MODELS
from app.utils.inference import run_inference

opus_mt_en_es_bp = Blueprint("opus_mt_en_es_bp", __name__)

@opus_mt_en_es_bp.route("/translate", methods=["POST"])
def opus_mt_en_es_translate():
    """
    Endpoint para traducción inglés->español usando Helsinki-NLP/opus-mt-en-es.
    Espera un JSON como:
    { "input_text": "Hello, how are you?" }
    """
    return _handle_operation("opus_mt_en_es", "translate")

@opus_mt_en_es_bp.route("/summary", methods=["POST"])
def opus_mt_en_es_summary():
    return _handle_operation("opus_mt_en_es", "summary")

@opus_mt_en_es_bp.route("/generate", methods=["POST"])
def opus_mt_en_es_generate():
    return _handle_operation("opus_mt_en_es", "generate")

def _handle_operation(model_name, operation):
    data = request.get_json()
    input_text = data.get("input_text", "")

    if not input_text:
        return jsonify({"error": "Falta el parámetro 'input_text'"}), 400

    if model_name not in AVAILABLE_MODELS:
        return jsonify({"error": f"Modelo '{model_name}' no está en AVAILABLE_MODELS"}), 400

    model, tokenizer, model_type = get_model(model_name)
    result = run_inference(model, tokenizer, input_text, operation, model_type)

    return jsonify({
        "model": model_name,
        "operation": operation,
        "result": result
    })