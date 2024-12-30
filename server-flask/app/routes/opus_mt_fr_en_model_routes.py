# app/routes/opus_mt_fr_en_model_routes.py
from flask import Blueprint, request, jsonify
from app.utils.model_loader import get_model, AVAILABLE_MODELS
from app.utils.inference import run_inference

opus_mt_fr_en_bp = Blueprint("opus_mt_fr_en_bp", __name__)

@opus_mt_fr_en_bp.route("/translate", methods=["POST"])
def opus_mt_fr_en_translate():
    """
    Endpoint para traducción francés->inglés usando Helsinki-NLP/opus-mt-fr-en
    Espera un JSON como:
    { "input_text": "Bonjour, comment ça va?" }
    """
    return _handle_operation("opus_mt_fr_en", "translate")

@opus_mt_fr_en_bp.route("/summary", methods=["POST"])
def opus_mt_fr_en_summary():
    return _handle_operation("opus_mt_fr_en", "summary")

@opus_mt_fr_en_bp.route("/generate", methods=["POST"])
def opus_mt_fr_en_generate():
    return _handle_operation("opus_mt_fr_en", "generate")

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