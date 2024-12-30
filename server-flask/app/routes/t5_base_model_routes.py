# app/routes/t5_base_model_routes.py
from flask import Blueprint, request, jsonify
from app.utils.model_loader import get_model, AVAILABLE_MODELS
from app.utils.inference import run_inference

t5_base_model_bp = Blueprint("t5_base_model_bp", __name__)

@t5_base_model_bp.route("/translate", methods=["POST"])
def t5_base_translate():
    return _handle_operation("t5-base", "translate")

@t5_base_model_bp.route("/summary", methods=["POST"])
def t5_base_summary():
    return _handle_operation("t5-base", "summary")

@t5_base_model_bp.route("/generate", methods=["POST"])
def t5_base_generate():
    return _handle_operation("t5-base", "generate")

def _handle_operation(model_name, operation):
    data = request.get_json()
    input_text = data.get("input_text")

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