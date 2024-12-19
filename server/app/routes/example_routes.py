from flask import Blueprint, jsonify, request
from server.app.utils import model_manager  # Importa el gestor del modelo

bp = Blueprint('example_routes', __name__)

@bp.route('/api/t5/translate', methods=['POST'])
def translate():
    data = request.get_json()
    text = data.get('text', '')
    source_lang = data.get('source_lang', 'English')
    target_lang = data.get('target_lang', 'French')

    if not text:
        return jsonify({"error": "El texto es obligatorio."}), 400

    model, tokenizer = model_manager.get_current_model_and_tokenizer()

    input_text = f"translate {source_lang} to {target_lang}: {text}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"translated_text": translated_text})

@bp.route('/api/t5/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({"error": "El texto es obligatorio."}), 400

    model, tokenizer = model_manager.get_current_model_and_tokenizer()

    input_text = f"summarize: {text}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=50, num_beams=4, early_stopping=True)
    summarized_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"summarized_text": summarized_text})

@bp.route('/api/t5/generate', methods=['POST'])
def generate_text():
    data = request.get_json()
    prompt = data.get('prompt', '')

    if not prompt:
        return jsonify({"error": "El prompt es obligatorio."}), 400

    model, tokenizer = model_manager.get_current_model_and_tokenizer()

    input_text = f"complete: {prompt}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"generated_text": generated_text})