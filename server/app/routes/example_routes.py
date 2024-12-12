from flask import Blueprint, jsonify, request
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Inicializa el modelo T5
model_name = "t5-small"  # Puedes usar otros modelos como t5-base o t5-large
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

bp = Blueprint('example_routes', __name__)

@bp.route('/api/t5/translate', methods=['POST'])
def translate():
    """
    Endpoint para traducir texto utilizando T5.
    """
    data = request.get_json()
    text = data.get('text', '')
    source_lang = data.get('source_lang', 'English')
    target_lang = data.get('target_lang', 'French')

    if not text:
        return jsonify({"error": "El texto es obligatorio."}), 400

    # Prepara la entrada para T5
    input_text = f"translate {source_lang} to {target_lang}: {text}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    # Genera la salida
    outputs = model.generate(input_ids)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"translated_text": translated_text})

@bp.route('/api/t5/summarize', methods=['POST'])
def summarize():
    """
    Endpoint para resumir texto utilizando T5.
    """
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({"error": "El texto es obligatorio."}), 400

    # Preparar entrada para T5
    input_text = f"summarize: {text}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    # Generar salida
    outputs = model.generate(input_ids, max_length=50, num_beams=4, early_stopping=True)
    summarized_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"summarized_text": summarized_text})

@bp.route('/api/t5/generate', methods=['POST'])
def generate_text():
    """
    Endpoint para generar texto utilizando T5.
    """
    data = request.get_json()
    prompt = data.get('prompt', '')

    if not prompt:
        return jsonify({"error": "El prompt es obligatorio."}), 400

    # Preparar entrada para T5
    input_text = f"complete: {prompt}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    # Generar salida
    outputs = model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"generated_text": generated_text})