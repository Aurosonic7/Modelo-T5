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