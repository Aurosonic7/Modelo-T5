from flask import Blueprint, jsonify, request
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    BartForConditionalGeneration,
    PegasusForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
import torch
from server.app.utils import model_manager  # Importa el gestor del modelo

bp = Blueprint("model_routes", __name__)

# Diccionario para modelos soportados
MODEL_REGISTRY = {
    "t5-small": "t5-small",
    "t5-base": "t5-base",
    "t5-large": "t5-large",
    "flan-t5": "google/flan-t5-base",
    "pegasus": "google/pegasus-xsum",
    "bart": "facebook/bart-large"
}

@bp.route("/api/models/select", methods=["POST"])
def select_model():
    data = request.get_json()
    new_model_name = data.get("model_name", "")

    if new_model_name not in MODEL_REGISTRY:
        return jsonify({"error": f"Modelo '{new_model_name}' no soportado."}), 400

    # Carga el nuevo modelo y tokenizador
    model_name = MODEL_REGISTRY[new_model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Actualiza el modelo global
    model_manager.set_model_and_tokenizer(model_name, tokenizer, model)

    return jsonify({"message": f"Modelo cambiado a '{new_model_name}'"}), 200

@bp.route("/api/models/fine_tune", methods=["POST"])
def fine_tune_model():
    data = request.get_json()
    dataset = data.get("dataset", [])
    epochs = data.get("epochs", 3)
    learning_rate = data.get("learning_rate", 5e-5)

    if not dataset:
        return jsonify({"error": "El dataset es obligatorio para el ajuste fino."}), 400

    model, tokenizer = model_manager.get_current_model_and_tokenizer()

    # Convertir dataset en tensores
    inputs = [example["input"] for example in dataset]
    targets = [example["target"] for example in dataset]

    input_encodings = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
    target_encodings = tokenizer(targets, padding=True, truncation=True, return_tensors="pt")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(input_ids=input_encodings["input_ids"], labels=target_encodings["input_ids"])
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    return jsonify({"message": f"Modelo ajustado durante {epochs} epochs con éxito."}), 200