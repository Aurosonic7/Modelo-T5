import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def run_inference(model, tokenizer, text, operation):
    if operation == "translate":
        prompt = f"translate English to Spanish: {text}"
        generation_params = dict(
            max_length=80,
            num_beams=4,
            early_stopping=True
        )
    elif operation == "summary":
        prompt = f"summarize: {text}"
        generation_params = dict(
            max_length=80,
            num_beams=4,
            early_stopping=True
        )
    else:
        prompt = text
        generation_params = dict(
            max_length=100,
            num_beams=4,
            early_stopping=True
        )

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = inputs.to(model.device)  # Asegura que los inputs estén en el mismo dispositivo que el modelo
    outputs = model.generate(**inputs, **generation_params)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("Prompt utilizado:", prompt)
    print("Entrada tokenizada:", inputs)
    print("Salida generada:", outputs)

    return result

if __name__ == "__main__":
    model_name = "/Users/christianvicente/Desktop/Flask-Server/server-flask/fine_tuned_opus_mt_en_es"  # Ruta correcta al modelo fine-tuned
    print("Ruta absoluta del modelo:", os.path.abspath(model_name))
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True)

    # Asegúrate de que el modelo está en el dispositivo correcto
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print("Modelo y tokenizer cargados correctamente.")

    input_text = "Hello, how are you today?"
    operation = "translate"
    translation = run_inference(model, tokenizer, input_text, operation)
    print(f"Translated Text: {translation}")