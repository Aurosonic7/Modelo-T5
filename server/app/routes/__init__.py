from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def preload_models():
    models = [
        "t5-small",
        "t5-base",
        "t5-large",
        "google/flan-t5-base",
        "google/pegasus-xsum",
        "facebook/bart-large"
    ]
    for model_name in models:
        print(f"Pre-cargando modelo: {model_name}")
        AutoTokenizer.from_pretrained(model_name)
        AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Llamar a preload_models() al inicio
preload_models()