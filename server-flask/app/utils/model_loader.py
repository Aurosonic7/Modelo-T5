# app/utils/model_loader.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

AVAILABLE_MODELS = {
    "t5-small": "t5-small",
    "t5-base": "t5-base",
    "t5-large": "t5-large",
    "google/flan-t5-base": "google/flan-t5-base",
    "google/pegasus-xsum": "google/pegasus-xsum",
    "facebook/bart-large": "facebook/bart-large",
    "google/mt5-small": "google/mt5-small",
    
    # Otros Helsinki
    "opus_mt_fr_en": "Helsinki-NLP/opus-mt-fr-en",

    # ─────────────────────────────────────────────────────────────────
    # NUEVO: Modelo para inglés -> español
    "opus_mt_en_es": "Helsinki-NLP/opus-mt-en-es",

    # Si conservas tu fine-tuned T5, etc.
    "fine_tuned_t5_small_translate": "/Users/.../fine_tuned_t5_small_translate",
}

model_cache = {}
tokenizer_cache = {}

def get_model(model_name: str):
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Modelo '{model_name}' no está soportado.")
    
    if model_name not in model_cache:
        model_path = AVAILABLE_MODELS[model_name]
        print(f"Cargando modelo desde: {model_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            add_prefix_space=False
        )

        # Fijar pad_token_id si es necesario
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        model.config.pad_token_id = tokenizer.pad_token_id

        # Guardar en caché
        model_cache[model_name] = model
        tokenizer_cache[model_name] = tokenizer
    else:
        print(f"Usando modelo en caché: {model_name}")
    
    # Determinar el tipo de modelo
    if "opus-mt" in model_name:
        model_type = "marian"  # Para MarianMT/Helsinki
    elif "t5" in model_name:
        model_type = "t5"
    elif "bart" in model_name:
        model_type = "bart"
    else:
        model_type = "other"
    
    return model_cache[model_name], tokenizer_cache[model_name], model_type

def clear_cache():
    global model_cache, tokenizer_cache
    model_cache = {}
    tokenizer_cache = {}
    print("Caché limpiado.")