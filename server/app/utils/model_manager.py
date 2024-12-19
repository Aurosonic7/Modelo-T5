from transformers import T5Tokenizer, T5ForConditionalGeneration

# Variables globales iniciales
_current_model_name = "t5-small"
_current_tokenizer = T5Tokenizer.from_pretrained(_current_model_name)
_current_model = T5ForConditionalGeneration.from_pretrained(_current_model_name)

def get_current_model_and_tokenizer():
    return _current_model, _current_tokenizer

def set_model_and_tokenizer(model_name, tokenizer, model):
    global _current_model_name, _current_model, _current_tokenizer
    _current_model_name = model_name
    _current_tokenizer = tokenizer
    _current_model = model