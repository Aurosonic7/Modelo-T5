# app/utils/inference.py
import torch

def run_inference(model, tokenizer, input_text, operation, model_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    if operation == "translate":
        if model_type == "marian":
            # Para Helsinki (opus-mt-en-es), manda SOLO el texto en inglés
            prompt = input_text
        else:
            # Para T5, BART, etc.
            prompt = f"translate English to Spanish: {input_text}"

    elif operation == "summary":
        prompt = f"summarize: {input_text}"
    elif operation == "generate":
        prompt = f"generate text: {input_text}"
    else:
        raise ValueError(f"Operación '{operation}' no soportada.")
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=128,
            num_beams=4,
            early_stopping=True,
            do_sample=False
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    result = result.replace("Traduce Inglés al Español: ", "")
    # print(result)

    return result