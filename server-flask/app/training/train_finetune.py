import argparse
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a T5 model for translation.")
    parser.add_argument("--model_name", type=str, required=True, help="Nombre del modelo preentrenado.")
    parser.add_argument("--data_file", type=str, required=True, help="Ruta al archivo CSV de datos.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directorio para guardar el modelo fine-tuned.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Número de épocas de entrenamiento.")
    parser.add_argument("--batch_size", type=int, default=16, help="Tamaño del batch.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Pasos de acumulación de gradiente.")
    parser.add_argument("--fp16", action='store_true', help="Habilitar entrenamiento de precisión mixta (fp16).")
    args = parser.parse_args()

    # Cargar el tokenizer y el modelo
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    # Cargar y preparar el dataset
    print(f"Loading dataset from {args.data_file}")
    dataset = load_dataset('csv', data_files=args.data_file, split='train')

    # Tokenización
    def tokenize_function(examples):
        # Tokenizar el texto de entrada
        inputs = tokenizer(
            examples["text"],
            max_length=128,
            padding="max_length",
            truncation=True,
        )
        # Tokenizar el texto de salida (labels)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["summary"],
                max_length=128,
                padding="max_length",
                truncation=True,
            )
        # Asignar las labels al conjunto de datos
        inputs["labels"] = labels["input_ids"]
        return inputs

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Configurar los argumentos de entrenamiento
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="no",  # Actualizado de 'evaluation_strategy' a 'eval_strategy'
        save_strategy="epoch",
        fp16=args.fp16,
        logging_steps=100,
        save_total_limit=2,
    )

    # Crear el Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
    )

    # Entrenar el modelo
    trainer.train()

    # Guardar el modelo
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Fine-tuned model saved to {args.output_dir}")

if __name__ == "__main__":
    main()