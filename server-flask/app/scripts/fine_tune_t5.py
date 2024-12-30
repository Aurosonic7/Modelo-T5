# app/scripts/fine_tune_t5.py

from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset
import pandas as pd
import os

def main():
    data_path = "app/scripts/data/translation_dataset_opus_books.csv"

    if not os.path.exists(data_path):
        print(f"Error: El archivo '{data_path}' no existe.")
        return

    df = pd.read_csv(data_path)

    if df.empty:
        print("Error: El archivo CSV está vacío.")
        return

    if 'source' not in df.columns or 'target' not in df.columns:
        print("Error: El archivo CSV debe contener las columnas 'source' y 'target'.")
        return

    if df['source'].isnull().any() or df['target'].isnull().any():
        print("Error: Hay valores faltantes en las columnas 'source' o 'target'.")
        return

    # Crear dataset de HF
    dataset = Dataset.from_pandas(df)
    dataset = dataset.train_test_split(test_size=0.2)

    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    def preprocess_function(examples):
        inputs = [f"translate English to Spanish: {text}" for text in examples["source"]]
        model_inputs = tokenizer(inputs, max_length=128, truncation=True)
        
        labels = tokenizer(
            text_target=examples["target"],
            max_length=128,
            truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    training_args = TrainingArguments(
        output_dir="./fine_tuned_t5_small_translate",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=2,  # <--- baja a 4 o incluso 2
        per_device_eval_batch_size=4,
        num_train_epochs=10,
        weight_decay=0.01,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
    )

    trainer.train()

    # Guardar el modelo
    trainer.save_model("./fine_tuned_t5_small_translate")
    tokenizer.save_pretrained("./fine_tuned_t5_small_translate")

    print("Fine-tuning completado. Modelo guardado en './fine_tuned_t5_small_translate'.")

if __name__ == "__main__":
    main()