# fine_tune_model.py

import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset, concatenate_datasets
from config import CLEANED_TEXT_DIR, MODEL_DIR, BASE_MODEL_NAME

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    
    # Set pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME)
    
    # Load and prepare datasets using the datasets library
    data_files = [
        os.path.join(CLEANED_TEXT_DIR, f)
        for f in os.listdir(CLEANED_TEXT_DIR)
        if f.endswith('.txt')
    ]
    datasets = []
    for file_path in data_files:
        # Check if the file is empty
        if os.path.getsize(file_path) == 0:
            print(f"Skipping empty file: {file_path}")
            continue
        try:
            dataset = load_dataset(
                'text',
                data_files={'train': file_path},
                encoding='utf-8',
                split='train'
            )
            if len(dataset) == 0:
                print(f"No examples found in dataset loaded from {file_path}, skipping.")
                continue
            datasets.append(dataset)
            print(f"Loaded dataset from {file_path} with {len(dataset)} examples.")
        except Exception as e:
            print(f'Error loading {file_path}: {e}')
            print('Skipping this file.')
            continue

    # Check if we have any datasets to concatenate
    if not datasets:
        print("No datasets to train on. Exiting.")
        return

    # Concatenate all datasets into one
    if len(datasets) > 1:
        train_dataset = concatenate_datasets(datasets)
    else:
        train_dataset = datasets[0]

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=512,
            padding='max_length',  # Ensure all sequences are of equal length
        )

    tokenized_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )

    # Initialize the Data Collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Set training arguments
    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,  # Adjust based on VRAM
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        fp16=True,  # Enable mixed precision if supported
        prediction_loss_only=True,
    )

    # Initialize the Trainer with the data collator
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print("Starting fine-tuning...")
    trainer.train()
    print("Fine-tuning completed.")
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

if __name__ == '__main__':
    main()
