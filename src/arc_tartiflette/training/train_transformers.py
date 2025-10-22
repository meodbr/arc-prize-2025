from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
import torch
from datasets import DatasetDict

def train_transformers(
        model, 
        tokenized_datasets: DatasetDict, 
        tokenizer,
        output_model="default_output_model",
    ):
    use_bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False

    # Define training arguments using Transformers
    training_args = TrainingArguments(
        output_dir="./data/models/smollm2_arc_kaggle_without_trl",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        learning_rate=5e-5,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        weight_decay=0.01,
        report_to="none",
        push_to_hub=True,

        # Precision config
        bf16=use_bf16,
        fp16=not use_bf16,
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
    )

    # Start training
    print("Train")
    trainer.train()

    # Eval
    print("Eval")
    results = trainer.evaluate()
    print(results)