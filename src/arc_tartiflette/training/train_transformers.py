from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
import torch
from datasets import DatasetDict
import os

def train_transformers(
        model, 
        tokenized_datasets: DatasetDict, 
        tokenizer,
        output_model="default_output_model",
    ):
    """
    Train a model using the Transformers Trainer API.
    """
    # Retrieve training args in environment variables
    batch_size = int(os.environ.get("BATCH_SIZE", "4"))
    gradient_accumulation_steps = int(os.environ.get("GRAD_ACC_STEPS", "2"))
    gradient_checkpointing = os.environ.get("GRAD_CHPT", "true").lower() == "true"
    learning_rate = float(os.environ.get("LR", "5e-5"))
    num_train_epochs = float(os.environ.get("NUM_EPOCHS", "3"))


    use_bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False

    print(f"---- Starting training with Transformers Trainer ----")
    print(f"Batch size: {batch_size}\nGradient Accumulation Steps: {gradient_accumulation_steps}\nGradient Checkpointing: {gradient_checkpointing}\nLearning Rate: {learning_rate}\nNum Train Epochs: {num_train_epochs}\nUsing bf16: {use_bf16}")

    # Define training arguments using Transformers
    training_args = TrainingArguments(
        output_dir="./data/models/smollm2_arc_kaggle_without_trl",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        learning_rate=learning_rate,
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