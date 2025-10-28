from trl import SFTConfig, SFTTrainer
from datasets import Dataset, DatasetDict
import torch

from arc_tartiflette.model_tools.tokenizer import get_architects_prompt_format
from arc_tartiflette.model_tools.data_collator import ExampleMaskingDataCollator

def train_trl(
        model, 
        tokenized_datasets: DatasetDict, 
        tokenizer,
        output_model="default_output_model",
    ):
    use_bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False

    fmt = get_architects_prompt_format(tokenizer)

    # Define training arguments using TRL's SFTConfig
    training_args = SFTConfig(
        output_dir=f"./data/models/{output_model}",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        learning_rate=5e-5,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        weight_decay=0.01,
        report_to="none",
        push_to_hub=True,

        # Precision config
        bf16=use_bf16,  # use bf16 if GPU supports it
        fp16=not use_bf16,  # fallback to fp16 if bf16 not supported
    )

    # Define TRL SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["eval"],
        dataset_text_field="text",
        packing=False,
        data_collator=ExampleMaskingDataCollator(
            response_template=fmt['output_beg'],
            mlm=False,
            tokenizer=tokenizer,
            mask_first_n_examples=1,
        ),
    )

    # Start training
    print("Train")
    trainer.train()

    # Eval
    print("Eval")
    results = trainer.evaluate()
    print(results)