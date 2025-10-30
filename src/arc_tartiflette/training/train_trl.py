from trl import SFTConfig, SFTTrainer
from datasets import Dataset, DatasetDict
import torch

from arc_tartiflette.model_tools.tokenizer import get_architects_prompt_format
from arc_tartiflette.model_tools.data_collator import ExampleMaskingDataCollator
from arc_tartiflette.config.settings import ENV_VARS

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
        num_train_epochs=ENV_VARS["TRAIN_EPOCHS"],
        max_steps=ENV_VARS["TRAIN_STEPS"],
        per_device_train_batch_size=ENV_VARS["BATCH_SIZE"],
        per_device_eval_batch_size=ENV_VARS["BATCH_SIZE"],
        gradient_accumulation_steps=ENV_VARS["GRAD_ACC_STEPS"],
        save_strategy="epoch",
        logging_steps=50,
        learning_rate=ENV_VARS["LR"],
        warmup_ratio=ENV_VARS["WARMUP_RATIO"],
        lr_scheduler_type=ENV_VARS["LR_SCHEDULER_TYPE"],
        weight_decay=ENV_VARS["WEIGHT_DECAY"],
        optim=ENV_VARS["OPTIM"],
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
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["eval"],
        # data_collator=ExampleMaskingDataCollator(
        #     response_template=fmt['output_beg'],
        #     mlm=False,
        #     tokenizer=tokenizer,
        #     mask_first_n_examples=1,
        # ),
    )

    # Start training
    print("Train")
    trainer.train()

    # Eval
    print("Eval")
    results = trainer.evaluate()
    print(results)