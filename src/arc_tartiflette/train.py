from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, setup_chat_format
import torch
from datasets import Dataset, DatasetDict
import huggingface_hub as hf
import os

from arc_tartiflette.utils.gpu_availability import print_gpu_availability
from arc_tartiflette.utils import load

hf.login(os.environ.get("HUGGING_FACE_TOKEN", ""))

print_gpu_availability()

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Configure model and tokenizer
model_name = "HuggingFaceTB/SmolLM2-135M"
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name).to(
    device
)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)


# Get dataset
input_dir = "data/kaggle_input"
dict_dataset = load.load_challenges_kaggle_format(input_dir)
hf_dataset = load.dict_to_transformers_dataset(dict_dataset)

hf_dataset.push_to_hub("meo-des/arc-agi-2_kaggle_prepared")

dataset_dict = DatasetDict({
    "train": hf_dataset.shuffle(seed=42).select(range(int(0.8*len(hf_dataset)))),
    "test": hf_dataset.shuffle(seed=42).select(range(int(0.8*len(hf_dataset)), len(hf_dataset)))
})

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="longest")

tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)

use_bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False

# Define training arguments using TRL's SFTConfig
training_args = SFTConfig(
    output_dir="./data/models/smolvlm_arc_kaggle_with_trl",
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
    eval_dataset=tokenized_datasets["test"],
    dataset_text_field="text",
    packing=False,
)

# Start training
print("Train")
trainer.train()

# Eval
print("Eval")
results = trainer.evaluate()
print(results)

# Generate text
print("Generate")

# Create inference pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

# Test prompt
prompt = "Once upon a time, the"
output = pipe(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)

print(output[0]["generated_text"])
