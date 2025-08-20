from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, setup_chat_format
import torch

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"


# Configure model and tokenizer
model_name = "HuggingFaceTB/SmolLM2-135M"
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name).to(
    device
)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

# Load dataset
dataset = load_dataset("HuggingFaceTB/smoltalk", "apigen-80k")

model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)

training_args = SFTConfig(
    output_dir="./sft_output",
    max_steps=10,
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    logging_steps=10,
    save_steps=100,
    eval_strategy="steps",
    eval_steps=50,
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"].select([i for i in range(0, len(dataset), 100)]),
    processing_class=tokenizer,
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
prompt = "Once upon a time,"
output = pipe(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)

print(output[0]["generated_text"])
