from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, setup_chat_format
import torch
from datasets import Dataset, DatasetDict
import huggingface_hub as hf

from arc_tartiflette.utils.gpu_availability import print_gpu_availability
from arc_tartiflette.utils import load
from arc_tartiflette.config.settings import settings

hf.login(token=settings.HUGGING_FACE_TOKEN)

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
input_dir = "data/kaggle_working"
dict_dataset = load.load_challenges_kaggle_format(input_dir)
hf_dataset = load.dict_to_transformers_dataset(dict_dataset)

hf_dataset.push_to_hub("meo-des/kaggle_input_prepared")

dataset_dict = DatasetDict({
    "train": hf_dataset.shuffle(seed=42).select(range(int(0.8*len(hf_dataset)))),
    "test": hf_dataset.shuffle(seed=42).select(range(int(0.8*len(hf_dataset)), len(hf_dataset)))
})

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="longest")

tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)


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
