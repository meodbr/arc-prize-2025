from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from datasets import Dataset, DatasetDict, load_dataset
import huggingface_hub as hf
import os

from arc_tartiflette.utils import utils, constants, gpu_availability
from arc_tartiflette.training.train_transformers import train_transformers
from arc_tartiflette.training.train_trl import train_trl

hf.login(token=os.environ.get("HUGGING_FACE_TOKEN"))

gpu_availability.print_gpu_availability()

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- MODEL ----
model_name = os.environ.get("HF_BASE_MODEL","HuggingFaceTB/SmolLM2-135M")
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name).to(
    device
)


# ---- DATASET ----
dataset_id = constants.HF_USER + "/" + os.environ.get("HF_DATASET", "arc-agi-2_kaggle_flattened")
hf_dataset = load_dataset(dataset_id, split="train")
dataset_dict = DatasetDict({
    "train": hf_dataset.shuffle(seed=42).select(range(int(0.8*len(hf_dataset)))),
    "test": hf_dataset.shuffle(seed=42).select(range(int(0.8*len(hf_dataset)), len(hf_dataset)))
})


# ---- TOKENIZE ----
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
tokenizer.pad_token = tokenizer.eos_token if not tokenizer.pad_token else tokenizer.pad_token
max_length = int(os.environ.get("TOKENIZER_MAX_LENGTH", "2048"))
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, max_length=max_length, padding="max_length")
tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)


# ---- TRAIN and PUSH ----
output_model_name = os.environ.get("HF_OUTPUT_MODEL", utils.default_output_model_name(model_name, dataset_id))
use_bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False

train_method = os.environ.get("TRAIN_METHOD", "default")
match train_method:
    case "transformers":
        model = train_transformers(model, tokenized_datasets, tokenizer, output_model=output_model_name)
    case "trl":
        model = train_trl(model, tokenized_datasets, tokenizer, output_model=output_model_name)
    case _:
        model = train_transformers(model, tokenized_datasets, tokenizer, output_model=output_model_name)

# ---- TEST generation ----
print("Generate")

# Create inference pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

# Test prompt
prompt = "Once upon a time, the"
output = pipe(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)

print(output[0]["generated_text"])
