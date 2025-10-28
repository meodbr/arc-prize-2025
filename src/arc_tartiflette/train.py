from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import LoraConfig, TaskType, get_peft_model
import torch
from datasets import Dataset, DatasetDict, load_dataset
import huggingface_hub as hf
import os

import arc_tartiflette.utils.model as model_tools
from arc_tartiflette.utils import utils, constants, gpu_availability
from arc_tartiflette.training.train_transformers import train_transformers
from arc_tartiflette.training.train_trl import train_trl

def train():
    gpu_availability.print_gpu_availability()

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- MODEL ----
    model_name = os.environ.get("HF_BASE_MODEL","HuggingFaceTB/SmolLM2-135M")
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name).to(
        device
    )
    print(f"---- Model {model_name} loaded. ----")
    print(f"Model has {utils.count_parameters(model)/1e9:.3f}B parameters.")
    # print(model.config)


    # ---- DATASET ----
    dataset_id = constants.HF_USER + "/" + os.environ.get("HF_DATASET", "arc-agi-2_kaggle_flattened")
    hf_dataset = load_dataset(dataset_id)
    dataset_dict = DatasetDict({
        "train": hf_dataset["train"],
        "eval": hf_dataset["eval"],
    })
    print(f"---- Dataset {dataset_id} loaded. ----")
    # print("Dataset average text length:", sum(len(x['text']) for x in dataset_dict['train'])/len(dataset_dict['train']))
    # print(f"Dataset max_length: {max(len(x['text']) for x in dataset_dict['train'])}")

    # ---- TOKENIZE ----
    print("Tokenizing dataset...")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
    print(f"Tokenizer fast? {tokenizer.is_fast()}")
    tokenizer.pad_token = tokenizer.eos_token if not tokenizer.pad_token else tokenizer.pad_token

    # Shrink vocab to only keep useful tokens
    print("Shrinking tokenizer vocabulary to only keep useful tokens...")
    keep_tok = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!?.:,;*+/-=')+tokenizer.tokenize('\n')
    model_tools.keep_single_char_tokens(model, tokenizer, keep_tok=keep_tok)
    print(f"New tokenizer vocab size: {len(tokenizer)}")

    max_length = int(os.environ.get("TOKENIZER_MAX_LENGTH", "2048"))
    def tokenize_function(example):
        tokenized = tokenizer(example["text"], truncation=True, max_length=max_length, padding="max_length")
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)
    print("---- Dataset tokenized. ----")
    # print("Tokenized dataset max length:", max(len(x['input_ids']) for x in tokenized_datasets['train']))
    # print("Tokenized dataset average length:", sum(len(x['input_ids']) for x in tokenized_datasets['train'])/len(tokenized_datasets['train']))
    print("Tokenized dataset example:", tokenized_datasets['train'][0] if len(tokenized_datasets['train']) > 0 else "N/A")

    # ---- PEFT ----
    use_peft = os.environ.get("USE_LORA", "true").lower() == "true"
    lora_target_modules = os.environ.get("LORA_TARGET_MODULES", "q_proj,k_proj,v_proj,o_proj,up_proj,down_proj").split(",")
    lora_r = int(os.environ.get("LORA_R", "32"))
    lora_alpha = int(os.environ.get("LORA_ALPHA", "24"))
    lora_dropout = float(os.environ.get("LORA_DROPOUT", "0.1"))
    use_rslora = bool(os.environ.get("USE_RSLORA", "true").lower() == "true")
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        use_rslora=use_rslora,
        task_type=TaskType.CAUSAL_LM,
    )
    if use_peft:
        print("Applying PEFT LoRA to the model...")
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        print(f"Model now has {utils.count_parameters(model)/1e6:.3f}M parameters.")

    # ---- TRAIN and PUSH ----
    output_model_name = os.environ.get("HF_OUTPUT_MODEL", utils.default_output_model_name(model_name, dataset_id))
    use_bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False

    train_method = os.environ.get("TRAIN_METHOD", "default")
    batch_size = int(os.environ.get("BATCH_SIZE", "4"))
    use_grad_checkpointing = os.environ.get("GRAD_CHPT", "true").lower() == "true"
    print(f"---- Training with method '{train_method}' (batch size: {batch_size}, bf16: {use_bf16}) ----")
    print(f"Estimated VRAM needed: {gpu_availability.estimate_vram_usage(model, batch_size, use_bf16, max_length)} GB")
    print(f"Free VRAM: {torch.cuda.mem_get_info()[0]/1e9 if torch.cuda.is_available() else 'N/A'} GB")
    print(f"Estimated number of steps: {len(tokenized_datasets['train']) // batch_size}")
    print(f"Estimated time per epoch: {utils.estimate_time_per_epoch(model, batch_size, use_bf16, max_length, len(tokenized_datasets['train']))/60:.2f} minutes")
    print("Using Gradient Checkpointing:", use_grad_checkpointing)
    print(f"Output model name: {output_model_name}")
    match train_method:
        case "transformers":
            train_transformers(model, tokenized_datasets, tokenizer, output_model=output_model_name)
        case "trl":
            train_trl(model, tokenized_datasets, tokenizer, output_model=output_model_name)
        case _:
            train_transformers(model, tokenized_datasets, tokenizer, output_model=output_model_name)

    # ---- TEST generation ----
    print("Generate")

    # Create inference pipeline
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

    # Test prompt
    prompt = "Once upon a time, the"
    output = pipe(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)

    print(output[0]["generated_text"])

if __name__ == "__main__":
    train()
