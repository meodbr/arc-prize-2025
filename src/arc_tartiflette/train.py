from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import LoraConfig, TaskType, get_peft_model
import torch
from datasets import Dataset, DatasetDict, load_dataset
import huggingface_hub as hf
import os

import arc_tartiflette.model_tools.tokenizer as tokenizer_tools
from arc_tartiflette.utils import utils, constants, gpu_availability, load
from arc_tartiflette.training.train_transformers import train_transformers
from arc_tartiflette.training.train_trl import train_trl
from arc_tartiflette.config.settings import ENV_VARS


def get_model(model_name: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name).to(
        device
    )
    print(f"---- Model {model_name} loaded. ----")
    print(f"Model has {utils.count_parameters(model)/1e9:.3f}B parameters.")
    return model


def get_dataset(dataset_id: str):
    hf_dataset = load_dataset(dataset_id)
    dataset_dict = DatasetDict({
        "train": hf_dataset["train"],
        "eval": hf_dataset["eval"],
    })
    print(f"---- Dataset {dataset_id} loaded. ----")
    # print("Dataset average text length:", sum(len(x['text']) for x in dataset_dict['train'])/len(dataset_dict['train']))
    # print(f"Dataset max_length: {max(len(x['text']) for x in dataset_dict['train'])}")
    return dataset_dict


def augment_dataset(dataset, tokenizer):
    for split, data in dataset.items():
        data = load.augment_transformers_dataset(
            data,
            format=tokenizer_tools.get_architects_prompt_format(tokenizer),
            multipliers = {
                "color": ENV_VARS["AUG_COLOR_NUM"],
                "order": ENV_VARS["AUG_ORDER_NUM"],
            },
        )
    return dataset


def get_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
    print(f"Tokenizer fast? {tokenizer.is_fast}")
    tokenizer.pad_token = tokenizer.eos_token if not tokenizer.pad_token else tokenizer.pad_token
    return tokenizer


def shrink_vocab(model, tokenizer):
    # Shrink vocab to only keep useful tokens
    print("Shrinking tokenizer vocabulary to only keep useful tokens...")
    keep_tok = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!?.:,;*+/-=')+tokenizer.tokenize('\n')
    tokenizer_tools.keep_single_char_tokens(model, tokenizer, keep_tok=keep_tok)
    print(f"New tokenizer vocab size: {len(tokenizer)}")


def tokenize_dataset(dataset_dict: DatasetDict, tokenizer: AutoTokenizer):
    print("Tokenizing dataset...")
    max_length = ENV_VARS["TOKENIZER_MAX_LENGTH"]
    def tokenize_function(example):
        tokenized = tokenizer(example["text"], truncation=True, max_length=max_length, padding="max_length")
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)
    print("---- Dataset tokenized. ----")
    # print("Tokenized dataset max length:", max(len(x['input_ids']) for x in tokenized_datasets['train']))
    # print("Tokenized dataset average length:", sum(len(x['input_ids']) for x in tokenized_datasets['train'])/len(tokenized_datasets['train']))
    print("Tokenized dataset example:", tokenized_datasets['train'][0] if len(tokenized_datasets['train']) > 0 else "N/A")
    return tokenized_datasets


def setup_peft_lora(model):
    lora_target_modules = ENV_VARS["LORA_TARGET_MODULES"]
    lora_r = ENV_VARS["LORA_R"]
    lora_alpha = ENV_VARS["LORA_ALPHA"]
    lora_dropout = ENV_VARS["LORA_DROPOUT"]
    use_rslora = ENV_VARS["USE_RSLORA"]

    # Configure PEFT LoRA
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        use_rslora=use_rslora,
        task_type=TaskType.CAUSAL_LM,
    )

    # Apply PEFT LoRA to the model
    print("Applying PEFT LoRA to the model...")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    print(f"Model now has {utils.count_parameters(model)/1e6:.3f}M parameters.")

    return model


def print_before_training_info(model, tokenized_datasets, use_bf16):
    use_grad_checkpointing = ENV_VARS["GRAD_CHPT"]
    batch_size = int(ENV_VARS["BATCH_SIZE"])
    max_length = ENV_VARS["TOKENIZER_MAX_LENGTH"]
    output_model_name = ENV_VARS["HF_OUTPUT_MODEL"]
    train_method = ENV_VARS["TRAIN_METHOD"]
    print(f"---- Training with method '{train_method}' (batch size: {batch_size}, bf16: {use_bf16}) ----")
    print(f"Estimated VRAM needed: {gpu_availability.estimate_vram_usage(model, batch_size, use_bf16, max_length)} GB")
    print(f"Free VRAM: {torch.cuda.mem_get_info()[0]/1e9 if torch.cuda.is_available() else 'N/A'} GB")
    print(f"Estimated number of steps: {len(tokenized_datasets['train']) // batch_size}")
    print(f"Estimated time per epoch: {utils.estimate_time_per_epoch(model, batch_size, use_bf16, max_length, len(tokenized_datasets['train']))/60:.2f} minutes")
    print("Using Gradient Checkpointing:", use_grad_checkpointing)
    print(f"Output model name: {output_model_name}")


def test_model(model, tokenizer):
    print("--- TEST ----")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

    # Test prompt
    fmt = tokenizer_tools.get_architects_prompt_format(tokenizer)
    prompt = fmt["bos_token"] + fmt["preprompt"] + fmt["input_beg"]
    output = pipe(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)

    print(output[0]["generated_text"])


def train():
    # ---- DEVICE ----
    gpu_availability.print_gpu_availability()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- MODEL ----
    model_name = ENV_VARS["HF_BASE_MODEL"]
    model = get_model(model_name, device)

    # ---- DATASET ----
    dataset_id = f"{constants.HF_USER}/{ENV_VARS['HF_DATASET']}"
    dataset_dict = get_dataset(dataset_id)

    # ---- PREPROCESS ----
    tokenizer = get_tokenizer(model_name)
    shrink_vocab(model, tokenizer)
    if len(dataset_dict) < 10000 or ENV_VARS['DO_AUG']:
        dataset_dict = augment_dataset(dataset_dict, tokenizer)
    tokenized_datasets = tokenize_dataset(dataset_dict, tokenizer)

    # ---- PEFT ----
    use_peft = ENV_VARS["USE_PEFT"]
    if use_peft:
        model = setup_peft_lora(model)

    # ---- TRAIN ----
    use_bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    print_before_training_info(model, tokenized_datasets, use_bf16)
    match ENV_VARS["TRAIN_METHOD"]:
        case "transformers":
            train_transformers(model, tokenized_datasets, tokenizer, output_model=ENV_VARS["HF_OUTPUT_MODEL"])
        case "trl":
            train_trl(model, tokenized_datasets, tokenizer, output_model=ENV_VARS["HF_OUTPUT_MODEL"])
        case _:
            train_transformers(model, tokenized_datasets, tokenizer, output_model=ENV_VARS["HF_OUTPUT_MODEL"])
    
    # ---- PUSH ----
    model.push_to_hub(ENV_VARS["HF_OUTPUT_MODEL"])

    # ---- TEST ----
    test_model(model, tokenizer)

if __name__ == "__main__":
    train()
