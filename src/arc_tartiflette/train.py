from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import LoraConfig, TaskType, get_peft_model
import torch
from datasets import Dataset, DatasetDict, load_dataset
import huggingface_hub as hf
import os

import arc_tartiflette.model_tools.tokenizer as tokenizer_tools
from arc_tartiflette.model_tools.tokenize_functions import tokenize_dataset_base, frac_dataset_dict
from arc_tartiflette.utils import utils, constants, gpu_availability, load
from arc_tartiflette.training.train_transformers import train_transformers
from arc_tartiflette.training.train_trl import train_trl
from arc_tartiflette.config.settings import ENV_VARS
from arc_tartiflette.inference.solvers.lm import LMSolver


def get_model(model_name: str, device: str, untie_lm_head: bool=None):
    if untie_lm_head is None:
        untie_lm_head = ENV_VARS["USE_LORA"]
    if untie_lm_head:
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name,
            tie_word_embeddings=False,
        ).to(device)
        print(f"Untying model head with embedding...")
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
    else:
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name).to(device)
    print(f"---- Model {model_name} loaded. ----")
    print(f"Model has {utils.count_parameters(model)/1e9:.3f}B parameters.")
    return model


def get_dataset(dataset_id: str):
    hf_dataset = load_dataset(dataset_id)
    dataset_dict = DatasetDict({
        "train": hf_dataset["train"],
        "eval": hf_dataset["eval"],
        "test": hf_dataset["test"],
    })
    print(f"---- Dataset {dataset_id} loaded. ----")
    frac = ENV_VARS["DATASET_FRAC"]
    if frac != 1.:
        return frac_dataset_dict(dataset_dict, frac)
    return dataset_dict


def augment_dataset(dataset, tokenizer, only_splits: list=None):
    print(f"Augmenting dataset (has {len(dataset['train'])} training examples)...")
    for split, data in dataset.items():
        if only_splits and split not in only_splits:
            continue
        data = load.augment_transformers_dataset(
            data,
            format=tokenizer_tools.get_architects_prompt_format(tokenizer),
            multipliers={
                "color": ENV_VARS["AUG_COLOR_NUM"],
                "order": ENV_VARS["AUG_ORDER_NUM"],
            },
        )
    print(f"Dataset now has {len(dataset['train'])} training examples after augmentation.")
    return dataset


def get_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
    print(f"Tokenizer fast? {tokenizer.is_fast}")
    tokenizer.pad_token = tokenizer.eos_token if not tokenizer.pad_token else tokenizer.pad_token
    print(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")
    return tokenizer


def shrink_vocab(model, tokenizer):
    # Shrink vocab to only keep useful tokens
    print("Shrinking tokenizer vocabulary to only keep useful tokens...")
    keep_tok = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!?.:,;*+/-=')+tokenizer.tokenize('\n')
    tokenizer_tools.keep_single_char_tokens(model, tokenizer, keep=keep_tok)
    print(f"New tokenizer vocab size: {len(tokenizer)}")
    print(f"Model parameters after vocab shrink: {utils.count_parameters(model)/1e9:.3f}B")


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
    print(f"Optimizer : {ENV_VARS['OPTIM']}")


def test_model_on_dataset(model, tokenizer, dataset_dict, splits: list=None):
    print("---- TEST SOLVE ----")
    num_solve_tests = ENV_VARS["NUM_SOLVE_TESTS"]
    batch_size = ENV_VARS["SOLVE_BATCH_SIZE"]
    solver = LMSolver(
        model=model,
        tokenizer=tokenizer,
    )

    if splits is None:
        splits = ["train", "test"]
    cards = {}
    for split in splits:
        hf_dataset = dataset_dict[split].shuffle(seed=42).select(
                range(min(
                    num_solve_tests//len(splits),
                    len(dataset_dict[split]),
                ))
            )
        cards[split] = solver.solve_hf_dataset(hf_dataset, split, batch_size)
    for split in splits:
        print(cards[split].summary)


def test_model_generation(model, tokenizer):
    print("---- TEST GENERATION ----")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

    # Test prompt
    fmt = tokenizer_tools.get_architects_prompt_format(tokenizer)
    prompt = fmt["bos_token"] + fmt["preprompt"] + fmt["input_beg"]
    output = pipe(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)

    print(output[0]["generated_text"])


def train(
    push=True,
    ):
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
    if ENV_VARS['DO_AUG']:
        dataset_dict = augment_dataset(dataset_dict, tokenizer)
    match ENV_VARS["MODEL_TYPE"]:
        case "base":
            tokenized_datasets = tokenize_dataset_base(dataset_dict, tokenizer)
        case _:
            tokenized_datasets = tokenize_dataset_base(dataset_dict, tokenizer)

    # ---- PEFT ----
    use_lora = ENV_VARS["USE_LORA"]
    if use_lora:
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
    if push:
        model.push_to_hub(ENV_VARS["HF_OUTPUT_MODEL"])
        merged_model = model.merge_and_unload()
        merged_name = ENV_VARS["HF_OUTPUT_MODEL"] + ENV_VARS["HF_OUTPUT_MERGED_SUFFIX"]
        merged_model.push_to_hub(merged_name)
        tokenizer.push_to_hub(merged_name)

    # ---- TEST ----
    test_model_on_dataset(model, tokenizer, dataset_dict)
    test_model_generation(model, tokenizer)

    return model

if __name__ == "__main__":
    train()
