import logging

import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
)
from peft import LoraConfig, TaskType, get_peft_model
import torch
from datasets import Dataset, DatasetDict, load_dataset
import huggingface_hub as hf
import os

import arc_tartiflette.model_tools.tokenizer as tokenizer_tools
from arc_tartiflette.model_tools.tokenize_functions import (
    tokenize_dataset_base,
    tokenize_dataset_2DPE,
    frac_dataset_dict,
)
from arc_tartiflette.model_tools.quantization import print_quantization_info
from arc_tartiflette.utils import utils, constants, gpu_availability, load
from arc_tartiflette.training.train_transformers import train_transformers
from arc_tartiflette.training.train_trl import train_trl
from arc_tartiflette.config.settings import ENV_VARS
from arc_tartiflette.inference.solvers.lm import LMSolver
from arc_tartiflette.inference.solvers.conv_embedding import ConvEmbeddingSolver
from arc_tartiflette.model_tools.custom_pe import CustomMistralModel2DPE
from arc_tartiflette.model_tools.conv_embeddings import (
    CustomMistralModelConvEmbedding,
    tokenize_dataset_conv,
)

logger = logging.getLogger(__name__)


def get_model(model_name: str, untie_lm_head: bool = None):
    if untie_lm_head is None:
        untie_lm_head = ENV_VARS["USE_LORA"]

    quantize_model = ENV_VARS["QUANTIZE_MODEL"]
    if quantize_model in [4, 8]:
        logger.info(
            "Loading quantized model with %d-bit quantization...", quantize_model
        )
        bnb_config = None
        if quantize_model == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=ENV_VARS["BNB_4BIT_QUANT_TYPE"],
                bnb_4bit_compute_type=torch.float16,
                llm_int8_enable_fp32_cpu_offload=True,
            )
        else:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True
            )
        if ENV_VARS["PRINT_QUANT_INFO"]:
            print_quantization_info(
                model_name=model_name,
                quantization_config=bnb_config,
                device_map="cpu",
                verbose=True,
            )
    else:
        bnb_config = None

    model_class = AutoModelForCausalLM
    match ENV_VARS["MODEL_TYPE"]:
        case "base":
            logger.info("Using base AutoModelForCausalLM...")
            model_class = AutoModelForCausalLM
        case "2DPE":
            logger.info("Using Custom Mistral Model with 2D PE...")
            model_class = CustomMistralModel2DPE
        case "conv":
            logger.info("Using Custom Mistral Model with Conv Embeddings...")
            model_class = CustomMistralModelConvEmbedding
        case _:
            model_class = AutoModelForCausalLM

    if untie_lm_head:
        model = model_class.from_pretrained(
            pretrained_model_name_or_path=model_name,
            tie_word_embeddings=False,
            quantization_config=bnb_config,
            device_map="auto",
        )
        logger.info("Untying model head with embedding...")
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
        logger.info(
            "Num non-quantized parameters: %.2fM",
            sum(p.numel() for p in model.parameters() if p.dtype in (torch.float32, torch.float16))
            / 1e6,
        )
    else:
        model = model_class.from_pretrained(
            pretrained_model_name_or_path=model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
    logger.info("Model %s loaded.", model_name)
    logger.info("Model has %.3fB parameters.", utils.count_parameters(model) / 1e9)
    logger.info("Model dtype: %s", next(model.parameters()).dtype)
    logger.debug("Model config: %s", model.config)
    logger.debug("Model generation config: %s", model.generation_config)
    return model


def get_dataset(dataset_id: str):
    hf_dataset = load_dataset(dataset_id)
    dataset_dict = DatasetDict(
        {
            "train": hf_dataset["train"],
            "eval": hf_dataset["eval"],
            "test": hf_dataset["test"],
        }
    )
    logger.info("Dataset %s loaded.", dataset_id)
    frac = ENV_VARS["DATASET_FRAC"]
    if frac != 1.0:
        return frac_dataset_dict(dataset_dict, frac)
    return dataset_dict


def augment_dataset(dataset, tokenizer, only_splits: list = None):
    logger.info("Augmenting dataset (has %d training examples)...", len(dataset["train"]))
    new_dataset = {}
    for split, data in dataset.items():
        if only_splits and split not in only_splits:
            new_dataset[split] = data
            continue
        logger.info("Augmenting split '%s' with %d examples...", split, len(data))
        new_dataset[split] = load.augment_transformers_dataset(
            data,
            fmt=tokenizer_tools.get_architects_prompt_format(tokenizer),
            multipliers={
                "color": ENV_VARS["AUG_COLOR_NUM"],
                "order": ENV_VARS["AUG_ORDER_NUM"],
            },
        )
        logger.info(
            "Augmented split '%s' now has %d examples.",
            split,
            len(new_dataset[split]),
        )
    logger.info(
        "Dataset now has %d training examples after augmentation.",
        len(new_dataset["train"]),
    )
    return DatasetDict(new_dataset)


def get_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
    logger.info("Tokenizer fast? %s", tokenizer.is_fast)
    tokenizer.pad_token = (
        tokenizer.eos_token if not tokenizer.pad_token else tokenizer.pad_token
    )
    logger.info("Tokenizer loaded. Vocab size: %d", len(tokenizer))
    logger.info("Tokenizer class: %s", type(tokenizer))
    return tokenizer


def shrink_vocab(model, tokenizer):
    # Shrink vocab to only keep useful tokens
    logger.info("Shrinking tokenizer vocabulary to only keep useful tokens...")
    logger.info("Original tokenizer vocab size: %d", len(tokenizer))
    logger.info("Original model parameters: %.3fB", utils.count_parameters(model) / 1e9)
    keep_tok = list(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!?.:,;*+/-="
    ) + tokenizer.tokenize("\n")
    logger.debug("Model config: %s", model.config)
    logger.debug("Model generation config: %s", model.generation_config)
    tokenizer_tools.keep_single_char_tokens(model, tokenizer, keep=keep_tok)
    logger.info("New tokenizer vocab size: %d", len(tokenizer))
    logger.info(
        "Model parameters after vocab shrink: %.3fB", utils.count_parameters(model) / 1e9
    )

    if ENV_VARS["MODEL_TYPE"] == "conv":
        logger.info("Extending tokenizer vocab for conv Embedding...")
        tokenizer_tools.extend_tokenizer_vocab_for_arc_grid(tokenizer)
        logger.info("Extended tokenizer vocab size for conv E: %d", len(tokenizer))
        tokenizer_tools.extend_model_embeddings_for_arc_grid(model, tokenizer)
        logger.info(
            "Model parameters after extending for conv E: %.3fB",
            utils.count_parameters(model) / 1e9,
        )


def setup_peft_lora(model):
    lora_target_modules = ENV_VARS["LORA_TARGET_MODULES"]
    lora_r = ENV_VARS["LORA_R"]
    lora_alpha = ENV_VARS["LORA_ALPHA"]
    lora_dropout = ENV_VARS["LORA_DROPOUT"]
    use_rslora = ENV_VARS["USE_RSLORA"]
    modules_to_save = ENV_VARS["LORA_MODULES_TO_SAVE"]

    # Configure PEFT LoRA
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        use_rslora=use_rslora,
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=modules_to_save if len(modules_to_save) > 0 else None,
    )

    # Apply PEFT LoRA to the model
    logger.info("Applying PEFT LoRA to the model...")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    logger.info("Model now has %.3fM parameters.", utils.count_parameters(model) / 1e6)
    logger.info("Target modules for LoRA: %s", lora_target_modules)
    logger.info(
        "LoRA config: R=%d, alpha=%d, dropout=%.2f, use_rslora=%s",
        lora_r,
        lora_alpha,
        lora_dropout,
        use_rslora,
    )

    return model


def print_before_training_info(model, tokenized_datasets, use_bf16):
    use_grad_checkpointing = ENV_VARS["GRAD_CHPT"]
    batch_size = int(ENV_VARS["BATCH_SIZE"])
    max_length = ENV_VARS["TOKENIZER_MAX_LENGTH"]
    output_model_name = ENV_VARS["HF_OUTPUT_MODEL"]
    train_method = ENV_VARS["TRAIN_METHOD"]
    logger.info(
        "---- Training with method '%s' (batch size: %d, bf16: %s) ----",
        train_method,
        batch_size,
        use_bf16,
    )
    logger.debug(
        "Estimated VRAM needed: %.2f GB",
        gpu_availability.estimate_vram_usage(
            model, batch_size, use_bf16, max_length
        ),
    )
    logger.info(
        "Free VRAM: %.2f GB",
        torch.cuda.mem_get_info()[0] / 1e9 if torch.cuda.is_available() else float("nan"),
    )
    logger.debug(
        "Estimated number of steps: %d",
        len(tokenized_datasets["train"]) // batch_size,
    )
    logger.debug(
        "Estimated time per epoch: %.2f minutes",
        utils.estimate_time_per_epoch(
            model,
            batch_size,
            use_bf16,
            max_length,
            len(tokenized_datasets["train"]),
        )
        / 60,
    )
    logger.info("Using Gradient Checkpointing: %s", use_grad_checkpointing)
    logger.info("Output model name: %s", output_model_name)
    logger.info("Optimizer : %s", ENV_VARS["OPTIM"])


def test_model_on_dataset(model, tokenizer, dataset_dict, splits: list = None):
    num_solve_tests = ENV_VARS["NUM_SOLVE_TESTS"]
    batch_size = ENV_VARS["SOLVE_BATCH_SIZE"]

    solver = LMSolver(
        model=model,
        tokenizer=tokenizer,
    )
    if ENV_VARS["MODEL_TYPE"] == "conv":
        solver = ConvEmbeddingSolver(
            model=model,
            tokenizer=tokenizer,
        )

    if splits is None:
        splits = ["train", "test"]
    cards = {}
    for split in splits:
        hf_dataset = (
            dataset_dict[split]
            .shuffle(seed=42)
            .select(
                range(
                    min(
                        num_solve_tests // len(splits),
                        len(dataset_dict[split]),
                    )
                )
            )
        )
        cards[split] = solver.solve_hf_dataset(hf_dataset, split, batch_size)
    for split in splits:
        logger.info(cards[split].summary)


def test_model_generation(model, tokenizer):
    logger.info("Testing model generation...")
    try:
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

        # Test prompt
        fmt = tokenizer_tools.get_architects_prompt_format(tokenizer)
        prompt = fmt["bos_token"] + fmt["preprompt"] + fmt["input_beg"]
        output = pipe(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)

        logger.info("Generation test successful.")
    except Exception as e:
        logger.error("Error during generation test:", exc_info=e)


def train(
    push=True,
):
    logger.debug(
        "Starting training with config: %s",
        {k: v for k, v in ENV_VARS.items() if "TOKEN" not in k and "PASSWORD" not in k},
    )
    # ---- DEVICE ----
    gpu_availability.print_gpu_availability()

    # ---- MODEL ----
    model_name = ENV_VARS["HF_BASE_MODEL"]
    model = get_model(model_name, untie_lm_head=ENV_VARS["UNTIE_LM_HEAD"])

    # ---- DATASET ----
    dataset_id = f"{constants.HF_USER}/{ENV_VARS['HF_DATASET']}"
    dataset_dict = get_dataset(dataset_id)

    # ---- PREPROCESS ----
    tokenizer = get_tokenizer(model_name)
    shrink_vocab(model, tokenizer)
    if ENV_VARS["DO_AUG"]:
        dataset_dict = augment_dataset(dataset_dict, tokenizer)
    match ENV_VARS["MODEL_TYPE"]:
        case "base":
            tokenized_datasets = tokenize_dataset_base(dataset_dict, tokenizer)
        case "2DPE":
            tokenized_datasets = tokenize_dataset_2DPE(dataset_dict, tokenizer)
        case "conv":
            tokenized_datasets = tokenize_dataset_conv(dataset_dict, tokenizer)
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
            train_transformers(
                model,
                tokenized_datasets,
                tokenizer,
                output_model=ENV_VARS["HF_OUTPUT_MODEL"],
            )
        case "trl":
            train_trl(
                model,
                tokenized_datasets,
                tokenizer,
                output_model=ENV_VARS["HF_OUTPUT_MODEL"],
            )
        case _:
            train_transformers(
                model,
                tokenized_datasets,
                tokenizer,
                output_model=ENV_VARS["HF_OUTPUT_MODEL"],
            )

    # ---- PUSH ----
    if push:
        model.push_to_hub(ENV_VARS["HF_OUTPUT_MODEL"])
        merged_model = model.merge_and_unload()
        merged_name = ENV_VARS["HF_OUTPUT_MODEL"] + ENV_VARS["HF_OUTPUT_MERGED_SUFFIX"]
        merged_model.push_to_hub(merged_name)
        tokenizer.push_to_hub(merged_name)

    # ---- TEST ----
    test_model_on_dataset(merged_model, tokenizer, dataset_dict)
    test_model_generation(merged_model, tokenizer)

    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    train()
