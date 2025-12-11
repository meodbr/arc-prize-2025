import logging.config
import os
import json

import torch
import huggingface_hub as hf
from peft import LoraConfig, TaskType, get_peft_model
from datasets import (
    Dataset,
    DatasetDict,
    load_dataset,
)
from transformers import (
    pipeline,
)

from arc_tartiflette.model import tokenizer_tools
from arc_tartiflette.utils import utils, constants, gpu_availability, load
from arc_tartiflette.training.train_trl import train_trl
from arc_tartiflette.config.settings import settings, get_logging_config
from arc_tartiflette.inference.solvers.lm import LMSolver
from arc_tartiflette.inference.solvers.conv_embedding import ConvEmbeddingSolver
from arc_tartiflette.dataset.builder import DatasetBuilder
from arc_tartiflette.model import ModelBuilder

logger = logging.getLogger(__name__)


def print_before_training_info(model, tokenized_datasets, use_bf16):
    use_grad_checkpointing = settings.GRAD_CHPT
    batch_size = settings.BATCH_SIZE
    max_length = settings.TOKENIZER_MAX_LENGTH
    output_model_name = settings.HF_OUTPUT_MODEL
    train_method = settings.TRAIN_METHOD
    logger.info(
        "---- Training with method '%s' (batch size: %d, bf16: %s) ----",
        train_method,
        batch_size,
        use_bf16,
    )
    logger.debug(
        "Estimated VRAM needed: %.2f GB",
        gpu_availability.estimate_vram_usage(model, batch_size, use_bf16, max_length),
    )
    logger.info(
        "Free VRAM: %.2f GB",
        (
            torch.cuda.mem_get_info()[0] / 1e9
            if torch.cuda.is_available()
            else float("nan")
        ),
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
    logger.info("Optimizer : %s", settings.OPTIM)


def test_model_on_dataset(model, tokenizer, dataset_dict, splits: list = None):
    num_solve_tests = settings.NUM_SOLVE_TESTS
    batch_size = settings.SOLVE_BATCH_SIZE

    solver = LMSolver(
        model=model,
        tokenizer=tokenizer,
    )
    if settings.MODEL_TYPE == "conv":
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
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
        )

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
        {k: v for k, v in settings.__dict__.items() if "TOKEN" not in k and "PASSWORD" not in k},
    )
    # ---- DEVICE ----
    gpu_availability.print_gpu_availability()

    # ---- MODEL / TOKENIZER ----
    model_name = "HuggingFaceTB/SmolLM2-135M"
    model, tokenizer = (
        ModelBuilder()
        .from_hf(model_name)
        .set_custom_class("base")
        .with_quantization(4)
        .with_lora(config="env")
        .with_untied_lm_head()
        .shrink_tokenizer_vocab()
        .build()
    )

    # ---- DATASET ----
    dataset_id = f"{constants.HF_USER}/{settings.HF_DATASET}"
    dataset_dict = (
        DatasetBuilder()
        .from_hf(dataset_id)
        .only_frac(settings.DATASET_FRAC)
        .tokenized(tokenizer, for_custom_class="base")
        .build()
    )

    # ---- TRAIN ----
    use_bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    print_before_training_info(model, dataset_dict, use_bf16)
    train_trl(
        model,
        dataset_dict,
        tokenizer,
        output_model= settings.HF_OUTPUT_MODEL,
        use_custom_data_collator=False,
    )

    # ---- PUSH ----
    if push:
        model.push_to_hub(settings.HF_OUTPUT_MODEL)
        merged_model = model.merge_and_unload()
        merged_name = (
            settings.HF_OUTPUT_MODEL + settings.HF_OUTPUT_MERGED_SUFFIX
        )
        merged_model.push_to_hub(merged_name)
        tokenizer.push_to_hub(merged_name)

    # ---- TEST ----
    test_model_on_dataset(merged_model, tokenizer, dataset_dict)
    test_model_generation(merged_model, tokenizer)

    return model


if __name__ == "__main__":
    logging.config.dictConfig(get_logging_config())
    train()
