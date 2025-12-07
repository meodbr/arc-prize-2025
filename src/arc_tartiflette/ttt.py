import logging
import os

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import LoraConfig, TaskType, get_peft_model
from datasets import Dataset, DatasetDict, load_dataset
import huggingface_hub as hf
import torch

import arc_tartiflette.model_tools.tokenizer as tokenizer_tools
from arc_tartiflette.model_tools.tokenize_functions import tokenize_dataset_base, frac_dataset_dict, tokenize_dataset_2DPE
from arc_tartiflette.model_tools.conv_embeddings import tokenize_dataset_conv
from arc_tartiflette.utils import utils, constants, gpu_availability, load
from arc_tartiflette.training.train_transformers import train_transformers
from arc_tartiflette.training.train_trl import train_trl
from arc_tartiflette.config.settings import ENV_VARS
from arc_tartiflette.inference.solvers.lm import LMSolver
from arc_tartiflette.train import get_model, get_tokenizer, augment_dataset, print_before_training_info, setup_peft_lora, test_model_on_dataset, test_model_generation

logger = logging.getLogger(__name__)

def get_kaggle_dataset(kaggle_dir: str, split: str, submission_run: bool=True, eval_split: str="eval", num_eval: int=40) -> DatasetDict:
    dataset_dict = load.load_challenges_kaggle_format(kaggle_dir)
    # take a portion of the train data as eval

    assert not (split != "test" and submission_run), "Submission run can only be done on test split."
    hf_dataset = DatasetDict({
        "train": load.dict_to_transformers_dataset(dataset_dict[split], keep_tests=False),
        "eval": load.dict_to_transformers_dataset(dataset_dict[eval_split]).shuffle(seed=42).select(range(min(num_eval, len(dataset_dict[eval_split])))),
    })
    logger.info("Kaggle dataset loaded from %s", kaggle_dir)

    frac = ENV_VARS["DATASET_FRAC"]
    if frac != 1.:
        return frac_dataset_dict(hf_dataset, frac)
    return hf_dataset


def get_dataset(dataset_id: str, split: str="test", eval_split: str="eval", num_eval: int=40):
    hf_dataset = load_dataset(dataset_id)
    dataset_dict = DatasetDict({
        "train": load.remove_tests_from_hf_dataset(hf_dataset[split]),
        "eval": hf_dataset[eval_split].shuffle(seed=42).select(range(min(num_eval, len(hf_dataset[eval_split])))),
    })
    logger.info("Non-kaggle dataset %s loaded.", dataset_id)
    frac = ENV_VARS["DATASET_FRAC"]
    if frac != 1.:
        return frac_dataset_dict(dataset_dict, frac)
    logger.info("Dataset example: %s", dataset_dict['train'][0] if len(dataset_dict['train']) > 0 else "N/A")
    return dataset_dict


def save_ttt_model(model, tokenizer, model_path: str, save_adapter: bool=True, save_merged: bool=True):
    if save_adapter:
        os.makedirs(model_path, exist_ok=True)
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
    if save_merged:
        os.makedirs(model_path, exist_ok=True)
        merged_model = model.merge_and_unload()
        merged_name = model_path + ENV_VARS["HF_OUTPUT_MERGED_SUFFIX"]
        merged_model.save_pretrained(merged_name)
        tokenizer.save_pretrained(merged_name)


def test_time_training(
        kaggle_dir: str=None,
        output_dir: str=None,
        save_adapter: bool=True,
        save_merged: bool=True,
        split: str="test",
        submission_run: bool=True,
        model_name: str=None,
        eval_split: str="eval",
        dataset_name: str="",
    ):
    # ---- DEVICE ----
    gpu_availability.print_gpu_availability()

    # ---- MODEL ----
    model_name = model_name if model_name else ENV_VARS["HF_BASE_MODEL"]
    model = get_model(model_name, untie_lm_head=ENV_VARS["UNTIE_LM_HEAD"])

    # ---- DATASET ----
    if dataset_name: # Test runs online on kaggle
        dataset_dict = get_dataset(dataset_name)
    else:
        dataset_dict = get_kaggle_dataset(kaggle_dir, split=split, submission_run=submission_run, eval_split=eval_split)

    # ---- PREPROCESS ----
    tokenizer = get_tokenizer(model_name)
    dataset_dict = augment_dataset(dataset_dict, tokenizer, only_splits=["train"])
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
    if ENV_VARS["USE_LORA"]:
        model = setup_peft_lora(model)

    # ---- TRAIN ----
    use_bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    print_before_training_info(model, tokenized_datasets, use_bf16)
    model_path = output_dir.rstrip("/") + "/" + ENV_VARS["HF_OUTPUT_MODEL"]
    match ENV_VARS["TRAIN_METHOD"]:
        case "transformers":
            train_transformers(model, tokenized_datasets, tokenizer, output_path=model_path, push=False)
        case "trl":
            train_trl(model, tokenized_datasets, tokenizer, output_path=model_path, push=False)
        case _:
            train_trl(model, tokenized_datasets, tokenizer, output_path=model_path, push=False)

    # ---- SAVE ----
    save_ttt_model(model, tokenizer, model_path, save_adapter=save_adapter, save_merged=save_merged)

    # ---- TEST ----
    test_model_on_dataset(model, tokenizer, dataset_dict, splits=["train", "eval"])
    test_model_generation(model, tokenizer)
    
    return model

if __name__ == "__main__":
    test_time_training(
        kaggle_dir="./data/kaggle_input",
        output_dir="./data/models/ttt_model",
        save_adapter=True,
        save_merged=True,
        split="test",
        submission_run=True,
        model_name="meo-des/nemo_arc_main_base_1s2e_m",
        dataset_name="",
    )
