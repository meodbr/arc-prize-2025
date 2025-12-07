import huggingface_hub as hf
from datasets import DatasetDict, concatenate_datasets
from transformers import AutoTokenizer
import os

from arc_tartiflette.utils import load, utils, constants
from arc_tartiflette.model import tokenizer_tools
from arc_tartiflette.model.tokenize_functions import tokenize_dataset_base
from arc_tartiflette.train import shrink_vocab

def kaggle_flatten_and_push(input_dir):
    # Get dataset
    dict_datasets = load.load_challenges_kaggle_format(input_dir)

    hf_train_dataset = load.dict_to_transformers_dataset(dict_datasets["train"])
    hf_eval_dataset = load.dict_to_transformers_dataset(dict_datasets["eval"])
    hf_test_dataset = load.dict_to_transformers_dataset(dict_datasets["test"])

    hf_datasetdict = DatasetDict({
        "train": hf_train_dataset,
        "eval": hf_eval_dataset,
        "test": hf_test_dataset,
    })

    hf_datasetdict.push_to_hub("meo-des/arc-agi-2_kaggle_flattened")

def neoneye_flatten_and_push(
        path_tuples: list, 
        output_name: str="meo-des/arc-agi-2_neoneye",
        neoneye_dir: str="data/neoneye",
        format: dict=constants.DEFAULT_PROMPT_FORMAT,
    ):
    # Get dataset
    dict_datasets = load.load_challenges_neoneye_format(path_tuples, neoneye_dir=neoneye_dir)

    datasetdict = {}
    for dataset_name, dataset in dict_datasets.items():
        hf_dataset = load.dict_to_transformers_dataset(dataset, format)
        datasetdict[dataset_name] = hf_dataset

    hf_concatenated = concatenate_datasets(list(datasetdict.values()))
    hf_datasetdict = DatasetDict({
        "train": hf_concatenated.shuffle(seed=42).select(range(int(0.8*len(hf_concatenated)))),
        "eval": hf_concatenated.shuffle(seed=42).select(range(int(0.8*len(hf_concatenated)), int(0.9*len(hf_concatenated)))),
        "test": hf_concatenated.shuffle(seed=42).select(range(int(0.9*len(hf_concatenated)), len(hf_concatenated))),
    })
    hf_datasetdict.push_to_hub(output_name)
    print(f"Dataset pushed to hub: {output_name}")

def neoneye_augment_and_push(
        path_tuples: list,
        output_name: str="meo-des/arc-agi-2_neoneye",
        neoneye_dir: str="data/neoneye",
        format: dict=constants.DEFAULT_PROMPT_FORMAT,
        use_arc_public_eval: bool=False,
        use_arc_public_eval_only_in_test: bool=True,
    ):
    # Get dataset
    dict_datasets = load.load_challenges_neoneye_format(path_tuples, neoneye_dir=neoneye_dir)

    datasetdict = {}
    print("Augmenting datasets...")
    for dataset_name, dataset in dict_datasets.items():
        hf_dataset = load.dict_to_transformers_dataset(dataset, format)
        print(f"  Augmenting dataset: {dataset_name} with {len(dataset)} tasks.")
        hf_dataset = load.augment_transformers_dataset(hf_dataset, format=format, augment_types=["rot_flip", "color", "order"])
        print(f"  Augmented dataset now has {len(dataset)} tasks.")
        datasetdict[dataset_name] = hf_dataset
    
    if use_arc_public_eval:
        dict_ds_eval = load.load_challenges_neoneye_format(path_tuples=[("ARC-AGI-2", "evaluation")], neoneye_dir=neoneye_dir)
        ds_eval = next(iter(dict_ds_eval.values()))
        print(f"  Adding ARC Public Evaluation dataset with {len(ds_eval)} tasks.")
        hf_ds_eval = load.dict_to_transformers_dataset(ds_eval, format)
        if use_arc_public_eval_only_in_test:
            hf_ds_eval_splitted = {"test": hf_ds_eval}
        else:
            frac = 0.4
            hf_ds_eval_splitted = hf_ds_eval.shuffle(seed=42).train_test_split(test_size=frac)

        arc_eval_test = hf_ds_eval_splitted["test"]
        if not use_arc_public_eval_only_in_test:
            datasetdict["arc_eval_train"] = load.augment_transformers_dataset(
                hf_ds_eval_splitted["train"],
                format=format,
                augment_types=["order", "color"],
                multipliers={"order": 2, "color": 2}, # x4 total
            )


    hf_concatenated = concatenate_datasets(list(datasetdict.values()))
    if use_arc_public_eval:
        hf_datasetdict = DatasetDict({
            "train": hf_concatenated.shuffle(seed=42).select(range(int(0.95*len(hf_concatenated)))),
            "eval": hf_concatenated.shuffle(seed=42).select(range(int(0.95*len(hf_concatenated)), len(hf_concatenated))),
            "test": arc_eval_test,
        })
    else:
        hf_datasetdict = DatasetDict({
            "train": hf_concatenated.shuffle(seed=42).select(range(int(0.8*len(hf_concatenated)))),
            "eval": hf_concatenated.shuffle(seed=42).select(range(int(0.8*len(hf_concatenated)), int(0.9*len(hf_concatenated)))),
            "test": hf_concatenated.shuffle(seed=42).select(range(int(0.9*len(hf_concatenated)), len(hf_concatenated))),
        })
    hf_datasetdict.push_to_hub(output_name)
    print(f"Dataset pushed to hub: {output_name}")


def neoneye_augment_tokenize_and_push(
        path_tuples: list,
        tokenizer: AutoTokenizer,
        output_name: str="meo-des/arc-agi-2_neoneye",
        neoneye_dir: str="data/neoneye",
        format: dict=constants.DEFAULT_PROMPT_FORMAT,
    ):
    # Get dataset
    dict_datasets = load.load_challenges_neoneye_format(path_tuples, neoneye_dir=neoneye_dir)

    datasetdict = {}
    print("Augmenting datasets...")
    for dataset_name, dataset in dict_datasets.items():
        hf_dataset = load.dict_to_transformers_dataset(dataset, format)
        print(f"  Augmenting dataset: {dataset_name} with {len(dataset)} tasks.")
        hf_dataset = load.augment_transformers_dataset(hf_dataset, format=format, augment_types=["rot_flip", "color", "order"])
        print(f"  Augmented dataset now has {len(dataset)} tasks.")
        datasetdict[dataset_name] = hf_dataset

    hf_concatenated = concatenate_datasets(list(datasetdict.values()))
    hf_datasetdict = DatasetDict({
        "train": hf_concatenated.shuffle(seed=42).select(range(int(0.8*len(hf_concatenated)))),
        "eval": hf_concatenated.shuffle(seed=42).select(range(int(0.8*len(hf_concatenated)), int(0.9*len(hf_concatenated)))),
        "test": hf_concatenated.shuffle(seed=42).select(range(int(0.9*len(hf_concatenated)), len(hf_concatenated))),
    })

    hf_datasetdict = tokenize_dataset_base(hf_datasetdict, tokenizer)

    hf_datasetdict.push_to_hub(output_name)
    print(f"Dataset pushed to hub: {output_name}")


if __name__ == "__main__":
    output_name = "meo-des/arc_main_v2_fmt_aug"
    kaggle_input_dir = "data/kaggle_input"
    model_name = "nvidia/Mistral-NeMo-Minitron-8B-Base"
    # model_name = "HuggingFaceTB/SmolLM2-135M"
    # kaggle_flatten_and_push(kaggle_input_dir)
    neoneye_path_tuples = load.NEONEYE_DATASETS["main_v2"]

    # Format
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    format = tokenizer_tools.get_architects_prompt_format(tokenizer)
    tokenizer_smol = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

    # neoneye_flatten_and_push(neoneye_path_tuples, output_name=output_name, format=format)
    # neoneye_augment_tokenize_and_push(neoneye_path_tuples, tokenizer_smol, output_name=output_name, format=format)
    neoneye_augment_and_push(
        neoneye_path_tuples,
        output_name=output_name, 
        format=format,
        use_arc_public_eval=True,
        use_arc_public_eval_only_in_test=False,
    )