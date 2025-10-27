import json
import re
import os
import random
import copy
from itertools import permutations
from datasets import Dataset
import numpy as np

from arc_tartiflette.utils import constants

NEONEYE_DATASETS = {
    "arc-agi-2": [
        ("ARC-AGI-2", "training"),
        ("ARC-AGI-2", "evaluation")
    ],   
    "main_v1": [
        ("ARC-AGI-2", "training"),
        ("RE-ARC", "easy"),
        ("RE-ARC", "hard"),
        ("ConceptARC", "all")
    ]
}

def load_arc_challenges(filename: str):
    with open(filename, 'r') as f:
        return json.load(f)

def collapse_last_lists(obj):
    # Dump nicely formatted JSON first
    dict_obj = convert_np_arrays_to_lists(obj)
    text = json.dumps(dict_obj, indent=4)
    # Collapse the innermost lists of numbers only
    text = re.sub(r'\[\s+([0-9,\s]+?)\s+\]', lambda m: '[' + ' '.join(m.group(1).split()) + ']', text)
    return text

def save_arc_challenges(data, filename: str):
    with open(filename, 'w') as f:
        f.write(collapse_last_lists(data))

def convert_np_arrays_to_lists(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_np_arrays_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_arrays_to_lists(v) for v in obj]
    else:
        return obj


def load_challenges_kaggle_format(input_dir):
    input_dict = {}
    out_dict = {}

    # Retrieve data from files
    for key, file_name in constants.ARC_INPUT_FILES.items():
        file_path = os.path.join(input_dir, file_name)
        input_dict[key] = load_arc_challenges(file_path)
    
    # Reorganize data : solutions must be contained in task
    for d_name in ["train", "eval"]:
        out_dataset = {}
        challenges = input_dict[d_name+"_challenges"]
        solutions = input_dict[d_name+"_solutions"]
        for task_name, task in challenges.items():
            task_test_outputs = solutions[task_name]
            assert len(task["test"]) == len(solutions[task_name])
            for i in range(len(task["test"])):
                assert "output" not in task["test"][i].keys()
                task["test"][i]["output"] = task_test_outputs[i]
            out_dataset[task_name] = task
        out_dict[d_name] = out_dataset
    
    out_dict["test"] = input_dict["test_challenges"]

    return out_dict


def load_challenges_neoneye_format(path_tuples, neoneye_dir: str="data/neoneye") -> dict:
    """
    neoneye format: neoneye_dir/dataset_name/data/split_name/task_name.json

    path_tuples: list of (dataset_name, split_name) tuples to load

    returns: dict of datasets, each dataset is a dict of tasks
    """
    datasets = {}
    expanded_tuples = []
    for dataset_name, split_name in path_tuples:
        splits_path = os.path.join(neoneye_dir, dataset_name, "data")
        if split_name == "all":
            for element in os.listdir(splits_path):
                if os.path.isdir(os.path.join(splits_path, element)):
                    expanded_tuples.append((dataset_name, element))
        else:
            expanded_tuples.append((dataset_name, split_name))


    for dataset_name, split_name in expanded_tuples:
        dataset_path = os.path.join(neoneye_dir, dataset_name, "data", split_name)
        dataset = {}
        for task_file in os.listdir(dataset_path):
            if task_file.endswith(".json"):
                task_name = task_file[:-5]
                task_path = os.path.join(dataset_path, task_file)
                with open(task_path, 'r') as f:
                    task_data = json.load(f)
                dataset[task_name] = task_data
        # leave only letters, numbers and dots in dataset key
        cleaned_dataset_name = re.sub(r'[^a-zA-Z0-9.]', '.', dataset_name)
        cleaned_split_name = re.sub(r'[^a-zA-Z0-9.]', '.', split_name)
        datasets_key = f"{cleaned_dataset_name}.{cleaned_split_name}"
        datasets[datasets_key] = dataset

    return datasets


def grid_to_str(grid: list[list[int]]) -> str:
    return "\n".join(["".join([str(c) for c in row]) for row in grid])

def flatten_task(task: dict, prompt: bool=False) -> str:
    # Prepare few-shot context from examples
    context = ""
    for ex in task["train"]:
        ex_input_str = grid_to_str(ex["input"])
        ex_output_str = grid_to_str(ex["output"])
        context += f"Input:\n{ex_input_str}\nOutput:\n{ex_output_str}\n\n"

    # Prepare test example
    for ex in task["test"]:
        ex_input_str = grid_to_str(ex["input"])
        if prompt:
            context += f"Input:\n{ex_input_str}\nOutput:\n"
            break

        if "output" in ex.keys():
            ex_output_str = grid_to_str(ex["output"])
            context += f"Input:\n{ex_input_str}\nOutput:\n{ex_output_str}\n\n"
        else:
            context += f"Input:\n{ex_input_str}\n\n"

    return context

def flatten_dataset(dataset: dict, prompt: bool=False) -> list[dict]:
    data = []

    for _, task_data in dataset.items():
        data.append({"text": flatten_task(task_data, prompt=prompt), "task": task_data})
    return data

def dict_to_transformers_dataset(dataset: dict) -> Dataset:
    return Dataset.from_list(flatten_dataset(dataset))

def sample_dict(data: dict, num_samples: int) -> dict:
    sampled_dict = dict(random.sample(list(data.items()), num_samples))
    return sampled_dict


def augment_rotations_flips(task: dict) -> list[dict]:
    """
    Generate augmented tasks by applying rotations and flips to the input and output grids.
    Returns a list of augmented tasks including the original.
    """
    augmented_tasks = []

    transformations = [
        lambda x: x,  # original
        lambda x: np.rot90(x, k=1), # 90 degrees
        lambda x: np.rot90(x, k=2), # 180 degrees
        lambda x: np.rot90(x, k=3), # 270 degrees
        lambda x: np.fliplr(x),     # vertical flip
        lambda x: np.fliplr(np.rot90(x, k=1)), # vertical flip + 90 degrees
        lambda x: np.fliplr(np.rot90(x, k=2)), # vertical flip + 180 degrees
        lambda x: np.fliplr(np.rot90(x, k=3)), # vertical flip + 270 degrees
    ]

    for transform in transformations:
        new_task = {"train": [], "test": []}
        for ex in task["train"]:
            new_ex = {
                "input": transform(ex["input"]),
                "output": transform(ex["output"])
            }
            new_task["train"].append(new_ex)
        for ex in task["test"]:
            new_ex = {
                "input": transform(ex["input"])
            }
            if "output" in ex:
                new_ex["output"] = transform(ex["output"])
            new_task["test"].append(new_ex)
        augmented_tasks.append(new_task)

    return augmented_tasks


def random_permutations(size=10, n_samples=5, seed=None):
    """
    Generate n_samples random permutations of range(n_colors),
    without constructing all possible permutations.
    """
    rng = np.random.default_rng(seed)
    return [rng.permutation(size) for _ in range(n_samples)]


def augment_color_permutations(task: dict, max_perm: int=3) -> list[dict]:
    """
    Generate augmented tasks by permuting the colors in the input and output grids.
    Returns a list of augmented tasks including the original.

    task: dict with train and test examples, each example has 'input' and 'output' np.arrays
    """

    augmented_tasks = []

    sampled_perms =  [np.array(p) for p in random_permutations(10, max_perm)]

    for perm in sampled_perms:
        new_task = {"train": [], "test": []}
        for ex in task["train"]:
            new_ex = {
                "input": perm[ex["input"]],
                "output": perm[ex["output"]],
            }
            new_task["train"].append(new_ex)
        for ex in task["test"]:
            new_ex = {
                "input": perm[ex["input"]],
            }
            if "output" in ex:
                new_ex["output"] = perm[ex["output"]]
            new_task["test"].append(new_ex)
        augmented_tasks.append(new_task)

    return augmented_tasks



def augment_examples_order_permutations(task: dict, max_perm: int=3) -> list[dict]:
    """
    Generate augmented tasks by permuting the order of train and test examples.
    Returns a list of augmented tasks including the original.

    task: dict with train and test examples, each example has 'input' and 'output' np.arrays
    """

    augmented_tasks = []


    combined = task["train"] + task["test"]
    length = len(combined)
    possible_permutations = length * (length - 1) // 2  # approximate number of unique permutations
    sampled_perms = random_permutations(length, min(max_perm, possible_permutations))

    for perm in sampled_perms:
        new_task = {
            "train": [combined[i] for i in perm[:len(task["train"])]],
            "test": [combined[i] for i in perm[len(task["train"]):]]
        }
        augmented_tasks.append(new_task)

    return augmented_tasks


def augment_dict(data: dict, augment_types: list[str]) -> dict:
    """
    Augment each task in the dataset according to the specified augmentation types.
    Supported types: "rot_flip", "color", "order"
    Returns a new dataset dict with augmented tasks.
    """
    # If grids are not numpy arrays, convert them
    print("Converting grids to numpy arrays...")
    for task in data.values():
        for split in ["train", "test"]:
            for example in task[split]:
                if not isinstance(example["input"], np.ndarray):
                    example["input"] = np.array(example["input"])
                if "output" in example and not isinstance(example["output"], np.ndarray):
                    example["output"] = np.array(example["output"])

    print("Augmenting tasks...")
    augmented_data = {}
    for i, (task_name, task) in enumerate(data.items()):

        if i % 10 == 0:
            print(f"  Augmented {i}/{len(data)} tasks.")

        augmented_tasks = [task]  # start with original task
        if "rot_flip" in augment_types:
            new_tasks = []
            for t in augmented_tasks:
                new_tasks.extend(augment_rotations_flips(t))
            augmented_tasks = new_tasks
        if "color" in augment_types:
            new_tasks = []
            for t in augmented_tasks:
                new_tasks.extend(augment_color_permutations(t))
            augmented_tasks = new_tasks
        if "order" in augment_types:
            new_tasks = []
            for t in augmented_tasks:
                new_tasks.extend(augment_examples_order_permutations(t))
            augmented_tasks = new_tasks

        # Add augmented tasks to the dataset with unique names
        for i, aug_task in enumerate(augmented_tasks):
            aug_task_name = f"{task_name}_aug{i}"
            augmented_data[aug_task_name] = aug_task

    return augmented_data


from arc_tartiflette.utils import plot
if __name__ == "__main__":
    dataset = {"0a938d79":{"train": [{"input": [[0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], "output": [[0, 0, 0, 0, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0], [0, 0, 0, 0, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0], [0, 0, 0, 0, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0], [0, 0, 0, 0, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0], [0, 0, 0, 0, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0], [0, 0, 0, 0, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0], [0, 0, 0, 0, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0], [0, 0, 0, 0, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0], [0, 0, 0, 0, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0], [0, 0, 0, 0, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0]]}, {"input": [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], "output": [[0, 0, 0, 0, 0, 1, 0, 0, 3, 0, 0, 1, 0, 0, 3, 0, 0, 1, 0, 0, 3, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 3, 0, 0, 1, 0, 0, 3, 0, 0, 1, 0, 0, 3, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 3, 0, 0, 1, 0, 0, 3, 0, 0, 1, 0, 0, 3, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 3, 0, 0, 1, 0, 0, 3, 0, 0, 1, 0, 0, 3, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 3, 0, 0, 1, 0, 0, 3, 0, 0, 1, 0, 0, 3, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 3, 0, 0, 1, 0, 0, 3, 0, 0, 1, 0, 0, 3, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 3, 0, 0, 1, 0, 0, 3, 0, 0, 1, 0, 0, 3, 0, 0]]}, {"input": [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 3], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]], "output": [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 2, 2, 2, 2, 2, 2, 2, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0], [3, 3, 3, 3, 3, 3, 3, 3, 3], [0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 2, 2, 2, 2, 2, 2, 2, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0], [3, 3, 3, 3, 3, 3, 3, 3, 3], [0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 2, 2, 2, 2, 2, 2, 2, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0], [3, 3, 3, 3, 3, 3, 3, 3, 3], [0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 2, 2, 2, 2, 2, 2, 2, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0], [3, 3, 3, 3, 3, 3, 3, 3, 3], [0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 2, 2, 2, 2, 2, 2, 2, 2]]}, {"input": [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [4, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]], "output": [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [4, 4, 4, 4, 4, 4, 4, 4], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [4, 4, 4, 4, 4, 4, 4, 4], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [4, 4, 4, 4, 4, 4, 4, 4]]}], "test": [{"input": [[0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], "output": [[0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0], [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0], [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0], [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0], [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0], [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0], [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0], [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0], [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0], [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0], [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0]]}]}}

    data = augment_dict(dataset, augment_types=["rot_flip"])
    plot.peek_dict(data, num_tasks=5)
    flattened_data = flatten_dataset(data)