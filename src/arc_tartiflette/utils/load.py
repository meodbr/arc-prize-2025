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
    ],
    "main_v2": [
        ("ARC-AGI-2", "training"),
        ("RE-ARC", "easy"),
        ("RE-ARC", "hard"),
        ("ConceptARC", "all"),
    ]
}

def convert_lists_to_np_arrays(task):
    if isinstance(task, list):
        return [convert_lists_to_np_arrays(v) for v in task]
    elif isinstance(task, dict):
        new_dict = {}
        for k, v in task.items():
            if k in ["input", "output"]:
                new_dict[k] = np.array(v)
            else:
                new_dict[k] = convert_lists_to_np_arrays(v)
        return new_dict
    else:
        return task


def load_arc_challenges(filename: str):
    with open(filename, 'r') as f:
        json_data = json.load(f)
        # Convert lists to numpy arrays
        json_data = convert_lists_to_np_arrays(json_data)
        return json_data

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

def print_arc_challenge(task):
    print(collapse_last_lists(task))

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


def grid_to_str(grid: list[list[int]], format: dict[str, str]) -> str:
    assert "row_end" in format.keys()
    return format["row_end"].join(["".join([str(c) for c in row]) for row in grid])

def flatten_task(task: dict, prompt: bool=False, format: dict=constants.DEFAULT_PROMPT_FORMAT) -> str:
    # Prepare few-shot context from examples
    context = format["bos_token"] + format["preprompt"]
    for i, ex in enumerate(task["train"]):
        if i > 0:
            context += format["bos_token"]
        context += format["input_beg"]
        context += grid_to_str(ex["input"], format=format)
        context += format["grid_end"]
        context += format["output_beg"]
        context += grid_to_str(ex["output"], format=format)
        context += format["grid_end"]
        context += format["eos_token"]

    # Prepare test example
    for ex in task["test"]:
        ex_input_str = grid_to_str(ex["input"], format=format)

        context += format["input_beg"]
        context += grid_to_str(ex["input"], format=format)
        context += format["grid_end"]

        if prompt:
            context += format["output_beg"]
            break

        if "output" in ex.keys():
            context += format["output_beg"]
            context += grid_to_str(ex["output"], format=format)
            context += format["grid_end"]

        context += format["eos_token"]

    return context

def flatten_dataset(dataset: dict, prompt: bool=False, format: dict=constants.DEFAULT_PROMPT_FORMAT) -> list[dict]:
    data = []

    for _, task_data in dataset.items():
        data.append({"text": flatten_task(task_data, prompt=prompt, format=format), "task": task_data})
    return data

def dict_to_transformers_dataset(dataset: dict, format: dict=constants.DEFAULT_PROMPT_FORMAT, keep_tests: bool=True) -> Dataset:
    if not keep_tests:
        # Remove outputs from test examples
        dataset = copy.deepcopy(dataset)
        for task in dataset.values():
            task["test"] = [test for test in task["test"] if "output" in test.keys()]
    return Dataset.from_list(flatten_dataset(dataset, format=format))

def transformers_dataset_to_dict(hf_dataset: Dataset) -> dict:
    dataset = {f"task_{i}": task["task"] for i, task in enumerate(hf_dataset)}
    for task in dataset.values():
        for ex in task["train"] + task["test"]:
            for key, grid in ex.items():
                if key in ["input", "output"]:
                        ex[key] = np.array(grid)
    return dataset

def sample_dict(data: dict, num_samples: int) -> dict:
    sampled_dict = dict(random.sample(list(data.items()), num_samples))
    return sampled_dict


def augment_rotations_flips(task: dict, max_transformations: int) -> list[dict]:
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

    for i, transform in enumerate(transformations):
        if i >= max_transformations:
            break
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


def augment_transformers_dataset(
        dataset: Dataset, 
        format: dict=constants.DEFAULT_PROMPT_FORMAT,
        augment_types: list[str]=["rot_flip", "color", "order"],
        multipliers: dict={},
    ) -> Dataset:
    """
    Efficiently applies a task augmentation function to a Hugging Face Dataset.

    Args:
        dataset (Dataset): Dataset with {"text": ..., "task": ...}.
        augment_fn (callable): Function that takes a task dict -> list of task dicts.
        format (dict, optional): Used to flatten each new task into text.
        keep_original (bool): If True, keep original examples as well.

    Returns:
        Dataset: Augmented dataset.
    """
    def gen():
        for item in dataset:
            for aug_task in augment_task(item["task"], augment_types=augment_types, multipliers=multipliers):
                yield {
                    "text": flatten_task(aug_task, prompt=False, format=format) if format else "",
                    "task": aug_task,
                }
    return Dataset.from_generator(gen)



def augment_task(task: dict, augment_types: list[str], multipliers: dict[str, int]) -> dict:
    default_multipliers = {
        "rot_flip": 8,
        "color": 3,
        "order": 3,
    }
    mul = {**default_multipliers, **multipliers}
    augmented_tasks = [task]  # start with original task
    if "rot_flip" in augment_types:
        new_tasks = []
        for t in augmented_tasks:
            new_tasks.extend(augment_rotations_flips(t, max_transformations=mul["rot_flip"]))
        augmented_tasks = new_tasks

    if "color" in augment_types:
        new_tasks = []
        for t in augmented_tasks:
            new_tasks.extend(augment_color_permutations(t, max_perm=mul["color"]))
        augmented_tasks = new_tasks

    if "order" in augment_types:
        new_tasks = []
        for t in augmented_tasks:
            new_tasks.extend(augment_examples_order_permutations(t, max_perm=mul["order"]))
        augmented_tasks = new_tasks
    
    return augmented_tasks


def augment_dict(data: dict, augment_types: list[str], multipliers: dict[str, int]) -> dict:
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
    
    default_multipliers = {
        "color": 3,
        "order": 3,
    }

    mul = {**default_multipliers, **multipliers}

    print("Augmenting tasks...")
    augmented_data = {}
    for i, (task_name, task) in enumerate(data.items()):
        if i % 10 == 0:
            print(f"  Augmented {i}/{len(data)} tasks.")

        augmented_tasks = augment_task(task, augment_types, mul)

        # Add augmented tasks to the dataset with unique names
        for i, aug_task in enumerate(augmented_tasks):
            aug_task_name = f"{task_name}_aug{i}"
            augmented_data[aug_task_name] = aug_task

    return augmented_data


from arc_tartiflette.utils import plot
if __name__ == "__main__":
    dataset = {"0a938d79":{"train": [{"input": [[0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], "output": [[0, 0, 0, 0, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0], [0, 0, 0, 0, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0], [0, 0, 0, 0, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0], [0, 0, 0, 0, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0], [0, 0, 0, 0, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0], [0, 0, 0, 0, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0], [0, 0, 0, 0, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0], [0, 0, 0, 0, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0], [0, 0, 0, 0, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0], [0, 0, 0, 0, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0, 2, 0, 8, 0]]}, {"input": [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], "output": [[0, 0, 0, 0, 0, 1, 0, 0, 3, 0, 0, 1, 0, 0, 3, 0, 0, 1, 0, 0, 3, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 3, 0, 0, 1, 0, 0, 3, 0, 0, 1, 0, 0, 3, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 3, 0, 0, 1, 0, 0, 3, 0, 0, 1, 0, 0, 3, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 3, 0, 0, 1, 0, 0, 3, 0, 0, 1, 0, 0, 3, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 3, 0, 0, 1, 0, 0, 3, 0, 0, 1, 0, 0, 3, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 3, 0, 0, 1, 0, 0, 3, 0, 0, 1, 0, 0, 3, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 3, 0, 0, 1, 0, 0, 3, 0, 0, 1, 0, 0, 3, 0, 0]]}, {"input": [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 3], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]], "output": [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 2, 2, 2, 2, 2, 2, 2, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0], [3, 3, 3, 3, 3, 3, 3, 3, 3], [0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 2, 2, 2, 2, 2, 2, 2, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0], [3, 3, 3, 3, 3, 3, 3, 3, 3], [0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 2, 2, 2, 2, 2, 2, 2, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0], [3, 3, 3, 3, 3, 3, 3, 3, 3], [0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 2, 2, 2, 2, 2, 2, 2, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0], [3, 3, 3, 3, 3, 3, 3, 3, 3], [0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 2, 2, 2, 2, 2, 2, 2, 2]]}, {"input": [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [4, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]], "output": [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [4, 4, 4, 4, 4, 4, 4, 4], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [4, 4, 4, 4, 4, 4, 4, 4], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [4, 4, 4, 4, 4, 4, 4, 4]]}], "test": [{"input": [[0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], "output": [[0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0], [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0], [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0], [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0], [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0], [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0], [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0], [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0], [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0], [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0], [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0]]}]}}

    data = augment_dict(
        dataset,
        augment_types=["rot_flip"],
        multipliers={"rot_flip": 8}
    )
    plot.peek_dict(data, num_tasks=5)
    flattened_data = flatten_dataset(data)