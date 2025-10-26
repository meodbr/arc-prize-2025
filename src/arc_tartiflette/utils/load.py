import json
import re
import os
import random
from datasets import Dataset

from arc_tartiflette.utils import constants

def load_arc_challenges(filename: str):
    with open(filename, 'r') as f:
        return json.load(f)

def collapse_last_lists(obj):
    # Dump nicely formatted JSON first
    text = json.dumps(obj, indent=4)
    # Collapse the innermost lists of numbers only
    text = re.sub(r'\[\s+([0-9,\s]+?)\s+\]', lambda m: '[' + ' '.join(m.group(1).split()) + ']', text)
    return text

def save_arc_challenges(data, filename: str):
    with open(filename, 'w') as f:
        f.write(collapse_last_lists(data))

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

def flatten_dataset(dataset: dict) -> list[dict]:
    data = []

    for _, task_data in dataset.items():
        data.append({"text": flatten_task(task_data)})
    return data

def dict_to_transformers_dataset(dataset: dict) -> Dataset:
    return Dataset.from_list(flatten_dataset(dataset))

def sample_dict(data: dict, num_samples: int) -> dict:
    sampled_dict = dict(random.sample(list(data.items()), num_samples))
    return sampled_dict

if __name__ == "__main__":
    dataset = {
        "gqzfqf": {
            "train": [
                {
                    "input": [[1, 2], [3, 4]],
                    "output": [[1, 2], [3, 4]],
                },
                {
                    "input": [[1, 2], [3, 4]],
                    "output": [[1, 2], [3, 4]],
                }
            ],
            "test": [
                {
                    "input": [[1, 2], [3, 4]],
                    "output": [[1, 2], [3, 4]],
                }
            ]
        }
    }
    flattened_data = flatten_dataset(dataset)
    print(flattened_data)