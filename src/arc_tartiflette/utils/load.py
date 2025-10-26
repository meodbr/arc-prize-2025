import json
import re
import os
from arc_tartiflette.utils import constants
from datasets import Dataset

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

def flatten_task(task: dict) -> str:
    # Prepare few-shot context from examples
    context = ""
    for ex in task["train"]:
        ex_input_str = grid_to_str(ex["input"])
        ex_output_str = grid_to_str(ex["output"])
        context += f"Input:\n{ex_input_str}\nOutput:\n{ex_output_str}\n\n"

    # Prepare test example
    for ex in task["test"]:
        ex_input_str = grid_to_str(ex["input"])
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

def extract_output_from_text(text, auto_correct: bool=False, strict_format: bool=False) -> list[list[int]]:
    """
    Extract the output rectangle grid from the model's generated text.
    Parameters:
    - text (str): The generated text containing the output grid in the format:
        "Output:\n123\n456\n789\n\n"

    It must be robust against non regular outputs e.g. non consistent line lengths, extra text, etc.
    """
    # Find the "Output:" section
    output_match = re.search(r'Output:\s*(.*?)\s*(Input:|$)', text, re.DOTALL)
    if output_match:
        output_text = output_match.group(1).strip()
    else:
        if strict_format:
            raise ValueError("Strict format enforced and no 'Output:' section found in the text.")
        else:
            # find the first grid-like structure in the text
            grid_match = re.search(r'(\d+\n)+\d+', text)
            if not grid_match:
                raise ValueError("No grid-like structure found in the text.")
            output_text = grid_match.group(0)

    lines = output_text.splitlines()

    # Determine the expected line length (most common length)
    line_lengths = [len(line) for line in lines if line.strip()]
    if not line_lengths:
        raise ValueError("No valid lines found in the 'Output:' section.")
    expected_length = max(set(line_lengths), key=line_lengths.count)

    grid = []
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            continue  # Skip empty lines
        if auto_correct:
            # Adjust line to expected length
            if len(stripped_line) < expected_length:
                stripped_line = stripped_line.ljust(expected_length, '0')  # Pad with '0's
            elif len(stripped_line) > expected_length:
                stripped_line = stripped_line[:expected_length]  # Truncate
        elif len(stripped_line) != expected_length:
            raise ValueError(f"Inconsistent line length in output: '{stripped_line}'")

        row = [int(char) for char in stripped_line if char.isdigit()]
        grid.append(row)

    assert all(len(row) == expected_length for row in grid), "Internal error: Inconsistent row lengths in the output grid."
    assert len(grid) > 0, "Output grid is empty."
    return grid

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