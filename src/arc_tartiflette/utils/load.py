import json
import re
import os
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

