ARC_INPUT_FILES = {
    "train_challenges": "arc-agi_training_challenges.json",
    "train_solutions": "arc-agi_training_solutions.json",
    "eval_challenges": "arc-agi_evaluation_challenges.json",
    "eval_solutions": "arc-agi_evaluation_solutions.json",
    "test_challenges": "arc-agi_test_challenges.json",
}

HF_USER = "meo-des"

COLOR_MAP = {
    0: "#000000",  # Black
    1: "#0074D9",  # Blue
    2: "#FF4136",  # Red
    3: "#2ECC40",  # Green
    4: "#FFDC00",  # Yellow
    5: "#AAAAAA",  # Grey
    6: "#F012BE",  # Pink
    7: "#FF851B",  # Orange
    8: "#7FDBFF",  # Light Blue
    9: "#870C25",  # Dark Red
}

DEFAULT_PROMPT_FORMAT = {
    "preprompt": "",
    "input_beg": "Input:\n",
    "output_beg": "Output:\n",
    "row_end": "\n",
    "grid_end": "\n",
    "bos_token": "",
    "eos_token": "\n",
}