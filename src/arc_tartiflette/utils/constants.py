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
    "grid_end": "",
    "bos_token": "",
    "eos_token": "\n",
}

DEFAULT_ENV_VARS = {
    "HF_BASE_MODEL": {
        "value": "HuggingFaceTB/SmolLM2-135M",
        "type": str
    },
    "HF_DATASET": {
        "value": "arc-agi-2_kaggle_flatten",
        "type": str
    },
    "HF_OUTPUT_MODEL": {
        "value": "default_output_model",
        "type": str
    },
    "HF_OUTPUT_MERGED_SUFFIX": {
        "value": "_m",
        "type": str
    },
    "TOKENIZER_MAX_LENGTH": {
        "value": "2048",
        "type": int
    },
    "TRAIN_METHOD": {
        "value": "default",
        "type": str
    },
    "TRAIN_EPOCHS": {
        "value": "0.1",
        "type": float,
    },
    "TRAIN_STEPS": {
        "value": "-1",
        "type": int,
    },
    "BATCH_SIZE": {
        "value": "4",
        "type": int
    },
    "LR": {
        "value": "5e-5",
        "type": float
    },
    "GRAD_ACC_STEPS": {
        "value": "2",
        "type": int
    },
    "GRAD_CHPT": {
        "value": "False",
        "type": bool
    },
    "USE_LORA": {
        "value": "True",
        "type": bool
    },
    "LORA_TARGET_MODULES": {
        "value": "q_proj,k_proj,v_proj,o_proj,up_proj,down_proj",
        "type": list[str]
    },
    "LORA_R": {
        "value": "32",
        "type": int
    },
    "LORA_ALPHA": {
        "value": "24",
        "type": int
    },
    "LORA_DROPOUT": {
        "value": "0.1",
        "type": float
    },
    "USE_RSLORA": {
        "value": "True",
        "type": bool
    },
    "DO_AUG": {
        "value": "False",
        "type": bool
    },
    "AUG_COLOR_NUM": {
        "value": "3",
        "type": int
    },
    "AUG_ORDER_NUM": {
        "value": "3",
        "type": int
    },
    "DATASET_FRAC": {
        "value": "1.",
        "type": float
    },
    "SAVE_EVERY_N_STEPS":{
        "value": "1000",
        "type": int
    },
    "OPTIM": {
        "value": "adamw_8bit",
        "type": str
    },
    "LR_SCHEDULER_TYPE": {
        "value": "cosine",
        "type": str
    },
    "WEIGHT_DECAY": {
        "value": 0.0,
        "type": float
    },
    "LR": {
        "value": 5e-5,
        "type": float
    },
    "EMBEDDING_LR": {
        "value": 5e-5,
        "type": float
    },
    "WARMUP_RATIO": {
        "value": 0.25,
        "type": float
    },
}