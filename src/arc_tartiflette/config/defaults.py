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
        "value": "q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,embed_tokens",
        "type": list[str]
    },
    "LORA_R": {
        "value": "128",
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
    "USE_COMPLETION_MASK": {
        "value": "True",
        "type": bool
    },
    "NUM_SOLVE_TESTS": {
        "value": "100",
        "type": int
    },
    "SOLVE_BATCH_SIZE": {
        "value": "4",
        "type": int
    },
    "UNTIE_LM_HEAD": {
        "value": "False",
        "type": bool
    },
    "MODEL_TYPE": {
        "value": "base",
        "type": str
    },
    "QUANTIZE_MODEL": {
        "value": "0",
        "type": int
    },
    "PRINT_QUANT_INFO": {
        "value": "False",
        "type": bool
    },
    "BNB_4BIT_QUANT_TYPE": {
        "value": "nf4",
        "type": str
    },
    "LORA_MODULES_TO_SAVE": {
        "value": "",
        "type": list[str]
    }
}