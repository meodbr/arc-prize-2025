import os
import json
from dataclasses import dataclass, field

from dotenv import load_dotenv

from arc_tartiflette.config import defaults


def convert_env_var(value, var_type):
    if value == "None":
        return None
    if var_type is bool:
        return value.lower() in ("true", "1", "yes")
    elif var_type is list[str]:
        out = value.split(",")
        if out == ['']:
            return []
        return out
    else:
        return var_type(value)


def get_env_vars_with_defaults():
    env_vars = defaults.DEFAULT_ENV_VARS
    returned_vars = {}
    for var, default in env_vars.items():
        returned_vars[var] = os.environ.get(var, default["value"])
        returned_vars[var] = convert_env_var(returned_vars[var], default["type"])
    return returned_vars


def refresh_env_vars():
    load_dotenv()
    return get_env_vars_with_defaults()


def get_logging_config():
    log_config_path = os.environ.get("LOGGING_CONFIG_PATH", "configs/logging.json")
    with open(log_config_path, "r") as f:
        return json.load(f)


ENV_VARS = refresh_env_vars()


@dataclass(frozen=True)
class Settings:
    HF_BASE_MODEL: str = "HuggingFaceTB/SmolLM2-135M"
    HF_DATASET: str = "arc-agi-2_kaggle_flatten"
    HF_OUTPUT_MODEL: str = "default_output_model"
    HF_OUTPUT_MERGED_SUFFIX: str = "_m"
    TOKENIZER_MAX_LENGTH: int = 2048
    TRAIN_METHOD: str = "default"
    TRAIN_EPOCHS: float = 0.1
    TRAIN_STEPS: int = -1
    BATCH_SIZE: int = 4
    LR: float = 5e-5
    GRAD_ACC_STEPS: int = 2
    GRAD_CHPT: bool = False
    USE_LORA: bool = True
    LORA_TARGET_MODULES: list[str] = field(default_factory=lambda: ["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","embed_tokens"])
    LORA_R: int = 128
    LORA_ALPHA: int = 24
    LORA_DROPOUT: float = 0.1
    USE_RSLORA: bool = True
    DO_AUG: bool = False
    AUG_COLOR_NUM: int = 3
    AUG_ORDER_NUM: int = 3
    DATASET_FRAC: float = 1.0
    SAVE_EVERY_N_STEPS: int = 1000
    OPTIM: str = "adamw_8bit"
    LR_SCHEDULER_TYPE: str = "cosine"
    WEIGHT_DECAY: float = 0.0
    EMBEDDING_LR: float = 5e-5
    WARMUP_RATIO: float = 0.25
    USE_COMPLETION_MASK: bool = True
    NUM_SOLVE_TESTS: int = 100
    SOLVE_BATCH_SIZE: int = 4
    UNTIE_LM_HEAD: bool = False
    MODEL_TYPE: str = "base"
    QUANTIZE_MODEL: int = 0
    PRINT_QUANT_INFO: bool = False
    BNB_4BIT_QUANT_TYPE: str = "nf4"
    LORA_MODULES_TO_SAVE: list[str] = field(default_factory=list)

    @classmethod
    def from_env(cls):
        load_dotenv()
        init_values = {}
        for field_name, field_def in cls.__dataclass_fields__.items():
            env_value = os.environ.get(field_name, None)
            if env_value is None:
                continue
            converted_value = convert_env_var(env_value, field_def.type)
            init_values[field_name] = converted_value
        return cls(**init_values)

settings = Settings.from_env()
