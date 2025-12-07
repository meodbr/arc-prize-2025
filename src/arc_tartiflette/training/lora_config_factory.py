import logging

from peft import LoraConfig, TaskType

from arc_tartiflette.config.settings import ENV_VARS


class LoraConfigFactory:
    """Factory class to create LoRA configuration dictionaries based on presets."""

    PRESETS = {
        "default": {
            "r": 128,
            "lora_alpha": 16,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "up_proj",
                "down_proj",
                "embed_tokens",
            ],
            "lora_dropout": 0.05,
            "bias": "none",
            "use_rslora": False,
            "task_type": "CAUSAL_LM",
            "modules_to_save": None,
        },
        "small": {
            "r": 64,
            "lora_alpha": 8,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "up_proj",
                "down_proj",
                "embed_tokens",
            ],
            "lora_dropout": 0.1,
            "bias": "none",
            "use_rslora": False,
            "task_type": "CAUSAL_LM",
            "modules_to_save": None,
        },
    }

    @classmethod
    def from_preset(cls, preset_name: str) -> LoraConfig:
        if preset_name not in cls.PRESETS:
            raise ValueError(f"Preset '{preset_name}' not found in LoraConfigFactory.")
        return LoraConfig(**cls.PRESETS[preset_name])

    @staticmethod
    def from_env() -> LoraConfig:
        return LoraConfig(
            r=ENV_VARS["LORA_R"],
            lora_alpha=ENV_VARS["LORA_ALPHA"],
            target_modules=ENV_VARS["LORA_TARGET_MODULES"],
            lora_dropout=ENV_VARS["LORA_DROPOUT"],
            bias="none",
            use_rslora=ENV_VARS["USE_RSLORA"],
            task_type=TaskType.CAUSAL_LM,
            modules_to_save=(
                ENV_VARS["LORA_MODULES_TO_SAVE"]
                if len(ENV_VARS["LORA_MODULES_TO_SAVE"]) > 0
                else None
            ),
        )
