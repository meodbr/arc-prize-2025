import logging
from typing import Any, Tuple

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers.modeling_utils import PreTrainedModel
from transformers import (
    AutoModelForCausalLM,
    PreTrainedTokenizerBase,
    BitsAndBytesConfig,
)

from arc_tartiflette.config.settings import ENV_VARS
from arc_tartiflette.model_tools.conv_embeddings import CustomMistralModelConvEmbedding
from arc_tartiflette.model_tools.custom_pe import CustomMistralModel2DPE
from arc_tartiflette.utils import utils
from arc_tartiflette.tokenizer import TokenizerBuilder
from arc_tartiflette.model_tools import tokenizer as tokenizer_tools
from arc_tartiflette.training import LoraConfigFactory

logger = logging.getLogger(__name__)


class ModelBuilder:
    def __init__(self):
        self.model_name_or_path = None
        self.quantization_bits = None
        self.use_lora = False
        self.print_quant_info = False
        self.untie_lm_head = False
        self.bnb_4bit_quant_type = "nf4"
        self.custom_class = "base"
        self.device = "auto"
        self.shrink_vocab = False
        self.lora_config = None

    def from_hf(self, model_name: str):
        self.model_name_or_path = model_name
        return self

    def from_local(self, model_path: str):
        self.model_name_or_path = model_path
        return self

    def set_custom_class(self, custom_class: Any):
        self.custom_class = custom_class
        return self

    def with_quantization(self, bits: int):
        self.quantization_bits = bits
        return self

    def with_lora(self, use_lora: bool = True, config: LoraConfig | str = "env"):
        self.use_lora = use_lora
        if isinstance(config, str):
            match config:
                case "env":
                    self.lora_config = LoraConfigFactory.from_env()
                    return self
                case preset_name:
                    self.lora_config = LoraConfigFactory.from_preset(preset_name)
            return self
        else:
            self.lora_config = config
            return self


    def with_untied_lm_head(self, untie: bool = True):
        self.untie_lm_head = untie
        return self

    def set_bnb_4bit_quant_type(self, quant_type: str):
        self.bnb_4bit_quant_type = quant_type
        return self

    def on_device(self, device: str):
        self.device = device
        return self

    def with_print_quant_info(self, print_info: bool = True):
        self.print_quant_info = print_info
        return self
    

    def shrink_tokenizer_vocab(self, shrink: bool = True):
        self.shrink_vocab = shrink
        return self

    def build_bnb_config(self) -> BitsAndBytesConfig | None:
        if self.print_quant_info:
            utils.print_quantization_info(
                model_name=self.model_name_or_path,
                quantization_config=self.quantization_bits,
                device_map="cpu",
                verbose=True,
            )
        if self.quantization_bits == 4:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.bnb_4bit_quant_type,
                bnb_4bit_compute_type=torch.float16,
                llm_int8_enable_fp32_cpu_offload=True,
            )
        elif self.quantization_bits == 8:
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )
        return None

    def get_model_class(self) -> Any:
        match self.custom_class:
            case "base":
                logger.info("Using base AutoModelForCausalLM...")
                return AutoModelForCausalLM
            case "2DPE":
                logger.info("Using Custom Mistral Model with 2D PE...")
                return CustomMistralModel2DPE
            case "conv":
                logger.info("Using Custom Mistral Model with Conv Embeddings...")
                return CustomMistralModelConvEmbedding
            case _:
                return AutoModelForCausalLM

    def build_again_lm_head(self, model: PreTrainedModel):
        logger.info("Model head untied, re-initializing lm_head weights...")
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
        logger.info(
            "Num non-quantized parameters: %.2fM",
            sum(
                p.numel()
                for p in model.parameters()
                if p.dtype in (torch.float32, torch.float16)
            )
            / 1e6,
        )

    def _log_model_info(self, model: PreTrainedModel):
        logger.info("Model %s loaded.", self.model_name_or_path)
        logger.info("Model has %.3fB parameters.", utils.count_parameters(model) / 1e9)
        logger.info(
            "Trainableable parameters: %.3fM",
            utils.count_trainable_parameters(model) / 1e6,
        )
        logger.info("Model dtype: %s", next(model.parameters()).dtype)
        logger.debug("Model config: %s", model.config)
        logger.debug("Model generation config: %s", model.generation_config)
    
    def build_shrinked_vocab(
        self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase
    ):
        logger.info("Shrinking tokenizer vocabulary to only keep useful tokens...")
        logger.info("Original tokenizer vocab size: %d", len(tokenizer))
        logger.info(
            "Original model parameters: %.3fB", utils.count_parameters(model) / 1e9
        )
        keep_tok = list(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!?.:,;*+/-="
        ) + tokenizer.tokenize("\n")
        logger.debug("Model config: %s", model.config)
        logger.debug("Model generation config: %s", model.generation_config)
        tokenizer_tools.keep_single_char_tokens(model, tokenizer, keep=keep_tok)
        logger.info("New tokenizer vocab size: %d", len(tokenizer))
        logger.info(
            "Model parameters after vocab shrink: %.3fB",
            utils.count_parameters(model) / 1e9,
        )

    def build_extended_vocab_for_conv(
        self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase
    ):
        logger.info("Extending tokenizer vocab for conv Embedding...")
        tokenizer_tools.extend_tokenizer_vocab_for_arc_grid(tokenizer)
        logger.info("Extended tokenizer vocab size for conv E: %d", len(tokenizer))
        tokenizer_tools.extend_model_embeddings_for_arc_grid(model, tokenizer)
        logger.info(
            "Model parameters after extending for conv E: %.3fB",
            utils.count_parameters(model) / 1e9,
        )
        return model, tokenizer

    def build_peft_model(self, model: PreTrainedModel) -> PreTrainedModel:
        if self.lora_config is None:
            raise ValueError("LoRA config is not set. Cannot build PEFT model.")
        logger.info("Applying PEFT LoRA to the model...")
        model = get_peft_model(model, self.lora_config)
        model.print_trainable_parameters()
        logger.info("Model now has %.3fM parameters.", utils.count_parameters(model) / 1e6)
        logger.info("Target modules for LoRA: %s", self.lora_config.target_modules)
        logger.info(
            "LoRA config: R=%d, alpha=%d, dropout=%.2f, use_rslora=%s",
            self.lora_config.r,
            self.lora_config.lora_alpha,
            self.lora_config.lora_dropout,
            self.lora_config.use_rslora,
        )
        return model

    def build(self) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        logger.info("Building model with config: %s", self.model_config)
        logger.info("Model built successfully")

        model_class = self.get_model_class()
        bnb_config = self.build_bnb_config()

        model = model_class.from_pretrained(
            pretrained_model_name_or_path=self.model_name_or_path,
            tie_word_embeddings=not self.untie_lm_head,
            quantization_config=bnb_config,
            device_map=self.device,
        )

        if self.untie_lm_head:
            self.build_again_lm_head(model)

        self._log_model_info(model)

        tokenizer = TokenizerBuilder().from_pretrained(self.model_name_or_path).build()

        if self.shrink_vocab:
            model, tokenizer = self.build_shrinked_vocab(model, tokenizer)
        if self.custom_class == "conv":
            model, tokenizer = self.build_extended_vocab_for_conv(model, tokenizer)
        
        if self.use_lora and self.lora_config is not None:
            model = self.build_peft_model(model)

        return model


# BELOW LIES PREVIOUS IMPLEMENTATION FOR REFERENCE


def shrink_vocab(model, tokenizer):
    # Shrink vocab to only keep useful tokens
    logger.info("Shrinking tokenizer vocabulary to only keep useful tokens...")
    logger.info("Original tokenizer vocab size: %d", len(tokenizer))
    logger.info("Original model parameters: %.3fB", utils.count_parameters(model) / 1e9)
    keep_tok = list(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!?.:,;*+/-="
    ) + tokenizer.tokenize("\n")
    logger.debug("Model config: %s", model.config)
    logger.debug("Model generation config: %s", model.generation_config)
    tokenizer_tools.keep_single_char_tokens(model, tokenizer, keep=keep_tok)
    logger.info("New tokenizer vocab size: %d", len(tokenizer))
    logger.info(
        "Model parameters after vocab shrink: %.3fB",
        utils.count_parameters(model) / 1e9,
    )

    if ENV_VARS["MODEL_TYPE"] == "conv":
        logger.info("Extending tokenizer vocab for conv Embedding...")
        tokenizer_tools.extend_tokenizer_vocab_for_arc_grid(tokenizer)
        logger.info("Extended tokenizer vocab size for conv E: %d", len(tokenizer))
        tokenizer_tools.extend_model_embeddings_for_arc_grid(model, tokenizer)
        logger.info(
            "Model parameters after extending for conv E: %.3fB",
            utils.count_parameters(model) / 1e9,
        )


def setup_peft_lora(model):
    lora_target_modules = ENV_VARS["LORA_TARGET_MODULES"]
    lora_r = ENV_VARS["LORA_R"]
    lora_alpha = ENV_VARS["LORA_ALPHA"]
    lora_dropout = ENV_VARS["LORA_DROPOUT"]
    use_rslora = ENV_VARS["USE_RSLORA"]
    modules_to_save = ENV_VARS["LORA_MODULES_TO_SAVE"]

    # Configure PEFT LoRA
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        use_rslora=use_rslora,
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=modules_to_save if len(modules_to_save) > 0 else None,
    )

    # Apply PEFT LoRA to the model
    logger.info("Applying PEFT LoRA to the model...")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    logger.info("Model now has %.3fM parameters.", utils.count_parameters(model) / 1e6)
    logger.info("Target modules for LoRA: %s", lora_target_modules)
    logger.info(
        "LoRA config: R=%d, alpha=%d, dropout=%.2f, use_rslora=%s",
        lora_r,
        lora_alpha,
        lora_dropout,
        use_rslora,
    )

    return model


def get_model(model_name: str, untie_lm_head: bool = None):
    if untie_lm_head is None:
        untie_lm_head = ENV_VARS["USE_LORA"]

    quantize_model = ENV_VARS["QUANTIZE_MODEL"]
    if quantize_model in [4, 8]:
        logger.info(
            "Loading quantized model with %d-bit quantization...", quantize_model
        )
        bnb_config = None
        if quantize_model == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=ENV_VARS["BNB_4BIT_QUANT_TYPE"],
                bnb_4bit_compute_type=torch.float16,
                llm_int8_enable_fp32_cpu_offload=True,
            )
        else:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True
            )
        if ENV_VARS["PRINT_QUANT_INFO"]:
            print_quantization_info(
                model_name=model_name,
                quantization_config=bnb_config,
                device_map="cpu",
                verbose=True,
            )
    else:
        bnb_config = None

    model_class = AutoModelForCausalLM
    match ENV_VARS["MODEL_TYPE"]:
        case "base":
            logger.info("Using base AutoModelForCausalLM...")
            model_class = AutoModelForCausalLM
        case "2DPE":
            logger.info("Using Custom Mistral Model with 2D PE...")
            model_class = CustomMistralModel2DPE
        case "conv":
            logger.info("Using Custom Mistral Model with Conv Embeddings...")
            model_class = CustomMistralModelConvEmbedding
        case _:
            model_class = AutoModelForCausalLM

    if untie_lm_head:
        model = model_class.from_pretrained(
            pretrained_model_name_or_path=model_name,
            tie_word_embeddings=False,
            quantization_config=bnb_config,
            device_map="auto",
        )
        logger.info("Untying model head with embedding...")
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
        logger.info(
            "Num non-quantized parameters: %.2fM",
            sum(
                p.numel()
                for p in model.parameters()
                if p.dtype in (torch.float32, torch.float16)
            )
            / 1e6,
        )
    else:
        model = model_class.from_pretrained(
            pretrained_model_name_or_path=model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
    logger.info("Model %s loaded.", model_name)
    logger.info("Model has %.3fB parameters.", utils.count_parameters(model) / 1e9)
    logger.info("Model dtype: %s", next(model.parameters()).dtype)
    logger.debug("Model config: %s", model.config)
    logger.debug("Model generation config: %s", model.generation_config)
    return model


def get_dataset(dataset_id: str):
    hf_dataset = load_dataset(dataset_id)
    dataset_dict = DatasetDict(
        {
            "train": hf_dataset["train"],
            "eval": hf_dataset["eval"],
            "test": hf_dataset["test"],
        }
    )
    logger.info("Dataset %s loaded.", dataset_id)
    frac = ENV_VARS["DATASET_FRAC"]
    if frac != 1.0:
        return frac_dataset_dict(dataset_dict, frac)
    return dataset_dict
