import logging
from typing import Any, Tuple

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers.modeling_utils import PreTrainedModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    BitsAndBytesConfig,
)

from arc_tartiflette.config.settings import ENV_VARS
from arc_tartiflette.model_tools.conv_embeddings import CustomMistralModelConvEmbedding
from arc_tartiflette.model_tools.custom_pe import CustomMistralModel2DPE
from arc_tartiflette.utils import utils
from arc_tartiflette.model_tools.quantization import print_quantization_info
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
            print_quantization_info(
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
        matchings = ["mistral", "nemo"]
        if (
            not any(m in self.model_name_or_path.lower() for m in matchings)
            and self.custom_class != "base"
        ):
            logger.warning(
                "Model name or path does not seem to be a Mistral model, but custom class %s is requested.",
                self.custom_class,
            )

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
            sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6,
        )
        logger.info("Model dtype: %s", next(model.parameters()).dtype)
        logger.debug("Model config: %s", model.config)
        logger.debug("Model generation config: %s", model.generation_config)

    def build_shrinked_vocab(
        self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase
    ):
        if len(tokenizer) < 1000:
            logger.warning(
                "Tokenizer vocab size is already small (%d). Shrinking may not be what you want to do.",
                len(tokenizer),
            )
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
        return model, tokenizer

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
        logger.debug("LoRA config: %s", self.lora_config)
        model = get_peft_model(model, self.lora_config)
        model.print_trainable_parameters()
        logger.info(
            "Model now has %.3fM parameters.", utils.count_parameters(model) / 1e6
        )
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
        logger.info("Building model from %s...", self.model_name_or_path)
        logger.debug("Builder config: %s", self.__dict__)

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

        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

        if self.shrink_vocab:
            model, tokenizer = self.build_shrinked_vocab(model, tokenizer)
        if self.custom_class == "conv":
            model, tokenizer = self.build_extended_vocab_for_conv(model, tokenizer)

        if self.use_lora and self.lora_config is not None:
            model = self.build_peft_model(model)

        return model, tokenizer
