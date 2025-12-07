import logging

import torch
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, AutoConfig

logger = logging.getLogger(__name__)


def print_quantization_info(
    model_name: str, quantization_config=None, device_map="cpu", verbose=True
):
    """
    Load a model in dry-run mode (on CPU) and print quantization details:
      • Which layers are quantized
      • Fraction of quantized vs total layers
      • Estimated VRAM use after quantization

    Args:
        model_name (str): Model name or path
        quantization_config (BitsAndBytesConfig): quantization config
        device_map (str): typically "cpu" for dry run
        verbose (bool): whether to print detailed layer info
    """
    # Load config and model structure only
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        quantization_config=quantization_config,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )

    n_total = 0
    n_quant = 0
    quantized_layers = []
    float_layers = []

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # leaf only
            n_total += 1
            if isinstance(module, (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)):
                n_quant += 1
                quantized_layers.append(name)
            elif isinstance(module, torch.nn.Linear):
                float_layers.append(name)

    total_params = sum(p.numel() for p in model.parameters())
    quantized_params = sum(
        p.numel()
        for m in model.modules()
        if isinstance(m, (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt))
        for p in m.parameters()
    )

    non_quantized_params = total_params - quantized_params

    param_bytes = {"float32": 4, "float16": 2, "bfloat16": 2, "int8": 1, "int4": 0.5}

    # Determine quant level
    quant_type = None
    if any(isinstance(m, bnb.nn.Linear4bit) for m in model.modules()):
        quant_type = "4-bit"
        bytes_per_param = param_bytes["int4"]
    elif any(isinstance(m, bnb.nn.Linear8bitLt) for m in model.modules()):
        quant_type = "8-bit"
        bytes_per_param = param_bytes["int8"]
    else:
        quant_type = "float16"
        bytes_per_param = param_bytes["float16"]

    vram_bytes = 0
    for m in model.modules():
        if isinstance(m, bnb.nn.Linear4bit):
            vram_bytes += sum(p.numel() * 0.5 for p in m.parameters())
        elif isinstance(m, bnb.nn.Linear8bitLt):
            vram_bytes += sum(p.numel() * 1 for p in m.parameters())
        else:
            vram_bytes += sum(p.numel() * 2 for p in m.parameters())  # assume fp16
    est_vram_gb = vram_bytes / (1024**3)

    logger.debug("───────────────────────────────────────────────")
    logger.debug("Model: %s", model_name)
    logger.debug("Quantization type: %s", quant_type)
    logger.debug("Total modules: %d", n_total)
    logger.debug("Quantized modules: %d (%.1f%%)", n_quant, 100 * n_quant / n_total)
    logger.debug("Non-quantized modules: %d", n_total - n_quant)
    logger.debug("Total parameters: %.2fB", total_params / 1e9)
    logger.debug("Quantized parameters: %.2fB", quantized_params / 1e9)
    logger.debug("Non-quantized parameters: %.2fB", non_quantized_params / 1e9)
    logger.debug("Estimated VRAM after quantization: %.2f GB", est_vram_gb)
    logger.debug("───────────────────────────────────────────────")

    if verbose:
        logger.debug("Quantized layers:")
        for name in quantized_layers[:20]:
            logger.debug("  - %s", name)
        if len(quantized_layers) > 20:
            logger.debug("  ... (+%d more)", len(quantized_layers) - 20)

        if float_layers:
            logger.debug("Unquantized layers (first 10):")
            for name in float_layers[:10]:
                logger.debug("  - %s", name)
            if len(float_layers) > 10:
                logger.debug("  ... (+%d more)", len(float_layers) - 10)

    return {
        "model_name": model_name,
        "quant_type": quant_type,
        "n_total": n_total,
        "n_quant": n_quant,
        "total_params": total_params,
        "estimated_vram_gb": est_vram_gb,
        "quantized_layers": quantized_layers,
        "unquantized_layers": float_layers,
    }
