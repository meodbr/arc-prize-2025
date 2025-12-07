import os
import platform
import subprocess
import sys
import logging

logger = logging.getLogger(__name__)


def print_gpu_availability():
    """
    Print detailed information about the system and GPU availability.
    """
    logger.debug("%s SYSTEM INFO %s", "=" * 40, "=" * 40)
    logger.debug("Python version: %s", sys.version)
    logger.debug("Platform: %s", platform.platform())
    logger.debug("Processor: %s", platform.processor())
    logger.debug("Machine: %s", platform.machine())
    logger.debug("Hostname: %s", platform.node())
    logger.info("System information logged.")

    logger.debug("%s ENVIRONMENT VARIABLES %s", "=" * 40, "=" * 40)
    for k, v in sorted(os.environ.items()):
        logger.debug("%s=%s", k, v)
    logger.info("Environment variables logged.")

    logger.debug("%s GPU / NVIDIA INFO %s", "=" * 40, "=" * 40)
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        if result.returncode == 0:
            logger.debug(result.stdout)
            logger.info("nvidia-smi output logged.")
        else:
            logger.debug("nvidia-smi command returned error: %s", result.stderr)
    except FileNotFoundError:
        logger.info("nvidia-smi command not found.")

    logger.debug("%s CUDA VISIBLE DEVICES %s", "=" * 40, "=" * 40)
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    logger.info("CUDA_VISIBLE_DEVICES: %s", cuda_visible)

    logger.debug("%s NVIDIA-SMI INFO %s", "=" * 40, "=" * 40)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        if result.returncode == 0:
            logger.debug(result.stdout)
            logger.info("nvidia-smi GPU info logged.")
        else:
            logger.debug("nvidia-smi command returned error: %s", result.stderr)
    except FileNotFoundError:
        logger.info("nvidia-smi command not found.")

    logger.info("CUDA / TORCH INFO:")
    logger.debug("%s CUDA / TORCH INFO %s", "=" * 40, "=" * 40)
    try:
        import torch

        logger.info("Torch version: %s", torch.__version__)
        logger.info("CUDA available: %s", torch.cuda.is_available())
        logger.info("CUDA device count: %d", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            logger.info("Device %d: %s", i, torch.cuda.get_device_name(i))
            logger.info("  Memory allocated: %d", torch.cuda.memory_allocated(i))
            logger.info("  Memory reserved: %d", torch.cuda.memory_reserved(i))
            logger.info("  Is bf16 supported: %s", torch.cuda.is_bf16_supported())
    except ImportError:
        logger.warning("PyTorch not installed.")


def estimate_vram_usage(
    model, batch_size: int, use_bf16: bool, seq_length: int
) -> float:
    """
    Estimate VRAM usage in GB for training a model.
    """
    param_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    total_model_size_bytes = param_size_bytes + buffer_size_bytes

    dtype_multiplier = 2 if use_bf16 else 4

    activation_size_bytes = (
        batch_size * seq_length * model.config.hidden_size * dtype_multiplier
    )

    optimizer_overhead_bytes = total_model_size_bytes * 2

    total_vram_bytes = (
        total_model_size_bytes + activation_size_bytes + optimizer_overhead_bytes
    )

    total_vram_gb = total_vram_bytes / (1024**3)

    return round(total_vram_gb, 2)
