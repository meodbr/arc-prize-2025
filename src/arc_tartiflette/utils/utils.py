import huggingface_hub as hf
import os

import arc_tartiflette.utils.gpu_availability as gpu_availability


def default_output_model_name(base_model: str, dataset: str):
    base_model_stem = base_model.split("/")[-1].strip("/")
    dataset_stem = dataset.split("/")[-1].strip("/")
    return f"{base_model_stem}_{dataset_stem}"


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_time_per_epoch(
    model, batch_size: int, use_bf16: bool, seq_length: int, dataset_size: int
) -> float:
    """
    Estimate time per epoch in seconds for training a model.
    """
    vram_usage_gb = gpu_availability.estimate_vram_usage(
        model, batch_size, use_bf16, seq_length
    )
    base_time_per_batch = 0.5  # Base time in seconds per batch for 8GB VRAM
    time_multiplier = vram_usage_gb / 8.0
    time_per_batch = base_time_per_batch * time_multiplier
    num_batches = dataset_size / batch_size
    return time_per_batch * num_batches


def hf_login():
    if os.environ.get("HUGGING_FACE_TOKEN", None) is not None:
        hf.login(token=os.environ.get("HUGGING_FACE_TOKEN"))


hf_login()
