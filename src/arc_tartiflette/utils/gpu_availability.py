import os
import platform
import subprocess
import sys

def print_gpu_availability():
    """
    Print detailed information about the system and GPU availability.
    """
    print("="*40, "SYSTEM INFO", "="*40)
    print("Python version:", sys.version)
    print("Platform:", platform.platform())
    print("Processor:", platform.processor())
    print("Machine:", platform.machine())
    print("Hostname:", platform.node())

    print("\n", "="*40, "ENVIRONMENT VARIABLES", "="*40)
    for k, v in sorted(os.environ.items()):
        print(f"{k}={v}")

    print("\n", "="*40, "GPU / NVIDIA INFO", "="*40)
    try:
        result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("nvidia-smi not available:", result.stderr)
    except FileNotFoundError:
        print("nvidia-smi command not found")

    print("\n", "="*40, "CUDA VISIBLE DEVICES", "="*40)
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    print("CUDA_VISIBLE_DEVICES:", cuda_visible)

    print("\n", "="*40, "NVIDIA-SMI (Physical GPU indices)", "="*40)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("nvidia-smi error:", result.stderr)
    except FileNotFoundError:
        print("nvidia-smi not found")

    print("\n", "="*40, "CUDA / TORCH INFO", "="*40)
    try:
        import torch
        print("Torch version:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        print("CUDA device count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
            print("  Memory allocated:", torch.cuda.memory_allocated(i))
            print("  Memory reserved:", torch.cuda.memory_reserved(i))
            print("  Is bf16 supported:", torch.cuda.is_bf16_supported())
    except ImportError:
        print("PyTorch not installed")

    # print("\n", "="*40, "CUDA / TENSORFLOW INFO", "="*40)
    # try:
    #     import tensorflow as tf
    #     print("TensorFlow version:", tf.__version__)
    #     print("Built with CUDA:", tf.test.is_built_with_cuda())
    #     print("GPU devices:", tf.config.list_physical_devices('GPU'))
    #     pass
    # except ImportError:
    #     print("TensorFlow not installed")

def estimate_vram_usage(model, batch_size: int, use_bf16: bool, seq_length: int) -> float:
    """
    Estimate VRAM usage in GB for training a model.
    """
    param_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    total_model_size_bytes = param_size_bytes + buffer_size_bytes

    dtype_multiplier = 2 if use_bf16 else 4

    activation_size_bytes = batch_size * seq_length * model.config.hidden_size * dtype_multiplier

    optimizer_overhead_bytes = total_model_size_bytes * 2

    total_vram_bytes = total_model_size_bytes + activation_size_bytes + optimizer_overhead_bytes

    total_vram_gb = total_vram_bytes / (1024 ** 3)

    return round(total_vram_gb, 2)