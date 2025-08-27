import os
import platform
import subprocess
import sys

def print_gpu_availability():
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
    except ImportError:
        print("PyTorch not installed")

    print("\n", "="*40, "CUDA / TENSORFLOW INFO", "="*40)
    try:
        import tensorflow as tf
        print("TensorFlow version:", tf.__version__)
        print("Built with CUDA:", tf.test.is_built_with_cuda())
        print("GPU devices:", tf.config.list_physical_devices('GPU'))
    except ImportError:
        print("TensorFlow not installed")