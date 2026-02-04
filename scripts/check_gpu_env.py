"""
GPU/CUDA 环境检查
用于排查 VLM/嵌入 未使用 GPU 的问题。

用法: 在项目根目录执行
  python scripts/check_gpu_env.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    print("=" * 60)
    print("  GPU / CUDA 环境检查")
    print("=" * 60)
    print(f"Python: {sys.version.split()[0]}")
    print()

    # PyTorch
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"torch.cuda.is_available(): {cuda_available}")
        if cuda_available:
            print(f"CUDA 版本 (PyTorch 编译): {getattr(torch.version, 'cuda', 'N/A')}")
            print(f"GPU 数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print()
            print("  [未检测到 CUDA] 可能原因:")
            print("    1. 安装的是 CPU 版 PyTorch")
            print("    2. 显卡驱动未装或版本过旧")
            print("    3. CUDA 与驱动不匹配")
            print()
            print("  建议:")
            print("    - 安装 GPU 版 PyTorch (按 CUDA 版本选):")
            print("      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
            print("      或 cu121: https://download.pytorch.org/whl/cu121")
            print("    - 检查驱动: nvidia-smi")
    except ImportError as e:
        print(f"PyTorch: 未安装 ({e})")
        print("  安装: pip install torch torchvision")
        print("  GPU 版: pip install torch --index-url https://download.pytorch.org/whl/cu118")
    print()

    # nvidia-smi
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            print("nvidia-smi (本机显卡):")
            for line in out.stdout.strip().split("\n"):
                print(f"  {line}")
        else:
            print("nvidia-smi: 未找到或未安装 NVIDIA 驱动")
    except FileNotFoundError:
        print("nvidia-smi: 未找到（未安装 NVIDIA 驱动或不在 PATH）")
    except Exception as e:
        print(f"nvidia-smi: {e}")
    print()

    # 本项目会用到的 device
    try:
        import torch
        if torch.cuda.is_available():
            dev = "cuda"
            name = torch.cuda.get_device_name(0)
            print(f"本项目将使用: device={dev} ({name})")
        else:
            dev = "cpu"
            print(f"本项目将使用: device={dev}（请按上方建议安装 GPU 版 PyTorch）")
    except Exception:
        print("本项目将使用: device=cpu（需先安装 PyTorch）")
    print("=" * 60)


if __name__ == "__main__":
    main()
