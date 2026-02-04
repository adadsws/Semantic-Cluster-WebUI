#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查 VLM 环境是否可用，可选测试加载模型。用法：python scripts/check_vlm.py [--load]"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    import argparse
    p = argparse.ArgumentParser(description="检查 VLM（Qwen2-VL）环境")
    p.add_argument("--load", action="store_true", help="尝试加载模型（会下载若未缓存）")
    p.add_argument("--model", default="Qwen/Qwen2-VL-7B-Instruct", help="模型名")
    args = p.parse_args()

    print("=" * 50)
    print("VLM 环境检查")
    print("=" * 50)

    # transformers 版本
    try:
        import transformers
        print(f"transformers 版本: {transformers.__version__}")
    except ImportError:
        print("transformers: 未安装。请执行 pip install -U transformers")
        return 1

    # 可用性
    from models.vlm_models import is_vlm_available, check_vlm_ready
    if not is_vlm_available():
        print("VLM 可用性: 否（需 transformers 4.37+ 支持 Qwen2-VL）")
        print("建议: pip install -U transformers")
        return 1
    print("VLM 可用性: 是")

    if args.load:
        print(f"\n尝试加载模型: {args.model} ...")
        from utils import ConfigLoader
        loader = ConfigLoader()
        config = loader.to_dict()
        config.setdefault("vlm", {})["model_name"] = args.model
        ready, msg = check_vlm_ready(config)
        if ready:
            print(f"加载结果: {msg}")
        else:
            print(f"加载失败: {msg}")
            return 1

    print("=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(main())
