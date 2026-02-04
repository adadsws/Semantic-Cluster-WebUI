"""
单图测试：并行描述（Step-4）与语义蒸馏（Step-5）
用一张图跑完 Step-0～Step-5，验证 VLM 描述与簇标签蒸馏。

与正常流程一致：同一套 Step-0～Step-5、同一 config（config.yaml），仅覆盖单图必需项
（输入目录、设备、min_samples=1、top_k=1 等）；model_source、max_image_size 等用 config 默认。

用法: 在项目根目录执行
  python scripts/test_one_image_caption.py
  python scripts/test_one_image_caption.py path/to/one.jpg   # 指定图片

📅 2026-01-31
"""

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 尽早检测 GPU，供后续配置使用
try:
    import torch
    USE_CUDA = torch.cuda.is_available()
    if USE_CUDA:
        DEVICE_STR = "cuda"
        _gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() else "GPU"
    else:
        DEVICE_STR = "cpu"
        _gpu_name = None
except Exception:
    USE_CUDA = False
    DEVICE_STR = "cpu"
    _gpu_name = None

from utils import ConfigLoader
from core.step0_indexing import run_step0
from core.step1_embedding import run_step1
from core.step2_clustering import run_step2
from core.step3_sampling import run_step3
from core.step4_caption import run_step4
from core.step5_label import run_step5


def main():
    # 单图输入目录
    one_image_dir = ROOT / "data" / "one_image_test" / "input"
    one_image_dir.mkdir(parents=True, exist_ok=True)

    # 若命令行指定了图片，复制过去；否则从 test_pics 取第一张
    if len(sys.argv) >= 2:
        src = Path(sys.argv[1])
        if not src.is_absolute():
            src = ROOT / src
        if not src.exists():
            print(f"[ERROR] 文件不存在: {src}")
            sys.exit(1)
        # 清空 input 后只放这一张
        for f in one_image_dir.iterdir():
            f.unlink()
        dest = one_image_dir / src.name
        shutil.copy2(src, dest)
        print(f"[*] 使用指定图片: {src.name} -> {dest}")
    else:
        test_pics = ROOT / "test_pics"
        if not test_pics.exists():
            print(f"[ERROR] test_pics 不存在: {test_pics}")
            sys.exit(1)
        exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff")
        images = [f for f in test_pics.iterdir() if f.suffix.lower() in exts and f.is_file()]
        if not images:
            print(f"[ERROR] test_pics 下无图片")
            sys.exit(1)
        # 清空后只放一张
        for f in one_image_dir.iterdir():
            f.unlink()
        src = images[0]
        dest = one_image_dir / src.name
        shutil.copy2(src, dest)
        print(f"[*] 使用单图: {src.name} -> {one_image_dir / src.name}")

    # 输出目录
    from datetime import datetime
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ROOT / "data" / "one_image_test" / f"out_{session_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] 输出目录: {output_dir}")

    # 强制使用 GPU（若可用）
    if USE_CUDA:
        print(f"[*] 使用设备: {DEVICE_STR} ({_gpu_name})")
    else:
        print("[*] 使用设备: cpu（未检测到 CUDA，请确认已安装 PyTorch GPU 版: pip install torch --index-url https://download.pytorch.org/whl/cu118）")
    print()

    # 配置：与正常流程同源 config.yaml，仅覆盖单图测试必需项；其余（model_source、max_image_size、model_scale 等）用 config 默认
    loader = ConfigLoader()
    loader.set("data.input_directory", str(one_image_dir.resolve()))
    loader.set("embedding.device", DEVICE_STR)
    loader.set("embedding.backbone", "dinov2_vitl14")  # 与 UI 一致：provider=dinov2 时用 vitl14
    loader.set("vlm.device", DEVICE_STR)
    loader.set("vlm.model_scale", "small")
    loader.set("vlm.caption_batch_size", 4)
    loader.set("clustering.backend", "sklearn")
    loader.set("clustering.min_samples", 1)
    loader.set("clustering.epsilon", 0.5)
    loader.set("postprocessing.caption_mode", "representative")
    loader.set("postprocessing.top_k_sampling", 1)
    config = loader.to_dict()

    try:
        # Step-0 索引
        print("[Step-0] 图像索引（1 张）…")
        index_path = run_step0(config, output_dir)
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        assert len(index) == 1, f"期望 1 张图，实际 {len(index)}"
        print(f"  -> {list(index.keys())[0]}\n")

        # Step-1 嵌入
        print("[Step-1] 特征嵌入…")
        run_step1(config, index_path, output_dir)
        print()

        # Step-2 聚类（1 点 → 1 簇）
        print("[Step-2] 聚类（1 簇）…")
        run_step2(
            config,
            output_dir / "S1_embeddings.npy",
            output_dir / "S1_image_ids.json",
            output_dir,
        )
        print()

        # Step-3 采样（1 簇 1 代表）
        print("[Step-3] 多点采样…")
        sampled_path = run_step3(
            config,
            output_dir / "S1_embeddings.npy",
            output_dir / "S1_image_ids.json",
            output_dir / "S2_clustering.csv",
            output_dir,
        )
        print()

        # Step-4 并行描述（VLM 描述这一张图）
        print("[Step-4] 并行描述（VLM 单图描述）…")
        s4_path = run_step4(
            config,
            index_path,
            output_dir / "S2_clustering.csv",
            output_dir,
            mode="representative",
            sampled_path=sampled_path,
        )
        with open(s4_path, "r", encoding="utf-8") as f:
            captions = json.load(f)
        print("\n[Step-4] 描述结果 S4_captions.json:")
        for iid, cap in captions.items():
            preview = (cap or "").strip()[:200]
            print(f"  {iid}: {preview}..." if len((cap or "")) > 200 else f"  {iid}: {preview}")
        print()

        # Step-5 语义蒸馏（1 簇 → 1 标签）
        print("[Step-5] 语义蒸馏…")
        s5_path = run_step5(config, s4_path, sampled_path, output_dir)
        labels_df = pd.read_csv(s5_path)
        print("\n[Step-5] 簇标签 S5_cluster_labels.csv:")
        print(labels_df.to_string(index=False))
        print()

        print("=" * 60)
        print("  单图测试完成：并行描述 + 语义蒸馏 已跑通")
        print("=" * 60)
        print(f"  输出目录: {output_dir}")
        print(f"  S4: {s4_path.name}")
        print(f"  S5: {s5_path.name}")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
