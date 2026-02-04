"""
Step-4: 并行描述
使用 VLM 对图像生成语义描述，输出 S4_captions.json。

模式1: 仅描述代表图像（需 S3_sampled_images.json）
模式2: 描述所有图片（跳过 Step-3，从 S2_clustering 取全部 image_id）

VLM 来源（config.vlm）:
- model_source: huggingface（默认）| modelscope
  ModelScope 通义千问2-VL: https://www.modelscope.cn/models/qwen/Qwen2-VL-2B-Instruct/summary
  国内使用 model_source=modelscope 可加速下载；需 pip install modelscope
- model_scale: small(2B) | large(7B)，或 model_name 指定具体 ID
- 本地加载: transformers AutoModelForImageTextToText + AutoProcessor

📅 Last Updated: 2026-01-31
📖 Reference: docs/workflow-structure.md
"""

import json
import pandas as pd
from pathlib import Path
from typing import Callable, Dict, List, Optional

from tqdm import tqdm

ROOT = Path(__file__).parent.parent


def _load_image_list(
    mode: str,
    index_path: Path,
    clustering_path: Path,
    sampled_path: Optional[Path],
) -> List[str]:
    """根据模式返回待描述 image_id 列表"""
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    clustering = pd.read_csv(clustering_path)

    if mode == "representative" and sampled_path and sampled_path.exists():
        with open(sampled_path, "r", encoding="utf-8") as f:
            sampled = json.load(f)
        ids = []
        for cid, img_list in sampled.items():
            ids.extend(img_list)
        return list(dict.fromkeys(ids))

    # 模式2 或 无 S3: 所有图像（不含 noise 可选，这里含 noise）
    return clustering["image_id"].astype(str).tolist()


def _caption_with_placeholder(image_ids: List[str], clustering_path: Path) -> Dict[str, str]:
    """无 VLM 时生成占位描述"""
    clustering = pd.read_csv(clustering_path)
    cid_map = dict(zip(clustering["image_id"].astype(str), clustering["cluster_id"]))
    return {
        iid: f"[Placeholder] Image in cluster {cid_map.get(iid, -1)}."
        for iid in image_ids
    }


def _caption_with_vlm(
    image_ids: List[str],
    index: Dict,
    config: dict,
    caption_prompt: str,
    device: str,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, str]:
    """使用本地 VLM（Qwen2-VL 等）生成描述；失败或未安装时返回空 dict 走占位逻辑。"""
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        from models.vlm_models import (
            get_vlm_model_and_processor,
            caption_batch,
            resolve_vlm_model_name,
            is_vlm_available,
        )
    except ImportError as e:
        print(f"[Step-4] 导入 VLM 模块失败: {e}")
        return {}
    if not is_vlm_available():
        return {}
    vlm_cfg = config.get("vlm", {})
    model_name = resolve_vlm_model_name(config)
    torch_dtype = vlm_cfg.get("torch_dtype", "bfloat16")
    use_flash_attn = vlm_cfg.get("use_flash_attn", True)
    quantization = (vlm_cfg.get("quantization") or "none").strip().lower()
    batch_size = max(1, int(vlm_cfg.get("caption_batch_size", 4)))
    max_image_size = max(0, int(vlm_cfg.get("max_image_size", 512)))
    max_new_tokens = min(512, max(64, int(config.get("postprocessing", {}).get("caption_length", 50)) * 3))
    if max_image_size > 0:
        print(f"[Step-4] 描述前缩小图像：长边 ≤ {max_image_size} px（加速）")
    print(f"[Step-4] 正在加载 VLM 模型（首次可能较慢）: {model_name}")
    try:
        model, processor = get_vlm_model_and_processor(
            model_name,
            device=device,
            torch_dtype=torch_dtype,
            use_flash_attn=use_flash_attn,
            quantization=quantization if quantization not in ("", "none") else None,
            config=config,
        )
    except Exception as e:
        print(f"[Step-4] VLM 加载失败: {e}")
        return {}
    base_dir = Path(config.get("data", {}).get("input_directory", "."))
    items: List[tuple] = []
    for iid in image_ids:
        if iid not in index:
            continue
        path = Path(index[iid]["path"])
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        if path.exists():
            items.append((iid, path))
    total = len(image_ids)
    captions: Dict[str, str] = {iid: "" for iid in image_ids}
    if not items:
        return captions
    print(f"[Step-4] 模型已加载，开始批量描述（共 {len(items)} 张，batch_size={batch_size}）")
    pbar = tqdm(
        total=len(items),
        desc="Step-4 批量描述",
        unit="张",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} ({percentage:3.0f}%) [已用 {elapsed}, 剩余 {remaining}]",
    )
    done = 0
    total_batches = (len(items) + batch_size - 1) // batch_size
    for batch_idx, start in enumerate(range(0, len(items), batch_size)):
        chunk = items[start : start + batch_size]
        if batch_idx == 0 or (batch_idx + 1) % max(1, total_batches // 5) == 0 or batch_idx == total_batches - 1:
            print(f"[Step-4] 批次 {batch_idx + 1}/{total_batches}，本批 {len(chunk)} 张")
        batch_result = caption_batch(
            model, processor, chunk, caption_prompt,
            max_new_tokens=max_new_tokens, batch_size=len(chunk),
            max_image_size=max_image_size,
        )
        for iid, cap in batch_result.items():
            captions[iid] = cap or ""
        done += len(chunk)
        if progress_callback:
            progress_callback(min(done, total), total, chunk[-1][0] if chunk else "")
        pbar.update(len(chunk))
        log_interval = max(1, total // 10)
        if done % log_interval < len(chunk) or done == len(items):
            print(f"[Step-4] 进度 {done}/{len(items)} ({100 * done // len(items)}%)")
    pbar.close()
    return captions if any(captions.values()) else {}


def run_step4(
    config: dict,
    index_path: Path,
    clustering_path: Path,
    output_dir: Path,
    mode: str = "representative",
    sampled_path: Optional[Path] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Path:
    """
    运行 Step-4: 并行描述

    Args:
        config: 配置字典
        index_path: S0_image_index.json
        clustering_path: S2_clustering.csv
        output_dir: 输出目录
        mode: "representative" 用 S3 代表图，"all" 用全部图
        sampled_path: S3_sampled_images.json（模式1 必选）

    Returns:
        S4_captions.json 路径
    """
    print("=" * 60)
    print("Step-4: 并行描述")
    print("=" * 60)

    post = config.get("postprocessing", {})
    caption_tpl = post.get("caption_prompt", "Describe this image in about {caption_length} words.")
    caption_len = int(post.get("caption_length", 50))
    caption_prompt = caption_tpl.format(caption_length=caption_len)

    image_ids = _load_image_list(mode, index_path, clustering_path, sampled_path)
    total_images = len(image_ids)
    device = (config.get("vlm", {}).get("device") or "cuda").strip().lower() or "cuda"
    try:
        import torch
        if device == "cpu" and torch.cuda.is_available():
            device = "cuda"
            print("[Step-4] 检测到 GPU，使用 cuda（config 中 device 已覆盖）")
    except Exception:
        pass
    try:
        from models.vlm_models import resolve_vlm_model_name
        model_name = resolve_vlm_model_name(config)
    except ImportError:
        model_name = config.get("vlm", {}).get("model_name", "Qwen/Qwen2-VL-2B-Instruct")
    batch_size = max(1, int(config.get("vlm", {}).get("caption_batch_size", 4)))
    print(f"[Step-4] 模式: {mode}，待描述: {total_images} 张")
    print(f"[Step-4] 模型: {model_name}，设备: {device}，批量: {batch_size}，描述长度约 {caption_len} 词")

    # 立即通知 UI 总数，避免长时间显示 0/?
    if progress_callback and total_images > 0:
        progress_callback(0, total_images, "")

    # 先检测模型可用并且已下载
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        from models.vlm_models import check_vlm_ready
    except ImportError:
        check_vlm_ready = None
    vlm_ready = False
    if check_vlm_ready is not None:
        vlm_ready, vlm_msg = check_vlm_ready(config)
        if vlm_ready:
            print(f"[Step-4] {vlm_msg}")
        else:
            print(f"[Step-4] VLM 未就绪：{vlm_msg}")

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    if vlm_ready:
        captions = _caption_with_vlm(
            image_ids, index, config, caption_prompt, device,
            progress_callback=progress_callback,
        )
        if not captions:
            captions = _caption_with_placeholder(image_ids, clustering_path)
            print("[Step-4] VLM 推理未返回结果，使用占位描述。")
    else:
        captions = _caption_with_placeholder(image_ids, clustering_path)
        print("[Step-4] 使用占位描述（VLM 未就绪）。")

    out_path = output_dir / "S4_captions.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(captions, f, indent=2, ensure_ascii=False)

    n_ok = sum(1 for v in captions.values() if (v or "").strip())
    print(f"[Step-4] 完成: 共 {len(captions)} 条，有效描述 {n_ok} 条 -> {out_path.name}")

    # 将完整描述 .txt 输出到 output 目录（config.output.write_caption_txt 为 true 时）
    write_txt = config.get("output", {}).get("write_caption_txt", True)
    if write_txt and captions:
        txt_dir = output_dir / "caption_txt"
        txt_dir.mkdir(parents=True, exist_ok=True)
        txt_count = 0
        for iid, text in captions.items():
            if not (text or "").strip():
                continue
            txt_path = txt_dir / f"{iid}.txt"
            try:
                txt_path.write_text(text.strip(), encoding="utf-8")
                txt_count += 1
            except Exception as e:
                print(f"[Step-4] 写入 {txt_path} 失败: {e}")
        if txt_count:
            print(f"[Step-4] 已输出 {txt_count} 个描述 .txt -> {txt_dir.relative_to(output_dir)}/")

    print("=" * 60)
    return out_path
