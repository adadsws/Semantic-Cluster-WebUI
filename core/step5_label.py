"""
Step-5: 语义蒸馏
根据 S4 描述为每个簇生成语义标签，输出 S5_cluster_labels.csv

输入: S4_captions.json, S3_sampled_images.json（簇→代表图ID）
输出: S5_cluster_labels.csv (cluster_id, label)
无真实 LLM 时：用首条描述截断或占位 "Cluster_N"，并做文件名安全化

📅 Last Updated: 2026-01-31
📖 Reference: docs/workflow-structure.md
"""

import json
import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).parent.parent


def _sanitize_label(label: str, max_length: int = 512) -> str:
    """替换文件名非法字符，保留中文、字母、数字、空格、连字符、下划线；超过 max_length 字符截断（默认 512，可在 config 的 postprocessing.label_max_length 调整）。"""
    s = re.sub(r'[<>:"/\\|?*\n\r\t]', "_", label)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:max_length] if len(s) > max_length else s


def _sentence_to_keywords(raw: str) -> str:
    """若模型返回完整句（如 The image captures...），尝试提取为逗号分隔的关键词。"""
    s = (raw or "").strip()
    if not s or len(s) > 300:
        return s
    # 已像关键词：含逗号且无典型句子开头
    if "," in s and not re.match(r"^(The|This|It)\s+(image\s+)?(captures|shows|depicts|features)", s, re.I):
        return s
    # 去掉句首 "The image captures/shows ..." 等
    s = re.sub(r"^(The|This)\s+image\s+(captures|shows|depicts|features|presents)\s+", "", s, flags=re.I).strip()
    s = re.sub(r"^(The|This)\s+", "", s, count=1, flags=re.I).strip()
    s = s.rstrip(".")
    # 去掉冠词、介词等，按空格/逗号拆成词，保留有意义的
    stop = {"a", "an", "the", "in", "on", "at", "is", "are", "of", "to", "and", "or", "for"}
    parts = re.split(r"[\s,]+", s)
    words = [w for w in parts if w and len(w) > 1 and w.lower() not in stop]
    if not words:
        return (raw or "").strip()
    return ", ".join(words[:15])  # 最多 15 个词作为关键词


def _is_placeholder_caption(text: str) -> bool:
    """判断是否为 Step-4 占位描述（无 VLM 时生成），应使用 Cluster_N 而非原文。"""
    s = text.strip()
    if s.startswith("["):
        idx = s.find("]")
        if idx >= 0:
            s = s[idx + 1 :].strip()
    # "Image in cluster 0." / "Image in cluster -1." 等
    return bool(re.match(r"^Image in cluster (-?\d+)\.?$", s, re.I))


def _distill_placeholder(
    cluster_id: int,
    captions: List[str],
    label_length_min: int,
    label_length_max: int,
    label_max_length: int = 512,
) -> str:
    """无 LLM 时：从首条描述截取前 N 词；若为占位描述则直接返回 Cluster_N。"""
    if not captions:
        return f"Cluster_{cluster_id:02d}"
    first = captions[0].strip()
    if _is_placeholder_caption(first):
        return f"Cluster_{cluster_id:02d}"
    # 去掉 [Placeholder] 等前缀
    if first.startswith("["):
        idx = first.find("]")
        if idx >= 0:
            first = first[idx + 1 :].strip()
    if _is_placeholder_caption(first):
        return f"Cluster_{cluster_id:02d}"
    words = first.split()
    n = min(label_length_max, max(label_length_min, len(words)))
    label = " ".join(words[:n]) if words else f"Cluster_{cluster_id:02d}"
    return _sanitize_label(label, max_length=label_max_length)


def _merge_keywords(
    keyword_lists: List[str],
    label_max_len: int = 512,
    max_keywords: int = 8,
) -> str:
    """合并多条关键词串：按逗号/空格拆词，去重（保序），最多保留 max_keywords 个，再拼成一条。"""
    seen: set = set()
    out: List[str] = []
    for s in keyword_lists:
        if not (s or "").strip():
            continue
        s = _sentence_to_keywords(s)
        for part in re.split(r"[\s,]+", s):
            if len(out) >= max_keywords:
                break
            w = part.strip()
            if not w or len(w) < 2:
                continue
            key = w.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(w)
        if len(out) >= max_keywords:
            break
    return ", ".join(out) if out else ""


def _distill_with_llm(
    cluster_id: int,
    captions: List[str],
    config: dict,
    output_dir: Optional[Path] = None,
) -> str:
    """先对每条描述提取关键词，再合并同簇关键词作为簇标签；失败则走占位逻辑。"""
    post = config.get("postprocessing", {})
    label_min = int(post.get("label_length_min", 5))
    label_max = int(post.get("label_length_max", 10))
    label_max_len = int(post.get("label_max_length", 512))
    label_keyword_max = max(1, int(post.get("label_keyword_max", 8)))
    if not captions:
        return _distill_placeholder(cluster_id, captions, label_min, label_max, label_max_len)
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        from models.vlm_models import (
            get_vlm_model_and_processor,
            generate_text,
            is_vlm_available,
        )
    except ImportError as e:
        print(f"[Step-5] 导入 VLM 模块失败: {e}")
        return _distill_placeholder(cluster_id, captions, label_min, label_max, label_max_len)
    if not is_vlm_available():
        return _distill_placeholder(cluster_id, captions, label_min, label_max, label_max_len)
    # 单条描述关键词提取 prompt，占位符 {description}；无则用默认
    kw_tpl = (post.get("keyword_extract_prompt") or "").strip()
    if not kw_tpl or "{description}" not in kw_tpl:
        kw_tpl = (
            "Extract 3-8 keywords from the following description. "
            "Reply with words separated by commas only. No sentences.\n\n{description}"
        )
    # 过滤占位描述与空
    valid_captions = [
        c.strip() for c in captions
        if (c or "").strip() and not _is_placeholder_caption((c or "").strip())
    ]
    if not valid_captions:
        return _distill_placeholder(cluster_id, captions, label_min, label_max, label_max_len)
    vlm_cfg = config.get("vlm", {})
    try:
        from models.vlm_models import resolve_vlm_model_name
        model_name = resolve_vlm_model_name(config)
    except ImportError:
        model_name = vlm_cfg.get("model_name", "Qwen/Qwen2-VL-2B-Instruct")
    device = (vlm_cfg.get("device") or "cuda").strip().lower() or "cuda"
    try:
        import torch
        if device == "cpu" and torch.cuda.is_available():
            device = "cuda"
    except Exception:
        pass
    quantization = (vlm_cfg.get("quantization") or "none").strip().lower()
    try:
        model, processor = get_vlm_model_and_processor(
            model_name,
            device=device,
            torch_dtype=vlm_cfg.get("torch_dtype", "bfloat16"),
            use_flash_attn=vlm_cfg.get("use_flash_attn", True),
            quantization=quantization if quantization not in ("", "none") else None,
            config=config,
        )
        # 逐条描述提取关键词
        keyword_strs: List[str] = []
        for cap in valid_captions:
            prompt = kw_tpl.replace("{description}", cap)
            raw = generate_text(model, processor, prompt, max_new_tokens=32)
            raw = (raw or "").strip()
            keyword_strs.append(raw or "")
        # 每句得到的关键词保存到 txt（config.output.save_keyword_txt 为 true 时）
        if output_dir and config.get("output", {}).get("save_keyword_txt", True) and keyword_strs:
            kw_dir = output_dir / "step5_keywords"
            kw_dir.mkdir(parents=True, exist_ok=True)
            fname = "noise_keywords.txt" if cluster_id == -1 else f"cluster_{cluster_id:02d}_keywords.txt"
            kw_path = kw_dir / fname
            lines = [f"{kw}\n" for kw in keyword_strs]
            try:
                kw_path.write_text("".join(lines), encoding="utf-8")
            except Exception as e:
                print(f"[Step-5] 写入 {kw_path} 失败: {e}")
        # 合并同簇关键词并去重，最多保留 label_keyword_max 个
        merged = _merge_keywords(keyword_strs, label_max_len, max_keywords=label_keyword_max)
        if merged:
            return _sanitize_label(merged, max_length=label_max_len)
    except Exception as e:
        print(f"[Step-5] 簇 {cluster_id} LLM 蒸馏失败: {e}")
    return _distill_placeholder(cluster_id, captions, label_min, label_max, label_max_len)


def run_step5(
    config: dict,
    captions_path: Path,
    sampled_path: Path,
    output_dir: Path,
    progress_callback=None,
) -> Path:
    """
    运行 Step-5: 语义蒸馏

    Args:
        config: 配置字典
        captions_path: S4_captions.json
        sampled_path: S3_sampled_images.json
        output_dir: 输出目录
        progress_callback: 可选，回调 (current, total) 用于 UI 进度

    Returns:
        S5_cluster_labels.csv 路径
    """
    print("=" * 60)
    print("Step-5: 语义蒸馏")
    print("=" * 60)

    print(f"[Step-5] 加载 S4 描述与 S3 采样…")
    with open(captions_path, "r", encoding="utf-8") as f:
        captions_by_image = json.load(f)
    with open(sampled_path, "r", encoding="utf-8") as f:
        sampled = json.load(f)
    post = config.get("postprocessing", {})
    n_clusters = len(sampled)
    label_min = int(post.get("label_length_min", 5))
    label_max = int(post.get("label_length_max", 10))
    label_max_len = int(post.get("label_max_length", 512))
    print(f"[Step-5] 共 {n_clusters} 个簇待蒸馏标签（label_length: {label_min}-{label_max} 词，最大 {label_max_len} 字符）")
    rows = []
    for idx, (cid_str, image_ids) in enumerate(sampled.items()):
        cid = int(cid_str)
        captions = [
            captions_by_image.get(iid, "")
            for iid in image_ids
            if captions_by_image.get(iid, "").strip()
        ]
        label = _distill_with_llm(cid, captions, config, output_dir=output_dir)
        if not label:
            label = _distill_placeholder(cid, captions, label_min, label_max, label_max_len)
        rows.append({"cluster_id": cid, "label": label})
        if progress_callback:
            try:
                progress_callback(idx + 1, n_clusters)
            except Exception:
                pass
        if (idx + 1) % max(1, n_clusters // 5) == 0 or idx == n_clusters - 1:
            print(f"[Step-5] 已蒸馏 {idx + 1}/{n_clusters} 簇")

    df = pd.DataFrame(rows)
    out_path = output_dir / "S5_cluster_labels.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[Step-5] Saved {len(rows)} cluster labels -> {out_path.name}")
    print("=" * 60)
    return out_path
