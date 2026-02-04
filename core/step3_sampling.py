"""
Step-3: 多点采样
按策略从每簇选取代表图像，输出 S3_sampled_images.json

📅 Last Updated: 2026-01-31
📖 Reference: docs/workflow-structure.md
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).parent.parent


def _sample_nearest(
    features: np.ndarray,
    image_ids: List[str],
    cluster_ids: np.ndarray,
    top_k: int,
    seed: int,
) -> Dict[int, List[str]]:
    """每簇取距离簇中心最近的 top_k 张"""
    rng = np.random.default_rng(seed)
    out = {}
    for cid in np.unique(cluster_ids):
        if cid == -1:
            continue
        mask = cluster_ids == cid
        idx = np.where(mask)[0]
        cen = features[mask].mean(axis=0)
        dist = np.linalg.norm(features[mask] - cen, axis=1)
        order = np.argsort(dist)
        k = min(top_k, len(order))
        chosen = order[:k]
        if len(order) > k and seed >= 0:
            rng.shuffle(chosen)
        out[int(cid)] = [image_ids[idx[i]] for i in chosen[:k]]
    return out


def _sample_farthest(
    features: np.ndarray,
    image_ids: List[str],
    cluster_ids: np.ndarray,
    top_k: int,
) -> Dict[int, List[str]]:
    """每簇取距离簇中心最远的 top_k 张（多样性）"""
    out = {}
    for cid in np.unique(cluster_ids):
        if cid == -1:
            continue
        mask = cluster_ids == cid
        idx = np.where(mask)[0]
        cen = features[mask].mean(axis=0)
        dist = np.linalg.norm(features[mask] - cen, axis=1)
        order = np.argsort(dist)[::-1]
        k = min(top_k, len(order))
        out[int(cid)] = [image_ids[idx[i]] for i in order[:k]]
    return out


def _sample_random(
    image_ids: List[str],
    cluster_ids: np.ndarray,
    top_k: int,
    seed: int,
) -> Dict[int, List[str]]:
    """每簇随机取 top_k 张"""
    rng = np.random.default_rng(seed)
    out = {}
    for cid in np.unique(cluster_ids):
        if cid == -1:
            continue
        idx = np.where(cluster_ids == cid)[0]
        ids_in_cluster = [image_ids[i] for i in idx]
        k = min(top_k, len(ids_in_cluster))
        chosen = rng.choice(len(ids_in_cluster), size=k, replace=False)
        out[int(cid)] = [ids_in_cluster[i] for i in chosen]
    return out


def run_step3(
    config: dict,
    features_path: Path,
    image_ids_path: Path,
    clustering_path: Path,
    output_dir: Path,
) -> Path:
    """
    运行 Step-3: 多点采样

    Args:
        config: 配置字典
        features_path: S1_embeddings.npy
        image_ids_path: S1_image_ids.json
        clustering_path: S2_clustering.csv
        output_dir: 输出目录

    Returns:
        S3_sampled_images.json 路径
    """
    print("=" * 60)
    print("Step-3: 多点采样")
    print("=" * 60)

    print(f"[Step-3] 加载嵌入与聚类文件…")
    features = np.load(features_path)
    with open(image_ids_path, "r", encoding="utf-8") as f:
        image_ids = json.load(f)
    clustering = pd.read_csv(clustering_path)
    cluster_ids = clustering["cluster_id"].values
    n_clusters = len(set(c for c in cluster_ids if c >= 0))
    print(f"[Step-3] 特征 shape={features.shape}，共 {n_clusters} 个簇")

    strategy = config.get("postprocessing", {}).get("sampling_strategy", "nearest")
    top_k = int(config.get("postprocessing", {}).get("top_k_sampling", 5))
    seed = int(config.get("system", {}).get("seed", 42))
    if seed < 0:
        seed = np.random.randint(0, 2**31)

    print(f"[Step-3] Strategy: {strategy}, top_k: {top_k}")
    print(f"[Step-3] 开始采样…")

    if strategy == "nearest":
        sampled = _sample_nearest(features, image_ids, cluster_ids, top_k, seed)
    elif strategy == "farthest":
        sampled = _sample_farthest(features, image_ids, cluster_ids, top_k)
    elif strategy in ("random", "stratified"):
        sampled = _sample_random(image_ids, cluster_ids, top_k, seed)
    else:
        sampled = _sample_nearest(features, image_ids, cluster_ids, top_k, seed)

    out_path = output_dir / "S3_sampled_images.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sampled, f, indent=2, ensure_ascii=False)

    n_images = sum(len(v) for v in sampled.values())
    print(f"[Step-3] Sampled {n_images} images from {len(sampled)} clusters -> {out_path.name}")
    print("=" * 60)
    return out_path
