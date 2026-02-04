"""
HDBSCAN 参数敏感性测试
逐个变化 HDBSCAN 参数，测量对聚类结果的影响

用法:
  python scripts/test_hdbscan_params.py [--embeddings PATH] [--no-plot]
  不指定 --embeddings 则使用合成数据
  --no-plot 跳过绘图
"""

import sys
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")  # 无 GUI 模式
    # 设置中文字体（若可用），避免中文显示为方块
    try:
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# 添加项目根目录
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import hdbscan


def _cluster_stats(labels: np.ndarray) -> Dict[str, Any]:
    """计算聚类统计信息"""
    n = len(labels)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    noise_ratio = n_noise / n * 100 if n > 0 else 0
    
    # 簇大小分布
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    cluster_sizes = dict(zip(unique.tolist(), counts.tolist())) if len(unique) > 0 else {}
    
    return {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_ratio": noise_ratio,
        "cluster_sizes": cluster_sizes,
        "max_cluster_size": max(cluster_sizes.values()) if cluster_sizes else 0,
        "min_cluster_size": min(cluster_sizes.values()) if cluster_sizes else 0,
    }


def _silhouette_with_features(features: np.ndarray, labels: np.ndarray) -> float:
    """使用特征计算 silhouette score"""
    from sklearn.metrics import silhouette_score
    mask = labels >= 0
    n_clusters = len(set(labels[mask]))
    if np.sum(mask) < 2 or n_clusters < 2:
        return float("nan")
    try:
        return float(silhouette_score(features[mask], labels[mask], metric="euclidean"))
    except Exception:
        return float("nan")


def run_hdbscan(
    features: np.ndarray,
    min_cluster_size: int = 2,
    min_samples: int = 2,
    cluster_selection_method: str = "leaf",
    cluster_selection_epsilon: float = 0.0,
    cluster_selection_persistence: float = 0.0,
    alpha: float = 1.0,
    allow_single_cluster: bool = False,
) -> np.ndarray:
    """运行 HDBSCAN 并返回标签"""
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_persistence=cluster_selection_persistence,
        alpha=alpha,
        allow_single_cluster=allow_single_cluster,
    )
    return clusterer.fit_predict(features)


def load_embeddings(path: Path) -> np.ndarray:
    """加载嵌入矩阵"""
    arr = np.load(path)
    if arr.dtype != np.float32 and arr.dtype != np.float64:
        arr = arr.astype(np.float32)
    return arr


def generate_synthetic(n_samples: int = 500, n_features: int = 64, n_clusters: int = 5, seed: int = 42):
    """生成合成聚类数据"""
    rng = np.random.default_rng(seed)
    X = []
    labels_true = []
    centers = rng.standard_normal((n_clusters, n_features)) * 2
    per_cluster = n_samples // n_clusters
    for i in range(n_clusters):
        size = per_cluster if i < n_clusters - 1 else n_samples - per_cluster * (n_clusters - 1)
        pts = centers[i] + rng.standard_normal((size, n_features)) * 0.5
        X.append(pts)
        labels_true.extend([i] * size)
    X = np.vstack(X)
    # 打乱顺序
    perm = rng.permutation(len(X))
    X = X[perm]
    labels_true = np.array(labels_true)[perm]
    # L2 归一化（模拟真实嵌入）
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    X = X / norms
    return X


def _plot_param_impact(report: pd.DataFrame, n_samples: int, out_dir: Path) -> None:
    """绘制各参数对 n_clusters 和 n_noise 的影响"""
    out_dir.mkdir(parents=True, exist_ok=True)
    params = report["param"].unique()
    n_params = len(params)
    fig, axes = plt.subplots(n_params, 2, figsize=(12, 2.5 * n_params), sharex="col")
    if n_params == 1:
        axes = axes.reshape(1, -1)

    for i, param in enumerate(params):
        sub = report[report["param"] == param].copy()
        sub = sub.sort_values("value")
        x_vals = sub["value"].tolist()
        x_labels = [str(v) for v in x_vals]
        x_nums = np.arange(len(x_vals))

        ax1, ax2 = axes[i, 0], axes[i, 1]
        ax1.plot(x_nums, sub["n_clusters"].values, "o-", color="C0", linewidth=2, markersize=8)
        ax1.set_ylabel("n_clusters", fontsize=9)
        ax1.set_title(f"{param}", fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(x_nums)
        ax1.set_xticklabels(x_labels, rotation=20, ha="right")

        ax2.plot(x_nums, sub["n_noise"].values, "s-", color="C1", linewidth=2, markersize=8)
        ax2.set_ylabel("n_noise", fontsize=9)
        ax2.set_title(f"{param}", fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(x_nums)
        ax2.set_xticklabels(x_labels, rotation=20, ha="right")

    axes[0, 0].set_title("n_clusters vs param", fontsize=11)
    axes[0, 1].set_title("n_noise vs param", fontsize=11)
    plt.tight_layout()
    fig_path = out_dir / "hdbscan_param_impact.png"
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[图] {fig_path}")

    # 双 Y 轴图：每参数一图，左轴簇数、右轴噪声数
    fig2, axes2 = plt.subplots(n_params, 1, figsize=(10, 2.2 * n_params), sharex=False)
    if n_params == 1:
        axes2 = [axes2]

    for i, param in enumerate(params):
        sub = report[report["param"] == param].copy()
        sub = sub.sort_values("value")
        x_labels = [str(v) for v in sub["value"].tolist()]
        x_nums = np.arange(len(x_labels))
        ax = axes2[i]
        ax2_twin = ax.twinx()
        b1 = ax.bar(x_nums - 0.2, sub["n_clusters"].values, 0.35, color="C0", alpha=0.8)
        b2 = ax2_twin.bar(x_nums + 0.2, sub["n_noise"].values, 0.35, color="C1", alpha=0.8)
        ax.set_ylabel("n_clusters", color="C0")
        ax2_twin.set_ylabel("n_noise", color="C1")
        ax.set_title(f"{param}")
        ax.set_xticks(x_nums)
        ax.set_xticklabels(x_labels, rotation=20, ha="right")
        ax.tick_params(axis="y", labelcolor="C0")
        ax2_twin.tick_params(axis="y", labelcolor="C1")
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend([b1[0], b2[0]], ["n_clusters", "n_noise"], loc="upper right", fontsize=8)

    plt.suptitle(f"HDBSCAN param impact (n={n_samples})", fontsize=12, y=1.02)
    plt.tight_layout()
    fig2_path = out_dir / "hdbscan_param_dual_y.png"
    fig2.savefig(fig2_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[图] {fig2_path}")


def test_parameter(
    features: np.ndarray,
    param_name: str,
    values: List[Any],
    base_config: Dict[str, Any],
) -> pd.DataFrame:
    """测试单个参数不同取值的影响"""
    rows = []
    hdb_params = ["min_cluster_size", "min_samples", "cluster_selection_method",
                  "cluster_selection_epsilon", "cluster_selection_persistence", "alpha", "allow_single_cluster"]
    for v in values:
        cfg = base_config.copy()
        cfg[param_name] = v
        kwargs = {k: cfg[k] for k in hdb_params if k in cfg}
        labels = run_hdbscan(features, **kwargs)
        stats = _cluster_stats(labels)
        stats["silhouette"] = _silhouette_with_features(features, labels)
        stats["param"] = param_name
        stats["value"] = str(v)  # 统一为字符串便于展示
        rows.append({k: stats[k] for k in ["param", "value", "n_clusters", "n_noise", "noise_ratio", "silhouette"]})
    return pd.DataFrame(rows)


# 多组数据配置：(n_samples, n_clusters, seed)
MULTI_DATASETS = [
    (100, 3, 1),
    (150, 4, 2),
    (200, 5, 3),
    (300, 5, 4),
    (500, 5, 5),
    (500, 8, 6),
    (800, 10, 7),
    (1000, 12, 8),
]


def run_full_test(features: np.ndarray, base: Dict) -> pd.DataFrame:
    """在给定数据上运行完整参数测试"""
    all_results = []
    for param_name, values, base_cfg in [
        ("min_cluster_size", [2, 5, 10, 20, 50], base),
        ("min_samples", [2, 3, 5, 10, 20], base),
        ("cluster_selection_method", ["leaf", "eom"], {**base, "min_cluster_size": 5}),
        ("cluster_selection_epsilon", [0.0, 0.1, 0.2, 0.3, 0.5], base),
        ("cluster_selection_persistence", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], base),
        ("alpha", [0.5, 0.8, 1.0, 1.2, 1.5, 2.0], base),
        ("allow_single_cluster", [False, True], base),
    ]:
        df = test_parameter(features, param_name, values, base_cfg)
        all_results.append(df)
    return pd.concat(all_results, ignore_index=True)


def _apply_recommendations(rec: Dict[str, Any]) -> None:
    """将推荐默认值应用到 config 和 UI"""
    import yaml
    config_path = ROOT / "config" / "config.yaml"
    if not config_path.exists():
        return
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    c = cfg.get("clustering", {})
    for param, v in rec.items():
        default = v.get("default")
        if default is None:
            continue
        if param == "min_cluster_size":
            c["min_cluster_size"] = int(default) if isinstance(default, (int, float)) else default
        elif param == "min_samples":
            c["min_samples"] = int(default) if isinstance(default, (int, float)) else default
        elif param == "cluster_selection_method":
            c["cluster_selection_method"] = str(default)
        elif param == "cluster_selection_epsilon":
            c["cluster_selection_epsilon"] = float(default) if isinstance(default, (int, float)) else default
        elif param == "cluster_selection_persistence":
            c["cluster_selection_persistence"] = float(default) if isinstance(default, (int, float)) else default
        elif param == "alpha":
            c["alpha"] = float(default) if isinstance(default, (int, float)) else default
        elif param == "allow_single_cluster":
            c["allow_single_cluster"] = bool(default) if isinstance(default, bool) else str(default).lower() == "true"
    cfg["clustering"] = c
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    print(f"[OK] 已更新 config/config.yaml")


def recommend_defaults(agg: pd.DataFrame) -> Dict[str, Any]:
    """根据多组数据聚合结果推荐默认值和范围"""
    # 对每个 (param, value) 计算平均 silhouette 和 noise_ratio，并打分
    scores = []
    for (param, value), g in agg.groupby(["param", "value"]):
        sil_mean = g["silhouette"].replace([np.inf, -np.inf], np.nan).dropna().mean()
        sil_mean = sil_mean if not np.isnan(sil_mean) else 0
        noise_mean = g["noise_ratio"].mean()
        n_cl_mean = g["n_clusters"].mean()
        # 偏好: 高 silhouette、低 noise_ratio、n_clusters 适中(不为0不接近样本数)
        score = sil_mean * 0.5 - noise_mean * 0.003 + min(n_cl_mean, 20) * 0.02
        if noise_mean > 80:
            score -= 10
        if n_cl_mean < 1:
            score -= 5
        scores.append({"param": param, "value": value, "sil_mean": sil_mean, "noise_mean": noise_mean, "n_cl_mean": n_cl_mean, "score": score})
    scores_df = pd.DataFrame(scores)

    recommendations = {}
    for param in scores_df["param"].unique():
        sub = scores_df[scores_df["param"] == param].sort_values("score", ascending=False)
        if len(sub) > 0:
            best = sub.iloc[0]
            recommendations[param] = {
                "default": best["value"],
                "score": float(best["score"]),
                "noise_mean": float(best["noise_mean"]),
                "sil_mean": float(best["sil_mean"]),
            }
    return recommendations


def main():
    parser = argparse.ArgumentParser(description="HDBSCAN 参数敏感性测试")
    parser.add_argument("--embeddings", type=str, default=None, help="S1_embeddings.npy 路径")
    parser.add_argument("--output", type=str, default=None, help="报告输出路径")
    parser.add_argument("--synthetic-n", type=int, default=500, help="合成数据样本数（无 embeddings 时）")
    parser.add_argument("--no-plot", action="store_true", help="跳过绘图")
    parser.add_argument("--multi", action="store_true", help="多组数据测试并推荐默认值")
    parser.add_argument("--no-apply", action="store_true", help="--multi 时只输出推荐，不更新 config")
    args = parser.parse_args()

    base = {
        "min_cluster_size": 2,
        "min_samples": 2,
        "cluster_selection_method": "leaf",
        "cluster_selection_epsilon": 0.0,
        "cluster_selection_persistence": 0.0,
        "alpha": 1.0,
        "allow_single_cluster": False,
    }

    if args.multi:
        # 多组数据测试
        print("\n[多组数据测试] 共 %d 组" % len(MULTI_DATASETS))
        all_reports = []
        for n_s, n_c, seed in MULTI_DATASETS:
            print(f"  数据: n={n_s}, clusters={n_c}, seed={seed}")
            features = generate_synthetic(n_samples=n_s, n_clusters=n_c, seed=seed)
            df = run_full_test(features, base)
            df["dataset"] = f"n{n_s}_c{n_c}"
            all_reports.append(df)
        agg = pd.concat(all_reports, ignore_index=True)
        rec = recommend_defaults(agg)
        # 参数范围建议（基于测试覆盖）
        rec["_param_ranges"] = {
            "min_cluster_size": {"min": 2, "max": 50, "step": 1},
            "min_samples": {"min": 2, "max": 30, "step": 1},
            "cluster_selection_epsilon": {"min": 0.0, "max": 0.5, "step": 0.05},
            "cluster_selection_persistence": {"min": 0.0, "max": 1.0, "step": 0.1},
            "alpha": {"min": 0.5, "max": 2.0, "step": 0.1},
        }
        out_dir = ROOT / "data" / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        rec_path = out_dir / "hdbscan_recommendations.json"
        with open(rec_path, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in rec.items() if not k.startswith("_")}, f, indent=2, ensure_ascii=False)
        ranges_path = out_dir / "hdbscan_param_ranges.json"
        with open(ranges_path, "w", encoding="utf-8") as f:
            json.dump(rec.get("_param_ranges", {}), f, indent=2)
        print("\n[推荐默认值]")
        for p, v in rec.items():
            if p.startswith("_") or not isinstance(v, dict) or "default" not in v:
                continue
            print(f"  {p}: default={v['default']}, score={v['score']:.2f}, noise_mean={v.get('noise_mean', 0):.1f}%")
        print(f"\n[OK] 推荐已保存: {rec_path}")
        if not getattr(args, "no_apply", False):
            _apply_recommendations(rec)
        return

    # 单组数据测试
    # 加载数据
    if args.embeddings:
        emb_path = Path(args.embeddings)
        if not emb_path.exists():
            print(f"[ERROR] 嵌入文件不存在: {emb_path}")
            sys.exit(1)
        features = load_embeddings(emb_path)
        print(f"[加载] {emb_path} | shape={features.shape}")
    else:
        features = generate_synthetic(n_samples=args.synthetic_n)
        print(f"[合成数据] shape={features.shape}")

    n_samples = features.shape[0]

    all_results = []
    for param_name, values, base_cfg in [
        ("min_cluster_size", [2, 5, 10, 20, 50], base),
        ("min_samples", [2, 3, 5, 10, 20], base),
        ("cluster_selection_method", ["leaf", "eom"], {**base, "min_cluster_size": 5}),
        ("cluster_selection_epsilon", [0.0, 0.1, 0.2, 0.3, 0.5], base),
        ("cluster_selection_persistence", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], base),
        ("alpha", [0.5, 0.8, 1.0, 1.2, 1.5, 2.0], base),
        ("allow_single_cluster", [False, True], base),
    ]:
        print(f"[测试] {param_name}...")
        df = test_parameter(features, param_name, values, base_cfg)
        all_results.append(df)

    report = pd.concat(all_results, ignore_index=True)

    # 计算每个参数的影响幅度
    impact_rows = []
    for param in report["param"].unique():
        sub = report[report["param"] == param]
        if len(sub) < 2:
            continue
        n_clusters_range = int(sub["n_clusters"].max() - sub["n_clusters"].min())
        noise_range = float(sub["noise_ratio"].max() - sub["noise_ratio"].min())
        impact_rows.append({
            "param": param,
            "n_clusters_range": n_clusters_range,
            "noise_ratio_range": f"{noise_range:.1f}%",
            "impact_level": "高" if n_clusters_range >= 3 or noise_range >= 20 else ("中" if n_clusters_range >= 1 or noise_range >= 5 else "低"),
        })
    impact_df = pd.DataFrame(impact_rows)

    # 绘图：参数对噪声数和簇个数的影响
    if HAS_MATPLOTLIB and not getattr(args, "no_plot", False):
        _plot_param_impact(report, n_samples, ROOT / "data" / "output")
    elif not HAS_MATPLOTLIB:
        print("[WARN] 未安装 matplotlib，跳过绘图 (pip install matplotlib)")

    # 输出
    out_lines = [
        "=" * 70,
        "HDBSCAN 参数敏感性测试报告",
        "=" * 70,
        f"样本数: {n_samples}",
        "",
        "--- 各参数详细结果 ---",
        report.to_string(index=False),
        "",
        "--- 参数影响总结 ---",
        impact_df.to_string(index=False),
        "",
        "说明:",
        "  - n_clusters: 簇数量",
        "  - n_noise: 噪音点数量",
        "  - noise_ratio: 噪音比例 (%)",
        "  - silhouette: 轮廓系数 (-1~1，越高越好)",
        "  - impact_level: 高=该参数对结果影响大，需谨慎调参",
        "=" * 70,
    ]
    text = "\n".join(out_lines)

    # 默认输出到文件（避免 Windows 控制台 GBK 编码问题）
    out_path = Path(args.output) if args.output else ROOT / "data" / "output" / "hdbscan_param_test_report.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"\n[OK] 报告已保存: {out_path}")

    # 同时保存 CSV
    csv_path = (ROOT / "data" / "output" / "hdbscan_param_test.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[CSV] {csv_path}")


if __name__ == "__main__":
    main()
