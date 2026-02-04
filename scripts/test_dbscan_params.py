"""
DBSCAN 参数敏感性测试
测试 sklearn DBSCAN 各参数对聚类结果的影响

用法:
  python scripts/test_dbscan_params.py [--embeddings PATH] [--no-plot]
"""

import sys
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
    try:
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from sklearn.cluster import DBSCAN


def _cluster_stats(labels: np.ndarray) -> Dict[str, Any]:
    n = len(labels)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    noise_ratio = n_noise / n * 100 if n > 0 else 0
    return {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_ratio": noise_ratio,
    }


def _silhouette_with_features(features: np.ndarray, labels: np.ndarray, metric: str = "euclidean") -> float:
    from sklearn.metrics import silhouette_score
    mask = labels >= 0
    n_clusters = len(set(labels[mask]))
    if np.sum(mask) < 2 or n_clusters < 2:
        return float("nan")
    try:
        return float(silhouette_score(features[mask], labels[mask], metric=metric))
    except Exception:
        return float("nan")


def run_dbscan(
    features: np.ndarray,
    eps: float = 1.0,
    min_samples: int = 2,
    metric: str = "euclidean",
    algorithm: str = "auto",
    n_jobs: int = None,
    leaf_size: int = 30,
) -> np.ndarray:
    kwargs = dict(
        eps=eps,
        min_samples=min_samples,
        metric=metric,
        algorithm=algorithm,
        leaf_size=leaf_size,
    )
    if n_jobs is not None:
        kwargs["n_jobs"] = n_jobs
    clusterer = DBSCAN(**kwargs)
    return clusterer.fit_predict(features)


def load_embeddings(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.dtype not in (np.float32, np.float64):
        arr = arr.astype(np.float32)
    return arr


def generate_synthetic(n_samples: int = 500, n_features: int = 64, n_clusters: int = 5, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = []
    centers = rng.standard_normal((n_clusters, n_features)) * 2
    per_cluster = n_samples // n_clusters
    for i in range(n_clusters):
        size = per_cluster if i < n_clusters - 1 else n_samples - per_cluster * (n_clusters - 1)
        pts = centers[i] + rng.standard_normal((size, n_features)) * 0.5
        X.append(pts)
    X = np.vstack(X)
    perm = rng.permutation(len(X))
    X = X[perm]
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    X = X / norms
    return X


def test_param(
    features: np.ndarray,
    param_name: str,
    values: List[Any],
    base: Dict[str, Any],
) -> pd.DataFrame:
    rows = []
    for v in values:
        cfg = base.copy()
        cfg[param_name] = v
        t0 = time.perf_counter()
        labels = run_dbscan(features, **{k: cfg[k] for k in ["eps", "min_samples", "metric", "algorithm", "n_jobs", "leaf_size"] if k in cfg})
        elapsed = time.perf_counter() - t0
        stats = _cluster_stats(labels)
        stats["silhouette"] = _silhouette_with_features(features, labels, cfg.get("metric", "euclidean"))
        stats["time_sec"] = round(elapsed, 4)
        stats["param"] = param_name
        stats["value"] = str(v)
        rows.append({k: stats[k] for k in ["param", "value", "n_clusters", "n_noise", "noise_ratio", "silhouette", "time_sec"]})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="DBSCAN parameter sensitivity test")
    parser.add_argument("--embeddings", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--synthetic-n", type=int, default=500)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    if args.embeddings:
        emb_path = Path(args.embeddings)
        if not emb_path.exists():
            print(f"[ERROR] File not found: {emb_path}")
            sys.exit(1)
        features = load_embeddings(emb_path)
        print(f"[Load] {emb_path} | shape={features.shape}")
    else:
        features = generate_synthetic(n_samples=args.synthetic_n)
        print(f"[Synthetic] shape={features.shape}")

    n_samples = features.shape[0]
    base = {"eps": 1.0, "min_samples": 2, "metric": "euclidean", "algorithm": "auto"}

    all_results = []

    # 1. eps
    print("\n[Test] eps...")
    df = test_param(features, "eps", [0.3, 0.5, 0.7, 1.0, 1.2, 1.5], base)
    all_results.append(df)

    # 2. min_samples
    print("[Test] min_samples...")
    df = test_param(features, "min_samples", [2, 3, 5, 10, 20], base)
    all_results.append(df)

    # 3. metric
    print("[Test] metric...")
    df = test_param(features, "metric", ["euclidean", "cosine", "manhattan"], base)
    all_results.append(df)

    # 4. algorithm
    print("[Test] algorithm...")
    df = test_param(features, "algorithm", ["auto", "ball_tree", "kd_tree", "brute"], base)
    all_results.append(df)

    # 5. n_jobs (speed only)
    print("[Test] n_jobs...")
    df = test_param(features, "n_jobs", [None, 1, 2, 4, -1], base)
    all_results.append(df)

    # 6. leaf_size
    print("[Test] leaf_size...")
    base_leaf = base.copy()
    base_leaf["algorithm"] = "ball_tree"
    df = test_param(features, "leaf_size", [10, 30, 50, 100], base_leaf)
    all_results.append(df)

    report = pd.concat(all_results, ignore_index=True)

    # Impact summary
    impact_rows = []
    for param in report["param"].unique():
        sub = report[report["param"] == param]
        if len(sub) < 2:
            continue
        ncr = int(sub["n_clusters"].max() - sub["n_clusters"].min())
        nr = float(sub["noise_ratio"].max() - sub["noise_ratio"].min())
        tr = float(sub["time_sec"].max() - sub["time_sec"].min()) if "time_sec" in sub else 0
        impact_rows.append({
            "param": param,
            "n_clusters_range": ncr,
            "noise_range": f"{nr:.1f}%",
            "time_range_sec": f"{tr:.3f}" if tr > 0.001 else "-",
            "affects_result": "yes" if ncr > 0 or nr > 1 else "no (speed only)",
        })
    impact_df = pd.DataFrame(impact_rows)

    out_dir = ROOT / "data" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else out_dir / "dbscan_param_test_report.txt"

    text = "\n".join([
        "=" * 70,
        "DBSCAN Parameter Sensitivity Report",
        "=" * 70,
        f"Samples: {n_samples}",
        "",
        "--- Results ---",
        report.to_string(index=False),
        "",
        "--- Impact Summary ---",
        impact_df.to_string(index=False),
        "",
        "Recommendations:",
        "  - eps, min_samples, metric: affect clustering results",
        "  - algorithm, n_jobs, leaf_size: mainly affect speed",
        "=" * 70,
    ])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"\n[OK] Report: {out_path}")

    report.to_csv(out_dir / "dbscan_param_test.csv", index=False, encoding="utf-8-sig")

    if HAS_MATPLOTLIB and not args.no_plot:
        _plot(report, n_samples, out_dir)


def _plot(report: pd.DataFrame, n_samples: int, out_dir: Path):
    params = report["param"].unique()
    n_params = len(params)
    fig, axes = plt.subplots(n_params, 2, figsize=(12, 2.5 * n_params))
    if n_params == 1:
        axes = axes.reshape(1, -1)
    for i, param in enumerate(params):
        sub = report[report["param"] == param].sort_values("value" if report["value"].dtype == object or param in ("eps", "min_samples", "leaf_size") else "value")
        try:
            sub = sub.sort_values("value", key=lambda x: pd.to_numeric(x, errors="coerce"))
        except Exception:
            pass
        x_vals = sub["value"].tolist()
        x_nums = np.arange(len(x_vals))
        ax1, ax2 = axes[i, 0], axes[i, 1]
        ax1.plot(x_nums, sub["n_clusters"].values, "o-", color="C0")
        ax1.set_ylabel("n_clusters")
        ax1.set_title(param)
        ax1.set_xticks(x_nums)
        ax1.set_xticklabels([str(v) for v in x_vals], rotation=20, ha="right")
        ax1.grid(True, alpha=0.3)
        ax2.plot(x_nums, sub["n_noise"].values, "s-", color="C1")
        ax2.set_ylabel("n_noise")
        ax2.set_title(param)
        ax2.set_xticks(x_nums)
        ax2.set_xticklabels([str(v) for v in x_vals], rotation=20, ha="right")
        ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "dbscan_param_impact.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[Plot] {out_dir / 'dbscan_param_impact.png'}")


if __name__ == "__main__":
    main()
