"""
Step-2: Clustering
使用HDBSCAN聚类算法对图像特征进行聚类，生成 S2_clustering.csv

📅 Last Updated: 2026-01-31
📖 Reference: docs/workflow-structure.md
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
import hdbscan
from sklearn.cluster import DBSCAN


class ImageClusterer:
    """
    图像聚类器 - 使用DBSCAN/HDBSCAN进行聚类
    """
    
    def __init__(self, config: dict):
        """
        初始化聚类器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.backend = config['clustering']['backend']
        self.metric = config['clustering']['metric']
        self.dbscan_metric = config['clustering'].get('dbscan_metric') or self.metric
        self.dbscan_algorithm = config['clustering'].get('dbscan_algorithm', 'auto')
        self.mode = config['clustering']['mode']
        self.epsilon = config['clustering']['epsilon']
        self.max_noise_ratio = config['clustering']['max_noise_ratio']
        self.min_samples = config['clustering']['min_samples']
        # [B7] HDBSCAN簇选择方法: eom=保守/簇少, leaf=细粒度/噪音少
        self.cluster_selection_method = config['clustering'].get('cluster_selection_method', 'leaf')
        # [B8-B12] HDBSCAN 专用参数
        self.min_cluster_size = config['clustering'].get('min_cluster_size') or self.min_samples
        self.cluster_selection_epsilon = config['clustering'].get('cluster_selection_epsilon', 0.0)
        self.cluster_selection_persistence = config['clustering'].get('cluster_selection_persistence', 0.0)
        self.alpha = config['clustering'].get('alpha', 1.0)
    
    def cluster_fixed_eps(self, features: np.ndarray) -> np.ndarray:
        """
        使用固定eps进行聚类
        
        Args:
            features: 特征矩阵
            
        Returns:
            聚类标签数组
        """
        print(f"[Step-2] Clustering with fixed eps={self.epsilon}")
        
        if self.backend == "hdbscan":
            # HDBSCAN (lmcinnes/hdbscan) 不支持 random_state，算法本身较确定性
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=int(self.min_cluster_size),
                min_samples=int(self.min_samples),
                metric=self.metric,
                cluster_selection_method=self.cluster_selection_method,
                cluster_selection_epsilon=float(self.cluster_selection_epsilon),
                cluster_selection_persistence=float(self.cluster_selection_persistence),
                alpha=float(self.alpha),
            )
        else:
            algo = self.dbscan_algorithm
            if self.dbscan_metric == "cosine" and algo in ("ball_tree", "kd_tree"):
                algo = "brute"
            clusterer = DBSCAN(
                eps=self.epsilon,
                min_samples=self.min_samples,
                metric=self.dbscan_metric,
                algorithm=algo,
            )
        
        labels = clusterer.fit_predict(features)
        
        return labels
    
    def cluster_noise_control(self, features: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        自动搜索eps，控制噪音比例
        
        Args:
            features: 特征矩阵
            
        Returns:
            (聚类标签数组, 最终eps值)
        """
        print(f"[Step-2] Searching eps to keep noise <= {self.max_noise_ratio}%")
        
        max_noise_fraction = self.max_noise_ratio / 100.0
        n_samples = features.shape[0]
        
        # eps搜索范围
        eps_min = 0.05
        eps_max = 2.0
        eps_step = 0.05
        
        best_labels = None
        best_eps = self.epsilon
        best_noise_ratio = 100.0
        
        eps_values = np.arange(eps_min, eps_max, eps_step)
        
        algo = self.dbscan_algorithm
        if self.dbscan_metric == "cosine" and algo in ("ball_tree", "kd_tree"):
            algo = "brute"
        for eps in eps_values:
            clusterer = DBSCAN(
                eps=eps,
                min_samples=self.min_samples,
                metric=self.dbscan_metric,
                algorithm=algo,
            )
            labels = clusterer.fit_predict(features)
            
            # 计算噪音比例
            noise_count = np.sum(labels == -1)
            noise_ratio = noise_count / n_samples
            
            # 如果噪音比例满足要求
            if noise_ratio <= max_noise_fraction:
                if best_labels is None or noise_ratio > best_noise_ratio:
                    # 选择噪音比例最接近目标的
                    best_labels = labels
                    best_eps = eps
                    best_noise_ratio = noise_ratio
                    
                    # 如果已经很接近目标，可以停止
                    if abs(noise_ratio - max_noise_fraction) < 0.01:
                        break
        
        if best_labels is None:
            print(f"[Warning] Could not find eps with noise <= {self.max_noise_ratio}%, using fixed eps")
            best_labels = self.cluster_fixed_eps(features)
            best_eps = self.epsilon
        else:
            print(f"[Step-2] Found optimal eps={best_eps:.3f} (noise={best_noise_ratio*100:.1f}%)")
        
        return best_labels, best_eps
    
    def cluster(self, features: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        执行聚类
        
        Args:
            features: 特征矩阵
            
        Returns:
            (聚类标签, 统计信息)
        """
        print(f"[Step-2] Clustering {features.shape[0]} images")
        print(f"[Step-2] Backend: {self.backend}")
        print(f"[Step-2] Mode: {self.mode}")
        print(f"[Step-2] Metric: {self.metric}")
        print(f"[Step-2] Min samples: {self.min_samples}")
        if self.backend == "hdbscan":
            print(f"[Step-2] HDBSCAN: min_cluster_size={self.min_cluster_size}, method={self.cluster_selection_method}")
            print(f"[Step-2] HDBSCAN: cluster_sel_eps={self.cluster_selection_epsilon}, persistence={self.cluster_selection_persistence}")
        
        # 根据模式选择聚类方法
        if self.mode == "fixed_eps":
            labels = self.cluster_fixed_eps(features)
            used_eps = self.epsilon
        elif self.mode == "noise_control":
            labels, used_eps = self.cluster_noise_control(features)
        else:
            print(f"[Warning] Unknown mode {self.mode}, using fixed_eps")
            labels = self.cluster_fixed_eps(features)
            used_eps = self.epsilon
        
        # 统计信息
        n_samples = features.shape[0]
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)
        noise_ratio = n_noise / n_samples * 100
        
        print(f"\n[Step-2] Clustering results:")
        print(f"  Total images: {n_samples}")
        print(f"  Number of clusters: {n_clusters}")
        print(f"  Noise images: {n_noise} ({noise_ratio:.1f}%)")
        
        # 每个簇的大小
        unique_labels = sorted(set(labels))
        cluster_sizes = {}
        for label in unique_labels:
            if label != -1:
                size = np.sum(labels == label)
                cluster_sizes[int(label)] = int(size)  # 转换为Python int
                print(f"  Cluster {label}: {size} images")
        
        stats = {
            'n_samples': int(n_samples),
            'n_clusters': int(n_clusters),
            'n_noise': int(n_noise),
            'noise_ratio': float(noise_ratio),
            'used_eps': float(used_eps),
            'cluster_sizes': cluster_sizes
        }
        
        return labels, stats


def run_step2(config: dict, features_path: Path, image_ids_path: Path, output_dir: Path) -> Path:
    """
    运行Step-2: 聚类
    
    Args:
        config: 配置字典
        features_path: 特征文件路径
        image_ids_path: image_ids文件路径
        output_dir: 输出目录
        
    Returns:
        聚类结果文件路径
    """
    print("=" * 60)
    print("Step-2: Clustering")
    print("=" * 60)

    print(f"[Step-2] 加载特征与图像 ID…")
    # 加载特征
    features = np.load(features_path)
    print(f"[Step-2] Loaded features: {features.shape}")
    
    # 加载image_ids
    with open(image_ids_path, 'r', encoding='utf-8') as f:
        image_ids = json.load(f)
    print(f"[Step-2] Loaded {len(image_ids)} image IDs")
    
    # 创建聚类器
    clusterer = ImageClusterer(config)
    
    # 执行聚类
    labels, stats = clusterer.cluster(features)
    
    # 创建结果DataFrame
    results = pd.DataFrame({
        'image_id': image_ids,
        'cluster_id': labels
    })
    
    # 保存聚类结果
    output_path = output_dir / "S2_clustering.csv"
    results.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n[Step-2] Clustering results saved to: {output_path}")
    
    # 保存统计信息
    stats_path = output_dir / "S2_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[Step-2] Statistics saved to: {stats_path}")
    
    print("=" * 60)
    print(f"[Step-2] Complete! Found {stats['n_clusters']} clusters")
    print("=" * 60)
    
    return output_path


# ============================================
# Testing
# ============================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils import ConfigLoader
    
    print("Testing Step-2: Clustering...")
    
    # 加载配置
    loader = ConfigLoader()
    
    # 设置聚类参数（使用HDBSCAN）
    loader.set("clustering.backend", "hdbscan")
    loader.set("clustering.mode", "fixed_eps")
    loader.set("clustering.min_samples", 2)  # 更小的min_samples
    
    config = loader.to_dict()
    
    # 输入输出路径
    output_dir = Path(__file__).parent.parent / "data" / "output" / "test_run"
    features_path = output_dir / "S1_embeddings.npy"
    image_ids_path = output_dir / "S1_image_ids.json"
    
    # 检查文件
    if not features_path.exists():
        print(f"[ERROR] Features file not found: {features_path}")
        print(f"[INFO] Please run step1_embedding.py first")
        sys.exit(1)
    
    if not image_ids_path.exists():
        print(f"[ERROR] Image IDs file not found: {image_ids_path}")
        print(f"[INFO] Please run step1_embedding.py first")
        sys.exit(1)
    
    # 运行Step-2
    try:
        clustering_path = run_step2(config, features_path, image_ids_path, output_dir)
        print(f"\n[SUCCESS] Clustering file: {clustering_path}")
        
        # 读取并显示聚类摘要
        results = pd.read_csv(clustering_path)
        print(f"\nClustering Summary:")
        print(f"  Total images: {len(results)}")
        print(f"  Unique clusters: {results['cluster_id'].nunique()}")
        print(f"  Noise images: {(results['cluster_id'] == -1).sum()}")
        
        # 显示簇大小分布
        print(f"\nCluster size distribution:")
        cluster_counts = results[results['cluster_id'] != -1]['cluster_id'].value_counts().sort_index()
        for cluster_id, count in cluster_counts.head(10).items():
            print(f"  Cluster {cluster_id}: {count} images")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
