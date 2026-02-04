"""
Step-1: Feature Embedding
使用视觉模型提取图像特征，生成 S1_embeddings.npy

📅 Last Updated: 2026-01-31
📖 Reference: docs/workflow-structure.md
"""

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
from sklearn.decomposition import PCA


class ImageDataset(Dataset):
    """
    图像数据集类
    """
    
    def __init__(self, index: Dict[str, dict], transform=None):
        """
        初始化数据集
        
        Args:
            index: 图像索引字典
            transform: 图像变换
        """
        self.index = index
        self.image_ids = list(index.keys())
        self.transform = transform
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = self.index[image_id]['path']
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, image_id


class FeatureExtractor:
    """
    特征提取器 - 使用预训练视觉模型
    """
    
    def __init__(self, config: dict, device: str = None):
        """
        初始化特征提取器
        
        Args:
            config: 配置字典
            device: 设备 (cuda/cpu)
        """
        self.config = config
        self.provider = config['embedding']['provider']
        self.backbone = config['embedding']['backbone']
        self.batch_size = config['embedding']['batch_size']
        self.pca_components = config['embedding']['pca_components']
        
        # 设置设备: 优先使用 config 的 embedding.device；config 为 cpu 但本机有 GPU 时自动用 cuda
        device_str = (device or config.get('embedding', {}).get('device') or 'cuda').strip().lower() or 'cuda'
        if device_str == 'cpu' and torch.cuda.is_available():
            device_str = 'cuda'
            print("[Step-1] 检测到 GPU，使用 cuda（config 中 device 已覆盖）")
        if device_str == 'cuda' and not torch.cuda.is_available():
            print("[Step-1] Warning: CUDA not available, falling back to CPU")
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(device_str)
        
        print(f"[Step-1] Using device: {self.device}")
        
        # 加载模型
        self.model, self.processor = self._load_model()
        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()
        
        # 图像变换 (CLIP 使用 processor，DINOv2 使用 transforms)
        self.transform = self._get_transform()
    
    def _get_dinov2_model_name(self) -> str:
        """DINOv2 backbone -> HuggingFace 模型名 (兼容 Python 3.9，避免 torch.hub)"""
        mapping = {
            'dinov2_vits14': 'facebook/dinov2-small',
            'dinov2_vitb14': 'facebook/dinov2-base',
            'dinov2_vitl14': 'facebook/dinov2-large',
            'dinov2_vitg14': 'facebook/dinov2-giant',
        }
        return mapping.get(self.backbone, 'facebook/dinov2-base')
    
    def _get_clip_model_name(self) -> str:
        """CLIP backbone -> HuggingFace 模型名"""
        mapping = {
            'clip_vitb32': 'openai/clip-vit-base-patch32',
            'clip_vitb16': 'openai/clip-vit-base-patch16',
            'clip_vitl14': 'openai/clip-vit-large-patch14',
            'clip_vitl14_336': 'openai/clip-vit-large-patch14-336',
        }
        return mapping.get(self.backbone, 'openai/clip-vit-base-patch16')
    
    def _load_model(self) -> tuple:
        """
        加载预训练模型
        
        Returns:
            (model, processor) - CLIP 时 processor 非空，DINOv2 时 processor 为 None
        """
        print(f"[Step-1] Loading model: {self.provider}/{self.backbone}")
        
        if self.provider == "dinov2":
            from transformers import AutoModel, AutoImageProcessor
            model_name = self._get_dinov2_model_name()
            print(f"[Step-1] Loading DINOv2 from HuggingFace: {model_name}")
            processor = AutoImageProcessor.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            return model, processor
            
        elif self.provider == "clip":
            from transformers import CLIPModel, CLIPProcessor
            model_name = self._get_clip_model_name()
            print(f"[Step-1] Loading CLIP from HuggingFace: {model_name}")
            processor = CLIPProcessor.from_pretrained(model_name)
            model = CLIPModel.from_pretrained(model_name)
            return model, processor
        
        else:
            print(f"[Warning] Unknown provider {self.provider}, using DINOv2")
            self.provider = "dinov2"
            self.backbone = "dinov2_vitb14"
            return self._load_model()
    
    def _get_transform(self):
        """
        获取图像变换 (DINOv2/CLIP 均使用 processor 在提取时处理)
        """
        if self.provider in ("clip", "dinov2"):
            return None  # 使用 processor 在 extract_features 中处理
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _collate_clip(self, batch):
        """CLIP 用 collate: 返回 (list_of_pil, list_of_ids)"""
        images = [b[0] for b in batch]
        ids = [b[1] for b in batch]
        return images, ids
    
    def extract_features(
        self,
        index: Dict[str, dict],
        progress_callback=None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        提取所有图像的特征
        
        Args:
            index: 图像索引
            progress_callback: 可选，每 batch 调用 callback(batch_idx, total_batches, n_done, n_total)
            
        Returns:
            (features, image_ids)
        """
        print(f"[Step-1] Extracting features from {len(index)} images")
        print(f"[Step-1] Batch size: {self.batch_size}")
        
        use_processor = self.provider in ("clip", "dinov2")
        dataset = ImageDataset(index, transform=self.transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False,
            collate_fn=self._collate_clip if use_processor else None
        )
        
        all_features = []
        all_image_ids = []
        total_batches = len(dataloader)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting features")):
                if use_processor:
                    images, image_ids = batch
                    inputs = self.processor(images=images, return_tensors="pt", padding=True)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    outputs = self.model(**inputs)
                    if self.provider == "clip":
                        features = self.model.get_image_features(**inputs)
                    else:
                        # DINOv2: 使用 [CLS] token
                        features = outputs.last_hidden_state[:, 0]
                else:
                    images, image_ids = batch
                    images = images.to(self.device)
                    features = self.model(images)
                
                if features.dim() > 2:
                    features = features.view(features.size(0), -1)
                
                all_features.append(features.cpu().numpy())
                all_image_ids.extend(image_ids)
                n_done = len(all_image_ids)
                # 每5个batch或最后一个batch打印进度
                if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == total_batches:
                    print(f"  [Step-1] 已处理 {n_done}/{len(index)} 张图像 (batch {batch_idx+1}/{total_batches})")
                if progress_callback:
                    progress_callback(batch_idx + 1, total_batches, n_done, len(index))
        
        all_features = np.vstack(all_features)
        print(f"[Step-1] Extracted features shape: {all_features.shape}")
        return all_features, all_image_ids
    
    def apply_pca(self, features: np.ndarray) -> np.ndarray:
        """
        应用PCA降维
        
        Args:
            features: 原始特征
            
        Returns:
            降维后的特征
        """
        n_samples, n_features = features.shape
        max_components = min(n_samples, n_features)
        # 单样本时无法做 PCA（n_components 至少为 1 且须 < n_samples），直接跳过
        if max_components <= 1:
            print(f"[Step-1] Skipping PCA (single sample or zero dim, keeping shape {features.shape})")
            return features
        
        if self.pca_components <= 0:
            print(f"[Step-1] Skipping PCA (components={self.pca_components})")
            return features
        
        if self.pca_components >= max_components:
            print(f"[Step-1] Adjusting PCA components: {self.pca_components} -> {max_components - 1}")
            pca_components = max_components - 1
        else:
            pca_components = self.pca_components
        
        print(f"[Step-1] Applying PCA: {n_features} -> {pca_components}")
        
        seed = self.config.get('system', {}).get('seed', -1)
        random_state = int(seed) if seed is not None and int(seed) >= 0 else None
        pca = PCA(n_components=pca_components, random_state=random_state)
        features_pca = pca.fit_transform(features)
        
        explained_variance = np.sum(pca.explained_variance_ratio_)
        print(f"[Step-1] PCA explained variance: {explained_variance:.2%}")
        
        return features_pca
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        L2归一化特征
        
        Args:
            features: 特征矩阵
            
        Returns:
            归一化后的特征
        """
        print(f"[Step-1] Normalizing features (L2)")
        
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1  # 避免除以0
        features_normalized = features / norms
        
        return features_normalized


def run_step1(
    config: dict,
    index_path: Path,
    output_dir: Path,
    progress_callback=None,
) -> Path:
    """
    运行Step-1: 特征提取
    
    Args:
        config: 配置字典
        index_path: 索引文件路径
        output_dir: 输出目录
        progress_callback: 可选，提取时调用 callback(batch_idx, total, n_done, n_total)
        
    Returns:
        特征文件路径
    """
    print("=" * 60)
    print("Step-1: Feature Embedding")
    print("=" * 60)
    
    # 加载索引
    with open(index_path, 'r', encoding='utf-8') as f:
        index = json.load(f)
    
    print(f"[Step-1] Loaded index with {len(index)} images")

    # 创建特征提取器
    extractor = FeatureExtractor(config)

    # 提取特征
    print(f"[Step-1] 开始提取特征…")
    features, image_ids = extractor.extract_features(index, progress_callback=progress_callback)
    print(f"[Step-1] 特征提取完成，shape={features.shape}")

    # PCA降维
    print(f"[Step-1] 开始 PCA 降维…")
    features = extractor.apply_pca(features)

    # L2归一化
    print(f"[Step-1] 开始 L2 归一化…")
    features = extractor.normalize_features(features)

    # 保存特征
    output_path = output_dir / "S1_embeddings.npy"
    np.save(output_path, features)
    print(f"[Step-1] Features saved to: {output_path}")
    
    # 保存image_ids映射
    mapping_path = output_dir / "S1_image_ids.json"
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(image_ids, f, indent=2)
    print(f"[Step-1] Image IDs saved to: {mapping_path}")
    
    # 保存 path 列表（与 embeddings 同序，供缓存复用时按路径 remap）
    paths = [index[iid]["path"] for iid in image_ids]
    paths_path = output_dir / "S1_paths.json"
    with open(paths_path, 'w', encoding='utf-8') as f:
        json.dump(paths, f, indent=2)
    
    # 保存统计信息
    stats = {
        'num_images': len(image_ids),
        'feature_dim': features.shape[1],
        'feature_mean': float(np.mean(features)),
        'feature_std': float(np.std(features)),
        'feature_min': float(np.min(features)),
        'feature_max': float(np.max(features)),
    }
    
    stats_path = output_dir / "S1_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    print(f"[Step-1] Statistics saved to: {stats_path}")
    
    print("=" * 60)
    print(f"[Step-1] Complete! Features shape: {features.shape}")
    print("=" * 60)
    
    return output_path


# ============================================
# Testing
# ============================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils import ConfigLoader
    
    print("Testing Step-1: Feature Embedding...")
    
    # 加载配置
    loader = ConfigLoader()
    
    loader.set("embedding.provider", "dinov2")
    loader.set("embedding.backbone", "dinov2_vitb14")
    loader.set("embedding.batch_size", 16)  # 较小的batch size
    loader.set("embedding.pca_components", 256)
    
    config = loader.to_dict()
    
    # 输入输出路径
    output_dir = Path(__file__).parent.parent / "data" / "output" / "test_run"
    index_path = output_dir / "S0_image_index.json"
    
    # 检查索引文件
    if not index_path.exists():
        print(f"[ERROR] Index file not found: {index_path}")
        print(f"[INFO] Please run step0_indexing.py first")
        sys.exit(1)
    
    # 运行Step-1
    try:
        features_path = run_step1(config, index_path, output_dir)
        print(f"\n[SUCCESS] Features file: {features_path}")
        
        # 读取并显示特征摘要
        features = np.load(features_path)
        print(f"\nFeatures Summary:")
        print(f"  Shape: {features.shape}")
        print(f"  Mean: {np.mean(features):.4f}")
        print(f"  Std: {np.std(features):.4f}")
        print(f"  Range: [{np.min(features):.4f}, {np.max(features):.4f}]")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
