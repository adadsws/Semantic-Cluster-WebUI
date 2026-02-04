"""
Step-0: Image Indexing
扫描输入目录，生成图像索引文件 S0_image_index.json

📅 Last Updated: 2026-01-31
📖 Reference: docs/workflow-structure.md
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image
from tqdm import tqdm


class ImageIndexer:
    """
    图像索引器 - 扫描目录并生成索引
    """
    
    def __init__(self, config: dict):
        """
        初始化索引器
        
        Args:
            config: 配置字典，包含数据源配置
        """
        self.config = config
        self.input_dir = Path(config['data']['input_directory'])
        self.supported_formats = config['data']['supported_formats']
        self.min_size_kb = config['data']['min_file_size_kb']
        self.max_size_mb = config['data']['max_file_size_mb']
        self.exclude_folders = config['data'].get('exclude_folders', '').split(',')
        self.exclude_folders = [f.strip() for f in self.exclude_folders if f.strip()]
    
    def _should_exclude_path(self, path: Path) -> bool:
        """
        检查路径是否应该被排除
        
        Args:
            path: 文件路径
            
        Returns:
            True if should exclude, False otherwise
        """
        # 检查是否在排除文件夹中
        for exclude_folder in self.exclude_folders:
            if exclude_folder in str(path):
                return True
        return False
    
    def _is_valid_image(self, image_path: Path) -> Tuple[bool, Optional[str]]:
        """
        检查图像是否有效
        
        Args:
            image_path: 图像路径
            
        Returns:
            (is_valid, error_message)
        """
        # 检查文件大小
        size_bytes = image_path.stat().st_size
        size_kb = size_bytes / 1024
        size_mb = size_kb / 1024
        
        # 大小过滤：min_size_kb<=0 表示不设下限，max_size_mb<=0 表示不设上限
        if self.min_size_kb > 0 and size_kb < self.min_size_kb:
            return False, f"File too small: {size_kb:.2f} KB < {self.min_size_kb} KB"
        if self.max_size_mb > 0 and size_mb > self.max_size_mb:
            return False, f"File too large: {size_mb:.2f} MB > {self.max_size_mb} MB"
        
        # 尝试打开图像验证
        try:
            with Image.open(image_path) as img:
                img.verify()  # 验证图像完整性
            return True, None
        except Exception as e:
            return False, f"Invalid image: {str(e)}"
    
    def _calculate_hash(self, image_path: Path) -> str:
        """
        计算图像的SHA-256哈希值（用于去重）
        
        Args:
            image_path: 图像路径
            
        Returns:
            SHA-256哈希值
        """
        sha256_hash = hashlib.sha256()
        with open(image_path, "rb") as f:
            # 分块读取以处理大文件
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def scan_directory(self) -> Dict[str, dict]:
        """
        扫描目录并生成索引
        
        Returns:
            图像索引字典 {image_id: {path, size, width, height, hash}}
        """
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
        
        print(f"[Step-0] Scanning directory: {self.input_dir}")
        print(f"[Step-0] Supported formats: {', '.join(self.supported_formats)}")
        
        # 收集所有图像文件
        image_files = []
        for ext in self.supported_formats:
            # 支持大小写
            image_files.extend(self.input_dir.rglob(f"*.{ext}"))
            image_files.extend(self.input_dir.rglob(f"*.{ext.upper()}"))
        
        print(f"[Step-0] Found {len(image_files)} potential image files")
        
        # 去重（处理大小写扩展名重复）
        image_files = list(set(image_files))
        print(f"[Step-0] After deduplication: {len(image_files)} files")
        
        # 处理每个图像
        index = {}
        valid_count = 0
        excluded_count = 0
        invalid_size_count = 0
        hash_dup_count = 0
        error_count = 0
        seen_hashes = {}
        
        for image_path in tqdm(image_files, desc="Indexing images"):
            # 检查排除路径
            if self._should_exclude_path(image_path):
                excluded_count += 1
                continue
            
            # 验证图像
            is_valid, error_msg = self._is_valid_image(image_path)
            if not is_valid:
                invalid_size_count += 1
                continue
            
            # 计算哈希值
            try:
                file_hash = self._calculate_hash(image_path)
                
                # 检查重复（内容相同只保留第一张）
                if file_hash in seen_hashes:
                    hash_dup_count += 1
                    continue
                
                seen_hashes[file_hash] = str(image_path)
                
                # 获取图像信息
                with Image.open(image_path) as img:
                    width, height = img.size
                
                # 生成唯一ID
                image_id = f"img_{valid_count:06d}"
                
                # 添加到索引
                index[image_id] = {
                    'path': str(image_path),
                    'filename': image_path.name,
                    'size_bytes': image_path.stat().st_size,
                    'width': width,
                    'height': height,
                    'hash': file_hash,
                    'extension': image_path.suffix.lower()
                }
                
                valid_count += 1
                
            except Exception as e:
                print(f"[Warning] Error processing {image_path}: {e}")
                error_count += 1
                continue
        
        # 统计信息（排除/无效/重复/异常 互不重叠）
        total_skipped = excluded_count + invalid_size_count + hash_dup_count + error_count
        print(f"\n[Step-0] Indexing complete:")
        print(f"  Valid images: {valid_count}")
        print(f"  Skipped: {total_skipped} (excluded: {excluded_count}, invalid/size: {invalid_size_count}, hash duplicates: {hash_dup_count}, errors: {error_count})")
        
        return index
    
    def save_index(self, index: Dict[str, dict], output_path: Path) -> None:
        """
        保存索引到JSON文件
        
        Args:
            index: 图像索引
            output_path: 输出文件路径
        """
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存索引
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2, ensure_ascii=False)
        
        print(f"[Step-0] Index saved to: {output_path}")
        print(f"[Step-0] Total images: {len(index)}")
        
        # 保存统计信息
        stats = {
            'total_images': len(index),
            'total_size_mb': sum(img['size_bytes'] for img in index.values()) / (1024 * 1024),
            'avg_size_mb': sum(img['size_bytes'] for img in index.values()) / len(index) / (1024 * 1024) if index else 0,
            'formats': {},
            'total_pixels': sum(img['width'] * img['height'] for img in index.values()),
        }
        
        # 统计每种格式的数量
        for img in index.values():
            ext = img['extension']
            stats['formats'][ext] = stats['formats'].get(ext, 0) + 1
        
        # 保存统计信息
        stats_path = output_path.parent / "S0_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"[Step-0] Statistics saved to: {stats_path}")


def run_step0(config: dict, output_dir: Path) -> Path:
    """
    运行Step-0: 图像索引
    
    Args:
        config: 配置字典
        output_dir: 输出目录
        
    Returns:
        索引文件路径
    """
    print("=" * 60)
    print("Step-0: Image Indexing")
    print("=" * 60)

    input_dir = Path(config['data']['input_directory'])
    print(f"[Step-0] 输入目录: {input_dir}")
    min_kb = config['data'].get('min_file_size_kb', 0)
    max_mb = config['data'].get('max_file_size_mb', -1)
    if min_kb <= 0 and (max_mb is None or max_mb <= 0):
        print("[Step-0] 不进行大小过滤")
    else:
        print(f"[Step-0] 大小过滤: {min_kb} KB ~ {max_mb} MB" + (" (无上限)" if max_mb <= 0 else ""))
    if config['data'].get('exclude_folders'):
        print(f"[Step-0] 排除文件夹: {config['data']['exclude_folders']}")

    # 创建索引器
    indexer = ImageIndexer(config)

    # 扫描目录
    print(f"[Step-0] 开始扫描…")
    index = indexer.scan_directory()

    if not index:
        raise ValueError("No valid images found in input directory")

    # 保存索引
    output_path = output_dir / "S0_image_index.json"
    indexer.save_index(index, output_path)

    print("=" * 60)
    print(f"[Step-0] Complete! Indexed {len(index)} images -> {output_path.name}")
    print("=" * 60)
    
    return output_path


# ============================================
# Testing
# ============================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils import ConfigLoader
    
    print("Testing Step-0: Image Indexing...")
    
    # 加载配置
    loader = ConfigLoader()
    
    # 设置test_pics作为输入
    test_pics_path = Path(__file__).parent.parent / "test_pics"
    loader.set("data.input_directory", str(test_pics_path))
    
    config = loader.to_dict()
    
    # 创建输出目录
    output_dir = Path(__file__).parent.parent / "data" / "output" / "test_run"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 运行Step-0
    try:
        index_path = run_step0(config, output_dir)
        print(f"\n[SUCCESS] Index file: {index_path}")
        
        # 读取并显示索引摘要
        with open(index_path, 'r', encoding='utf-8') as f:
            index = json.load(f)
        
        print(f"\nIndex Summary:")
        print(f"  Total images: {len(index)}")
        if index:
            first_key = list(index.keys())[0]
            print(f"  Sample entry: {first_key}")
            print(f"    Path: {index[first_key]['path']}")
            print(f"    Size: {index[first_key]['size_bytes'] / 1024:.2f} KB")
            print(f"    Dimensions: {index[first_key]['width']}x{index[first_key]['height']}")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
