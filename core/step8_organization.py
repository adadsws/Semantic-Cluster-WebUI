"""
Step-8: File Organization
按照聚类结果整理文件，生成整理后的文件夹结构

输出结构：默认 簇序号/簇序号@簇标签@原名（id@label@original）；或按 file_naming_rule（如 label@original）

📅 Last Updated: 2026-01-31
📖 Reference: docs/workflow-structure.md
"""

import json
import re
import shutil
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
from tqdm import tqdm


def _sanitize_dirname(name: str, max_length: int = 512) -> str:
    """替换文件夹名非法字符，保留中文、字母、数字、空格、连字符、下划线；与 Step-5 label 最大长度一致。"""
    s = re.sub(r'[<>:"/\\|?*\n\r\t]', "_", name)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.rstrip(" .")  # Windows 不允许文件夹名以空格或点结尾
    return s[:max_length] if len(s) > max_length else s if s else "unnamed"


class FileOrganizer:
    """
    文件整理器 - 按簇整理图像文件
    """
    
    def __init__(self, config: dict):
        """
        初始化文件整理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.naming_rule = config['output']['file_naming_rule']
    
    def generate_filename(
        self,
        cluster_id: int,
        cluster_label: str,
        original_filename: str,
        cluster_id_width: int = 2,
    ) -> str:
        """
        根据命名规则生成新文件名

        Args:
            cluster_id: 簇ID
            cluster_label: 簇标签（语义标签，Phase-2才有）
            original_filename: 原始文件名
            cluster_id_width: 簇序号格式化宽度（与最大簇数一致，如 100+ 簇用 3）

        Returns:
            新文件名
        """
        # 获取原始文件名和扩展名
        original_stem = Path(original_filename).stem
        original_ext = Path(original_filename).suffix

        # 簇ID格式化为统一位数（与最大簇数一致）
        cluster_str = f"{cluster_id:0{cluster_id_width}d}" if cluster_id >= 0 else "noise"
        
        # 根据命名规则生成
        if self.naming_rule == "id@label@original":
            # 格式: 簇序号/簇序号@簇标签@原名，如 00/00@Mountain_Landscape@IMG_1234.jpg
            if cluster_label:
                new_name = f"{cluster_str}@{cluster_label}@{original_stem}{original_ext}"
            else:
                new_name = f"{cluster_str}@{original_stem}{original_ext}"
        
        elif self.naming_rule == "label@original":
            # 格式: Cluster_Label@original_name.jpg
            if cluster_label:
                new_name = f"{cluster_label}@{original_stem}{original_ext}"
            else:
                new_name = f"{cluster_str}@{original_stem}{original_ext}"
        
        elif self.naming_rule == "cluster_id@label":
            # 格式: 01@Cluster_Label.jpg
            if cluster_label:
                new_name = f"{cluster_str}@{cluster_label}{original_ext}"
            else:
                new_name = f"{cluster_str}{original_ext}"
        
        elif self.naming_rule == "cluster_id@label@original":
            # 格式: 01@Cluster_Label@original_name.jpg（文件夹仍为 label）
            if cluster_label:
                new_name = f"{cluster_str}@{cluster_label}@{original_stem}{original_ext}"
            else:
                new_name = f"{cluster_str}@{original_stem}{original_ext}"
        
        else:
            # 默认: 簇序号@簇标签@原名
            if cluster_label:
                new_name = f"{cluster_str}@{cluster_label}@{original_stem}{original_ext}"
            else:
                new_name = f"{cluster_str}@{original_stem}{original_ext}"
        
        return new_name
    
    def organize_files(
        self,
        index: Dict[str, dict],
        clustering: pd.DataFrame,
        output_base_dir: Path,
        dry_run: bool = False,
        cluster_labels: Optional[Dict[int, str]] = None,
    ) -> Dict:
        """
        整理文件到输出目录

        Args:
            index: 图像索引
            clustering: 聚类结果DataFrame
            output_base_dir: 输出基础目录
            dry_run: 是否只预览不实际移动
            cluster_labels: 可选，簇ID→语义标签（Phase-3 S5 提供）

        Returns:
            整理日志
        """
        print(f"[Step-8] Organizing files to: {output_base_dir}")
        print(f"[Step-8] Naming rule: {self.naming_rule}")
        print(f"[Step-8] Dry run: {dry_run}")

        # 簇序号宽度：与最大簇 ID 位数一致（至少 2 位）
        non_noise = [int(c) for c in clustering['cluster_id'].unique() if int(c) >= 0]
        cluster_id_width = max(2, len(str(max(non_noise)))) if non_noise else 2

        # 创建输出目录
        if not dry_run:
            output_base_dir.mkdir(parents=True, exist_ok=True)

        # 整理日志
        log = {
            'moved': [],
            'skipped': [],
            'errors': [],
            'conflicts': []
        }

        # 按簇整理
        for cluster_id in tqdm(sorted(clustering['cluster_id'].unique()), desc="Organizing clusters"):
            # 获取该簇的所有图像
            cluster_images = clustering[clustering['cluster_id'] == cluster_id]

            # 文件夹名：id@label@original 时为簇序号（统一位数），否则有语义标签用 label，无则 cluster_00 / noise
            label = (cluster_labels or {}).get(int(cluster_id), "")
            cluster_str = f"{cluster_id:0{cluster_id_width}d}" if cluster_id >= 0 else "noise"
            if self.naming_rule == "id@label@original":
                cluster_dir = output_base_dir / cluster_str
            elif cluster_id == -1:
                cluster_dir = output_base_dir / "noise"
            elif label:
                safe_label = _sanitize_dirname(label)
                cluster_dir = output_base_dir / safe_label
            else:
                cluster_dir = output_base_dir / f"cluster_{cluster_id:0{cluster_id_width}d}"
            
            if not dry_run:
                cluster_dir.mkdir(parents=True, exist_ok=True)
            
            # 整理该簇的图像
            for _, row in cluster_images.iterrows():
                image_id = row['image_id']
                
                if image_id not in index:
                    log['errors'].append({
                        'image_id': image_id,
                        'error': 'Image ID not found in index'
                    })
                    continue
                
                # 获取原始路径
                original_path = Path(index[image_id]['path'])
                
                if not original_path.exists():
                    log['errors'].append({
                        'image_id': image_id,
                        'original_path': str(original_path),
                        'error': 'Original file not found'
                    })
                    continue
                
                # 生成新文件名（有 S5 时使用语义标签；label 已在上面按簇取过）
                new_filename = self.generate_filename(
                    cluster_id=cluster_id,
                    cluster_label=label,
                    original_filename=original_path.name,
                    cluster_id_width=cluster_id_width,
                )
                
                new_path = cluster_dir / new_filename
                
                # 检查冲突
                if new_path.exists():
                    # 添加编号避免冲突
                    counter = 1
                    while new_path.exists():
                        stem = Path(new_filename).stem
                        ext = Path(new_filename).suffix
                        new_filename_numbered = f"{stem}_{counter}{ext}"
                        new_path = cluster_dir / new_filename_numbered
                        counter += 1
                    
                    log['conflicts'].append({
                        'original': new_filename,
                        'resolved': new_filename_numbered
                    })
                
                # 复制文件
                if not dry_run:
                    try:
                        shutil.copy2(original_path, new_path)
                        log['moved'].append({
                            'image_id': image_id,
                            'from': str(original_path),
                            'to': str(new_path),
                            'cluster_id': int(cluster_id)
                        })
                    except Exception as e:
                        log['errors'].append({
                            'image_id': image_id,
                            'original_path': str(original_path),
                            'error': str(e)
                        })
                else:
                    # 预览模式
                    log['moved'].append({
                        'image_id': image_id,
                        'from': str(original_path),
                        'to': str(new_path),
                        'cluster_id': int(cluster_id)
                    })
        
        # 统计
        print(f"\n[Step-8] Organization complete:")
        print(f"  Files moved: {len(log['moved'])}")
        print(f"  Conflicts resolved: {len(log['conflicts'])}")
        print(f"  Errors: {len(log['errors'])}")
        
        return log


def run_step8(
    config: dict,
    index_path: Path,
    clustering_path: Path,
    output_dir: Path,
    organized_output_dir: Path,
    dry_run: bool = False,
    labels_path: Optional[Path] = None,
) -> Path:
    """
    运行Step-8: 文件整理

    Args:
        config: 配置字典
        index_path: 索引文件路径
        clustering_path: 聚类结果路径
        output_dir: 输出目录（保存日志）
        organized_output_dir: 整理后文件的输出目录
        dry_run: 是否只预览
        labels_path: 可选，S5_cluster_labels.csv（Phase-3 语义标签）

    Returns:
        日志文件路径
    """
    print("=" * 60)
    print("Step-8: File Organization")
    print("=" * 60)

    # 加载索引
    print(f"[Step-8] 加载索引与聚类…")
    with open(index_path, 'r', encoding='utf-8') as f:
        index = json.load(f)
    print(f"[Step-8] Loaded index with {len(index)} images")

    # 加载聚类结果
    clustering = pd.read_csv(clustering_path)
    print(f"[Step-8] Loaded clustering results: {len(clustering)} images")

    # 加载簇语义标签（若有 S5）
    cluster_labels = None
    if labels_path and labels_path.exists():
        labels_df = pd.read_csv(labels_path)
        if "cluster_id" in labels_df.columns and "label" in labels_df.columns:
            cluster_labels = dict(zip(labels_df["cluster_id"].astype(int), labels_df["label"].astype(str)))
            print(f"[Step-8] Loaded cluster labels: {len(cluster_labels)} labels")

    # 创建整理器
    organizer = FileOrganizer(config)
    print(f"[Step-8] 开始整理文件（命名规则: {organizer.naming_rule}，dry_run={dry_run}）…")

    # 整理文件
    log = organizer.organize_files(
        index, clustering, organized_output_dir,
        dry_run=dry_run,
        cluster_labels=cluster_labels,
    )
    
    # 保存日志
    log_path = output_dir / "S8_organization_log.json"
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"\n[Step-8] Log saved to: {log_path}")
    
    # 保存简化的移动清单
    if log['moved']:
        moves_df = pd.DataFrame(log['moved'])
        moves_csv_path = output_dir / "S8_file_moves.csv"
        moves_df.to_csv(moves_csv_path, index=False, encoding='utf-8-sig')
        print(f"[Step-8] File moves list saved to: {moves_csv_path}")
    
    print("=" * 60)
    print(f"[Step-8] Complete! Organized {len(log['moved'])} files")
    print("=" * 60)
    
    return log_path


# ============================================
# Testing
# ============================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils import ConfigLoader
    
    print("Testing Step-8: File Organization...")
    
    # 加载配置
    loader = ConfigLoader()
    config = loader.to_dict()
    
    # 输入输出路径
    work_dir = Path(__file__).parent.parent / "data" / "output" / "test_run"
    index_path = work_dir / "S0_image_index.json"
    clustering_path = work_dir / "S2_clustering.csv"
    organized_output_dir = work_dir / "organized"
    
    # 检查文件
    if not index_path.exists():
        print(f"[ERROR] Index file not found: {index_path}")
        sys.exit(1)
    
    if not clustering_path.exists():
        print(f"[ERROR] Clustering file not found: {clustering_path}")
        sys.exit(1)
    
    # 运行Step-8（先dry run预览）
    try:
        print("\n=== DRY RUN (Preview) ===")
        log_path = run_step8(
            config,
            index_path,
            clustering_path,
            work_dir,
            organized_output_dir,
            dry_run=True
        )
        
        # 询问是否执行
        print("\n" + "=" * 60)
        response = input("Execute file organization? (y/n): ")
        
        if response.lower() == 'y':
            print("\n=== EXECUTING ===")
            log_path = run_step8(
                config,
                index_path,
                clustering_path,
                work_dir,
                organized_output_dir,
                dry_run=False
            )
            print(f"\n[SUCCESS] Files organized to: {organized_output_dir}")
        else:
            print("\n[INFO] Organization cancelled")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
