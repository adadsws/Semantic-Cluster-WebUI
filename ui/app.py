"""
Gradio Web UI for Semantic-Cluster-WebUI
集成所有步骤的Web界面

📅 Last Updated: 2026-01-31
"""

import sys
import os
import subprocess
import platform
import io
import hashlib
import warnings
from pathlib import Path
from typing import Optional
from contextlib import redirect_stdout
from threading import Thread
from queue import Queue, Empty

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 消除 numexpr/bottleneck 版本警告
warnings.filterwarnings("ignore", category=UserWarning, message=".*numexpr.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*bottleneck.*")
# 消除 VLM 加载时的提示：meta device 参数 offload、Qwen2VL 快速 image processor
warnings.filterwarnings("ignore", category=UserWarning, message=".*meta device.*offloaded.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Qwen2VLImageProcessor.*fast processor.*")
# 消除 boto3 Python 3.9 弃用提示（accelerate 依赖 boto3，2026 年后需 Python 3.10+）
try:
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="boto3.compat")
    if hasattr(warnings, "PythonDeprecationWarning"):
        warnings.filterwarnings("ignore", category=warnings.PythonDeprecationWarning, module="boto3.compat")
except Exception:
    pass

import gradio as gr
import json
import numpy as np
import shutil
import yaml

from datetime import datetime
from utils import ConfigLoader


def _embed_cache_key(input_path: Path, index: dict, embedding_provider: str, run_device: str) -> str:
    """计算嵌入缓存键：输入路径+图像路径集合+嵌入配置（使用路径而非 image_id，避免重启后扫描顺序变化导致索引错位）"""
    paths = tuple(sorted(index[k]["path"] for k in index))
    data = f"{input_path}|{embedding_provider}|{run_device}|{paths}"
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def get_latest_organized_dir() -> Optional[Path]:
    """返回最近的 organized 文件夹路径，不存在则返回 None"""
    project_root = Path(__file__).parent.parent
    output_base = project_root / "data" / "output"
    if not output_base.exists():
        return None
    candidates = []
    for p in output_base.iterdir():
        if p.is_dir() and not p.name.startswith("."):
            organized = p / "organized"
            if organized.exists() and organized.is_dir():
                candidates.append((organized.stat().st_mtime, organized))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def open_folder_in_explorer(path: Path) -> bool:
    """在系统文件管理器中打开文件夹，返回是否成功"""
    path = Path(path).resolve()
    if not path.exists() or not path.is_dir():
        return False
    path_str = str(path)
    try:
        if platform.system() == "Windows":
            os.startfile(path_str)
        elif platform.system() == "Darwin":
            subprocess.run(["open", path_str], check=False)
        else:
            subprocess.run(["xdg-open", path_str], check=False)
        return True
    except Exception:
        return False


def open_latest_organized() -> str:
    """打开最近的 organized 文件夹，返回状态消息"""
    latest = get_latest_organized_dir()
    if latest is None:
        return "❌ 未找到结果文件夹 (data/output/*/organized/)"
    if open_folder_in_explorer(latest):
        return f"✅ 已打开: {latest}"
    return f"❌ 无法打开: {latest}"


def get_gpu_status() -> str:
    """检测 PyTorch GPU 可用性，返回状态文本"""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "Unknown"
            count = torch.cuda.device_count()
            cuda_ver = torch.version.cuda or "N/A"
            return f"✅ GPU 可用 | {name} × {count} | CUDA {cuda_ver}"
        return "⚠️ GPU 不可用 (使用 CPU 模式)"
    except Exception as e:
        return f"❌ 检测失败: {e}"


# 默认参数（用于判断是否已修改）
UI_DEFAULTS = {
    "input_dir": "test_pics",
    "embedding_provider": "dinov2",
    "run_device": "cuda",
    "cluster_backend": "hdbscan",
    "min_samples": 2,
    "epsilon": 1.0,
    "dbscan_metric": "euclidean",
    "dbscan_algorithm": "auto",
    "cluster_selection_method": "leaf",
    "cluster_selection_epsilon": 0.0,
    "cluster_selection_persistence": 0.4,
    "batch_size": 16,
    "caption_mode": "representative",
    "top_k_sampling": 2,
    "vlm_model_scale": "small",
    "caption_batch_size": 4,
    "vlm_quantization": "none",
    "max_image_size": 512,
}
UI_PARAM_NAMES = (
    "input_dir", "embedding_provider", "run_device", "cluster_backend",
    "min_samples", "epsilon", "dbscan_metric", "dbscan_algorithm", "cluster_selection_method",
    "cluster_selection_epsilon", "cluster_selection_persistence",
    "batch_size", "caption_mode", "top_k_sampling", "vlm_model_scale", "caption_batch_size", "vlm_quantization", "max_image_size",
)
UI_PARAM_LABELS = {
    "input_dir": "输入目录",
    "embedding_provider": "特征模型",
    "run_device": "G10 运行设备 (嵌入+VLM)",
    "cluster_backend": "聚类算法",
    "min_samples": "Min Samples",
    "epsilon": "Epsilon",
    "dbscan_metric": "DBSCAN 度量",
    "dbscan_algorithm": "DBSCAN 算法",
    "cluster_selection_method": "簇选择方法",
    "cluster_selection_epsilon": "Cluster Sel Epsilon",
    "cluster_selection_persistence": "Cluster Sel Persistence",
    "batch_size": "批量大小",
    "caption_mode": "描述模式",
    "top_k_sampling": "D7 Top-K 采样",
    "vlm_model_scale": "D2 模型规模",
    "caption_batch_size": "D6 描述批量",
    "vlm_quantization": "D9 量化",
    "max_image_size": "D10 最大分辨率",
}


def get_config_choices() -> list:
    """扫描 config/*.yaml，返回配置名列表（排除 prompts.yaml），默认含 default_cfg"""
    project_root = Path(__file__).parent.parent
    config_dir = project_root / "config"
    choices = ["default_cfg"]
    if config_dir.exists():
        for f in sorted(config_dir.glob("*.yaml")):
            if f.name != "prompts.yaml" and f.stem not in choices:
                choices.append(f.stem)
    return choices


def reset_to_defaults() -> tuple:
    """返回默认参数元组，用于恢复默认配置"""
    return (
        UI_DEFAULTS["input_dir"],
        UI_DEFAULTS["embedding_provider"],
        UI_DEFAULTS["run_device"],
        UI_DEFAULTS["cluster_backend"],
        UI_DEFAULTS["min_samples"],
        UI_DEFAULTS["epsilon"],
        UI_DEFAULTS["dbscan_metric"],
        UI_DEFAULTS["dbscan_algorithm"],
        UI_DEFAULTS["cluster_selection_method"],
        UI_DEFAULTS["cluster_selection_epsilon"],
        UI_DEFAULTS["cluster_selection_persistence"],
        UI_DEFAULTS["batch_size"],
        UI_DEFAULTS["caption_mode"],
        UI_DEFAULTS["top_k_sampling"],
        UI_DEFAULTS["vlm_model_scale"],
        UI_DEFAULTS["caption_batch_size"],
        UI_DEFAULTS["vlm_quantization"],
        UI_DEFAULTS["max_image_size"],
        False,  # force_rerun_step1_2
        1,  # random_seed
        "<span style='font-size:0.9em'><b>当前已修改:</b> 无（均为默认）</span>",
    )


def save_ui_config(
    config_name,
    input_dir, embedding_provider, embedding_device, cluster_backend,
    min_samples, epsilon, dbscan_metric, dbscan_algorithm,
    cluster_selection_method, cluster_selection_epsilon, cluster_selection_persistence,
    batch_size, caption_mode, top_k_sampling, vlm_model_scale, caption_batch_size, vlm_quantization, max_image_size, run_device, force_rerun_step1_2, random_seed,
) -> str:
    """将当前 UI 参数保存到 config/{config_name}.yaml"""
    try:
        name = (config_name or "default_cfg").strip() or "default_cfg"
        project_root = Path(__file__).parent.parent
        config_dir = project_root / "config"
        target_path = config_dir / f"{name}.yaml"
        base_config = config_dir / "config.yaml"

        # 若目标文件不存在，从 config.yaml 加载作为模板
        load_path = target_path if target_path.exists() else base_config
        loader = ConfigLoader(config_path=str(load_path))

        input_path = Path((input_dir or "").strip())
        if not input_path.is_absolute():
            input_path = project_root / input_path
        loader.set("data.input_directory", str(input_path))

        loader.set("embedding.provider", embedding_provider)
        loader.set("embedding.device", run_device)
        backbone = "dinov2_vitl14" if embedding_provider == "dinov2" else "clip_vitb16"
        loader.set("embedding.backbone", backbone)
        loader.set("embedding.batch_size", int(batch_size))

        loader.set("clustering.backend", cluster_backend)
        loader.set("clustering.min_samples", int(min_samples))
        loader.set("clustering.epsilon", float(epsilon))
        loader.set("clustering.dbscan_metric", dbscan_metric)
        loader.set("clustering.dbscan_algorithm", dbscan_algorithm)
        loader.set("clustering.cluster_selection_method", cluster_selection_method)
        loader.set("clustering.min_cluster_size", None)
        loader.set("clustering.cluster_selection_epsilon", float(cluster_selection_epsilon))
        loader.set("clustering.cluster_selection_persistence", float(cluster_selection_persistence))
        loader.set("postprocessing.caption_mode", (caption_mode or "representative").strip() or "representative")
        loader.set("postprocessing.top_k_sampling", int(top_k_sampling) if top_k_sampling is not None else 2)
        loader.set("vlm.model_scale", (vlm_model_scale or "small").strip().lower() or "small")
        loader.set("vlm.caption_batch_size", int(caption_batch_size) if caption_batch_size is not None else 4)
        loader.set("vlm.quantization", (vlm_quantization or "none").strip().lower() or "none")
        loader.set("vlm.max_image_size", int(max_image_size) if max_image_size is not None else 512)
        loader.set("vlm.device", (run_device or "cuda").strip() or "cuda")
        loader.set("system.seed", int(random_seed) if random_seed is not None and int(random_seed) >= 0 else -1)

        loader.save_config(output_path=str(target_path))
        return f"✅ 已保存到 config/{name}.yaml"
    except Exception as e:
        return f"❌ 保存失败: {str(e)}"


def load_ui_config(
    config_name,
    input_dir, embedding_provider, embedding_device, cluster_backend,
    min_samples, epsilon, dbscan_metric, dbscan_algorithm,
    cluster_selection_method, cluster_selection_epsilon, cluster_selection_persistence,
    batch_size, caption_mode, top_k_sampling, vlm_model_scale, caption_batch_size, vlm_quantization, max_image_size, run_device, force_rerun_step1_2, random_seed,
) -> tuple:
    """从 config/{config_name}.yaml 读取配置并返回 UI 参数元组；失败时返回当前值并带错误信息"""
    fallback = (
        (input_dir or "").strip(), embedding_provider, (run_device or "cuda").strip() or "cuda", cluster_backend,
        int(min_samples), float(epsilon), dbscan_metric, dbscan_algorithm, cluster_selection_method,
        float(cluster_selection_epsilon), float(cluster_selection_persistence),
        int(batch_size), (caption_mode or "representative").strip() or "representative",
        int(top_k_sampling) if top_k_sampling is not None else 2,
        (vlm_model_scale or "small").strip().lower() or "small",
        int(caption_batch_size) if caption_batch_size is not None else 4,
        (vlm_quantization or "none").strip().lower() or "none",
        int(max_image_size) if max_image_size is not None else 512,
        bool(force_rerun_step1_2), int(random_seed) if random_seed is not None else 1,
    )
    fallback_hint = get_modified_hint(*fallback)

    try:
        name = (config_name or "default_cfg").strip() or "default_cfg"
        project_root = Path(__file__).parent.parent
        target_path = project_root / "config" / f"{name}.yaml"
        if not target_path.exists() and name == "default_cfg":
            target_path = project_root / "config" / "config.yaml"
        if not target_path.exists():
            return fallback + (fallback_hint,), f"❌ 配置文件不存在: config/{name}.yaml"

        loader = ConfigLoader(config_path=str(target_path))

        input_dir_raw = loader.get("data.input_directory", "test_pics") or "test_pics"
        try:
            inp = Path(input_dir_raw)
            if inp.is_absolute() and str(inp).startswith(str(project_root)):
                input_dir = str(inp.relative_to(project_root))
            else:
                input_dir = input_dir_raw
        except Exception:
            input_dir = input_dir_raw

        values = (
            input_dir,
            loader.get("embedding.provider", "dinov2"),
            loader.get("embedding.device", "cuda") or loader.get("vlm.device", "cuda") or "cuda",
            loader.get("clustering.backend", "hdbscan"),
            int(loader.get("clustering.min_samples", 2)),
            float(loader.get("clustering.epsilon", 1.0)),
            loader.get("clustering.dbscan_metric", "euclidean"),
            loader.get("clustering.dbscan_algorithm", "auto"),
            loader.get("clustering.cluster_selection_method", "leaf"),
            float(loader.get("clustering.cluster_selection_epsilon", 0.0)),
            float(loader.get("clustering.cluster_selection_persistence", 0.4)),
            int(loader.get("embedding.batch_size", 16)),
            loader.get("postprocessing.caption_mode", "representative") or "representative",
            int(loader.get("postprocessing.top_k_sampling", 2)),
            (loader.get("vlm.model_scale", "small") or "small").strip().lower(),
            int(loader.get("vlm.caption_batch_size", 4)),
            (loader.get("vlm.quantization", "none") or "none").strip().lower(),
            int(loader.get("vlm.max_image_size", 512)),
            False,  # force_rerun_step1_2 不保存到配置，加载时默认不勾选
            int(loader.get("system.seed", 1)),
        )
        hint = get_modified_hint(*values)
        return values + (hint,), f"✅ 已加载 config/{name}.yaml"
    except Exception as e:
        return fallback + (fallback_hint,), f"❌ 读取失败: {str(e)}"


def get_modified_hint(
    input_dir,
    embedding_provider,
    run_device,
    cluster_backend,
    min_samples,
    epsilon,
    dbscan_metric,
    dbscan_algorithm,
    cluster_selection_method,
    cluster_selection_epsilon,
    cluster_selection_persistence,
    batch_size,
    caption_mode,
    top_k_sampling,
    vlm_model_scale,
    caption_batch_size,
    vlm_quantization,
    max_image_size,
    force_rerun_step1_2,
    random_seed,
) -> str:
    """返回一行提示：当前已修改的参数（橙色标记）"""
    values = {
        "input_dir": (input_dir or "").strip(),
        "embedding_provider": embedding_provider,
        "run_device": (run_device or "cuda").strip() or "cuda",
        "cluster_backend": cluster_backend,
        "min_samples": int(min_samples),
        "epsilon": float(epsilon),
        "dbscan_metric": dbscan_metric,
        "dbscan_algorithm": dbscan_algorithm,
        "cluster_selection_method": cluster_selection_method,
        "cluster_selection_epsilon": float(cluster_selection_epsilon),
        "cluster_selection_persistence": float(cluster_selection_persistence),
        "batch_size": int(batch_size),
        "caption_mode": (caption_mode or "representative").strip() or "representative",
        "top_k_sampling": int(top_k_sampling) if top_k_sampling is not None else 2,
        "vlm_model_scale": (vlm_model_scale or "small").strip().lower() or "small",
        "caption_batch_size": int(caption_batch_size) if caption_batch_size is not None else 4,
        "vlm_quantization": (vlm_quantization or "none").strip().lower() or "none",
        "max_image_size": int(max_image_size) if max_image_size is not None else 512,
    }
    modified = [UI_PARAM_LABELS[k] for k in UI_PARAM_NAMES if values[k] != UI_DEFAULTS[k]]
    if not modified:
        return "<span style='font-size:0.9em'><b>当前已修改:</b> 无（均为默认）</span>"
    return "<span style='font-size:0.9em'><b>当前已修改:</b> <span style='color:#e65100;font-weight:bold'>" + "、".join(modified) + "</span></span>"


from core.step0_indexing import run_step0
from core.step1_embedding import run_step1
from core.step2_clustering import run_step2
from core.step3_sampling import run_step3
from core.step4_caption import run_step4
from core.step5_label import run_step5
from core.step8_organization import run_step8


class SemanticClusterApp:
    """
    Semantic Cluster Web应用
    """
    
    def __init__(self):
        """初始化应用"""
        self.config_loader = ConfigLoader()
        project_root = Path(__file__).parent.parent
        self.output_base = project_root / "data" / "output"
        self.output_base.mkdir(parents=True, exist_ok=True)
    
    def run_pipeline(
        self,
        input_dir: str,
        embedding_provider: str,
        run_device: str,
        cluster_backend: str,
        min_samples: int,
        epsilon: float,
        dbscan_metric: str,
        dbscan_algorithm: str,
        cluster_selection_method: str,
        cluster_selection_epsilon: float,
        cluster_selection_persistence: float,
        batch_size: int,
        caption_mode: str,
        top_k_sampling: int,
        vlm_model_scale: str,
        caption_batch_size: int,
        vlm_quantization: str,
        max_image_size: int = 512,
        force_rerun_step1_2: bool = False,
        random_seed: int = 1,
    ):
        """
        运行完整流程
        
        Args:
            input_dir: 输入目录
            cluster_backend: 聚类算法 hdbscan/dbscan
            min_samples: 最小样本数
            epsilon: DBSCAN距离阈值
            cluster_selection_method: HDBSCAN簇选择方法
            batch_size: 批量大小
            
        Returns:
            结果消息和统计信息
        """
        try:
            # 验证输入 - 支持相对路径（如test_pics）从项目根解析
            input_path = Path(input_dir.strip())
            if not input_path.is_absolute():
                project_root = Path(__file__).parent.parent
                input_path = project_root / input_path
            if not input_dir or not input_path.exists():
                yield 0, f"❌ 错误: 输入目录不存在！({input_path})", "", ""
                return
            
            # 创建会话输出目录
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = self.output_base / session_id
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 更新配置
            self.config_loader.set("data.input_directory", str(input_path))
            self.config_loader.set("embedding.provider", embedding_provider)
            self.config_loader.set("embedding.device", run_device)
            backbone = "dinov2_vitl14" if embedding_provider == "dinov2" else "clip_vitb16"
            self.config_loader.set("embedding.backbone", backbone)
            self.config_loader.set("clustering.backend", cluster_backend)
            self.config_loader.set("clustering.min_samples", min_samples)
            self.config_loader.set("clustering.epsilon", epsilon)
            self.config_loader.set("clustering.dbscan_metric", dbscan_metric)
            self.config_loader.set("clustering.dbscan_algorithm", dbscan_algorithm)
            self.config_loader.set("clustering.cluster_selection_method", cluster_selection_method)
            self.config_loader.set("clustering.min_cluster_size", None)
            self.config_loader.set("clustering.cluster_selection_epsilon", cluster_selection_epsilon)
            self.config_loader.set("clustering.cluster_selection_persistence", cluster_selection_persistence)
            self.config_loader.set("embedding.batch_size", batch_size)
            self.config_loader.set("postprocessing.caption_mode", (caption_mode or "representative").strip() or "representative")
            self.config_loader.set("postprocessing.top_k_sampling", int(top_k_sampling) if top_k_sampling is not None else 2)
            self.config_loader.set("vlm.model_scale", (vlm_model_scale or "small").strip().lower() or "small")
            self.config_loader.set("vlm.caption_batch_size", int(caption_batch_size) if caption_batch_size is not None else 4)
            self.config_loader.set("vlm.quantization", (vlm_quantization or "none").strip().lower() or "none")
            self.config_loader.set("vlm.max_image_size", int(max_image_size) if max_image_size is not None else 512)
            self.config_loader.set("vlm.device", (run_device or "cuda").strip() or "cuda")
            seed_val = int(random_seed) if random_seed is not None else 1
            self.config_loader.set("system.seed", seed_val if seed_val >= 0 else -1)
            config = self.config_loader.to_dict()
            
            # 控制随机性：seed>=0 时设置全局种子，-1 则随机
            if seed_val >= 0:
                import numpy as np
                np.random.seed(seed_val)
                try:
                    import torch
                    torch.manual_seed(seed_val)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed_val)
                except ImportError:
                    pass
            
            # 保存完整配置到输出目录
            config_path = output_dir / "run_config.yaml"
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            results = []
            stats_summary = []
            cluster_info = []
            current_step = 0  # 0=等待, 1=Step-0, 2=Step-1, 3=Step-2, 4=完成
            
            def _yield():
                return current_step, "\n".join(results), "\n".join(stats_summary), "\n".join(cluster_info)
            
            def _run_captured(fn, *args, **kwargs):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    return fn(*args, **kwargs), buf.getvalue().strip()
            
            # Step-0: 索引
            current_step = 1
            results.append("=" * 60)
            results.append("Step-0: 图像索引")
            results.append("=" * 60)
            results.append(f"📋 配置已保存: {config_path.name}")
            if seed_val >= 0:
                results.append(f"🎲 随机种子: {seed_val} (可复现)")
            else:
                results.append(f"🎲 随机种子: -1 (随机)")
            results.append("")
            yield _yield()
            
            try:
                index_path, log_out = _run_captured(run_step0, config, output_dir)
                if log_out:
                    results.append(log_out)
                with open(index_path, 'r', encoding='utf-8') as f:
                    index = json.load(f)
                results.append(f"✅ 索引完成: {len(index)} 张图像")
                stats_summary.append(f"总图像数: {len(index)}")
            except Exception as e:
                results.append(f"❌ 索引失败: {str(e)}")
                yield _yield()
                return
            yield _yield()

            # 检查 A/B 未改时是否可复用嵌入缓存
            project_root = Path(__file__).parent.parent
            cache_base = project_root / "data" / ".cache" / "embedding"
            cache_key = _embed_cache_key(input_path, index, embedding_provider, run_device)
            cache_dir = cache_base / cache_key
            features_path = output_dir / "S1_embeddings.npy"
            image_ids_path = output_dir / "S1_image_ids.json"
            stats_path = output_dir / "S1_stats.json"
            cache_hit = (
                not force_rerun_step1_2
                and (cache_dir / "S1_embeddings.npy").exists()
                and (cache_dir / "S1_image_ids.json").exists()
                and (cache_dir / "S1_paths.json").exists()  # 需要路径映射才能正确 remap
            )

            # Step-1: 嵌入（缓存命中则跳过）
            current_step = 2
            results.append("\n" + "=" * 60)
            results.append("Step-1: 特征嵌入")
            results.append("=" * 60)
            if cache_hit:
                results.append("♻️ A/B 未改，复用上次嵌入结果...")
                yield _yield()
                try:
                    # 按路径 remap，避免重启后扫描顺序变化导致索引错位
                    cached_embeddings = np.load(cache_dir / "S1_embeddings.npy")
                    with open(cache_dir / "S1_paths.json", 'r', encoding='utf-8') as f:
                        cached_paths = json.load(f)
                    path_to_embedding = {p: cached_embeddings[i] for i, p in enumerate(cached_paths)}
                    # 检查当前 index 的路径是否都在缓存中
                    current_paths = {index[k]["path"] for k in index}
                    if current_paths != set(path_to_embedding.keys()):
                        raise ValueError(
                            "缓存路径与当前索引不一致（图像可能已变更），将重新计算嵌入"
                        )
                    # 按当前 index 路径排序，保证与 S0_image_index.json 一致
                    sorted_ids = sorted(index.keys(), key=lambda k: index[k]["path"])
                    new_features = np.array([path_to_embedding[index[k]["path"]] for k in sorted_ids])
                    new_image_ids = sorted_ids
                    np.save(features_path, new_features)
                    with open(image_ids_path, 'w', encoding='utf-8') as f:
                        json.dump(new_image_ids, f, indent=2)
                    if (cache_dir / "S1_stats.json").exists():
                        shutil.copy2(cache_dir / "S1_stats.json", stats_path)
                    else:
                        stats = {'feature_dim': new_features.shape[1]}
                        with open(stats_path, 'w', encoding='utf-8') as f:
                            json.dump(stats, f, indent=2)
                    with open(stats_path, 'r') as f:
                        stats = json.load(f)
                    results.append(f"✅ 复用完成 | 维度: {stats['feature_dim']}")
                    stats_summary.append(f"特征维度: {stats['feature_dim']}")
                except Exception as e:
                    results.append(f"⚠️ 复用失败，重新计算: {e}")
                    cache_hit = False
            if not cache_hit:
                results.append("[1/3] 加载视觉模型...")
                yield _yield()
                results.append("[2/3] 提取特征中（请稍候）...")
                yield _yield()
                try:
                    step1_queue = Queue()
                    result_holder = [None]

                    def step1_progress(batch_idx, total_batches, n_done, n_total):
                        step1_queue.put(("progress", batch_idx, total_batches, n_done, n_total))

                    def step1_thread():
                        try:
                            buf = io.StringIO()
                            with redirect_stdout(buf):
                                path = run_step1(config, index_path, output_dir, progress_callback=step1_progress)
                            result_holder[0] = (path, buf.getvalue().strip(), None)
                        except Exception as e:
                            result_holder[0] = (None, None, e)
                        step1_queue.put(("done",))

                    t = Thread(target=step1_thread)
                    t.start()

                    while t.is_alive():
                        try:
                            msg = step1_queue.get(timeout=0.3)
                            if msg[0] == "done":
                                break
                            _, batch_idx, total, n_done, n_total = msg
                            results[-1] = f"[2/3] 已处理 {n_done}/{n_total} 张图像 (batch {batch_idx}/{total})"
                            yield _yield()
                        except Empty:
                            yield _yield()
                    t.join()

                    features_path, log_out, err = result_holder[0]
                    if err is not None:
                        raise err
                    results[-1] = "[2/3] 提取特征完成"
                    if log_out:
                        results.append("")
                        results.append(log_out)
                    stats_path = output_dir / "S1_stats.json"
                    with open(stats_path, 'r') as f:
                        stats = json.load(f)
                    results.append(f"✅ 特征提取完成 | 维度: {stats['feature_dim']}")
                    stats_summary.append(f"特征维度: {stats['feature_dim']}")
                    # 写入嵌入缓存供下次 A/B 未改时复用（含 S1_paths.json 供 remap）
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(output_dir / "S1_embeddings.npy", cache_dir / "S1_embeddings.npy")
                    shutil.copy2(output_dir / "S1_image_ids.json", cache_dir / "S1_image_ids.json")
                    shutil.copy2(output_dir / "S1_stats.json", cache_dir / "S1_stats.json")
                    paths_path = output_dir / "S1_paths.json"
                    if paths_path.exists():
                        shutil.copy2(paths_path, cache_dir / "S1_paths.json")
                except Exception as e:
                    results.append(f"❌ 特征提取失败: {str(e)}")
                    yield _yield()
                    return
            yield _yield()
            
            # Step-2: 聚类
            current_step = 3
            results.append("\n" + "=" * 60)
            results.append("Step-2: 聚类")
            results.append("=" * 60)
            
            try:
                clustering_path, log_out = _run_captured(
                    run_step2,
                    config, features_path,
                    output_dir / "S1_image_ids.json",
                    output_dir
                )
                if log_out:
                    results.append(log_out)
                stats_path = output_dir / "S2_stats.json"
                with open(stats_path, 'r') as f:
                    stats = json.load(f)
                results.append(f"✅ 聚类完成")
                results.append(f"   簇数量: {stats['n_clusters']} | 噪音: {stats['n_noise']} ({stats['noise_ratio']:.1f}%)")
                stats_summary.append(f"簇数量: {stats['n_clusters']}")
                stats_summary.append(f"噪音图像: {stats['n_noise']} ({stats['noise_ratio']:.1f}%)")
                cluster_info = [f"簇 {cid}: {size}张" for cid, size in stats['cluster_sizes'].items()]
            except Exception as e:
                results.append(f"❌ 聚类失败: {str(e)}")
                yield _yield()
                return
            yield _yield()

            # Phase-3: Step-3 多点采样 → Step-4 并行描述 → Step-5 语义蒸馏（D2=跳过 时省略，用簇序号命名）
            labels_path = None
            vlm_scale_val = (vlm_model_scale or "small").strip().lower()
            if vlm_scale_val == "skip":
                results.append("\n" + "=" * 60)
                results.append("跳过 Step-3/4/5（D2=跳过，使用簇序号作为标签）")
                results.append("=" * 60)
                yield _yield()
            else:
                try:
                    results.append("\n" + "=" * 60)
                    results.append("Step-3: 多点采样")
                    results.append("=" * 60)
                    sampled_path, log_out = _run_captured(
                        run_step3,
                        config,
                        output_dir / "S1_embeddings.npy",
                        output_dir / "S1_image_ids.json",
                        clustering_path,
                        output_dir,
                    )
                    if log_out:
                        results.append(log_out)
                    results.append("✅ 采样完成")
                    yield _yield()

                    results.append("\n" + "=" * 60)
                    results.append("Step-4: 并行描述")
                    results.append("=" * 60)
                    caption_mode_val = (caption_mode or "representative").strip() or "representative"
                    self.config_loader.set("postprocessing.caption_mode", caption_mode_val)
                    config = self.config_loader.to_dict()
                    results.append("Step-4: 正在描述… 0/?")
                    yield _yield()
                    step4_queue = Queue()
                    step4_holder = [None]

                    def step4_progress(n: int, total: int, _iid: str):
                        step4_queue.put(("progress", n, total))

                    def step4_thread():
                        try:
                            buf = io.StringIO()
                            with redirect_stdout(buf):
                                path = run_step4(
                                    config,
                                    index_path,
                                    clustering_path,
                                    output_dir,
                                    mode=caption_mode_val,
                                    sampled_path=sampled_path,
                                    progress_callback=step4_progress,
                                )
                            step4_holder[0] = (path, buf.getvalue().strip(), None)
                        except Exception as e:
                            step4_holder[0] = (None, None, e)
                        step4_queue.put(("done",))

                    t4 = Thread(target=step4_thread)
                    t4.start()
                    while t4.is_alive():
                        try:
                            msg = step4_queue.get(timeout=0.3)
                            if msg[0] == "done":
                                break
                            _, n, total = msg
                            pct = (100 * n // total) if total else 0
                            results[-1] = f"Step-4: 已描述 {n}/{total} ({pct}%)"
                            yield _yield()
                        except Empty:
                            yield _yield()
                    t4.join()
                    _, log_out, err4 = step4_holder[0]
                    if err4 is not None:
                        raise err4
                    results[-1] = "Step-4: 并行描述完成"
                    if log_out:
                        results.append("")
                        results.append(log_out)
                    results.append("✅ 描述完成")
                    yield _yield()

                    results.append("\n" + "=" * 60)
                    results.append("Step-5: 语义蒸馏")
                    results.append("=" * 60)
                    labels_path, log_out = _run_captured(
                        run_step5,
                        config,
                        output_dir / "S4_captions.json",
                        sampled_path,
                        output_dir,
                    )
                    if log_out:
                        results.append(log_out)
                    results.append("✅ 簇标签完成")
                    yield _yield()
                except Exception as e:
                    results.append(f"⚠️ Phase-3 某步失败（继续用无标签整理）: {str(e)}")
                    labels_path = None
                    yield _yield()
            
            # Step-8: 整理（有 S5 时使用语义标签命名）
            results.append("\n" + "=" * 60)
            results.append("Step-8: 文件整理")
            results.append("=" * 60)
            
            try:
                organized_dir = output_dir / "organized"
                _, log_out = _run_captured(
                    run_step8,
                    config, index_path, clustering_path, output_dir, organized_dir,
                    dry_run=False,
                    labels_path=labels_path,
                )
                if log_out:
                    results.append(log_out)
                results.append(f"✅ 文件整理完成 | 输出: {organized_dir}")
            except Exception as e:
                results.append(f"❌ 文件整理失败: {str(e)}")
                yield _yield()
                return
            yield _yield()
            
            # 完成
            current_step = 4
            results.append("\n" + "=" * 60)
            results.append("🎉 所有步骤完成!")
            results.append("=" * 60)
            results.append(f"\n会话ID: {session_id}")
            results.append(f"输出目录: {output_dir}")
            yield _yield()
            
        except Exception as e:
            import traceback
            error_msg = f"❌ 发生错误:\n{str(e)}\n\n{traceback.format_exc()}"
            yield 0, error_msg, "", ""


def create_ui():
    """
    创建Gradio界面
    """
    app = SemanticClusterApp()
    
    # L1=主节 L2=子节 L3=单参数 - 按级别调整字体、框大小、间隔
    PARAM_BOX_CSS = """
    #config-column { font-size: 0.9rem; }
    #config-column .gr-form, #config-column .gr-box { min-height: auto !important; }

    /* L1: 主节 - 最大字体、大框、窄间隔 */
    #config-column .param-l1 { margin-bottom: 6px; }
    #config-column .param-l1 > .wrap { margin-bottom: 3px !important; }
    #config-column .param-l1 button.label-wrap {
        font-size: 0.98rem !important; font-weight: 600 !important;
        padding: 5px 8px !important; min-height: 32px !important;
        display: flex !important; flex-direction: row !important;
        justify-content: space-between !important; width: 100% !important;
        user-select: text !important; -webkit-user-select: text !important;
    }
    #config-column .param-l1 .gr-form { padding: 2px 0 2px 6px !important; }
    #config-column .param-l1 .param-l2 { margin-bottom: 4px !important; }

    /* L2: 子节 - 中字体、中框、窄间隔，外框黑边 */
    #config-column .param-l2 {
        margin-bottom: 4px;
        border: 2px solid #0d1b4d;
        border-radius: 4px;
        background: rgba(13, 27, 77, 0.8);
        padding: 3px 4px;
    }
    #config-column .param-l2 > .wrap { margin-bottom: 2px !important; }
    #config-column .param-l2 button.label-wrap {
        font-size: 0.88rem !important; font-weight: 500 !important;
        padding: 4px 6px !important; min-height: 28px !important;
        display: flex !important; flex-direction: row !important;
        justify-content: space-between !important; width: 100% !important;
        user-select: text !important; -webkit-user-select: text !important;
    }
    #config-column .param-l2 .gr-form { padding: 1px 0 1px 4px !important; }
    #config-column .param-l2 .param-l3 { margin-bottom: 2px !important; }

    /* L3: 单参数 - 小字体、紧凑框、窄间隔 */
    #config-column .param-l3 { margin-bottom: 2px; }
    #config-column .param-l3 .wrap, #config-column .param-l3 > div { margin-bottom: 0 !important; }
    #config-column .param-l3 label { font-size: 0.82rem !important; font-weight: 500 !important; }
    #config-column .param-l3 .gr-form, #config-column .param-l3 .gr-box {
        margin-bottom: 1px !important; min-height: auto !important; padding: 1px 0 !important;
    }
    #config-column .param-l3 input:not([type="range"]), #config-column .param-l3 select,
    #config-column .param-l3 textarea {
        font-size: 0.82rem !important; padding: 4px 6px !important; min-height: 28px !important;
    }
    #config-column .param-l3 .block-info { font-size: 0.74rem !important; }
    #config-column .param-l3 .wrap.slider { align-items: center !important; }
    #config-column .param-l3 input[type="range"] { min-height: unset !important; padding: 0 !important; }

    /* L3 内嵌套在 L2 中时更紧凑 */
    #config-column .param-l2 .param-l3 label { font-size: 0.8rem !important; }
    #config-column .param-l2 .param-l3 input:not([type="range"]), #config-column .param-l2 .param-l3 select,
    #config-column .param-l2 .param-l3 textarea {
        font-size: 0.8rem !important; padding: 3px 5px !important; min-height: 26px !important;
    }
    #config-column .param-l2 .param-l3 input[type="range"] { min-height: unset !important; padding: 0 !important; }
    """
    # 用 gr.Button.click(js=...) 在应用上下文中执行，Gradio Accordion 为 button.label-wrap，展开时有 .open
    JS_COLLAPSE_ALL = """() => {
        var root = document.getElementById('config-column') || document.body;
        var toggles = root.querySelectorAll('button.label-wrap.open');
        if (!toggles.length) toggles = root.querySelectorAll('[aria-expanded="true"]');
        Array.from(toggles).forEach((el, i) => setTimeout(() => el.click(), i * 100));
    }"""
    JS_EXPAND_ALL = """() => {
        var root = document.getElementById('config-column') || document.body;
        var toggles = root.querySelectorAll('button.label-wrap:not(.open)');
        if (!toggles.length) toggles = root.querySelectorAll('[aria-expanded="false"]');
        Array.from(toggles).forEach((el, i) => setTimeout(() => el.click(), i * 100));
    }"""
    with gr.Blocks(title="Semantic Cluster WebUI", theme=gr.themes.Soft(), css=PARAM_BOX_CSS) as demo:
        gr.Markdown("""
        # 🎨 Semantic Cluster WebUI
        
        索引（indexing） → 嵌入（embedding） → 聚类（clustering） → 按簇整理：采样（sampling）/描述（captioning）
        """)
        
        with gr.Row():
            with gr.Column(scale=1, elem_id="config-column"):
                gr.Markdown("### ⚙️ 配置参数")
                with gr.Row():
                    btn_collapse_all = gr.Button("折叠所有", size="sm", scale=0)
                    btn_expand_all = gr.Button("展开所有", size="sm", scale=0)
                btn_collapse_all.click(None, None, None, js=JS_COLLAPSE_ALL)
                btn_expand_all.click(None, None, None, js=JS_EXPAND_ALL)
                with gr.Group(elem_classes=["param-box", "param-l3"]):
                    gpu_status = gr.Markdown(
                        value=f"**GPU 状态**: {get_gpu_status()}",
                        elem_id="gpu-status"
                    )
                with gr.Accordion("A. 数据源", open=False, elem_classes=["param-l1"]):
                    with gr.Group(elem_classes=["param-box", "param-l3"]):
                        input_dir = gr.Textbox(
                            label="A1 输入目录 (默认: test_pics)",
                            placeholder="例如: D:/images 或使用test_pics测试",
                            value="test_pics",
                            info="待聚类的图像所在目录，支持相对路径（如 test_pics）"
                        )
                    with gr.Accordion("固定项", open=False, elem_classes=["param-l2"]):
                        with gr.Group(elem_classes=["param-box", "param-l3"]):
                            gr.Textbox(label="A2 支持格式", value="jpg, jpeg, png, webp, bmp, tiff", interactive=False)
                        with gr.Group(elem_classes=["param-box", "param-l3"]):
                            gr.Textbox(label="A3 最小文件大小", value="0（不过滤）", interactive=False)
                        with gr.Group(elem_classes=["param-box", "param-l3"]):
                            gr.Textbox(label="A4 最大文件大小", value="-1（无限制）", interactive=False)
                        with gr.Group(elem_classes=["param-box", "param-l3"]):
                            gr.Textbox(label="A5 排除文件夹", value="（空）", interactive=False)
                
                with gr.Accordion("B. 嵌入", open=False, elem_classes=["param-l1"]):
                    with gr.Group(elem_classes=["param-box", "param-l3"]):
                        embedding_provider = gr.Dropdown(
                            choices=[("DINOv2 (推荐)", "dinov2"), ("CLIP", "clip")],
                            value="dinov2",
                            label="B1 特征模型 (默认: dinov2)",
                            info="DINOv2 视觉语义更强、聚类更稳；CLIP 支持图文对齐，适合有标签场景"
                        )
                    with gr.Group(elem_classes=["param-box", "param-l3"]):
                        batch_size = gr.Slider(
                            minimum=4,
                            maximum=64,
                            value=16,
                            step=4,
                            label="B4 批量大小 (默认: 16)",
                            info="越大越快但占更多显存；显存不足时可降到 4–8"
                        )
                    with gr.Accordion("固定项", open=False, elem_classes=["param-l2"]):
                        with gr.Group(elem_classes=["param-box", "param-l3"]):
                            gr.Textbox(label="B2 Backbone", value="dinov2_vitl14 / clip_vitb16（由 B1 决定）", interactive=False)
                        with gr.Group(elem_classes=["param-box", "param-l3"]):
                            gr.Textbox(label="B5 PCA Components", value="256", interactive=False)
                
                with gr.Accordion("C. 聚类", open=False, elem_classes=["param-l1"]):
                    with gr.Accordion("通用", open=False, elem_classes=["param-l2"]):
                        with gr.Group(elem_classes=["param-box", "param-l3"]):
                            cluster_backend = gr.Dropdown(
                                choices=[("HDBSCAN (默认)", "hdbscan"), ("DBSCAN", "sklearn")],
                                value="hdbscan",
                                label="C1 聚类算法 (默认: HDBSCAN)",
                                info="HDBSCAN 自动发现簇数量、噪音少；DBSCAN 需手动调 Epsilon，簇数更可控"
                            )
                        with gr.Group(elem_classes=["param-box", "param-l3"]):
                            min_samples = gr.Slider(
                                minimum=2,
                                maximum=30,
                                value=2,
                                step=1,
                                label="C6 Min Samples (默认: 2)",
                                info="越高簇越「紧密」、数量越少、噪音越多；越低簇越多、噪音越少"
                            )
                    with gr.Accordion("HDBSCAN", open=False, elem_classes=["param-l2"]):
                        with gr.Group(elem_classes=["param-box", "param-l3"]):
                            cluster_selection_method = gr.Dropdown(
                                choices=[("leaf - 噪音少", "leaf"), ("eom - 簇少", "eom")],
                                value="leaf",
                                label="C7 簇选择方法 (默认: leaf)",
                                info="leaf 细粒度、簇多噪音少；eom 保守、簇少但噪音数往往更多"
                            )
                        with gr.Group(elem_classes=["param-box", "param-l3"]):
                            cluster_selection_epsilon = gr.Slider(
                                minimum=0.0,
                                maximum=0.5,
                                value=0.0,
                                step=0.05,
                                label="C9 Cluster Sel Epsilon (默认: 0.0)",
                                info="两簇距离小于此值会合并；越大簇越少；0-0.5 范围"
                            )
                        with gr.Group(elem_classes=["param-box", "param-l3"]):
                            cluster_selection_persistence = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.4,
                                step=0.1,
                                label="C10 Cluster Sel Persistence (默认: 0.4)",
                                info="簇在层次树中的存活长度阈值：越高簇越少；建议 0.2-0.4"
                            )
                    with gr.Accordion("DBSCAN", open=False, elem_classes=["param-l2"]):
                        with gr.Group(elem_classes=["param-box", "param-l3"]):
                            epsilon = gr.Slider(
                                minimum=0.5,
                                maximum=1.5,
                                value=1.0,
                                step=0.05,
                                label="C4 Epsilon (默认: 1.0)",
                                info="DBSCAN 邻域半径：越大簇越少越大；越小簇越多越小，噪音可能增多"
                            )
                        with gr.Group(elem_classes=["param-box", "param-l3"]):
                            dbscan_metric = gr.Dropdown(
                                choices=[("euclidean（欧氏）", "euclidean"), ("cosine（余弦）", "cosine")],
                                value="euclidean",
                                label="C4b 距离度量 (默认: euclidean)",
                                info="L2 归一化特征可试 cosine；euclidean 通用"
                            )
                        with gr.Group(elem_classes=["param-box", "param-l3"]):
                            dbscan_algorithm = gr.Dropdown(
                                choices=[("auto（自动）", "auto"), ("ball_tree", "ball_tree"), ("kd_tree", "kd_tree"), ("brute（暴力）", "brute")],
                                value="auto",
                                label="C4c 最近邻算法 (默认: auto)",
                                info="影响速度：大数据集可试 ball_tree/kd_tree"
                            )
                    with gr.Accordion("固定项", open=False, elem_classes=["param-l2"]):
                        with gr.Group(elem_classes=["param-box", "param-l3"]):
                            gr.Textbox(label="C2 距离度量", value="euclidean", interactive=False)
                        with gr.Group(elem_classes=["param-box", "param-l3"]):
                            gr.Textbox(label="C3 聚类模式", value="fixed_eps", interactive=False)
                        with gr.Group(elem_classes=["param-box", "param-l3"]):
                            gr.Textbox(label="C5 最大噪音比例", value="20 %", interactive=False)
                        with gr.Group(elem_classes=["param-box", "param-l3"]):
                            gr.Textbox(label="C11 Alpha", value="1.0", interactive=False)
                
                with gr.Accordion("D. VLM", open=False, elem_classes=["param-l1"]):
                    with gr.Group(elem_classes=["param-box", "param-l3"]):
                        caption_mode = gr.Dropdown(
                            choices=[
                                ("代表图（模式1，需 Step-3 采样）", "representative"),
                                ("全部图（模式2，跳过 Step-3）", "all"),
                            ],
                            value="representative",
                            label="D5 描述模式 (Step-4)",
                            info="模式1 仅描述代表图；模式2 描述全部图像"
                        )
                    with gr.Group(elem_classes=["param-box", "param-l3"]):
                        vlm_model_scale = gr.Dropdown(
                            choices=[
                                ("2B (快，默认)", "small"),
                                ("7B (准)", "large"),
                                ("跳过（用簇序号）", "skip"),
                            ],
                            value="small",
                            label="D2 模型规模 (Step-4/5)",
                            info="小/大模型用于描述与标签；选「跳过」则跳过 Step-3/4/5，直接用簇序号命名"
                        )
                    with gr.Group(elem_classes=["param-box", "param-l3"]):
                        caption_batch_size = gr.Slider(
                            minimum=1,
                            maximum=16,
                            value=4,
                            step=1,
                            label="D6 描述批量 (Caption Batch Size)",
                            info="Step-4 每批图像数，默认 4"
                        )
                    with gr.Group(elem_classes=["param-box", "param-l3"]):
                        max_image_size = gr.Number(
                            value=512,
                            minimum=0,
                            maximum=2048,
                            step=64,
                            label="D10 最大分辨率 (长边像素)",
                            info="描述前长边缩至此像素以加速；0=不缩小，默认 512"
                        )
                    with gr.Group(elem_classes=["param-box", "param-l3"]):
                        top_k_sampling = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=2,
                            step=1,
                            label="D7 Top-K 采样 (原 E2，每簇代表图数)",
                            info="Step-3 每簇采样张数，默认 2；仅代表图模式时生效"
                        )
                    with gr.Group(elem_classes=["param-box", "param-l3"]):
                        vlm_quantization = gr.Dropdown(
                            choices=[
                                ("无", "none"),
                                ("int8", "int8"),
                                ("int4", "int4"),
                            ],
                            value="none",
                            label="D9 量化 (int8/int4)",
                            info="需安装 bitsandbytes，仅 CUDA；省显存、可提速"
                        )
                    with gr.Accordion("固定项", open=False, elem_classes=["param-l2"]):
                        with gr.Group(elem_classes=["param-box", "param-l3"]):
                            gr.Textbox(label="D1 Provider", value="local_qwen2vl", interactive=False)
                        with gr.Group(elem_classes=["param-box", "param-l3"]):
                            gr.Textbox(label="D2 Model Name", value="由 D2 规模决定 2B/7B", interactive=False)
                        with gr.Group(elem_classes=["param-box", "param-l3"]):
                            gr.Textbox(label="D4 API Key", value="（未使用）", interactive=False)
                        with gr.Group(elem_classes=["param-box", "param-l3"]):
                            gr.Textbox(label="D8 Sampling Strategy (原 E1)", value="nearest", interactive=False)
                
                with gr.Accordion("E. 后处理", open=False, elem_classes=["param-l1"]):
                    with gr.Accordion("固定项", open=False, elem_classes=["param-l2"]):
                        with gr.Group(elem_classes=["param-box", "param-l3"]):
                            gr.Textbox(label="E3/E4 Caption/Label Prompt", value="（见 config）", interactive=False)
                        with gr.Group(elem_classes=["param-box", "param-l3"]):
                            gr.Textbox(label="E5/E6 Caption/Label Length", value="50 / 5-10", interactive=False)
                        with gr.Group(elem_classes=["param-box", "param-l3"]):
                            gr.Textbox(label="E7 Rescue Threshold", value="0.60", interactive=False)
                        with gr.Group(elem_classes=["param-box", "param-l3"]):
                            gr.Textbox(label="E8 Similarity Algorithm", value="cosine", interactive=False)
                
                with gr.Accordion("F. 输出", open=False, elem_classes=["param-l1"]):
                    with gr.Accordion("固定项", open=False, elem_classes=["param-l2"]):
                        with gr.Group(elem_classes=["param-box", "param-l3"]):
                            gr.Textbox(label="F1 Dimensionality Reduction", value="umap", interactive=False)
                        with gr.Group(elem_classes=["param-box", "param-l3"]):
                            gr.Textbox(label="F2 File Naming Rule", value="id@label@original（簇序号/簇序号@簇标签@原名）", interactive=False)
                        with gr.Group(elem_classes=["param-box", "param-l3"]):
                            gr.Textbox(label="F3 描述 .txt 到 output", value="true", interactive=False)
                        with gr.Group(elem_classes=["param-box", "param-l3"]):
                            gr.Textbox(label="F4 每句关键词 .txt", value="true", interactive=False)
                
                with gr.Accordion("G. 优化", open=False, elem_classes=["param-l1"]):
                    with gr.Group(elem_classes=["param-box", "param-l3"]):
                        run_device = gr.Dropdown(
                            choices=[("GPU (cuda)", "cuda"), ("CPU", "cpu")],
                            value="cuda",
                            label="G10 运行设备 (嵌入+VLM)",
                            info="Step-1 特征提取与 Step-4/5 图像描述/标签蒸馏共用；cuda / cpu"
                        )
                    with gr.Group(elem_classes=["param-box", "param-l3"]):
                        random_seed = gr.Number(
                            value=1,
                            minimum=-1,
                            maximum=2147483647,
                            step=1,
                            label="G8 随机数种子 (默认: 1)",
                            info="-1 表示每次随机；≥0 表示固定种子，结果可复现"
                        )
                    with gr.Group(elem_classes=["param-box", "param-l3"]):
                        force_rerun_step1_2 = gr.Checkbox(
                            value=False,
                            label="G9 重新进行前2步（索引+嵌入）",
                            info="勾选后强制重新执行 Step-0 和 Step-1，即使 A/B 未改可复用缓存；不勾选则复用上次嵌入结果"
                        )
                    with gr.Accordion("固定项", open=False, elem_classes=["param-l2"]):
                        with gr.Group(elem_classes=["param-box", "param-l3"]):
                            gr.Textbox(label="G1 Enable Acceleration", value="True", interactive=False)
                        with gr.Group(elem_classes=["param-box", "param-l3"]):
                            gr.Textbox(label="G2 Num Workers", value="4", interactive=False)
                        with gr.Group(elem_classes=["param-box", "param-l3"]):
                            gr.Textbox(label="G3 Thumbnail Cache", value="True", interactive=False)
                        with gr.Group(elem_classes=["param-box", "param-l3"]):
                            gr.Textbox(label="G4 Mixed Precision", value="True", interactive=False)
                        with gr.Group(elem_classes=["param-box", "param-l3"]):
                            gr.Textbox(label="G5 Model Compile", value="False", interactive=False)
                        with gr.Group(elem_classes=["param-box", "param-l3"]):
                            gr.Textbox(label="G6 Embedding Cache", value="True", interactive=False)
                        with gr.Group(elem_classes=["param-box", "param-l3"]):
                            gr.Textbox(label="G7 Prefetch Factor", value="2", interactive=False)
                        with gr.Group(elem_classes=["param-box", "param-l3"]):
                            gr.Textbox(label="G11 输出根目录", value="data/output", interactive=False)
                        with gr.Group(elem_classes=["param-box", "param-l3"]):
                            gr.Textbox(label="G12 缓存目录", value="data/.cache", interactive=False)
                        with gr.Group(elem_classes=["param-box", "param-l3"]):
                            gr.Textbox(label="G13 日志级别", value="INFO", interactive=False)
                pipeline_inputs = [
                    input_dir, embedding_provider, run_device, cluster_backend,
                    min_samples, epsilon, dbscan_metric, dbscan_algorithm,
                    cluster_selection_method, cluster_selection_epsilon, cluster_selection_persistence,
                    batch_size, caption_mode, top_k_sampling, vlm_model_scale, caption_batch_size, vlm_quantization, max_image_size, force_rerun_step1_2, random_seed,
                ]
                modified_hint = gr.HTML(
                    value="<span style='font-size:0.9em'><b>当前已修改:</b> 无（均为默认）</span>",
                    elem_id="modified-hint",
                )

                with gr.Accordion("配置保存", open=False, elem_classes=["param-l1"]):
                    config_selector = gr.Dropdown(
                        choices=get_config_choices(),
                        value="default_cfg",
                        label="配置名",
                        allow_custom_value=True,
                        info="选择或输入名称，保存为 config/xxx.yaml",
                    )
                    with gr.Row():
                        btn_load = gr.Button("加载配置", size="sm")
                        btn_reset = gr.Button("恢复默认", size="sm")
                        btn_save = gr.Button("保存配置", size="sm")
                    config_status = gr.Textbox(label=None, value="", interactive=False, show_label=False, lines=1)

                with gr.Row():
                    run_btn = gr.Button("🚀 开始处理", variant="primary", size="lg")
                    btn_open_latest = gr.Button("📂 打开最近结果", size="sm", variant="secondary")
                
                gr.Markdown("""
                ### 📋 使用步骤
                
                1. **输入目录**: 选择包含图像的文件夹
                2. **调整参数**: （可选）调整聚类参数
                3. **开始处理**: 点击按钮运行完整流程
                4. **查看结果**: 在右侧查看处理进度和结果
                """)
            
            with gr.Column(scale=2):
                with gr.Accordion("统计与簇分布", open=True):
                    with gr.Row():
                        stats_output = gr.Textbox(label="总体统计", lines=6, scale=1)
                        cluster_output = gr.Textbox(label="簇大小分布", lines=6, scale=1)
                
                with gr.Group():
                    gr.Markdown("#### 📈 进度")
                    progress_bar = gr.Slider(
                        0, 4, value=0, step=1,
                        label="阶段 (0-4)",
                        interactive=False,
                        show_label=True
                    )
                
                with gr.Group():
                    gr.Markdown("#### 📋 日志")
                    log_output = gr.Textbox(
                        show_label=False,
                        lines=20,
                        max_lines=35,
                        placeholder="点击「开始处理」后，日志将在此处实时更新..."
                    )
        
        # 绑定事件
        btn_open_latest.click(fn=open_latest_organized, outputs=[])
        run_btn.click(
            fn=app.run_pipeline,
            inputs=pipeline_inputs,
            outputs=[progress_bar, log_output, stats_output, cluster_output]
        )

        def _load_wrapper(config_name, *args):
            vals, msg = load_ui_config(config_name, *args)
            return list(vals) + [msg]

        btn_load.click(
            fn=_load_wrapper,
            inputs=[config_selector] + pipeline_inputs,
            outputs=pipeline_inputs + [modified_hint, config_status],
        )

        btn_reset.click(
            fn=reset_to_defaults,
            inputs=None,
            outputs=pipeline_inputs + [modified_hint],
        )

        btn_save.click(
            fn=save_ui_config,
            inputs=[config_selector] + pipeline_inputs,
            outputs=[config_status],
        )

        # 参数变更时更新“当前已修改”提示（非默认项橙色显示）
        for inp in pipeline_inputs:
            inp.change(fn=get_modified_hint, inputs=pipeline_inputs, outputs=[modified_hint])
        
        gr.Markdown("""
        ---
        ### 💡 提示
        
        - **输出位置**: `data/output/{会话ID}/organized/`
        - **会话保留**: 所有中间文件都保存在会话目录中
        
        ### 📁 输出结构
        
        ```
        organized/
        ├── cluster_00/ (簇0的图像)
        ├── cluster_01/ (簇1的图像)
        ├── ...
        └── noise/ (未分类的图像)
        ```
        
        ---
        **Version**: Phase-1 MVP | **Date**: 2026-01-31
        """)
    
    return demo


if __name__ == "__main__":
    print("=" * 60)
    print("  Semantic Cluster WebUI - Phase 1")
    print("=" * 60)
    print("\n正在启动Web界面...")
    print("请在浏览器中打开显示的URL\n")
    
    demo = create_ui()
    demo.queue().launch(  # queue() 启用生成器流式输出，实现进度和日志实时更新
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )
