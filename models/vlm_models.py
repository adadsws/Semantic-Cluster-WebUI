"""
VLM/LLM 本地推理封装
- Step-4: Qwen2-VL 图像描述（image + prompt -> caption）
- Step-5: 同一模型文本生成（captions + label_prompt -> cluster label）

支持 HuggingFace 与 ModelScope（通义千问2-VL，见 https://www.modelscope.cn/models/qwen/Qwen2-VL-2B-Instruct/summary）。
使用 transformers AutoProcessor + AutoModelForImageTextToText 本地加载。

📅 Last Updated: 2026-01-31
"""

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).parent.parent


def _resize_image_for_caption(image_path: Path, max_size: int) -> Tuple[Path, Optional[Path]]:
    """
    描述前缩小图像以加速：将长边缩至 max_size 以内，保持宽高比。
    max_size <= 0 时直接返回原路径。

    Returns:
        (path_to_use, temp_path_or_None): 若生成了临时文件，第二个为临时路径（调用方负责删除）
    """
    if max_size <= 0:
        return image_path, None
    try:
        from PIL import Image
    except ImportError:
        return image_path, None
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return image_path, None
    w, h = img.size
    if max(w, h) <= max_size:
        return image_path, None
    if w >= h:
        new_w, new_h = max_size, int(h * max_size / w)
    else:
        new_w, new_h = int(w * max_size / h), max_size
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    fd, temp_path = tempfile.mkstemp(suffix=".jpg", prefix="vlm_resized_")
    import os
    try:
        img.save(temp_path, "JPEG", quality=85)
    finally:
        os.close(fd)
    return Path(temp_path), Path(temp_path)

# 全局缓存，避免重复加载
_cached_model: Any = None
_cached_processor: Any = None
_cached_model_name: Optional[str] = None

# HuggingFace 默认模型 ID
VLM_MODEL_SMALL = "Qwen/Qwen2-VL-2B-Instruct"
VLM_MODEL_LARGE = "Qwen/Qwen2-VL-7B-Instruct"

# ModelScope 默认模型 ID（通义千问2-VL，国内下载更快）
VLM_MODEL_SMALL_MODELSCOPE = "qwen/Qwen2-VL-2B-Instruct"
VLM_MODEL_LARGE_MODELSCOPE = "qwen/Qwen2-VL-7B-Instruct"


def resolve_vlm_model_name(config: dict) -> str:
    """从 config 解析实际使用的 VLM 模型 ID：model_scale + model_source，或显式 model_name。"""
    vlm = config.get("vlm", {})
    explicit = (vlm.get("model_name") or "").strip()
    source = (vlm.get("model_source") or "huggingface").strip().lower()
    scale = (vlm.get("model_scale") or "").strip().lower()

    if explicit:
        return explicit
    if source == "modelscope":
        if scale == "small":
            return VLM_MODEL_SMALL_MODELSCOPE
        if scale == "large":
            return VLM_MODEL_LARGE_MODELSCOPE
        return VLM_MODEL_LARGE_MODELSCOPE
    if scale == "small":
        return VLM_MODEL_SMALL
    if scale == "large":
        return VLM_MODEL_LARGE
    return VLM_MODEL_LARGE


def _get_model_load_path(model_id: str, config: dict) -> str:
    """
    根据 model_source 返回用于 from_pretrained 的路径：
    - huggingface: 直接返回 model_id（从 HF 下载）
    - modelscope: 先 snapshot_download 到本地，返回本地目录路径
    """
    source = (config.get("vlm", {}).get("model_source") or "huggingface").strip().lower()
    if source != "modelscope":
        return model_id
    try:
        from modelscope import snapshot_download
    except ImportError:
        print("[VLM] 未安装 modelscope，从 HuggingFace 加载。可选: pip install modelscope")
        return model_id
    cache_dir = Path(config.get("system", {}).get("cache_directory", "data/.cache"))
    cache_dir = ROOT / cache_dir if not str(cache_dir).startswith("/") and ":" not in str(cache_dir) else Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_dir = snapshot_download(model_id, cache_dir=str(cache_dir))
    return local_dir


def _get_device_map(device: str) -> str:
    if device == "cuda":
        return "auto"
    if device == "cpu":
        return "cpu"
    return "auto"


def get_vlm_model_and_processor(
    model_name: str,
    device: str = "cuda",
    torch_dtype: Optional[str] = "bfloat16",
    use_flash_attn: bool = True,
    quantization: Optional[str] = None,
    config: Optional[dict] = None,
) -> Tuple[Any, Any]:
    """
    加载 VLM 模型与 processor（单例：同 model_name+quantization 复用缓存）。
    支持 HuggingFace（model_id）与 ModelScope（config.vlm.model_source=modelscope 时先 snapshot_download 再本地加载）。

    Args:
        model_name: HuggingFace 或 ModelScope 模型 ID，如 Qwen/Qwen2-VL-2B-Instruct、qwen/Qwen2-VL-2B-Instruct
        device: cuda / cpu
        torch_dtype: bfloat16 / float16 / float32（量化时部分忽略）
        use_flash_attn: 是否使用 flash_attention_2
        quantization: none / int8 / int4（需安装 bitsandbytes，仅 CUDA）
        config: 可选，含 vlm.model_source 时从 ModelScope 下载到本地再加载

    Returns:
        (model, processor)
    """
    global _cached_model, _cached_processor, _cached_model_name
    q = (quantization or "none").strip().lower()
    cache_key = f"{model_name}|{q}"
    if _cached_model is not None and _cached_model_name == cache_key:
        return _cached_model, _cached_processor

    load_path = _get_model_load_path(model_name, config or {}) if config else model_name
    if load_path != model_name:
        print(f"[VLM] 正在加载模型（ModelScope 本地）: {model_name} -> {load_path}")
    else:
        print(f"[VLM] 正在加载模型: {model_name} (device={device}, quantization={q})")
    import torch
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText
        model_cls = AutoModelForImageTextToText
    except ImportError:
        try:
            from transformers import Qwen2VLForConditionalGeneration as model_cls
        except ImportError:
            raise ImportError(
                "需要 transformers 支持 Qwen2-VL（AutoModelForImageTextToText 或 Qwen2VLForConditionalGeneration）"
            )

    device_map = _get_device_map(device)
    kw = {"device_map": device_map, "trust_remote_code": True}

    if q == "int8":
        kw["load_in_8bit"] = True
    elif q == "int4":
        kw["load_in_4bit"] = True
    else:
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        dtype = dtype_map.get(torch_dtype, torch.bfloat16)
        # 新版本 transformers 推荐 dtype，避免 torch_dtype 弃用警告
        kw["dtype"] = dtype
        if use_flash_attn:
            kw["attn_implementation"] = "flash_attention_2"

    def _load_model():
        return model_cls.from_pretrained(load_path, **kw)

    if q in ("int8", "int4"):
        try:
            model = _load_model()
        except Exception as e:
            raise RuntimeError(
                f"量化加载失败（需安装 bitsandbytes 且仅 CUDA）: {e}\n"
                "可设 vlm.quantization: none 或 pip install bitsandbytes"
            ) from e
    else:
        try:
            model = _load_model()
        except TypeError:
            # 旧版 transformers 只认 torch_dtype
            kw.pop("dtype", None)
            kw["torch_dtype"] = dtype
            model = _load_model()
        except Exception:
            if use_flash_attn:
                kw.pop("attn_implementation", None)
                print("[VLM] 使用默认注意力（可选安装 flash-attn 加速）")
                model = _load_model()
            else:
                raise
    print("[VLM] 正在加载 processor…")
    processor = AutoProcessor.from_pretrained(load_path, trust_remote_code=True)

    _cached_model = model
    _cached_processor = processor
    _cached_model_name = cache_key
    print(f"[VLM] 模型与 processor 加载完成（已缓存，后续复用）")
    return model, processor


def caption_single_image(
    model: Any,
    processor: Any,
    image_path: Path,
    prompt: str,
    max_new_tokens: int = 256,
) -> str:
    """
    单张图像描述：VLM(image + prompt) -> caption 文本。
    使用 Qwen2-VL 对话格式：apply_chat_template(conversation) -> generate -> decode。
    """
    import torch

    path_str = str(image_path.resolve())
    # 本地图：部分版本用 "path"，部分用 "image"；先试 path，失败再试 image
    for img_key in ("path", "image"):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", img_key: path_str},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        try:
            inputs = processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            break
        except (TypeError, KeyError, ValueError):
            if img_key == "image":
                raise
            continue
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    try:
        from transformers import GenerationConfig
        gen_cfg = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False)
    except ImportError:
        gen_cfg = None
    with torch.no_grad():
        out = model.generate(**inputs, generation_config=gen_cfg) if gen_cfg else model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    prompt_len = inputs["input_ids"].shape[1]
    generated = out[:, prompt_len:]
    decoded = processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return (decoded[0] or "").strip()


def _caption_batch_forward(
    model: Any,
    processor: Any,
    paths: List[Path],
    prompt: str,
    max_new_tokens: int,
) -> List[str]:
    """
    尝试 processor 原生 batch（若支持多 conversation 的 apply_chat_template）。
    若不支持或失败则返回空列表，由 caption_batch 逐张回退。
    """
    import torch
    path_strs = [str(p.resolve()) for p in paths]
    conversations = []
    for path_str in path_strs:
        conv = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": path_str},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        conversations.append(conv)
    try:
        inputs = processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
    except (TypeError, KeyError, ValueError):
        return []
    if not inputs or "input_ids" not in inputs:
        return []
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    try:
        from transformers import GenerationConfig
        gen_cfg = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False)
    except ImportError:
        gen_cfg = None
    with torch.no_grad():
        out = model.generate(**inputs, generation_config=gen_cfg) if gen_cfg else model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    prompt_len = inputs["input_ids"].shape[1]
    decoded_list = []
    for i in range(out.size(0)):
        generated = out[i : i + 1, prompt_len:]
        dec = processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        decoded_list.append((dec[0] or "").strip())
    return decoded_list


def caption_batch(
    model: Any,
    processor: Any,
    items: List[Tuple[str, Path]],
    prompt: str,
    max_new_tokens: int = 256,
    batch_size: int = 4,
    max_image_size: int = 0,
) -> Dict[str, str]:
    """
    批量图像描述：按 batch_size 成批推理，优先尝试 batch forward，失败则逐张回退。
    max_image_size > 0 时描述前将图像长边缩至此像素以加速（保持宽高比）。

    items: [(image_id, path), ...]
    Returns: {image_id: caption, ...}
    """
    import os
    result: Dict[str, str] = {}
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        iids = [x[0] for x in chunk]
        paths = [x[1] for x in chunk]
        if len(paths) == 0:
            continue
        # 可选：缩小分辨率以加速
        temp_paths: List[Path] = []
        use_paths: List[Path] = []
        for p in paths:
            use_p, temp_p = _resize_image_for_caption(p, max_image_size)
            use_paths.append(use_p)
            if temp_p is not None:
                temp_paths.append(temp_p)
        try:
            batch_caps = _caption_batch_forward(model, processor, use_paths, prompt, max_new_tokens)
        except Exception:
            batch_caps = []
        if len(batch_caps) == len(iids):
            for iid, cap in zip(iids, batch_caps):
                result[iid] = cap or ""
        else:
            for iid, use_p in zip(iids, use_paths):
                try:
                    cap = caption_single_image(model, processor, use_p, prompt, max_new_tokens=max_new_tokens)
                    result[iid] = cap or ""
                except Exception:
                    result[iid] = ""
        for tp in temp_paths:
            try:
                if tp.exists():
                    os.unlink(tp)
            except Exception:
                pass
    return result


def generate_text(
    model: Any,
    processor: Any,
    prompt: str,
    max_new_tokens: int = 128,
) -> str:
    """
    纯文本生成：LLM(prompt) -> 文本（用于 Step-5 簇标签蒸馏）。
    Qwen2-VL 纯文本时仅传 input_ids/attention_mask，避免 apply_chat_template 返回的多模态键导致 string indices 错误。
    """
    import torch

    # Qwen2-VL 要求 content 为 part 列表，否则 processor 遍历 content 时会把字符串当迭代项，part["type"] 报 string indices must be integers
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    raw = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    # 只保留张量并移到设备；accept dict 或 BatchFeature（transformers 可能返回 BatchFeature 而非 dict）
    if not hasattr(raw, "get"):
        raise TypeError(f"apply_chat_template 预期返回 dict 或 BatchFeature，得到 {type(raw)}")
    input_ids = raw.get("input_ids")
    attention_mask = raw.get("attention_mask")
    if input_ids is None:
        raise KeyError("apply_chat_template 返回中缺少 input_ids")
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if attention_mask is not None and attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)
    input_ids = input_ids.to(model.device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)
    kwargs = {"input_ids": input_ids, "max_new_tokens": max_new_tokens, "do_sample": False}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    try:
        from transformers import GenerationConfig
        gen_cfg = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False)
    except ImportError:
        gen_cfg = None
    with torch.no_grad():
        out = model.generate(**kwargs, generation_config=gen_cfg) if gen_cfg else model.generate(**kwargs)
    prompt_len = input_ids.shape[1]
    generated = out[:, prompt_len:]
    decoded = processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return (decoded[0] or "").strip()


def is_vlm_available() -> bool:
    """检查当前环境是否可加载 VLM（transformers 含 Qwen2-VL 等）。"""
    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        return True
    except ImportError:
        pass
    try:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        return True
    except ImportError:
        pass
    try:
        from transformers.models.qwen2_vl import Qwen2VLForConditionalGeneration
        from transformers import AutoProcessor
        return True
    except ImportError:
        pass
    return False


def _get_transformers_version() -> str:
    try:
        import transformers
        return getattr(transformers, "__version__", "未知")
    except Exception:
        return "未安装"


def check_vlm_ready(config: dict) -> Tuple[bool, str]:
    """
    先检测 VLM 是否可用且模型已下载/可加载。
    若可用则尝试加载模型（加载成功会缓存，后续 Step-4 直接复用）。

    Returns:
        (ready, message): ready=True 表示可用且已加载；False 时 message 说明原因及建议。
    """
    if not is_vlm_available():
        ver = _get_transformers_version()
        return (
            False,
            f"VLM 不可用：当前 transformers 版本 {ver}，需 4.37+ 才支持 Qwen2-VL。"
            "请执行：pip install -U transformers"
        )
    vlm_cfg = config.get("vlm", {})
    model_name = resolve_vlm_model_name(config)
    device = vlm_cfg.get("device", "cuda")
    torch_dtype = vlm_cfg.get("torch_dtype", "bfloat16")
    use_flash_attn = vlm_cfg.get("use_flash_attn", True)
    quantization = (vlm_cfg.get("quantization") or "none").strip().lower()
    try:
        get_vlm_model_and_processor(
            model_name,
            device=device,
            torch_dtype=torch_dtype,
            use_flash_attn=use_flash_attn,
            quantization=quantization if quantization not in ("", "none") else None,
            config=config,
        )
        return (True, f"VLM 已就绪，模型已加载：{model_name}")
    except Exception as e:
        return (
            False,
            f"模型未下载或加载失败：{e}\n"
            f"请先下载模型：huggingface-cli download {model_name}\n"
            "或检查显存/网络后重试。"
        )
