# Semantic-Cluster-WebUI - 工作流结构

> 📅 **最后更新**: 2026-02-04（C1 HDBSCAN、C10/0.4、A3 不过滤、Step-0 统计）  
> 📖 **配合阅读**: [todo.md](./todo.md) | [numbering-system.md](./numbering-system.md)

---

## 工作流步骤（9步）

### Step-0: 建立索引
**输入**: [A1] Input Directory  
**输出**: `S0_image_index.json`  
**操作**:
- 扫描图像文件（按[A2]格式过滤）
- 按[A3][A4]大小过滤，排除[A5]文件夹
- 生成ID→路径映射

**所用参数**: [A1] 输入目录，[A2] 支持格式，[A3] 最小文件大小，[A4] 最大文件大小，[A5] 排除文件夹

---

### Step-1: 特征嵌入
**输入**: `S0_image_index.json`  
**输出**: `S1_embeddings.npy`  
**操作**:
- 使用[B1][B2]视觉模型提取特征（DINOv2/CLIP 均通过 HuggingFace 加载，兼容 Python 3.9）
- [B3]批量推理，[G4]混合精度，[G5]模型编译
- [B4]PCA降维，L2归一化
- [G6]嵌入缓存

**所用参数**: [B1] Provider，[B2] Backbone，[B4] Batch Size，[B5] PCA Components，[G10] 运行设备（embedding.device），[G4] 混合精度，[G5] 模型编译，[G6] 嵌入缓存，[G8] 随机种子（system.seed）

---

### Step-2: 严苛聚类
**输入**: `S1_embeddings.npy`  
**输出**: `S2_clustering.csv`  
**操作**:
- 使用[C1]后端：**HDBSCAN**（默认）或 DBSCAN（sklearn）
- [C2]距离度量（euclidean），[C6]最小样本数

**DBSCAN**（backend=sklearn）:
- [C4]eps 邻域半径，[C4b]距离度量 euclidean/cosine，[C4c]最近邻算法 auto/ball_tree/kd_tree/brute

**HDBSCAN**（backend=hdbscan）:
- 无需eps，自动发现簇结构
- [C7]`cluster_selection_method`: `leaf`=细粒度/噪音少，`eom`=保守/簇少

**DBSCAN 聚类模式**（config 可配，UI 当前固定 fixed_eps）:
- **fixed_eps**: 使用[C4]指定 eps
- **noise_control**: 自动搜索 eps，使噪音≤[C5]（仅 config 生效）

**输出**: 生成ID→簇ID映射，标记噪音为-1

**所用参数**: [C1] Backend，[C2] Metric，[C3] mode（仅 config），[C4] Epsilon，[C4b] DBSCAN 距离度量，[C4c] DBSCAN 最近邻算法，[C5] Max Noise Ratio，[C6] Min Samples，[C7] Cluster Selection Method，[C8] Min Cluster Size，[C9] Cluster Selection Epsilon，[C10] Cluster Selection Persistence，[C11] Alpha

---

### Step-3: 多点采样
**输入**: `S1_embeddings.npy`, `S2_clustering.csv`  
**输出**: `S3_sampled_images.json`  
**操作**:
- 按[D8]策略（nearest/farthest/random/stratified）采样
- 每簇选[D7]个代表图像（原 E1/E2 已归入 D 类）

**所用参数**: [D7] Top-K 采样（每簇代表图数），[D8] Sampling Strategy，[G8] 随机种子（system.seed）

---

### Step-4: 并行描述
**输入**: `S3_sampled_images.json`（代表模式）或 `S0`+`S2`（全图模式）  
**输出**: `S4_captions.json`  
**操作**:
- 使用[D1][D2]加载VLM模型，运行设备见[G10]
- **模式1**: 仅描述代表图像（需 Step-3 采样）
- **模式2**: 语义描述所有图片（跳过 Step-3）
- [E3]Caption Prompt 模板，目标[E5]字数

**所用参数**: [A1] 输入目录（data.input_directory），[D1] Provider，[D2] 模型规模，[D5] 描述模式（caption_mode），[D6] 描述批量，[D9] 量化，[D10] 最大分辨率，[G10] 运行设备，[E3] Caption Prompt，[E5] Caption Length，[F3] 描述 .txt 到 output（write_caption_txt）；VLM 内部还使用 V7 torch_dtype、V8 use_flash_attn

**VLM 图像预处理（processor）**  
Qwen2-VL 的 `Qwen2VLImageProcessor` 会 **resize 图像**：默认 `do_resize=True`，使用 `smart_resize`，像素数限制在 `min_pixels`～`max_pixels`（默认约 56²～28²×1280 ≈ 3136～1,003,520），保持宽高比且边长为 `patch_size×merge_size`（28）的倍数。大图会被缩小、小图可能被放大；此外会做 rescale(1/255)、归一化、转 RGB。

**已实现的加速**（默认启用）:
| 方式 | 参数 | 默认值 | 说明 |
|------|------|--------|------|
| 小模型 | [D2] 模型规模 / `vlm.model_scale` | small (2B) | 2B 快、省显存；7B 更准 |
| Flash Attention | `vlm.use_flash_attn` | true | 加速注意力、省显存（未安装则自动回退） |
| 批量推理 | `vlm.caption_batch_size` | 4 | 每批 N 张图，processor 支持则一次 forward，否则批内逐张 |

**其他可选加速**:
| 方式 | 说明 |
|------|------|
| **图像预缩小** | config `vlm.max_image_size`（[D10]）：默认 512，描述前长边缩至此像素以加速；0=不缩小 |
| 缩短生成长度 | 减小 [E5] caption_length 或 config `postprocessing.caption_length` |
| 量化 | [D9] / config `vlm.quantization`: int8 / int4（需 bitsandbytes、仅 CUDA） |
| 多 GPU | 多卡数据并行，需多进程/多卡调度 |

---

### Step-5: 语义蒸馏
**输入**: `S4_captions.json`  
**输出**: `S5_cluster_labels.csv`  
**操作**:
- 先对每条描述用[E4]提取关键词，再合并同簇关键词作为簇标签
- 生成簇语义标签
- 检测并精炼冲突标签

**所用参数**: [D1] Provider，[D2] 模型规模，[G10] 运行设备，[E4] 关键词提取 prompt（keyword_extract_prompt），[E6] Label Length（label_length_min/max），[E6b] Label 最大长度，[E6c] 蒸馏后关键词个数上限，[E9] Label Prompt（可选，留空则用 E4），[F4] 每句关键词 .txt（save_keyword_txt）；VLM 内部还使用 V7、V8、V9、V10、V11 等

---

### Step-6: 噪音挽救
**输入**: `S2_clustering.csv`, `S5_cluster_labels.csv`  
**输出**: `S6_rescue_candidates.csv`  
**操作**:
- 提取噪音图像（簇ID=-1）
- 生成描述，用[E8]算法计算与簇标签相似度
- 按[E7]阈值筛选归类建议

**所用参数**: [E7] Rescue Threshold，[E8] Similarity Algorithm

---

### Step-7: 可视化确认
**输入**: `S2`, `S5`, `S6`, `S0`  
**输出**: `S7_confirmed_moves.json`, `S7_user_edits.log`  
**操作**:
- [G3]缩略图缓存
- [F1]降维（UMAP/t-SNE/PCA）生成2D分布图
- 列表视图 + 交互散点图
- 用户拖拽、编辑标签、确认噪音挽救
- 保存修改

**所用参数**: [F1] Dimensionality Reduction，[G3] Thumbnail Cache

---

### Step-8: 文件整理
**输入**: `S7_confirmed_moves.json`, `S5_cluster_labels.csv`  
**输出**: 整理后文件夹 + `S8_organization_log.txt`  
**操作**:
- 按[F2]命名规则生成新文件名
- 预览表格（原名→新名→路径）
- 冲突检测（自动添加_1, _2）
- [G2]多线程并行移动

**所用参数**: [F2] File Naming Rule（output.file_naming_rule）；默认 id@label@original 时文件夹名=簇序号（00/01/noise）、文件名=簇序号@簇标签@原名；其他规则时文件夹名=簇标签（label）或 cluster_00，文件名按 F2

---

## 配置选项（与 config/config.yaml 同步）

> **格式**: [X#] 选项名 | 类型 | 默认值  
> **config 键**: data.* / clustering.* / vlm.* / embedding.* / postprocessing.* / output.* / optimization.* / system.*

### A. 数据源（5个）

| 编号 | 选项 | 类型 | 默认值 | config 键 |
|------|------|------|--------|-----------|
| A1 | Input Directory | 路径 | - | data.input_directory |
| A2 | Supported Formats | 多选 | jpg,jpeg,png,webp,bmp,tiff | data.supported_formats |
| A3 | Min File Size | 数字(KB) | 0（0=不过滤） | data.min_file_size_kb |
| A4 | Max File Size | 数字(MB) | -1（-1=不限制） | data.max_file_size_mb |
| A5 | Exclude Folders | 文本 | "" | data.exclude_folders |

### B. 嵌入（4个 + 设备见 G）

| 编号 | 选项 | 类型 | 默认值 | config 键 |
|------|------|------|--------|-----------|
| B1 | Provider | 下拉 | DINOv2 | embedding.provider |
| B2 | Backbone | 下拉 | dinov2_vitl14 | embedding.backbone |
| B4 | Batch Size | 数字 | 16 | embedding.batch_size |
| B5 | PCA Components | 数字 | 256 | embedding.pca_components |

**B3 嵌入设备** 已合并至 **G10**（见 G. 优化）。config: `embedding.device`。

**B2选项**（根据B1）:
- DINOv2: dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14
- CLIP: clip_vitb32, clip_vitb16, clip_vitl14, clip_vitl14_336

### C. 聚类（12个，C3 仅 config）

| 编号 | 选项 | 类型 | 默认值 | config 键 |
|------|------|------|--------|-----------|
| C1 | Backend | 下拉 | sklearn / hdbscan | clustering.backend |
| C2 | Metric | 固定 | euclidean | clustering.metric |
| C4 | Epsilon (eps) | 滑块 | 0.5 (范围 0.5-1.5) | clustering.epsilon |
| C4b | DBSCAN 距离度量 | 下拉 | euclidean / cosine | clustering.dbscan_metric |
| C4c | DBSCAN 最近邻算法 | 下拉 | auto / ball_tree / kd_tree / brute | clustering.dbscan_algorithm |
| C5 | Max Noise Ratio | 滑块(%) | 20 | clustering.max_noise_ratio |
| C6 | Min Samples | 滑块 | 2 (范围 2-30) | clustering.min_samples |
| C7 | Cluster Selection Method | 下拉 | leaf | clustering.cluster_selection_method |
| C8 | Min Cluster Size | 数字 | 5(null 时用 C6) | clustering.min_cluster_size |
| C9 | Cluster Selection Epsilon | 滑块 | 0.0 (范围 0-0.5) | clustering.cluster_selection_epsilon |
| C10 | Cluster Selection Persistence | 滑块 | 0.4 (范围 0-1) | clustering.cluster_selection_persistence |
| C11 | Alpha | 数字 | 1.0 | clustering.alpha |

**说明**:
- **C1 Backend**: 默认 `hdbscan`（HDBSCAN，无需 eps 自动发现簇）；`sklearn` 为 DBSCAN
- **C7 Cluster Selection Method**: 仅 HDBSCAN
  - `leaf`: 细粒度簇，**噪音较少**
  - `eom`: 保守选择，簇更少、噪音可能更多
- **C8-C11 仅 HDBSCAN**: C8 最小簇大小(null 时用 C6)；C9 距离阈值，小于此的簇合并；C10 持久度阈值；C11 距离缩放
- **C4 Epsilon**: 仅 DBSCAN 邻域半径。**C4b 距离度量**: euclidean/cosine，L2 归一化特征可试 cosine。**C4c 最近邻算法**: 影响速度，大数据集可试 ball_tree/kd_tree
- **C3**（仅 config）: clustering.mode = fixed_eps / noise_control；UI 未暴露

### D. VLM（10个 + 设备见 G，含原 E1/E2）

| 编号 | 选项 | 类型 | 默认值 | config 键 |
|------|------|------|--------|-----------|
| D5 | 描述模式 (Step-4) | 下拉 | 代表图（模式1） | postprocessing.caption_mode |
| D2 | 模型规模 | 下拉 | 2B (快) / 7B (准) / 跳过（用簇序号） | vlm.model_scale |
| D6 | 描述批量 (Caption Batch Size) | 数字 | 4 | vlm.caption_batch_size |
| D10 | 最大分辨率 | 数字 | 512 | vlm.max_image_size |
| D7 | Top-K 采样 (原 E2) | 数字 | 2 | postprocessing.top_k_sampling |
| D9 | 量化 | 下拉 | 无 / int8 / int4 | vlm.quantization |
| D1 | Provider | 固定 | Local Qwen2-VL | vlm.provider |
| D4 | API Key | 固定 | - | vlm.api_key |
| D8 | Sampling Strategy (原 E1) | 固定 | nearest | postprocessing.sampling_strategy |

**D3 运行设备** 已合并至 **G10**（见 G. 优化，嵌入与 VLM 共用）。**D6** 对应 config `vlm.caption_batch_size`。**D10 最大分辨率**：描述前将图像长边缩至此像素（保持宽高比），默认 512 以加速；0 表示不缩小。config: `vlm.max_image_size`。**D7** 对应 config `postprocessing.top_k_sampling`（Step-3 每簇代表图数），**D8** 对应 config `postprocessing.sampling_strategy`（Step-3 采样策略）。**D9** 对应 config `vlm.quantization`（int8/int4 需安装 bitsandbytes，仅 CUDA；省显存、可提速）。

**D2 模型规模**: `small`=Qwen2-VL-2B（默认，快、省显存）、`large`=Qwen2-VL-7B（更准）、`skip`=跳过 Step-3/4/5，直接用簇序号命名（cluster_00、cluster_01…）。config: `vlm.model_scale`、`vlm.model_name`（可覆盖）。

**D5 描述模式**: `representative`（模式1，仅描述代表图，需 Step-3 采样）/ `all`（模式2，描述全部图像，可跳过 Step-3）。

**VLM 相关 config 参数**（与 config.yaml 一致）:

| 序号 | config 键 | 类型 | 默认值 | 说明 |
|------|-----------|------|--------|------|
| V1 | `vlm.provider` | 字符串 | local_qwen2vl | 固定本地 Qwen2-VL |
| V2 | `vlm.model_source` | 字符串 | huggingface | huggingface / **modelscope**（[通义千问2-VL-2B](https://www.modelscope.cn/models/qwen/Qwen2-VL-2B-Instruct/summary)） |
| V3 | `vlm.model_scale` | 字符串 | small | small=2B / large=7B / **skip**=跳过描述与标签（用簇序号），与 [D2] 对应 |
| V4 | `vlm.model_name` | 字符串 | "" | 留空则按 model_scale+model_source 选择；可覆盖为具体 ID |
| V5 | `vlm.device` | 字符串 | cuda | 与 [G10] 共用 |
| V6 | `vlm.api_key` | 字符串 | "" | 本地模型可留空 |
| V7 | `vlm.torch_dtype` | 字符串 | bfloat16 | bfloat16 / float16 / float32 |
| V8 | `vlm.use_flash_attn` | 布尔 | true | Flash Attention，未安装则回退 |
| V9 | `vlm.caption_batch_size` | 整数 | 4 | Step-4 每批图像数，与 [D6] 对应 |
| V10 | `vlm.max_image_size` | 整数 | 512 | 描述前长边最大像素（保持宽高比），默认 512 以加速；0=不缩小，与 [D10] 对应 |
| V11 | `vlm.quantization` | 字符串 | none | none / int8 / int4，与 [D9] 对应；需 bitsandbytes、仅 CUDA |

**本地 VLM/LLM**（Step-4 图像描述、Step-5 簇标签蒸馏）: 使用 `models/vlm_models.py`，按 `model_scale` 解析 `model_name`，默认 2B + Flash Attention + 批量推理。

### E. 后处理（9个，E1/E2 已归入 D 类）

| 编号 | 选项 | 类型 | 默认值 | config 键 |
|------|------|------|--------|-----------|
| E3 | Caption Prompt | 文本 | 见下方 | postprocessing.caption_prompt |
| E4 | Label / 关键词提取 | 文本 | 见下方 | postprocessing.keyword_extract_prompt |
| E5 | Caption Length | 数字 | 50 | postprocessing.caption_length |
| E6 | Label Length | 滑块 | 5-10 | postprocessing.label_length_min/max |
| E6b | Label 最大长度（字符） | 数字 | 512 | postprocessing.label_max_length |
| E6c | 蒸馏后关键词个数上限 | 数字 | 8 | postprocessing.label_keyword_max |
| E7 | Rescue Threshold | 滑块 | 0.60 | postprocessing.rescue_threshold |
| E8 | Similarity Algorithm | 下拉 | cosine | postprocessing.similarity_algorithm |
| E9 | Label Prompt（仅 config） | 文本 | "" | postprocessing.label_prompt |

**E6b**: 簇标签最终截断长度（字符）。**E6c**: 合并同簇关键词后最多保留的关键词个数，默认 8；调小可减少蒸馏后的关键词个数。**E9**: 可选；留空时 Step-5 使用 [E4] keyword_extract_prompt。**默认Prompt**:
- E3: `"Describe the main subject, action, lighting, and viewpoint of this image in detail (around {caption_length} words)."`
- E4: Step-5 先对每条描述用 `keyword_extract_prompt`（占位符 `{description}`）提取 3–8 个关键词，再将同簇多条描述的关键词合并去重，得到簇标签。config: `postprocessing.keyword_extract_prompt`。

### F. 输出（4个）

| 编号 | 选项 | 类型 | 默认值 | config 键 |
|------|------|------|--------|-----------|
| F1 | Dimensionality Reduction | 下拉 | UMAP | output.dimensionality_reduction |
| F2 | File Naming Rule | 下拉 | id@label@original（默认） | output.file_naming_rule |
| F3 | 描述 .txt 到 output | 复选 | true | output.write_caption_txt |
| F4 | 每句关键词 .txt | 复选 | true | output.save_keyword_txt |

**F3**: Step-4 在 output 目录下 `caption_txt/` 中按 image_id 输出描述 `.txt`。**F4**: Step-5 在 output 目录下 `step5_keywords/` 中按簇输出 `cluster_00_keywords.txt` 等，每行对应一条描述提取的关键词；关闭则仅输出 S5_cluster_labels.csv。**F2选项**:
- **`id@label@original`**（默认）: 簇序号/簇序号@簇标签@原名，如 `00/00@Mountain_Landscape@IMG_1234.jpg`；文件夹恒为簇序号（00、01、noise）
- `label@original`: 文件夹 `label`，文件 `label@original`，如 `Mountain_Landscape/Mountain_Landscape@IMG_1234.jpg`
- `cluster_id@label`: 文件夹 `label`，文件 `03@Mountain_Landscape.jpg`
- `cluster_id@label@original`: 文件夹 `label`，文件 `03@Mountain_Landscape@IMG_1234.jpg`  
无 S5 时除 id@label@original 外，文件夹为 `cluster_00`、`noise` 等。

### G. 优化（13个，含 B3/D3 合并设备）

| 编号 | 选项 | 类型 | 默认值 | config 键 |
|------|------|------|--------|-----------|
| G10 | 运行设备 (嵌入+VLM) | 下拉 | cuda | embedding.device / vlm.device |
| G8 | Random Seed | 数字 | 42 | system.seed |
| G9 | Force Rerun Step 0+1 | 复选 | False | （不持久化，加载时默认不勾选） |
| G1 | Enable Acceleration | 复选 | True | optimization.enable_acceleration |
| G2 | Num Workers | 数字 | 4 | optimization.num_workers |
| G3 | Thumbnail Cache | 复选 | True | optimization.thumbnail_cache |
| G4 | Mixed Precision | 复选 | True | optimization.mixed_precision |
| G5 | Model Compile | 复选 | False | optimization.model_compile |
| G6 | Embedding Cache | 复选 | True | optimization.embedding_cache |
| G7 | Prefetch Factor | 数字 | 2 | optimization.prefetch_factor |
| G11 | 输出根目录 | 路径 | data/output | system.output_base_directory |
| G12 | 缓存目录 | 路径 | data/.cache | system.cache_directory |
| G13 | 日志级别 | 字符串 | INFO | system.log_level |

**G10**: 嵌入与 VLM 共用运行设备。**G8**: -1 表示每次随机；≥0 表示固定种子。config: `system.seed`（默认 42）。**G9**: 勾选后强制重跑 Step-0/1。**G11**: 整理输出根目录；UI 当前使用固定 `data/output`。**G12**: 嵌入/缓存等存放目录。**G13**: 如 DEBUG/INFO/WARNING。


---

## 输出文件

| 序号 | 文件 | 步骤 | 说明 |
|------|------|------|------|
| 1 | run_config.yaml | - | 本次运行的完整配置（所有参数） |
| 2 | S0_image_index.json | 0 | ID→路径映射 |
| 3 | S1_embeddings.npy | 1 | 特征矩阵 |
| 4 | S2_clustering.csv | 2 | ID→簇ID |
| 5 | S3_sampled_images.json | 3 | 采样图像 |
| 6 | S4_captions.json | 4 | 图像描述 |
| 7 | S5_cluster_labels.csv | 5 | 簇→标签 |
| 8 | step5_keywords/*.txt | 5 | 每句描述提取的关键词（每行一条；config `output.save_keyword_txt`） |
| 9 | S6_rescue_candidates.csv | 6 | 噪音挽救建议 |
| 10 | S7_confirmed_moves.json | 7 | 最终移动清单 |
| 11 | S7_user_edits.log | 7 | 用户编辑日志 |
| 12 | S8_organization_log.txt | 8 | 移动操作日志 |

---

## 推荐配置

### 小型（<1K图）
```yaml
[C1] sklearn  [B1] DINOv2  [B2] vitb14  [B4] 32  [G4] False
```

### 中型（1K-10K）
```yaml
[C1] sklearn/rapids  [B1] DINOv2  [B2] vitl14  [B4] 64  [G4] True  [G5] True
```

### 大型（10K-100K）
```yaml
[C1] rapids/faiss  [B1] ConvNeXt  [B2] large  [B4] 128  [B5] 128  [G4] True
```

### 超大（>100K）
```yaml
[C1] faiss  [B1] ConvNeXt  [B2] base  [B4] 256  [B5] 64  [F1] PCA  [G2] 16
```
