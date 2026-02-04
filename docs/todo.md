# Semantic-Cluster-WebUI 开发计划

> 📅 **最后更新**: 2026-02-04 (C1/HDBSCAN、C10/0.4、A3 不过滤、Step-0 统计)  
> 📖 **配合阅读**: [workflow-structure.md](./workflow-structure.md) | [numbering-system.md](./numbering-system.md)

---

## 开发原则

1. **从简单到复杂** - 先核心流程，再高级功能
2. **UI驱动** - Phase-1就用Gradio边写边看
3. **边写边改** - 先跑通再优化，不追求完美
4. **直接参考** - 大量参考references/项目代码
5. **每改必更新文档** - 同步更新README、todo、workflow-structure、numbering-system

---

## 📝 文档维护 ⚠️

**改代码 → 测试 → 更新文档 → 标记日期**

| 文档 | 何时更新 |
|------|---------|
| README.md | 技术栈变更 |
| todo.md | 完成任务 |
| workflow-structure.md | Step/配置参数变更 |
| numbering-system.md | 编号变化 |

---

## 参考项目（14个）

运行 `references/克隆所有项目.ps1` 克隆所有项目。

| Phase | 项目 | 用途 |
|-------|------|------|
| 1 | DermNet | DINOv2特征提取、去重 |
| 1 | img2cluster | CLIP聚类、UMAP降维 |
| 1 | labelme | 文件扫描 |
| 1-2 | timm | 视觉模型库 |
| 3 | autodistill | 自动标注、文件整理 |
| 1-3 | Qwen2-VL | VLM推理示例 |
| 5 | Interactive-TSNE | Plotly可视化 |
| 5 | pix-plot | WebGL大规模可视化 |
| 5 | umap-streamlit-app | Streamlit UI |
| 6 | CVAT | 数据预览、标注界面 |
| 7 | vLLM | 高性能推理 |
| 7 | cuML | GPU加速聚类 |
| 7 | FAISS | 相似度搜索 |
| 7 | OpenTSNE | 快速降维 |

详见 [references/项目列表.md](./references/项目列表.md)


---

## 项目结构

```
semantic-cluster-webui/
├── data/
│   ├── input/           # 输入图像
│   └── output/          # 输出结果（S0-S8）
├── config/
│   ├── config.yaml      # 配置 [A1]-[G13]
│   └── prompts.yaml     # Prompt模板
├── core/                # Step-0 到 Step-8
│   ├── step0_indexing.py
│   ├── step1_embedding.py
│   └── ...
├── models/
│   ├── vision_models.py # 视觉模型
│   └── vlm_models.py    # VLM模型
├── utils/
│   ├── file_utils.py
│   └── similarity_utils.py
├── ui/
│   └── app.py           # Gradio主入口
└── requirements.txt
```

---

## 开发阶段

### Phase-0: 环境搭建 ✅

```
Phase-0
├── [x] 创建项目结构
├── [x] requirements.txt, config.yaml
└── [x] 配置加载器
```

**完成日期**: 2026-01-31 19:02

---

### Phase-1: MVP（核心流程）🎯 ✅

**目标**: 索引 → 嵌入 → 聚类 → 整理；聚类可切换 HDBSCAN（默认）/DBSCAN；参数可调、可保存加载；支持缓存复用

```
Phase-1
├── Step-0: 索引（默认不进行大小过滤，统计不重复计数）
│   └── [x] core/step0_indexing.py + 目录选择器 → S0_image_index.json
├── Step-1: 嵌入
│   ├── [x] DINOv2/CLIP (HuggingFace) → S1_embeddings.npy
│   ├── [x] PCA 降维、L2 归一化
│   └── [x] 嵌入缓存（按路径 key 复用）
├── Step-2: 聚类
│   ├── [x] HDBSCAN（默认）+ DBSCAN 双后端
│   ├── [x] DBSCAN: eps、C4b 度量 euclidean/cosine、C4c 算法 auto/ball_tree/kd_tree/brute
│   ├── [x] HDBSCAN: C7-C11 簇选择、persistence、alpha（已去除 C12）
│   └── [x] 固定 eps / 噪音比例控制 双模式
├── Step-8: 整理
│   └── [x] 按簇 ID 整理文件 → organized/
├── Gradio UI
│   ├── [x] 基础框架、GPU 检测、实时日志
│   ├── [x] 参数非默认值标记、A-G 多组参数（G 现含 G11/G12/G13）、折叠/展开
│   ├── [x] 配置保存/加载、恢复默认
│   ├── [x] 随机种子、强制重跑前 2 步、打开最近结果
│   └── [x] 参数按流程重排 (B 嵌入→C 聚类→D VLM)
├── 参数测试脚本
│   ├── [x] test_hdbscan_params.py（--multi 多组推荐默认值）
│   └── [x] test_dbscan_params.py（metric/algorithm 敏感性）
└── [x] 更新文档 (4 个文档)
```

**完成日期**: 2026-01-31

---

#### 📋 对话记录 (2026-01-31 Phase-1增强)


| 问题/需求 | 解决方案 |
|----------|----------|
| 全部被归为noise | config用sklearn+eps=0.15过小 → UI强制HDBSCAN，支持相对路径(test_pics) |
| 减少noise | cluster_selection_method改为leaf；增加B8-B12 HDBSCAN参数 |
| HDBSCAN/DBSCAN选择 | 改为UI可选参数，下拉选择 |
| output记录设置 | 每次运行保存run_config.yaml到输出目录 |
| 去掉ResNet，增加CLIP | 移除ResNet，添加CLIP(transformers)，特征模型: DINOv2/CLIP |
| 使用DINO嵌入 | 实现DINOv2，因Python 3.9兼容改用HuggingFace(非torch.hub) |
| Python 3.9报错 (type\|NoneType) | torch.hub的DINOv2用3.10+语法 → 改用HuggingFace加载facebook/dinov2-base等 |
| 清理多余文件 | PHASE0_COMPLETE、test_phase0、GRADIO_USAGE_GUIDE、QUICKSTART、test_pics_config、rename → ~dump |

#### 2026-01-31 Phase-1 UI增强

| 问题/需求 | 解决方案 |
|----------|----------|
| 从CPU切换到GPU | 修改config.yaml embedding.device: cuda，修改step1_embedding优先读config，添加requirements-cu128.txt |
| GPU检测与显示 | ui/app.py添加get_gpu_status()，UI显示GPU可用性(CUDA版本、显卡型号) |
| 实时日志更新 | run_pipeline改为生成器+yield，使用contextlib.redirect_stdout捕获print，demo.queue().launch()启用流式 |
| 进度条显示方式 | 改用Slider显示阶段(0-4)而非百分比，移除gr.Progress避免闪烁 |
| 参数非默认值标记 | 添加modified_hint HTML组件，动态显示已修改参数(橙色文字) |
| 显示所有参数 | 添加所有32个参数，可编辑/固定分离，固定项用子Accordion或interactive=False |
| 参数紧凑显示 | 自定义CSS减小padding/margin，去掉白框(移除border/background) |
| 折叠/展开所有手风琴 | 用gr.Button + .click(js=...)，JS查找button.label-wrap.open/.not(.open)并延迟点击 |
| 运行设备和GPU状态置顶 | 移到配置参数最上方(折叠展开按钮下)，从D.嵌入中移除 |
| 参数按流程顺序重排 | UI顺序改为 A数据源 → D嵌入 → B聚类 → C VLM → E后处理 → F输出 → G优化 |
| 全局参数字母调整 | D嵌入→B，B聚类→C，C VLM→D (workflow-structure.md和ui/app.py同步更新) |

**测试数据**: test_pics (123张图)  
**聚类结果**: 17个簇 + 32张噪音图像 (26%)  
**Web UI**: http://127.0.0.1:7860 ✅  
**可参考**: DermNet (DINOv2), img2cluster (聚类), labelme (扫描)  
**完成标志**: ✅ Gradio运行端到端，生成整理后文件夹，UI完整展示所有参数

---

#### 2026-01-31 聚类与参数增强（对话记录）

| 问题/需求 | 解决方案 |
|----------|----------|
| min_samples 默认值 | 改为 2（兼顾小数据集） |
| 去除 C12 Allow Single Cluster | 移除 UI 复选框，backend 固定为 false |
| HDBSCAN 参数敏感性测试 | `scripts/test_hdbscan_params.py`，支持 `--multi` 多组数据推荐默认值 |
| HDBSCAN 参数影响可视化 | 绘制 n_clusters、n_noise 随参数变化图 |
| 多组数据选默认值和范围 | MULTI_DATASETS 8 组，`recommend_defaults()` 聚合打分，输出 hdbscan_recommendations.json |
| DBSCAN 只有 eps 一个参数 | 新增 C4b 距离度量 euclidean/cosine，C4c 最近邻算法 auto/ball_tree/kd_tree/brute |
| DBSCAN 参数测试 | `scripts/test_dbscan_params.py`，验证 metric/algorithm 对结果和速度的影响 |
| cosine + ball_tree 报错 | step2_clustering 自动将 cosine 与 ball_tree/kd_tree 组合改为 brute |
| 默认使用 DBSCAN | config/UI 默认 backend: sklearn，C1 下拉默认 DBSCAN |

**新增脚本**: `test_hdbscan_params.py`（--multi 推荐）、`test_dbscan_params.py`  
**新增输出**: `data/output/hdbscan_recommendations.json`、`hdbscan_param_ranges.json`、`hdbscan_tuning_guide.md`  
**参数变更**: min_samples 2-30，persistence 默认 0.2；去除 C12；C4b/C4c DBSCAN 专用

---

### Phase-2: 配置系统 ✅

**目标**: 32+ 参数可配置；文档与编号统一；输出命名默认簇序号/簇序号@簇标签@原名；UI 副标题为简要流程

```
Phase-2
├── 配置文件
│   ├── [x] config.yaml (A-G 七组，G 含 G11/G12/G13 共 13 项)
│   └── [x] prompts.yaml
├── UI
│   ├── [x] 7个配置面板 (Accordion A-G)，F/E 固定项子手风琴
│   └── [x] 副标题：索引 → 嵌入 → 聚类 →（可选）采样/描述/标签 → 按簇整理
├── [x] 保存/加载配置
├── [x] workflow-structure.md 配置章节
├── [x] 文档更新：S1-S3→G11/G12/G13，numbering-system/README 同步
└── [x] F2 默认 id@label@original（簇序号/簇序号@簇标签@原名）
```

**完成标志**: ✅ 32+ 参数可配置；文档与编号统一（S1-S3→G11/G12/G13）；F2 默认簇序号命名；UI 简要流程副标题

#### 2026-01-31 文档更新（S1-S3 并入 G 类）

| 问题/需求 | 解决方案 |
|----------|----------|
| 系统参数 S1/S2/S3 归类 | 将 `system.output_base_directory`、`system.cache_directory`、`system.log_level` 从「系统/运行环境」表移至 G. 优化表，编号为 G11/G12/G13 |
| workflow-structure.md | G 表由 10 个改为 13 个，新增 G11 输出根目录、G12 缓存目录、G13 日志级别；系统小节仅保留合并说明（S1–S3→G11/G12/G13） |
| 4 个文档同步 | todo 记录本对话；numbering-system 更新 G 范围与合并说明；README/workflow-structure 更新「最后更新」 |

---

### Phase-3: VLM语义标签

**目标**: Step-3/4/5 生成语义标签；可选 D2=跳过 时用簇序号直通 Step-8

```
Phase-3
├── [x] Step-3: 多点采样（可选，代表模式需此步）
├── [x] Step-4: 并行描述（占位描述，VLM 待接入）
│   ├── [x] 模式1: 仅描述代表图像（需 Step-3 采样）
│   └── [x] 模式2: 语义描述所有图片（跳过 Step-3）
├── [x] Step-5: 语义蒸馏（占位蒸馏，LLM 待接入）
├── [x] 更新 Step-8：支持 S5_cluster_labels.csv 语义标签命名
├── [x] run_pipeline 接入 S3→S4→S5→S8
├── [x] D2 跳过选项：选「跳过（用簇序号）」时跳过 Step-3/4/5，Step-8 用簇序号命名
└── [x] 接入真实 VLM/LLM（本地开源：Qwen2-VL，models/vlm_models.py）
```

**可参考**: Qwen2-VL (VLM推理)  
**完成标志**: ✅ 流程跑通；有 VLM 时用真实描述与簇标签，无 VLM 时用占位；D2 可选「跳过」直通 Step-8

#### 2026-02-04 流程与输出增强（对话记录）

| 问题/需求 | 解决方案 |
|----------|----------|
| D2 增加跳过选项 | D2 下拉新增「跳过（用簇序号）」；选中时跳过 Step-3/4/5，Step-8 用簇序号命名（cluster_00、noise 等） |
| UI 副标题 | 「智能图像聚类与语义标注工具 - Phase 1 MVP」改为简要流程：「索引 → 嵌入 → 聚类 →（可选）采样/描述/标签 → 按簇整理」 |
| F. 输出 与 E 一样 | F. 输出 下增加「固定项」子手风琴，F1–F4 放入固定项；标题改为「F. 输出」 |
| 命名改为 簇序号/簇序号@簇标签@原名 | 新增 F2 规则 `id@label@original`：文件夹=簇序号（00/01/noise），文件名=簇序号@簇标签@原名；config 与文档默认改为该规则 |

#### 2026-02-04 聚类与索引增强（对话记录）

| 问题/需求 | 解决方案 |
|----------|----------|
| C1 默认 HDBSCAN | clustering.backend 默认改为 hdbscan；UI 下拉 HDBSCAN (默认)、DBSCAN |
| C10 默认 0.4 | cluster_selection_persistence 默认 0.2→0.4（config、UI、文档） |
| 统计与簇分布增加矩形图 | 曾添加 matplotlib 簇分布柱状图；后因 CJK 字体/webp 问题删除 |
| 不要大小过滤 | A3 min_file_size_kb 默认 0；Step-0 打印「不进行大小过滤」；min≤0 且 max≤0 时跳过大小校验 |
| Invalid/filtered 271 + Hash duplicates 271 | 原逻辑重复计数（hash 重复已含在 invalid 中）；改为互不重叠：Skipped (excluded, invalid/size, hash duplicates, errors) |

---

### Phase-4: 噪音挽救

```
Phase-4
├── [ ] Step-6: 噪音语义匹配
└── [ ] 更新文档
```

**完成标志**: ✅ 噪音挽救统计

---

### Phase-5: 可视化确认

```
Phase-5
├── [ ] Step-7: Plotly散点图 + UMAP降维
├── [ ] 图像悬停预览 + 拖拽重分配
└── [ ] 更新文档
```

**可参考**: Interactive-TSNE, pix-plot  
**完成标志**: ✅ 交互式可视化

---

### Phase-6: 文件预览

```
Phase-6
├── [ ] Step-8预览表格（原→新路径）
├── [ ] 冲突检测
└── [ ] 更新文档
```

**完成标志**: ✅ 预览功能完整

---

### Phase-7: 性能优化

```
Phase-7
├── [ ] [G1-G13]优化选项实现（G11-G13 仅 config）
├── [ ] 可选: cuML/FAISS/vLLM加速
└── [ ] 更新文档
```

**可参考**: vLLM, cuML, FAISS  
**完成标志**: ✅ 10倍速度提升

---

### Phase-8: 完善发布

```
Phase-8
├── [ ] README完善（安装、使用）
├── [ ] requirements.txt完整版
├── [ ] 示例数据集
└── [ ] 最终文档检查（4个文档）
```

**完成标志**: ✅ 用户可一键安装运行

---

### Phase-9: 高级功能（可选）

```
Phase-9
├── [ ] 会话管理
├── [ ] 批量处理
├── [ ] 3D可视化
├── [ ] 导出功能
└── [ ] 插件系统
```
