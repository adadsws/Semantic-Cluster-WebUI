# Semantic-Cluster-WebUI

> 📦 **发布版本**: v0.2 (2026-02-04)  
> 📅 **最后更新**: 2026-02-04  
> 🚀 **当前版本**: Phase-1 MVP + Phase-2 配置系统 + Phase-3 VLM 语义标签 (C1 HDBSCAN 默认、D2 跳过、F2 簇序号命名)

基于视觉模型和 VLM 的智能图像聚类与语义标注工具。

---


## 🎯 工作流程

```
输入图像 → Step-0:索引 → Step-1:嵌入 → Step-2:聚类 
         ↓
Step-3:采样 → Step-4:描述 → Step-5:标签生成

```

---

## 📦 安装

### 环境要求

- Python 3.9+
- CUDA 11.8+ (可选，用于GPU加速)
- 8GB+ RAM (16GB+ 推荐)

### Windows 快速安装

```bash
# 1. 克隆项目
git clone https://github.com/adadsws/Semantic-Cluster-WebUI.git
cd semantic-cluster-webui

# 2. 创建虚拟环境
python -m venv venv
venv\Scripts\activate # linux source venv/bin/activate

# 3. 安装依赖
pip install -r requirements-cu128.txt # If use GPU

# 4. Windows 下快速启动
restart-webui.bat   # 首次运行或重启 WebUI

# 5. 浏览器访问
打开 http://localhost:7860

```
---

## 📁 项目结构

```
semantic-cluster-webui/
├── data/
│   ├── input/           # 输入图像
│   └── output/          # 输出结果（S0-S8）
├── config/
│   ├── config.yaml      # 配置 [A1]-[G13]
│   └── prompts.yaml     # Prompt 模板
├── core/                # Step-0 到 Step-8
│   ├── step0_indexing.py
│   ├── step1_embedding.py
│   ├── step2_clustering.py
│   ├── step3_sampling.py
│   ├── step4_caption.py
│   ├── step5_label.py
│   └── step8_organization.py
├── models/
│   └── vlm_models.py    # VLM 模型 (Qwen2-VL)
├── scripts/             # 工具脚本
│   ├── check_gpu_env.py
│   ├── check_vlm.py
│   ├── test_hdbscan_params.py
│   └── test_dbscan_params.py
├── utils/
│   └── config_loader.py # 配置加载器
├── ui/
│   └── app.py           # Gradio主入口
├── requirements.txt        # Python 依赖
├── requirements-cu128.txt  # CUDA 12.8 额外依赖（可选）
├── restart-webui.bat       # Windows 启动脚本
└── README.md
```

---

## 📖 文档

- [docs/todo.md](./docs/todo.md) - 开发计划与任务清单
- [docs/workflow-structure.md](./docs/workflow-structure.md) - 工作流详细说明
- [docs/numbering-system.md](./docs/numbering-system.md) - 编号系统说明

---

## 🛠️ 技术栈

### 核心框架
- **UI**: Gradio 4.0+
- **配置**: OmegaConf
- **深度学习**: PyTorch 2.0+

### 视觉模型
- **DINOv2** (推荐) - HuggingFace facebook/dinov2-base 等，Python 3.9 兼容

### 聚类
- HDBSCAN - 聚类算法
- DBSCAN - 聚类算法

### VLM模型
- Qwen2-VL (本地推理)

---

## 🔧 开发进度

- [x] **Phase-0**: 环境搭建 ✅ (2026-01-31)
- [x] **Phase-1**: MVP 核心流程 ✅ (2026-01-31)
  - ✅ Step-0 索引 → Step-1 嵌入(DINOv2/CLIP) → Step-2 聚类(HDBSCAN/DBSCAN) → Step-8 整理
  - ✅ Gradio Web UI、GPU 检测、实时日志、配置保存/加载
- [x] **Phase-2**: 配置系统 ✅ (2026-01-31)
  - ✅ 32+ 参数 [A1]-[G13]、7 个配置面板、F2 簇序号命名
- [x] **Phase-3**: VLM 语义标签 ✅ (2026-02-04)
  - ✅ Step-3 采样 → Step-4 描述 → Step-5 标签生成
  - ✅ Qwen2-VL 本地推理；D2 跳过时用簇序号直通 Step-8
- [ ] **Phase-4**: 噪音挽救
- [ ] **Phase-5**: 可视化确认
- [ ] **Phase-6**: 文件预览
- [ ] **Phase-7**: 性能优化
- [ ] **Phase-8**: 完善发布



---

## 🙋 常见问题

### Q: 支持哪些图像格式？

A: 默认支持 jpg, png, webp, bmp, tiff。可在 `config.yaml` 中自定义。

### Q: 需要GPU吗？

A: 不是必需的，但强烈推荐。CPU模式下处理速度较慢。



