# RIVER Eval —— 长程视频交互评测框架

<p align="center">
  <b>面向长程视频理解、实时交互与历史记忆的统一多模态大模型评测平台</b>
</p>

---

## 项目定位

本框架基于 [RIVER Bench](https://github.com/OpenGVLab/RIVER) 构建，目标不是简单的"视频问答脚本集合"，而是一个**可长期复用、可扩展、可分析**的长程视频交互评测平台。

传统视频问答通常把完整视频一次性输入模型，但 RIVER 的核心价值在于**模拟真实在线交互**：

- **Retro-Memory**：用户在视频后期提问，模型需要回忆早期发生的事件
- **Live-Perception**：用户在当前时间点提问，模型只能看到当前窗口
- **Pro-Response**：模型需要主动监测视频流，在目标事件出现时及时报警

因此，本框架强调**时间维度上的可见性控制**和**交互轨迹的完整记录**。

---

## 当前功能（第一阶段）

✅ **Retro-Memory 离线评测**
- 读取 RIVER 官方标注（Retro-Memory / Live-Perception / Pro-Response）
- 根据 `question_time` 构建可见视频窗口 `[0, question_time]`
- 均匀抽帧，构造选择题 Prompt
- 调用多模态大模型（OpenAI GPT-4o / 阿里百炼 Qwen-VL）
- 解析答案、计算准确率、保存完整结果

✅ **工程级基础设施**
- **请求缓存**：相同请求直接命中缓存，避免重复花钱调 API
- **帧缓存**：同一视频同参数抽过的帧保存复用，加速二次评测
- **成本追踪**：精确记录 input/output token 数和预估费用
- **错误处理**：API 内容审核等异常被捕获，不中断评测流程
- **配置驱动**：YAML 统一管理实验参数，支持命令行覆盖

---

## 项目结构

```
RIVER/
├── river_eval/              # 核心评测框架（重点维护）
│   ├── datasets/            # 数据集适配器（RIVER / 电力场景）
│   ├── tasks/               # 任务定义（Retro-Memory / Live-Perception / Pro-Response）
│   ├── models/              # 模型适配器（OpenAI / Qwen / 本地模型）
│   ├── eval/                # 评测指标（准确率 / 响应时间 / 误报率）
│   ├── utils/               # 工具（抽帧 / 缓存 / 成本追踪 / 路径解析）
│   └── runners/             # 评测入口
│
├── configs/                 # 实验配置文件
│   ├── retro_memory_gpt.yaml
│   └── retro_memory_qwen.yaml
│
├── annotations/             # RIVER 官方标注 JSON
│   ├── Retro-Memory.json
│   ├── Live-Perception.json
│   ├── Pro-Response-Instant.json
│   └── Pro-Response-Streaming.json
│
├── data_manifests/          # 视频清单（各来源的视频 ID 列表）
├── scripts/                 # 运行脚本
├── docs/                    # 设计文档
├── data/videos/             # 视频数据（本地存放，不上传 git）
└── results/                 # 评测结果（运行时生成，不上传 git）
```

---

## 快速开始

### 1. 环境准备

```bash
# Python 3.10+
pip install openai pyyaml python-dotenv

# ffmpeg（项目中已提供静态二进制，或系统安装）
# ./bin/ffmpeg 或系统 PATH 中的 ffmpeg
```

### 2. 配置 API Key

创建 `.env` 文件（已加入 `.gitignore`，不会提交）：

```bash
# 阿里百炼（Qwen-VL）
DASHSCOPE_API_KEY=sk-your-key-here

# OpenAI（GPT-4o）
OPENAI_API_KEY=sk-your-key-here
```

### 3. 准备视频数据

视频下载到 `data/videos/RIVER/` 目录下，按来源分子文件夹：

```
data/videos/RIVER/
├── Vript-RR/
│   └── -n5eIlDgY5w.mp4
├── LongVideoBench/
│   └── NAJOZTNkhlI.mp4
├── LVBench/
│   └── ...
└── ...
```

### 4. 运行评测

**使用 Qwen-VL-Plus（推荐，成本低）：**

```bash
python3 -m river_eval.runners.run_eval \
  --config configs/retro_memory_qwen.yaml \
  --max-samples 10
```

**使用 GPT-4o：**

```bash
python3 -m river_eval.runners.run_eval \
  --config configs/retro_memory_gpt.yaml \
  --max-samples 10
```

### 5. 查看结果

```bash
cat results/retro_memory_qwen/summary.json
cat results/retro_memory_qwen/cost_report.json
```

---

## 配置说明

以 `configs/retro_memory_qwen.yaml` 为例：

```yaml
experiment:
  name: retro_memory_qwen          # 实验名称
  output_dir: results/retro_memory_qwen
  max_samples: 10                   # 评测样本数（覆盖全部则为空）

dataset:
  annotation_path: annotations/Retro-Memory.json
  video_root: data/videos/RIVER

task:
  name: retro_memory
  max_frames: 16                    # 每样本抽帧数（控制成本和速度）
  frame_resolution: 448             # 帧短边分辨率

model:
  name: qwen_api                    # qwen_api / openai
  model: qwen-vl-plus               # qwen-vl-plus / qwen-vl-max / gpt-4o
  temperature: 0
  max_tokens: 16

storage:
  use_cache: true                   # 是否启用请求缓存和帧缓存
  cache_dir: auto                   # auto = results/{exp}/cache

cost:
  track: true                       # 是否追踪费用
  pricing:                          # 单价（USD / 1K tokens）
    qwen-vl-plus: [0.00050, 0.00050]
    gpt-4o: [0.00250, 0.01000]
```

---

## 五阶段演进路线

| 阶段 | 目标 | 状态 |
|------|------|------|
| **第一阶段** | Retro-Memory 离线评测闭环 | ✅ 已完成 |
| **第二阶段** | Live-Perception + Pro-Response + 在线交互引擎 | ⏳ 待实现 |
| **第三阶段** | 多模型横向评测（GPT / Gemini / Claude / Qwen / InternVL） | ⏳ 待实现 |
| **第四阶段** | 记忆机制消融（SlidingWindow / Token / Event / RuleState） | ⏳ 待实现 |
| **第五阶段** | 电力作业场景适配（违规检测 / 风险预警 / 规程监督） | ⏳ 待实现 |

---

## 核心设计原则

1. **数据集可替换**：`BaseDataset` 接口统一，接入电力数据只需写一个适配器
2. **模型接入统一**：`BaseModel` 接口屏蔽 OpenAI / 百炼 / 本地模型的差异
3. **记忆机制可插拔**：`BaseMemory` 接口预留，后续改进记忆模块无需重写任务逻辑
4. **结果可追踪**：每条样本保存 `visible_range`、`frame_count`、`cached`、`cost_usd` 等完整上下文
5. **成本可控**：缓存机制 + 配置化定价，避免重复花钱

---

## 协作开发指南

```bash
# 1. 克隆仓库
git clone https://github.com/yourname/RIVER-eval.git
cd RIVER-eval

# 2. 安装依赖
pip install -r requirements.txt  # 待补充

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env，填入你的 API Key

# 4. 运行测试
python3 -m river_eval.runners.run_eval \
  --config configs/retro_memory_qwen.yaml \
  --max-samples 3
```

---

## 引用

如果你使用了本框架或 RIVER 数据集，请引用：

```bibtex
@misc{shi2026riverrealtimeinteractionbenchmark,
  title={RIVER: A Real-Time Interaction Benchmark for Video LLMs},
  author={Yansong Shi and Qingsong Zhao and Tianxiang Jiang and Xiangyu Zeng and Yi Wang and Limin Wang},
  year={2026},
  eprint={2603.03985},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2603.03985}
}
```

---

## License

MIT License
