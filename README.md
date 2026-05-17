# LV_benchmark

LV_benchmark 是一个面向长程视频理解与多模态大模型评测的通用实验框架。项目当前以 RIVER Bench 作为第一阶段接入对象，用于构建长程视频交互评测的基本闭环；后续可扩展到本地多模态模型、自建视频数据集、机械狗巡检任务和电力作业异常检测任务。

本框架的核心目标不是单独复现某一个数据集，而是将长程视频评测过程拆分为可替换的数据集、任务、模型、抽帧策略、记忆机制和评分模块，使不同任务和不同模型可以在同一套框架下进行统一评测和对比。

## 项目定位

传统视频问答通常将完整视频一次性输入模型，然后让模型回答问题。这种方式适合普通离线视频理解，但不适合长程任务、在线交互和历史记忆评测。

LV_benchmark 关注的是更接近真实应用的长程视频理解问题：模型只能在特定时间点看到允许可见的视频内容，需要根据当前画面、历史信息和任务上下文完成判断。对于机械狗巡检、电力作业监督等场景，模型不仅要识别当前动作，还需要判断任务执行阶段、历史步骤是否完成、是否存在流程遗漏或规则违反。

因此，本项目将长程视频评测抽象为一个配置驱动的评测流程。每次实验通过配置文件指定数据集、任务类型、模型类型、抽帧策略、评分方式和存储行为。评测入口只负责调度流程，具体逻辑由各模块独立实现。

当前主线任务是 RIVER Retro-Memory API baseline。该任务用于验证框架的基本能力，包括读取 RIVER 标注、根据问题时间控制视频可见范围、抽取长短记忆帧、调用多模态模型、解析选择题答案并保存评测结果。

## 项目结构

```text
LV_benchmark/
├── benchmark/
│   ├── config.py
│   ├── schema.py
│   ├── datasets/
│   │   ├── base.py
│   │   ├── registry.py
│   │   └── river.py
│   ├── tasks/
│   │   ├── base.py
│   │   ├── registry.py
│   │   └── river/
│   │       ├── retro_memory.py
│   │       ├── live_perception.py
│   │       ├── pro_response_instant.py
│   │       └── pro_response_streaming.py
│   ├── models/
│   │   ├── base.py
│   │   ├── registry.py
│   │   ├── api/
│   │   │   ├── qwen_api_model.py
│   │   │   └── openai_model.py
│   │   └── local/
│   │       └── qwen_local_model.py
│   ├── video/
│   │   ├── frame_policies.py
│   │   ├── frame_sampler.py
│   │   ├── registry.py
│   │   └── river_policies.py
│   ├── memory/
│   │   ├── base.py
│   │   ├── no_memory.py
│   │   └── sliding_window_memory.py
│   ├── eval/
│   │   ├── multiple_choice.py
│   │   ├── registry.py
│   │   └── river/
│   │       └── retro_memory_scoring.py
│   ├── storage/
│   │   ├── cache.py
│   │   ├── cost_tracker.py
│   │   ├── result_writer.py
│   │   └── trace_writer.py
│   ├── utils/
│   │   └── video_resolver.py
│   └── runners/
│       └── run_eval.py
├── river_eval/
│   └── runners/
│       └── run_eval.py
├── configs/
│   └── experiments/
│       └── river_retro_qwen_strict.yaml
├── annotations/
│   ├── Retro-Memory.json
│   ├── Live-Perception.json
│   ├── Pro-Response-Instant.json
│   └── Pro-Response-Streaming.json
├── data_manifests/
├── scripts/
├── docs/
├── data/
│   └── videos/
└── results/
```

`benchmark/` 是当前主包，后续开发都应围绕该目录进行。`river_eval/` 只保留旧命令兼容入口，不再作为主开发目录。

主要模块说明：

```text
benchmark/datasets/    数据集读取与样本构造
benchmark/tasks/       任务定义与 prompt 构造
benchmark/models/      API 模型和本地模型适配
benchmark/video/       抽帧策略与视频窗口控制
benchmark/memory/      记忆机制接口与实现
benchmark/eval/        输出解析与评分逻辑
benchmark/storage/     缓存、结果、成本和 trace 存储
benchmark/runners/     统一评测入口
```

## 配置方式

实验通过 YAML 文件配置。当前主线配置文件为：

```text
configs/experiments/river_retro_qwen_strict.yaml
```

一个实验配置需要显式声明以下部分：

```yaml
experiment:
  name: river_retro_qwen_strict
  output_dir: results/river_retro_qwen_strict
  max_samples: 10

dataset:
  name: river_retro_memory
  annotation_path: annotations/Retro-Memory.json
  video_root: data/videos/RIVER

task:
  name: river_retro_memory

model:
  name: qwen_api
  model: qwen-vl-plus
  temperature: 0
  max_tokens: 16

video:
  frame_policy: river_long_short
  frame_resolution: 448

prompt:
  style: river_longshort

evaluation:
  scoring: river_mcq_accuracy

storage:
  use_cache: true
  cache_dir: auto
  save_trace: true
  cache_errors: false

cost:
  track: true
```

各字段含义如下：

```text
experiment     实验名称、输出目录和样本数量控制
dataset        数据集名称、标注路径和视频根目录
task           任务类型，例如 RIVER Retro-Memory
model          模型适配器和具体模型名称
video          抽帧策略、分辨率和视频输入控制
prompt         prompt 风格
evaluation     评分方式
storage        缓存、结果保存和 trace 行为
cost           token 与费用统计
```

当前推荐运行入口：

```bash
python -m benchmark.runners.run_eval \
  --config configs/experiments/river_retro_qwen_strict.yaml \
  --max-samples 1
```

旧入口仍可使用，但只作为兼容 wrapper：

```bash
python -m river_eval.runners.run_eval \
  --config configs/experiments/river_retro_qwen_strict.yaml \
  --max-samples 1
```

新实验应优先使用 `benchmark.runners.run_eval`。

## 开发指南

新增数据集时，在 `benchmark/datasets/` 下实现数据集适配器，并在对应 registry 中注册。数据集模块应负责读取标注、解析视频路径、构造统一样本对象，不应包含模型调用和评分逻辑。

新增任务时，在 `benchmark/tasks/` 下实现任务模块。任务模块应负责根据样本构造模型输入，包括问题、选项、可见时间范围、prompt 格式和必要的任务上下文。不同任务 family 可以放在独立子目录中，例如 `benchmark/tasks/river/`。

新增模型时，在 `benchmark/models/` 下实现模型适配器。API 模型放在 `benchmark/models/api/`，本地模型放在 `benchmark/models/local/`。模型适配器应遵循统一的 `BaseModel` 接口，避免在任务模块中直接写具体模型调用代码。

新增抽帧策略时，在 `benchmark/video/` 下实现 frame policy，并通过 registry 注册。抽帧策略应只负责决定从哪些时间点取帧，不应处理 prompt、评分或模型调用逻辑。

新增记忆机制时，在 `benchmark/memory/` 下实现 `BaseMemory` 接口。记忆模块应负责保存、更新和检索历史信息。不同记忆机制应能通过配置切换，便于做消融实验。

新增评分方式时，在 `benchmark/eval/` 下实现 scorer，并通过 registry 注册。评分模块应只负责解析模型输出和计算指标，不应修改模型输入或样本内容。

新增实验时，应优先新增 YAML 配置文件，而不是修改 `run_eval.py`。`run_eval.py` 只负责读取配置、构建组件和调度执行，不应写入具体任务或具体模型的特殊逻辑。

提交代码前建议至少运行一次最小 smoke test：

```bash
python -m benchmark.runners.run_eval \
  --config configs/experiments/river_retro_qwen_strict.yaml \
  --max-samples 1
```

同时检查：

```bash
python -m compileall benchmark river_eval
```

## 引用

如果使用 RIVER 数据集，请引用：

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

## License

MIT License