# 基于 RIVER 的统一长程视频交互评测框架设计方案

## 1. 项目定位

本框架旨在构建一个面向 **长程视频理解、实时交互、历史记忆与主动响应能力** 的统一评测平台。第一阶段以 RIVER Bench 为基础数据与任务协议，接入当前主流多模态大模型，例如 GPT、Gemini、Claude、Qwen、GLM、InternVL、MiniCPM 等，对其在长程视频交互任务中的能力进行横向评测。

后续阶段，该框架可以进一步接入实验室自建的电力作业视频数据集，并用于评估改进记忆机制后的本地多模态模型，从而服务于“面向电力作业现场的智能巡查与安全监督大模型选型”。

RIVER 的核心价值在于它不是普通离线视频问答，而是将在线视频交互拆分为 **Retro-Memory、Live-Perception、Pro-Response** 三类任务，分别评估模型对过去、当前和未来事件的理解与响应能力。

因此，本框架的目标不是简单写几个模型调用脚本，而是构建一个可以长期复用的：

```text
长程视频交互评测平台
```

------

## 2. 设计目标

本框架需要满足以下目标：

第一，**数据集可替换**。
第一版接入 RIVER 官方 JSON 和视频，后续可以接入电力作业数据集、仿真视频数据集、生成式视频数据集。

第二，**任务协议可扩展**。
保留 RIVER 的 Retro-Memory、Live-Perception、Pro-Response，同时支持后续新增电力场景下的作业步骤识别、违规检测、风险预警、规程符合性判断等任务。

第三，**模型接入统一**。
无论是 GPT、Gemini、Claude 这类 API 模型，还是 Qwen、InternVL、MiniCPM 这类本地模型，都通过统一接口调用。

第四，**支持在线交互模拟**。
框架不能只把完整视频一次性输入模型，而要能够模拟视频流逐步到来，模型在每个时间点只能看到当前窗口和历史记忆。

第五，**记忆机制可插拔**。
后续你们如果改进本地模型的记忆模块，只需要实现新的 Memory 或 Model Adapter，不需要重写数据集和任务逻辑。

第六，**结果可追踪、可分析、可复现**。
不仅保存最终分数，还要保存每个时间步模型看到的视频范围、模型输出、响应时间、是否提前报警、是否延迟报警等完整 trace。

------

## 3. 总体架构设计

框架采用分层解耦设计，核心结构如下：

```text
river_modern_eval/
    Dataset Layer
        负责读取 RIVER 或电力场景数据，并统一样本格式

    Task Layer
        负责定义任务类型、prompt、可见性策略、输出格式和评分方式

    Interaction Engine Layer
        负责模拟离线问答、滑动窗口在线视频流、连续流式交互

    Video Layer
        负责视频读取、抽帧、裁剪、clip 构建和缓存

    Memory Layer
        负责维护历史视频信息、事件记忆、状态机记忆、规程记忆

    Model Layer
        负责统一调用不同多模态大模型，包括 API 模型和本地模型

    Evaluation Layer
        负责准确率、响应时间、误报率、漏报率等指标计算

    Storage / Trace Layer
        负责保存结果、完整交互轨迹、缓存和实验日志

    Runner Layer
        负责读取配置文件，组织完整评测流程
```

整体流程如下：

```text
RIVER JSON / 电力作业 JSON
        ↓
Dataset Adapter 统一样本格式
        ↓
Task Adapter 构建任务协议
        ↓
Interaction Engine 模拟视频流
        ↓
Video Module 构建当前可见视频上下文
        ↓
Memory Module 更新与检索历史信息
        ↓
Model Adapter 调用多模态大模型
        ↓
Evaluator 计算任务分数
        ↓
Trace Logger 保存完整过程
        ↓
Result Aggregator 输出模型评测报告
```

------

## 4. 推荐目录结构

```text
river_modern_eval/
│
├── configs/
│   ├── datasets/
│   │   ├── river.yaml
│   │   └── electric_safety.yaml
│   │
│   ├── models/
│   │   ├── gpt.yaml
│   │   ├── gemini.yaml
│   │   ├── claude.yaml
│   │   ├── qwen_api.yaml
│   │   ├── qwen_local.yaml
│   │   ├── internvl_local.yaml
│   │   ├── minicpm_local.yaml
│   │   └── custom_memory_model.yaml
│   │
│   ├── tasks/
│   │   ├── retro_memory.yaml
│   │   ├── live_perception.yaml
│   │   ├── pro_response_instant.yaml
│   │   └── pro_response_streaming.yaml
│   │
│   └── experiments/
│       ├── exp_river_gpt_retro.yaml
│       ├── exp_river_all_models.yaml
│       └── exp_electric_memory_ablation.yaml
│
├── datasets/
│   ├── base_dataset.py
│   ├── river_dataset.py
│   ├── electric_dataset.py
│   └── schema.py
│
├── tasks/
│   ├── base_task.py
│   ├── retro_memory.py
│   ├── live_perception.py
│   ├── pro_response_instant.py
│   ├── pro_response_streaming.py
│   └── visibility.py
│
├── interaction/
│   ├── base_engine.py
│   ├── offline_engine.py
│   ├── window_stream_engine.py
│   └── streaming_engine.py
│
├── protocols/
│   ├── base_protocol.py
│   ├── mcq_protocol.py
│   ├── open_ended_protocol.py
│   ├── pro_response_protocol.py
│   └── electric_safety_protocol.py
│
├── models/
│   ├── base_model.py
│   ├── openai_model.py
│   ├── gemini_model.py
│   ├── claude_model.py
│   ├── qwen_api_model.py
│   ├── qwen_local_model.py
│   ├── internvl_local_model.py
│   ├── minicpm_local_model.py
│   └── custom_memory_model.py
│
├── memory/
│   ├── base_memory.py
│   ├── no_memory.py
│   ├── sliding_window_memory.py
│   ├── token_memory.py
│   ├── event_memory.py
│   ├── graph_memory.py
│   └── rule_state_memory.py
│
├── video/
│   ├── video_loader.py
│   ├── frame_sampler.py
│   ├── clip_builder.py
│   └── video_cache.py
│
├── eval/
│   ├── multiple_choice.py
│   ├── open_ended_judge.py
│   ├── response_time_score.py
│   ├── temporal_metrics.py
│   └── aggregate.py
│
├── storage/
│   ├── result_writer.py
│   ├── trace_writer.py
│   ├── cache.py
│   └── cost_tracker.py
│
├── runners/
│   ├── run_eval.py
│   ├── run_ablation.py
│   └── run_model_compare.py
│
└── results/
    ├── raw/
    ├── traces/
    ├── summaries/
    └── reports/
```

------

## 5. 数据层 Dataset 设计

数据层负责读取原始数据，并将不同数据集统一转换成标准样本格式。

### 5.1 统一样本格式

无论是 RIVER 数据，还是后续电力作业数据，都应该转换成如下格式：

```json
{
  "sample_id": "river_000001",
  "dataset": "RIVER",
  "task_type": "retro_memory",
  "video_source": "Ego4D",
  "video_id": "xxx",
  "video_path": "/path/to/video.mp4",

  "question": "Where did the person put the bag?",
  "choices": ["A. on the chair", "B. on the table", "C. on the floor", "D. on the bed"],
  "answer": "B",

  "question_time": 64.7,
  "time_reference": [15.0, 19.0],

  "metadata": {
    "cue_type": "fine_grained",
    "duration": 120.0,
    "original_annotation": {}
  }
}
```

其中关键字段含义如下：

| 字段             | 含义                                     |
| ---------------- | ---------------------------------------- |
| `sample_id`      | 样本唯一编号                             |
| `dataset`        | 数据集名称                               |
| `task_type`      | 任务类型                                 |
| `video_path`     | 视频路径                                 |
| `question`       | 用户问题或指令                           |
| `choices`        | 选择题选项                               |
| `answer`         | 标准答案                                 |
| `question_time`  | 用户提问时间                             |
| `time_reference` | 关键视觉线索出现时间                     |
| `metadata`       | 额外信息，如线索类型、场景类别、风险等级 |

### 5.2 Dataset 基类

```python
class BaseDataset:
    def load_samples(self):
        raise NotImplementedError

    def get_sample(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
```

### 5.3 RIVER 数据适配器

```python
class RiverDataset(BaseDataset):
    def __init__(self, annotation_path, video_root):
        self.annotation_path = annotation_path
        self.video_root = video_root
        self.samples = self.load_samples()

    def load_samples(self):
        """
        读取 RIVER 官方 JSON，
        将原始字段统一转换成 StandardSample。
        """
        pass
```

### 5.4 电力数据适配器

后续接入电力作业数据时，只需要新增：

```python
class ElectricSafetyDataset(BaseDataset):
    def load_samples(self):
        """
        读取电力作业视频标注，
        转换为统一样本格式。
        """
        pass
```

电力场景样本示例：

```json
{
  "sample_id": "electric_000001",
  "dataset": "ElectricSafety",
  "task_type": "pro_response_instant",
  "video_path": "/path/to/electric_video.mp4",

  "question": "当工作人员未验电直接安装接地线时，请立即报警。",
  "answer": "工作人员跳过验电步骤，直接安装接地线，存在高风险违规。",

  "question_time": 0.0,
  "time_reference": [52.3, 56.8],

  "risk_level": "high",
  "procedure_id": "grounding_operation",
  "required_previous_steps": ["power_off", "verification"],

  "metadata": {
    "scene": "substation",
    "violation_type": "procedure_skip",
    "target_object": "grounding_wire"
  }
}
```

------

## 6. 任务层 Task 设计

任务层负责定义不同任务的输入协议、可见性策略、prompt 构造、输出解析和评分方法。

### 6.1 任务类型

第一阶段保留 RIVER 的三类任务：

```text
Retro-Memory
Live-Perception
Pro-Response
```

进一步细分为：

```text
retro_memory
live_perception
pro_response_instant
pro_response_streaming
```

后续电力场景可扩展为：

```text
procedure_state_recognition
violation_detection
risk_warning
procedure_compliance_check
long_horizon_safety_supervision
```

### 6.2 Task 基类

```python
class BaseTask:
    def get_visibility_policy(self, sample):
        raise NotImplementedError

    def build_prompt(self, sample, context):
        raise NotImplementedError

    def parse_output(self, raw_output):
        raise NotImplementedError

    def get_evaluator(self):
        raise NotImplementedError
```

------

## 7. 可见性策略 Visibility Policy

这是整个框架中非常关键的一层。
它决定模型在某个时间点到底能看到哪些视频内容，防止模型看到未来信息，造成评测泄露。

### 7.1 VisibilityPolicy 数据结构

```python
@dataclass
class VisibilityPolicy:
    mode: str
    window_size: float
    stride: float
    allow_future: bool = False
    use_memory: bool = True
    fps: float = 1.0
```

### 7.2 不同任务的可见范围

| 任务                   | 模型可见视频范围                   |
| ---------------------- | ---------------------------------- |
| Retro-Memory           | 当前窗口 + 历史记忆                |
| Live-Perception        | 当前窗口                           |
| Instant Pro-Response   | 当前时刻之前的视频流，不能看到未来 |
| Streaming Pro-Response | 视频流逐步推进，模型多次响应       |
| 电力违规预警           | 当前窗口 + 作业历史状态 + 规程约束 |

------

## 8. 交互模拟层 Interaction Engine 设计

这是相比普通视频问答框架最重要的新增模块。

普通视频问答通常是：

```text
输入完整视频
↓
输入问题
↓
模型回答
```

但 RIVER 式任务应该是：

```text
视频流逐步推进
↓
用户在某个时间点提出问题
↓
模型只能看到当前窗口和历史记忆
↓
模型判断是否回答
↓
对回答内容和回答时机同时评分
```

### 8.1 OfflineEngine

用于传统离线评测。

```text
模型一次性看到完整视频或指定片段
适合作为传统 VideoQA baseline
class OfflineEngine:
    def run(self, sample, task, model):
        video_context = build_full_context(sample)
        prompt = task.build_prompt(sample, video_context)
        raw_output = model.generate_once(prompt, video_context)
        return raw_output
```

### 8.2 WindowStreamEngine

用于滑动窗口式在线评测。

```text
按照固定 stride 推进视频
每次只给模型当前窗口
历史信息由 Memory 模块提供
class WindowStreamEngine:
    def run(self, sample, task, model, memory):
        memory.reset(sample.sample_id)

        for current_time in self.timeline(sample):
            video_context = self.build_context(sample, current_time)
            retrieved_memory = memory.retrieve(sample.question, current_time)

            prompt = task.build_prompt(sample, video_context)
            raw_output = model.step(
                prompt=prompt,
                video_context=video_context,
                memory=retrieved_memory,
                current_time=current_time
            )

            memory.update(video_context, raw_output)

            if task.should_stop(raw_output):
                return raw_output, current_time
```

### 8.3 StreamingEngine

用于连续流式响应任务。

```text
模型可以在多个时间点输出内容
适合实时描述、持续监督、连续风险提醒
class StreamingEngine:
    def run(self, sample, task, model, memory):
        outputs = []

        for current_time in self.timeline(sample):
            video_context = self.build_context(sample, current_time)
            retrieved_memory = memory.retrieve(sample.question, current_time)

            raw_output = model.step(
                prompt=task.build_prompt(sample, video_context),
                video_context=video_context,
                memory=retrieved_memory,
                current_time=current_time
            )

            memory.update(video_context, raw_output)

            if raw_output is not None:
                outputs.append({
                    "time": current_time,
                    "output": raw_output
                })

        return outputs
```

------

## 9. 视频处理层 Video Module 设计

视频层负责把原始视频转换成模型可用的输入。

### 9.1 VideoContext 标准对象

不同模型支持的视频输入不同，因此需要统一封装：

```python
@dataclass
class VideoContext:
    video_id: str
    current_time: float
    visible_start: float
    visible_end: float

    video_path: str | None = None
    clip_path: str | None = None

    frames: list | None = None
    frame_timestamps: list[float] | None = None

    fps: float = 1.0
    metadata: dict = field(default_factory=dict)
```

### 9.2 视频处理模块

```text
VideoLoader
    负责读取视频基本信息

FrameSampler
    按 fps、窗口范围抽帧

ClipBuilder
    根据 visible_start 和 visible_end 裁剪视频片段

VideoCache
    缓存已抽取帧和已裁剪片段，避免重复处理
```

对于不同模型：

| 模型类型      | 推荐输入                    |
| ------------- | --------------------------- |
| GPT 系        | 抽帧图片序列                |
| Claude 系     | 图片序列或关键帧            |
| Gemini 系     | 视频文件或视频片段          |
| Qwen 本地     | frames tensor               |
| InternVL 本地 | frames tensor               |
| MiniCPM 本地  | 高 FPS 抽帧或压缩视频 token |
| 自研记忆模型  | 当前帧 + memory tokens      |

------

## 10. 记忆层 Memory 设计

记忆层是为了支持长程视频任务和后续你们自己的模型改进。

### 10.1 Memory 基类

```python
class BaseMemory:
    def reset(self, sample_id: str):
        raise NotImplementedError

    def update(self, video_context, model_output=None):
        raise NotImplementedError

    def retrieve(self, query, current_time):
        raise NotImplementedError

    def serialize(self):
        raise NotImplementedError
```

### 10.2 记忆模块类型

| 记忆模块            | 说明                   | 作用                  |
| ------------------- | ---------------------- | --------------------- |
| NoMemory            | 不使用记忆             | 作为 baseline         |
| SlidingWindowMemory | 只保留最近若干窗口     | 短期上下文            |
| TokenMemory         | 保存压缩视觉 token     | 模拟 RIVER 式长期记忆 |
| EventMemory         | 保存结构化事件         | 适合长程动作链分析    |
| GraphMemory         | 构建事件图或对象关系图 | 支持因果和关系推理    |
| RuleStateMemory     | 保存作业步骤状态机     | 适合电力规程监督      |

### 10.3 电力场景中的记忆设计

电力作业不只是“记住画面”，还要记住：

```text
当前作业进行到哪一步；
哪些前置步骤已经完成；
哪些安全检查还没有完成；
人员是否进入过危险区域；
工具是否被正确使用；
设备状态是否发生过变化。
```

因此后续可以设计 `RuleStateMemory`：

```python
class RuleStateMemory(BaseMemory):
    def update(self, video_context, model_output=None):
        """
        从模型输出或事件检测结果中更新作业状态。
        例如：
        power_off = True
        verification = False
        grounding_wire_installing = True
        """
        pass

    def retrieve(self, query, current_time):
        """
        返回当前作业状态和潜在违规信息。
        """
        pass
```

------

## 11. 模型层 Model Adapter 设计

模型层负责屏蔽不同大模型的调用差异。

### 11.1 BaseModel 接口

```python
class BaseModel:
    def reset(self):
        pass

    def generate_once(self, prompt, video_context, memory=None):
        """
        单次回答。
        适用于 Retro-Memory 和 Live-Perception。
        """
        raise NotImplementedError

    def step(self, prompt, video_context, memory=None, current_time=None):
        """
        在线交互。
        适用于 Pro-Response。
        模型可以回答，也可以输出 <NO_RESPONSE>。
        """
        raise NotImplementedError
```

### 11.2 模型适配器

第一阶段建议接入：

```text
OpenAIModel
GeminiModel
ClaudeModel
QwenAPIModel
```

第二阶段接入：

```text
QwenLocalModel
InternVLLocalModel
MiniCPMLocalModel
GLMLocalModel
```

第三阶段接入：

```text
CustomMemoryModel
```

### 11.3 step 接口设计

对于支持流式或状态保持的模型：

```python
def step(self, prompt, video_context, memory=None, current_time=None):
    return self.streaming_generate(prompt, video_context, memory)
```

对于不支持真正流式的视频模型：

```python
def step(self, prompt, video_context, memory=None, current_time=None):
    raw_output = self.generate_once(prompt, video_context, memory)

    if "<NO_RESPONSE>" in raw_output:
        return None

    return raw_output
```

这样可以用伪在线方式兼容大多数闭源 API 模型。

------

## 12. Prompt 协议层 Protocol 设计

为了避免不同模型输出格式混乱，建议单独设计 Protocol 层。

### 12.1 选择题协议

```text
You are evaluating a video understanding task.
You must answer with only one option: A, B, C, or D.
Do not provide explanations.
```

### 12.2 开放式问答协议

```text
Answer the question based only on the visible video content and the provided memory.
Do not guess events that are not visible.
Keep the answer concise.
```

### 12.3 Pro-Response 协议

```text
You are watching a video stream.

Instruction:
{question}

If the target event has not occurred, output:
<NO_RESPONSE>

If the target event occurs, output:
<ALERT> followed by a short explanation.
```

### 12.4 电力安全协议

```text
你正在监督电力作业现场。

你需要根据当前视频画面、历史作业状态和安全规程判断是否存在风险。

如果没有发现违规或风险，输出：
<NO_RESPONSE>

如果发现风险，输出：
<ALERT>
风险类型：
风险等级：
证据：
建议：
```

------

## 13. 评测层 Evaluation 设计

评测层需要同时评估“答得对不对”和“答得是否及时”。

### 13.1 Retro-Memory 指标

| 指标                      | 含义                   |
| ------------------------- | ---------------------- |
| MC Accuracy               | 选择题准确率           |
| Open-ended Accuracy       | 开放式回答准确率       |
| Recall Interval Accuracy  | 不同记忆间隔下的准确率 |
| Fine-grained Cue Accuracy | 细粒度视觉线索准确率   |
| Causal Cue Accuracy       | 因果线索准确率         |
| Background Cue Accuracy   | 背景线索准确率         |

### 13.2 Live-Perception 指标

| 指标                      | 含义               |
| ------------------------- | ------------------ |
| Current State Accuracy    | 当前状态识别准确率 |
| Object Accuracy           | 目标识别准确率     |
| Action Accuracy           | 动作识别准确率     |
| Spatial Relation Accuracy | 空间关系理解准确率 |

### 13.3 Pro-Response 指标

| 指标                 | 含义                       |
| -------------------- | -------------------------- |
| Response Accuracy    | 是否在正确事件上响应       |
| Response Latency     | 响应延迟                   |
| Early Alarm Rate     | 提前报警率                 |
| Late Alarm Rate      | 延迟报警率                 |
| Miss Rate            | 漏报率                     |
| False Alarm Rate     | 误报率                     |
| Time-to-Detect       | 从事件发生到模型报警的时间 |
| Duplicate Alarm Rate | 重复报警率                 |

### 13.4 响应时间评分

Pro-Response 可以采用如下评分逻辑：

```text
如果模型提前报警：
    score = 0

如果模型在容忍窗口内报警：
    score = 1

如果模型延迟报警：
    score 随延迟线性下降

如果模型一直没有报警：
    score = 0
```

伪代码：

```python
def response_time_score(pred_time, gt_time, tolerance, max_delay):
    if pred_time is None:
        return 0.0

    if pred_time < gt_time:
        return 0.0

    delay = pred_time - gt_time

    if delay <= tolerance:
        return 1.0

    if delay >= max_delay:
        return 0.0

    return 1.0 - (delay - tolerance) / (max_delay - tolerance)
```

------

## 14. Trace Logger 设计

结果保存不能只保存最终分数，还要保存完整交互轨迹。

### 14.1 最终结果格式

```json
{
  "run_id": "exp_001",
  "sample_id": "river_000001",
  "model_name": "gpt-5.5",
  "task_type": "pro_response_instant",

  "question": "Tell me when the person picks up the bag.",
  "answer": "The person picks up the bag.",

  "ground_truth_time": 48.5,
  "prediction": "The person has picked up the bag.",
  "prediction_time": 51.0,

  "score": 0.83,
  "latency": 2.5,
  "is_early": false,
  "is_late": true,
  "is_missed": false
}
```

### 14.2 交互轨迹格式

```json
{
  "sample_id": "river_000001",
  "model_name": "gpt-5.5",
  "timeline_trace": [
    {
      "current_time": 12.0,
      "visible_range": [0.0, 12.0],
      "memory_summary": "",
      "model_output": "<NO_RESPONSE>"
    },
    {
      "current_time": 16.0,
      "visible_range": [0.0, 16.0],
      "memory_summary": "A person entered the room.",
      "model_output": "<NO_RESPONSE>"
    },
    {
      "current_time": 52.0,
      "visible_range": [36.0, 52.0],
      "memory_summary": "The bag was on the table.",
      "model_output": "<ALERT> The person has picked up the bag."
    }
  ]
}
```

这对于后续论文分析非常重要，因为你可以研究：

```text
模型是提前报警还是延迟报警；
模型是否在没有证据时幻觉响应；
模型是否对历史事件遗忘；
模型是否在关键事件附近响应不稳定；
模型是否存在重复报警问题。
```

------

## 15. 缓存与成本控制设计

评测闭源多模态模型时，必须考虑 API 费用、速率限制和失败重试。

### 15.1 请求缓存

每次模型请求生成唯一哈希：

```text
request_hash = hash(
    model_name
    + sample_id
    + prompt
    + visible_start
    + visible_end
    + frame_hash
)
```

如果同样请求已经运行过，直接读取缓存。

### 15.2 视频缓存

缓存内容包括：

```text
已抽取的视频帧；
已裁剪的视频 clip；
已上传到 API 平台的视频文件 ID；
已生成的视频摘要；
```

### 15.3 成本统计

记录：

```text
模型名称；
输入 token；
输出 token；
视频帧数量；
请求次数；
总费用估计；
平均单样本费用；
```

这样后续可以比较：

```text
能力最强模型；
性价比最高模型；
适合本地部署模型；
适合实时监督模型。
```

------

## 16. 实验配置设计

一个实验应该由 YAML 文件统一管理。

示例：

```yaml
experiment:
  name: exp_river_gpt_retro
  seed: 42
  output_dir: results/exp_river_gpt_retro

dataset:
  name: river
  annotation_path: data/river/annotations/retro_memory.json
  video_root: data/river/videos
  split: test
  max_samples: 100

task:
  name: retro_memory
  protocol: mcq
  visibility:
    mode: window
    window_size: 16
    stride: 4
    fps: 1
    allow_future: false
    use_memory: true

model:
  name: gpt
  version: gpt-5.5
  input_type: frames
  max_frames: 32
  temperature: 0

memory:
  name: sliding_window
  max_windows: 8

evaluation:
  metrics:
    - mc_accuracy
    - recall_interval_accuracy

storage:
  save_raw_output: true
  save_trace: true
  use_cache: true
```

------

## 17. 第一阶段：最小可行版本

第一阶段不要做得过大，目标是先跑通完整闭环。

### 17.1 第一阶段目标

```text
接入 RIVER 官方 JSON
接入一个 API 模型
跑通 Retro-Memory 小子集
支持视频抽帧
支持选择题准确率
保存 results.jsonl
```

### 17.2 第一阶段模块

```text
RiverDataset
RetroMemoryTask
FrameSampler
OpenAIModel 或 GeminiModel
MultipleChoiceEvaluator
ResultWriter
```

### 17.3 第一阶段流程

```text
读取 RIVER Retro-Memory JSON
↓
转换为统一 sample
↓
根据 question_time 构建可见视频窗口
↓
抽取关键帧
↓
构造选择题 prompt
↓
调用模型
↓
解析 A/B/C/D
↓
计算准确率
↓
保存结果
```

------

## 18. 第二阶段：扩展在线交互能力

第二阶段重点加入 Interaction Engine 和 Pro-Response。

### 18.1 第二阶段目标

```text
支持 Live-Perception
支持 Pro-Response-Instant
支持 WindowStreamEngine
支持 <NO_RESPONSE> / <ALERT> 输出协议
支持响应时间评分
支持 timeline trace
```

### 18.2 第二阶段重点

这一阶段是框架从“视频问答评测”升级为“在线视频交互评测”的关键。

核心流程变为：

```text
视频流逐步推进
↓
每个时间步构建可见窗口
↓
模型判断是否响应
↓
记录每个时间步输出
↓
根据真实事件时间计算响应分数
```

------

## 19. 第三阶段：接入多模型评测

第三阶段开始扩展模型池。

### 19.1 建议模型池

闭源模型：

```text
GPT 系列
Gemini 系列
Claude 系列
```

国产 API 模型：

```text
Qwen-VL
GLM-V
Doubao / Seed-VL
Kimi-VL
```

本地开源模型：

```text
Qwen-VL
InternVL
MiniCPM-V / MiniCPM-o
Llama 多模态模型
```

### 19.2 输出模型排行榜

按任务维度输出：

```text
Retro-Memory 排名
Live-Perception 排名
Pro-Response 排名
平均响应延迟排名
误报率排名
成本排名
综合性价比排名
```

这样可以服务于你的问题：

```text
电力作业场景到底应该选哪个多模态大模型？
```

------

## 20. 第四阶段：加入记忆机制消融实验

这一阶段开始体现研究创新。

### 20.1 记忆机制对比

| 方法                | 历史记忆 | 事件结构化 | 规程状态机 | 适用任务     |
| ------------------- | -------- | ---------- | ---------- | ------------ |
| NoMemory            | 否       | 否         | 否         | 基础对照     |
| SlidingWindowMemory | 弱       | 否         | 否         | 短期上下文   |
| TokenMemory         | 是       | 否         | 否         | 长程视觉记忆 |
| EventMemory         | 是       | 是         | 否         | 长程事件链   |
| RuleStateMemory     | 是       | 是         | 是         | 电力规程监督 |

### 20.2 消融实验设计

```text
Base VLM
Base VLM + Sliding Window
Base VLM + Token Memory
Base VLM + Event Memory
Base VLM + Rule State Memory
```

观察指标：

```text
长程记忆准确率是否提升；
因果线索理解是否提升；
违规检测是否提升；
主动报警是否更及时；
误报率是否下降；
```

------

## 21. 第五阶段：接入电力作业场景

最后再将框架迁移到你们自己的电力作业数据。

### 21.1 电力任务设计

可以定义以下任务：

```text
作业历史回忆 Retro-Memory
    刚才是否完成验电？
    前一步是否放置警示牌？
    工作人员是否佩戴绝缘手套？

当前状态识别 Live-Perception
    当前人员是否进入危险区域？
    当前设备柜门是否打开？
    当前是否正在安装接地线？

主动风险响应 Pro-Response
    如果未验电直接接地，请立即报警。
    如果人员靠近带电区域，请立即报警。
    如果跳过安全检查步骤，请立即报警。

规程符合性判断 Procedure Compliance
    当前作业流程是否符合标准操作规程？
    是否存在前置条件未满足的问题？
```

### 21.2 电力数据标注字段

建议增加：

```json
{
  "risk_level": "high",
  "violation_type": "procedure_skip",
  "procedure_id": "grounding_operation",
  "current_step": "install_grounding_wire",
  "required_previous_steps": ["power_off", "verification"],
  "object_list": ["worker", "insulating_gloves", "grounding_wire"],
  "danger_zone": true,
  "alarm_time": 52.3
}
```

### 21.3 电力场景指标

除了 RIVER 原有指标，还应加入：

```text
违规识别准确率
风险等级判断准确率
前置步骤缺失识别率
危险区域入侵检测率
报警及时性
误报率
漏报率
重复报警率
规程链条一致性
```

------

## 22. 框架最终价值

这个框架最终可以支持三类研究问题。

### 22.1 模型选型问题

回答：

```text
当前世界主流多模态大模型中，哪个最适合电力作业现场监督？
```

可以从以下维度比较：

```text
长程记忆能力
当前状态识别能力
主动风险响应能力
中文规程理解能力
响应延迟
部署成本
私有化可行性
```

### 22.2 记忆机制研究问题

回答：

```text
什么样的记忆机制更适合长程复杂作业监督？
```

可以比较：

```text
滑动窗口记忆
token 级长期记忆
事件级记忆
图结构记忆
规程状态机记忆
```

### 22.3 电力 benchmark 构建问题

回答：

```text
如何从通用在线视频交互 benchmark 扩展到专业电力作业安全监督 benchmark？
```

核心创新点可以是：

```text
从通用视频问答扩展到专业规程监督；
从视觉事件识别扩展到作业步骤链验证；
从回答正确率扩展到报警及时性和安全风险评价；
从视频 token 记忆扩展到事件状态机记忆；
从通用 Pro-Response 扩展到安全关键场景下的主动风险响应。
```

------

# 最终框架总结

你原来的设计是：

```text
Dataset + Task + Model
```

改进后建议升级为：

```text
Dataset
+ Task
+ Interaction Engine
+ Video Context
+ Memory
+ Model Adapter
+ Protocol
+ Evaluator
+ Trace Logger
+ Runner
```

其中最关键的三个模块是：

```text
Interaction Engine：
    让评测真正模拟在线视频流，而不是离线视频问答。

Memory：
    为长程历史理解和后续记忆机制改进提供统一接口。

Trace Logger：
    保存完整交互过程，为误报、漏报、延迟、遗忘分析提供依据。
```

最终，这个框架不仅能复现和扩展 RIVER，还能自然迁移到你的电力作业现场智能巡查场景，形成一个面向长程复杂作业的统一多模态大模型评测平台。