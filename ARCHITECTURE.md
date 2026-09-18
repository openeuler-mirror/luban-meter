# 架构与扩展指南

## 执行链路

```text
CLI / tasks.json
  → 显式导入内置模块 → 装饰器注册
  → 按场景与Profile参数定义校验 → 展开Case → 保存计划
  → RunCoordinator顺序启动任务子进程
  → Benchmark.prepare / samples / evaluate
  → 共享RequestScheduler → HTTP服务或Engine
  → 原始样本、请求和观测事件
  → 独立Metric计算 → Case / Task / Run结果 → 离线HTML
```

父进程和工作进程分别导入相同的内置实现；注册表保存类，在选中任务时才实例化。
导入期间不能加载模型、初始化GPU、启动Docker或联网。
Python反射只用于验证类继承与抽象方法，发现过程由显式导入控制。
不扫描任意目录，也不把已有脚本文件名当作接口。

## 职责与依赖

| 模块 | 负责 | 扩展点 |
|---|---|---|
| `core/parameters.py`、`planning.py` | 参数定义、独立合并、静态校验、Case展开 | 场景与Profile声明参数 |
| `core/benchmark.py` | 场景生命周期与两类评测基类 | `Benchmark`的类别子基类 |
| `data/` | 本地格式适配、抽样、完整记录保留 | `DataSource`与`sources` |
| `benchmarking/` | 输入输出模板、模型调用、领域事实 | 场景实现及共用领域函数 |
| `runtime/load_profiles.py`、`request_scheduler.py` | 到达策略、实际请求准入、并发、停止 | `Profile`与`profiles` |
| `runtime/evaluation_session.py` | 预热、正式采集、覆盖率、持久化、指标组合 | 注入的`RunContext` |
| `runtime/resource_manager.py` | 父进程持有的资源生命周期 | `Resource`与`resources` |
| `core/metric.py` | 指标接口与能力校验 | `Metric`与`metrics` |
| `reporting/` | 读取新结果、生成通用离线HTML | 通用数值、单位和样本数结构 |

场景负责“如何形成一份事实”，指标负责“如何从事实计算数值”。
调度器不知道C-Eval、HumanEval或在线性能的名称；配置加载器也不按场景名称分支。
C-Eval和CMMLU使用共同的选择题执行模板，以声明绑定数据集；GSM8K和LCSTS共享少样本执行框架。
WikiText窗口与HumanEval沙箱等特殊行为留在对应实现中。

场景可组合输入模板、答案解析器和指标，二者通过明确的事实字段衔接。
现有模板函数位于`benchmarking/model_service_quality/common/prompts.py`与`parsers.py`，
采集和评分函数按评测能力合并在同一模块，由统一运行时调用。
不再为每个数据集创建目录，也不复制`sampling.py`和`metrics.py`。

## 代码组织

```text
benchmarking/
├── benchmark_parameters.py       # 两类评测共享的参数组
├── benchmark_metrics.py          # 指标注册及事实到指标的适配
├── generation_performance/
│   ├── generation_benchmarks.py  # 性能场景声明与生命周期
│   ├── online_serving.py         # 在线请求事实与客户端指标
│   ├── vllm_engine.py            # 同步批量调用与Engine内部指标
│   ├── async_engine.py           # 异步Engine独立请求提交与结果接收
│   ├── service_observation.py    # 服务端观测指标
│   └── common/                   # 流式解析、Token处理、统计等
└── model_service_quality/
    ├── quality_benchmarks.py     # 质量场景声明与生命周期
    ├── arithmetic_qa.py          # 数值问答采集与评分
    ├── text_summarization.py     # 摘要采集与ROUGE评分
    ├── reading_comprehension.py  # 阅读理解采集与EM/F1评分
    ├── language_modeling.py      # 滚动窗口与语言模型损失
    ├── code_generation.py        # 代码候选采集与通过率
    ├── common/                   # 选择题、Prompt、解析与沙箱
    └── scripts/                  # 离线数据准备工具
```

公共数据集与许可证位于仓库根目录`dataset/`，由两类评测共同使用，
不随Python wheel内嵌分发。任务配置中的数据路径仍相对配置文件解析。

新增数据集优先复用已有能力，仅增加数据定义、模板或字段映射。
只有评测算法和事实结构确实不同，才新增能力模块。
C-Eval与CMMLU共同调用`common/choice.py`和`common/choice_result.py`，
无需各自的转发文件。质量指标的字段校验与数值封装共用`common/records.py`。
沙箱执行与容器内运行器放在`common/`，代码评分只通过执行器接口使用它们。

代码按数据加载、负载配置、请求调度、后端调用和结果输出划分职责。
跨两类评测的能力放在顶层`data/`、`runtime/`、`core/`和`reporting/`；
仅在某类评测内部复用的能力放在该类别的`common/`。

运行时目录按职责命名：

```text
runtime/
├── evaluation_session.py        # 单Case准备、预热、采集与指标计算
├── load_profiles.py             # 负载参数与策略创建
├── scheduling_strategies.py     # 时间计划、准入规则与完成反馈
├── request_scheduler.py         # 实际请求准入、排队、并发与停止
├── task_worker.py               # 任务子进程入口
├── case_recovery.py             # 中断后的Case结果恢复
├── resource_manager.py          # 本次运行的资源登记与清理
├── local_command.py             # 本地命令执行
├── prometheus_observer.py       # Prometheus服务指标采集
└── device_monitor.py            # 硬件指标监控
```

## 基类关系

```text
Benchmark
├── GenerationPerformanceBenchmark
└── ModelServiceQualityBenchmark

Metric                         Profile
└── 各种独立指标实现            └── rate / concurrent / batch及新增策略
```

`Benchmark`固定生命周期，采用模板方法组织采集；Profile创建调度策略，Metric独立组合。
注册表按名字创建已注册类，避免调用方理解具体类型。代价是开发者必须维护显式导入及唯一名称，
新增能力需测试父进程与子进程均能发现它。

## 新增仓库内场景与指标

在`src/luban_meter/benchmarking/example.py`加入以下完整示例：

```python
from luban_meter.api import (
    Metric,
    ModelServiceQualityBenchmark,
    Parameter,
    benchmarks,
    metrics,
)


@metrics.register("example_accuracy")
class ExampleAccuracy(Metric):
    requires = frozenset({"example_answer"})

    def compute(self, facts):
        rows = [
            row for row in facts["samples"]
            if row["state"] == "complete"
        ]
        return {
            "accuracy": {
                "value": (
                    sum(row["facts"]["correct"] for row in rows) / len(rows)
                    if rows else None
                ),
                "unit": "ratio",
                "sample_count": len(rows),
                "condition": "完整样本；未完成样本不计入分母",
            }
        }


@benchmarks.register("example")
class ExampleBenchmark(ModelServiceQualityBenchmark):
    parameters = {"sample_count": Parameter(int, 10, minimum=1)}
    capabilities = frozenset({"example_answer"})
    default_metrics = ("example_accuracy",)

    def prepare(self, context):
        context.total_samples = context.parameters["sample_count"]

    def samples(self, context):
        return ({"n": n} for n in range(context.total_samples))

    def evaluate(self, sample, context):
        # 此例只演示本地事实计算，不声称调用了模型。
        return {"correct": sample["n"] % 2 == 0}
```

然后在`benchmarking/__init__.py`显式加入`from . import example`：

```bash
luban-meter benchmarks list
luban-meter run --scenario example --sample-count 20
```

换成性能类别时继承`GenerationPerformanceBenchmark`。复用已有指标时只需选择匹配的
`default_metrics`或`--metrics`，不需要每个场景再定义一个指标类。
自定义指标与内置指标使用相同接口和报告规则，能力不匹配会在执行前报错。

示例展示A方案：代码属于当前仓库并随工具打包。外部插件安装、任意脚本路径和热加载不属于本方案。

## 场景接口

- 必须实现` samples(context)`与`evaluate(sample, context)`；样本和返回事实均为JSON可序列化对象。
- 可选`prepare(context)`初始化所选依赖；`metadata(context)`保存数据与观察边界；`close()`释放工作进程资源。
- `validate(parameters)`检查参数之间的关系；参数类型、范围、枚举值由`Parameter`处理。
- `stop_after(facts)`可根据已完成Case事实返回停止原因，跳过本任务后续Case。
- `context.total_samples`表达选定样本总数；离线批量样本的`size`表示其逻辑请求权重。
- 每次实际模型调用必须经过`context.request(operation)`。`operation(ticket)`接收请求编号和时间信息。
  已有`OpenAIClient(request_runner=context.request)`会在每次completion/chat调用时接入调度器。
- 调度停止会抛出`AdmissionStopped`。框架自动标记不完整样本，场景应允许其传播，且不得绕过调度器继续请求。
- 多线程执行下`evaluate`必须保证共享状态安全；不要把当前样本保存在实例字段。

公共基类不要求扩展开发者实现配置读取、子进程启动、报告导出或调度循环。

## 参数与数据

`Parameter`支持类型、默认值、必填、枚举、范围、列表元素类型、文件路径、凭据脱敏和`matrix=True`。
不同场景共享字段名时必须保持类型、列表元素类型与路径含义一致；默认值与取值范围可以不同。
新增字段会成为CLI选项。保留的运行级字段名如`scenario`、`config`、`task`、`result_dir`不能作为业务字段。

配置处理不持有具体场景类型。切换场景时从对应声明建立默认值，显式参数才参与覆盖。
路径解析、未知参数、非法Case组合在任务开始前检查；数据内容和可选运行依赖在工作进程准备阶段检查。

两类评测与few-shot统一调用公共`DatasetLoader`，通过注册的`DataSource.read(path)`
适配JSON、JSONL、CSV、ShareGPT、SQuAD和BurstGPT。加载器只负责路径、格式、
记录校验与抽样，返回完整记录；ShareGPT的首个人类Prompt由在线评测实现提取。

```python
from luban_meter.data import DatasetLoader

records = DatasetLoader(
    "dataset/ceval/val.jsonl",
    format="jsonl",
    max_samples=100,
    shuffle=True,
    seed=42,
).load()
```

省略`format`时根据扩展名识别JSON、JSONL、CSV；显式格式优先于扩展名。
`resolved_path`、`resolved_format`可用于检查实际输入。
直接调用时，`base_dir`明确指定相对路径的起点；不指定时优先使用当前目录，
再解析源码仓库中的`dataset/...`。显式`base_dir`下缺失的文件不会回退到别处。
CLI与配置文件在预检阶段按既定规则解析成绝对路径，加载器沿用这一结果。
few-shot同样通过加载器读取，默认按扩展名推断格式并保持顺序。
CSV返回字符串字段，不猜测评分语义；重复列名或列数不一致会报错。

`data.projections`单独提供`select_columns`、`extract_prompts`和`extract_traces`，
对应列选择、文本输入与轨迹字段提取。这些显式投影不会修改加载器返回的原始记录。
批量Prompt/Trace投影可筛选可用记录；逐样本在线调用使用`extract_prompt`，
无有效输入时记录该样本失败，不在加载阶段隐藏样本。
Prompt抽样先对完整候选集打乱，再截取，避免只打乱文件前几条。

PR !32的JSON/JSONL/CSV/ShareGPT/BurstGPT读取、路径解析、格式推断、
确定性抽样、few-shot选择和字段提取均由上述接口覆盖。
旧模块路径与转发函数不保留；数据路径使用公共`dataset/`，非法加载记录报错，
不沿用旧实现中悄悄过滤JSON或CSV异常记录的行为。

新增格式实现`DataSource.read(path)`并用`sources.register("name")`注册，
在`data/sources.py`中定义或显式导入即可，加载器无需新增格式分支。
自定义格式通过`format="name"`选择；源实现必须返回对象列表，不得悄悄丢弃非法记录。

BurstGPT通过公共加载器读取，轨迹适配器验证时间及正整数长度。默认定位文件、行号、字段和值报错；
`skip_invalid=true`显式跳过并保存诊断，不压缩剩余记录时间。`profile=replay`按时间稳定排序，
选取连续片段；其他Profile仅使用成对长度，允许确定性打乱。两者均通过公共TextGenerator构造文本。
请求数超过可用记录时报错，不循环补足。`trace_start`指定排序或打乱后的起始记录，`max_samples`限制候选数。
输入模板、输出解析与指标之间的字段约定由场景定义，不让通用文件读取器承担评分逻辑。

## 随机文本与长度条件

`generation_performance/common/tokenization.py`提供在线与Engine共用的`TextGenerator`和
`LengthSampler`。Faker文本经过Tokenizer编码、截取、解码和重新计数；最多尝试8次，
无法达到本地目标长度时明确报错。生成器按基础数据种子、阶段、样本序号和用途派生随机种子，
不访问调度策略的随机数状态。输入和输出长度分别支持固定、均匀或裁剪正态采样，
采样方式由目标值、可选std/min/max自动确定。合成输入在正式计时前准备，预热使用独立序列。

在线通过客户端Tokenizer构造文本并发送，Engine复用自身Tokenizer，将同样的合成文本编码后提交。
每条请求使用独立输出预算，Engine同一批次可以包含不同输入长度和不同SamplingParams。
原始样本保存目标、本地计数和输入文本；请求事实记录服务/Engine实际计数。

`common/lengths.py`独立验收负载条件，输出`conforming/nonconforming/unknown`。
严格策略逐条检查服务实际输入与目标相等；观测策略只记录输入偏差。
真实文本保持原文，仅按显式上下限验收。输出按ignore_eos选择精确预算或上限验收。
长度条件不修改请求成功状态，不将不符合条件的成功请求排除出性能统计。
缺失usage不使用目标值补齐；延迟可统计，无法完整计算的Token吞吐为null。

流式和Engine指标按实际长度计算；长度摘要复用通用统计结构，因此JSON和HTML都可展示
目标、实际、偏差分布及负载符合率。已知上下文超限提前报错，不自动改变负载条件。

## Profile与调度

`Profile.create_strategy()`创建一个Case/阶段独享的`SchedulingStrategy`；Profile负责参数声明，
策略负责调度规则，RequestScheduler负责等待、并发、停止和生命周期。用户只选择Profile，
策略无需额外注册或暴露CLI选项，可由多个Profile复用。

策略模式的接口为`plan(metadata)`、`ready(plan, elapsed)`，以及可选的`admitted(plan)`、
`completed(record)`通知。策略不休眠、不调用后端、不维护另一套并发队列。
`RequestPlan.offset_s=None`表示并发空位出现即可发送；非空值为相对正式采集起点的计划时间。
策略调用由调度器串行化。预热使用独立的即时策略，不消耗正式轨迹或速率随机序列。请求的`scheduling`元数据随样本绑定，轨迹包含`trace_index`和`offset_s`。

`rate`支持constant、poisson、gamma；`concurrent`完成后补充请求；`batch`串行批量执行；
`replay`用原始相对时间除以`replay_speed`。并发满时排队，不平移后续计划。
同一时刻按记录顺序准入，记录计划时间、准入时间与调度延迟。队列中的请求达到时长限制后不再发送，
已发请求按自身超时等待结束；正常达到预算、时长或数据末尾的Case仍保留。

场景通过`profile_modes`声明兼容的requests、trace或batch能力。HTTP与`offline_vllm_async`
支持逐请求策略；同步`offline_vllm_engine`仅支持batch，不兼容组合预检报错。
异步后端在独立事件循环中调用AsyncLLMEngine.generate，统一调度器决定提交时机；
生成取消或超时调用abort，Case关闭时释放Engine及事件循环。
提交时间、首个输出和完成时间使用客户端单调时钟，Engine内部时间原样单列；不可用时留空。
异步提交侧TTFT/E2EL不冒充同步Engine内部阶段指标，不与后者直接合并统计。

这一分层增加少量接口，换来跨HTTP与异步Engine复用算法；测试分别验证策略、生命周期和后端关联。

## 模板与字段契约

`core/template_contracts.py`公开InputTemplate、AnswerParser及独立注册表。
场景选择默认组合，也可通过`input_template`、`answer_parser`覆盖；模板声明`requires/provides`，
解析器同样声明字段名及类型。配置规划先检查组合，准备阶段验证并试渲染数据，运行时校验输出。
简单文本使用format_string及input_format，field_mapping使用`目标字段=源字段`列表，不执行表达式。
质量场景保留已有参考答案准备、few-shot与评分逻辑；选择题ppl、WikiText窗口和HumanEval沙箱
不强制改成单次文本调用。在线真实文本也可选择输入模板；合成文本直接按长度生成。

Metric可声明`required_fields`；场景通过`provided_fields(parameters)`声明可提供的事实字段。
规划阶段检查类型兼容，计算前校验完整样本事实。能力标签与字段契约并行，不以字段名代替指标语义。
扩展均在仓库内显式导入；不实现外部分发插件。

## 分层执行摘要

场景和Profile用`summary_fields`声明关键参数。CLI启动前输出每Case有效配置，凭据按参数声明脱敏；
工作进程准备完成后保存prepared摘要，父进程读取展示实际样本数、轨迹长度范围、原始时长及跳过诊断。
完整参数保存在plan.json，核心CLI不按场景名分支。

## 事实、指标与恢复

Case原始文件包含有效参数、样本、实际请求、覆盖率、时间窗口及场景元数据。
每条样本分别保存`source`与`facts`，通过`sample_id`关联一个或多个请求。
逐事件JSONL在每条写入后刷新，减少取消导致的数据丢失。

`Metric.compute(facts)`只读取事实，返回JSON可序列化对象，数值指标使用：

```json
{"ttft":{"value":12.5,"unit":"ms","sample_count":100}}
```

可嵌套组织指标；均值、分位数需给出单位与样本数；缺失用`null`并解释原因，不能用零代替。
非有限数值或指标异常会单独记录失败，不覆盖已经保存的原始数据。
观察窗口和失败分母属于指标含义，复用既有指标时沿用其约定。

重新计算某个已注册指标无需再调用模型：

```python
import json
from pathlib import Path
from luban_meter.core.planning import bootstrap
from luban_meter.api import metrics

bootstrap()
facts = json.loads(Path("runs/.../raw/case_1.json").read_text())
values = metrics.get("streaming_performance")().compute(facts)
```

首次Ctrl+C停止新请求，默认等待60秒，`interrupt_grace`可调整；再次Ctrl+C立即终止工作进程。
随后只整理结果，不续跑、不重发。正常结束采集的原始文件原子落盘时带collection_complete标记，
父进程以此为准，即使任务检查点未更新也保留Case，不覆盖事实。指标逐项保存；尚未计算的指标明确标记，
不伪造成功值。此边界针对进程退出，不承诺断电后的磁盘持久性。

未正常结束采集的Case删除原始请求、样本、指标、准备摘要、残留临时文件以及共享事件日志中的对应记录，
仅保存id、脱敏配置、discarded状态和原因。报告展示丢弃原因，排除该Case的指标和观测统计。
正常结束但存在请求失败的Case保留；指标失败不删除完整事实。未启动Case标记skipped。
父进程资源清理仅操作本次登记的资源句柄及容器标签。

## 验证边界

测试包括注册与配置、实际请求预算、评分公式、子进程扩展、多请求样本、中断、超时、
熔断、离线末批和HTML文本转义。HTTP与Engine模拟证明运行链路和指标衔接，
真实服务吞吐、GPU兼容性、Engine时间戳可用性及Docker隔离需要对应环境单独验证。
