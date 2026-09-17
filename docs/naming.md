# 命名约定与迁移说明

命名表达对象的实际含义。领域术语见[术语表](../CONTEXT.md)。
本次调整覆盖项目维护的Python代码、配置、目录、测试及文档。
数据集原文、第三方协议、生成的运行产物和本地技能不作为重命名对象。

## 1. 主要名称

| 原名称或用法 | 现名称 | 含义 |
|---|---|---|
| `benchmark/` | `benchmarking/` | 评测场景实现的集合 |
| `generate/` | `generation_performance/` | 生成式推理性能场景 |
| 模型服务质量类别目录 | `model_service_quality/` | 模型服务质量评测场景 |
| `serving-online/` | `online_serving/` | 在线服务性能场景 |
| `vllm-engine-offline/` | `offline_vllm_engine/` | vLLM离线引擎场景 |
| `vllm_metrics/` | `vllm_service_metrics/` | vLLM服务端指标场景 |
| 场景内的`benchmark.py` | `collect_raw.py` | 原始事实采集入口 |
| 场景内的`result.py` | `calculate_metrics.py` | 指标计算入口 |
| `core/models.py` | `core/run_contracts.py` | 运行请求、定义、产物和结果结构 |
| `suite/models.py` | `suite/suite_contracts.py` | 套件请求与结果结构 |
| `core/engine.py`、`CoreEngine` | `core/run_coordinator.py`、`RunCoordinator` | 单次运行编排，区别于模型推理引擎 |
| `core/registry.py`、`BenchmarkRegistry` | `core/benchmark_registry.py`、`BenchmarkRegistry` | 场景发现与解析 |
| `BenchmarkSpec` | `BenchmarkDefinition` | 场景定义对象 |
| `BenchmarkResult` | `RunResult` | 单次运行结果 |
| `ResolvedRun.benchmark` | `ResolvedRun.benchmark_definition` | 已解析的场景定义 |
| 数据结构中的`module` | `category_name` | 评测类别名称 |
| 数据结构中的`benchmark` | `benchmark_name` | 场景名称字符串 |
| 数据结构中的`config` | `config_path` | 参数文件路径 |
| `RunResult.model` | `RunResult.model_info` | 模型信息字典 |
| 在线调用中的`model` | `served_model_name` | 服务模型名称字符串 |
| `discover_model()` | `discover_served_model_name()` | 从服务发现模型名称 |
| `run_benchmark()` | `collect_raw_result()` | 执行采集并返回原始结果 |
| `reporting/data.py` | `reporting/result_reader.py` | 读取结果并构造报告数据 |
| `reporting/context.py` | `reporting/evaluation_context.py` | 读取评测条件 |
| `Task`、`Report`、`Table` | `ReportTask`、`ResultReport`、`MetricTable` | 明确报告数据对象的用途 |
| `task.data` | `task.result_payload` | 报告任务关联的结果内容 |

局部名称按照实际含义展开，例如`il/ol/rr`改为
`input_length/output_length/request_rate`，窗口计分中的`n/w`改为
`token_count/window_result`，Pass@k中的`n/c/k`改为
`sample_count/success_count/selection_count`。

`self`、`cls`、`main`、`__init__`、`process(raw_result)`等有明确语言或调用
协议含义的名称保留。HTTP、JSON、URL、CPU、GPU、TTFT、ITL、TPOT等通用
缩写保留；同一变量在所属作用域中已有明确含义时，无需机械叠加前缀。

## 2. 兼容边界

现有CLI选项、Suite YAML字段和JSON协议继续使用既有名称：

| 对外名称 | Python内部名称或目录 |
|---|---|
| `--module`、JSON/YAML的`module` | `category_name` |
| `--benchmark`、JSON/YAML的`benchmark` | `benchmark_name` |
| `--config`、JSON/YAML的`config` | `config_path` |
| 最终结果的`model`对象 | `model_info` |
| `--module generate` | `benchmarking/generation_performance/` |
| `--module model_service_quality` | `benchmarking/model_service_quality/` |
| `--benchmark serving-online` | `online_serving/` |
| `--benchmark vllm-engine-offline` | `offline_vllm_engine/` |
| `--benchmark vllm_metrics` | `vllm_service_metrics/` |

数据结构通过字段的`json_name`元数据和`to_jsonable()`输出原有协议。
嵌套的Suite任务同样遵循此约定。不要用`dataclasses.asdict()`代替协议序列化，
后者会输出Python内部字段名。

OpenAI兼容请求中的`model`、服务返回的`id`、vLLM参数、Prometheus标签与
指标名称，以及数据集字段保持原样。结果协议仍为raw/v1、result/v2和
suite-result/v2。模型服务质量类别值统一为`model_service_quality`；
现有配置和历史结果中的旧类别值需同步迁移，才能使用该类别的完整报告展示。
标准套件名称和文件名统一为`model_service_quality_standard`，旧类别和套件
名称不再作为别名接受。vLLM耗时聚合指标键改为`model_execution_time`；
采集时仍读取服务提供的原始Prometheus指标名。

新场景采用`collect_raw.py`与`calculate_metrics.py`。注册表仍接受自定义
场景中的旧`benchmark.py`与`result.py`文件对，以及生成性能类别的旧目录名。
一对文件必须完整存在；优先使用新的文件对。

**源码路径和内部Python接口已迁移。**直接导入旧模块、使用旧构造参数或直接
执行旧源码路径的调用方，需要按照上表同步修改；配置文件路径也应使用新的
实际路径。CLI选项名称、配置字段名和结果JSON结构保持兼容，不代表旧源码路径
继续存在。

## 3. PEP 8

- 函数、方法、变量、参数、Python模块和包目录使用`snake_case`。
- 类使用`CapWords`，常量使用`UPPER_CASE`，异常类以`Error`结尾。
- 名称包含对象用途；同一术语不混用字符串、路径、定义和执行结果。
- 代码行宽79字符，注释和文档字符串正文按72字符组织。
- Ruff统一启用`E`、`W`、`F`、`I`、`N`规则，并检查行长。

验证命令：

```bash
python -m pytest -q
python -m ruff check --select E,W,F,N --line-length 79 src tests
python -m ruff check src tests
python -m luban_meter benchmarks list
git diff --check
```
