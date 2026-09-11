# Benchmark 脚本开发指南

## 1. 目录选择

先根据评测目标选择模块：

| 模块 | 用途 |
|---|---|
| `generate` | 生成式推理时延、吞吐量、负载和引擎阶段性能 |
| `inference` | 基于在线推理服务的模型任务效果与质量评测 |

当前不开发算子层 Benchmark。

新增脚本必须放在：

```text
src/luban_meter/benchmark/<module>/<benchmark>/
├── benchmark.py
├── result.py
└── config.example.yaml
```

脚本身份由以下两个名称确定：

```text
module + benchmark
```

名称只允许小写字母、数字、连字符和下划线，并且必须以字母或数字开头。

## 2. 设计原则

1. **按测试场景组织脚本**：一次采集可复用的原始事实，再计算多个指标；
2. **统一脚本复用于不同硬件**：不得按硬件品牌复制同语义实现；
3. **差异通过配置表达**：服务地址、模型名、并行度和引擎参数写入配置；
4. **引擎专属能力显式命名**：例如 `vllm-engine-offline`；
5. **采集与计算分离**：硬件或服务调用位于 `benchmark.py`，纯数据处理位于
   `result.py`；
6. **结果可审计**：保留原始记录、参数、日志、失败原因和统计边界。
7. **硬件检测前置**：generate 模块的 Benchmark 在执行开头调用
   `print_hardware_info()` 输出设备摘要，便于复现问题。该工具由
   `common/device_monitor.py` 提供，支持多厂商硬件自动检测与监控采集。

## 3. 自动发现

当以下两个文件同时存在时，`BenchmarkRegistry` 自动发现脚本：

```text
benchmark/<module>/<benchmark>/benchmark.py
benchmark/<module>/<benchmark>/result.py
```

验证：

```bash
luban-meter benchmarks list
```

输出示例：

```text
generate    serving-online,vllm-engine-offline  Large-model generation benchmarks
inference   ceval,cmmlu,gsm8k                  Online-service model evaluation benchmarks
```

## 4. 配置文件

配置必须是 YAML Mapping，由 Core 加载后写入请求的 `parameters`：

```yaml
service_url: http://127.0.0.1:8000
request_timeout: 120
rounds: 10
```

Benchmark 必须主动校验：

- 必填字段是否存在；
- 类型、范围和列表是否有效；
- 模型长度、批量和并发是否超出边界；
- 固定语义参数是否被错误覆盖；
- 服务或引擎能力是否满足测试要求。

无效配置应抛出带字段名与原因的异常，禁止静默使用含义不同的默认值。

## 5. benchmark.py 协议

框架使用以下命令执行脚本：

```bash
python benchmark.py --request <request.json> --output <raw_result.json>
```

参数入口：

```python
import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()
```

请求结构：

```json
{
  "run_id": "generate-...",
  "module": "generate",
  "benchmark": "serving-online",
  "config": ".../serving-online.yaml",
  "model_path": null,
  "model_name": "Qwen3-8B",
  "output_dir": "runs",
  "timeout": 3600,
  "parameters": {}
}
```

请求中不包含硬件厂商字段。脚本应依据服务协议或显式引擎能力工作，而不是根据硬件
品牌分支。

成功输出至少包含：

```json
{
  "schema_version": "luban-meter.raw/v1",
  "status": "success",
  "environment": {},
  "metrics": {},
  "metadata": {},
  "artifacts": {}
}
```

失败输出：

```json
{
  "schema_version": "luban-meter.raw/v1",
  "status": "failed",
  "metrics": {},
  "metadata": {},
  "artifacts": {},
  "error": {
    "type": "RuntimeError",
    "message": "service request failed"
  }
}
```

即使捕获异常，也应写出失败 JSON，便于框架保留上下文；未捕获异常仍会由 Core
转换为标准失败结果。

## 6. result.py 协议

`result.py` 必须定义：

```python
def process(raw_result):
    return {
        "status": "success",
        "environment": raw_result.get("environment", {}),
        "metrics": raw_result["metrics"],
        "metadata": raw_result.get("metadata", {}),
    }
```

它负责：

- 校验原始记录的结构和数量；
- 拒绝非法时间线、Token 数或测试 Case；
- 从同一批原始样本计算 Mean、P50、P90、P99 等统计量；
- 将指标按 Request、Service、Engine 或 Task 视角分组；
- 输出明确的单位和样本数。

`result.py` 应尽量只依赖 Python 标准库或 Benchmark 公共工具，不重新调用模型、
服务或硬件运行时。

### 6.1 统一的 result.json v2

`ResultManager` 为所有脚本构造相同的外层结构，脚本无需自行拼接这些字段。
成功、部分失败和失败结果均使用 `luban-meter.result/v2`：

```json
{
  "schema_version": "luban-meter.result/v2",
  "run_id": "generate-example",
  "status": "success",
  "module": "generate",
  "benchmark": "custom",
  "config": "custom.yaml",
  "model": {"name": "model-a", "path": null},
  "environment": {},
  "parameters": {"batch_size": 4},
  "metrics": {"throughput": {"value": 120, "unit": "token/s"}},
  "artifacts": {},
  "metadata": {"display_name": "第一次测试"},
  "error": null
}
```

- `status` 为 `success`、`partial_failed` 或 `failed`。
- `model`、`environment`、`parameters`、`metrics`、`artifacts`、`metadata`
  均为字典；`error` 为字典或 null。
- **`metrics` 内部没有强制结构**，可以使用普通数值、嵌套字典和 Case 列表。
  指标名称、语义和计算方式由 Benchmark 负责。`{value, unit}` 和带 `unit`
  的统计字典是可选约定，报告会识别其单位。
- `raw_result.json` 是采集阶段的另一份文件，继续使用 `luban-meter.raw/v1`。
  本次升级的是最终结果协议；报告仅接受最终结果 v2，不兼容旧 v1。

Core 在运行异常处理范围内校验最终结果。脚本返回非法的结果字段（如非字典
`error`）时，保存 `status=failed` 的诊断结果，记录 `failure_stage` 和错误原因，
并保留已产生的原始产物链接、已读入的环境信息。Suite 将其视为失败任务，继续
执行后续任务或遵循 `--fail-fast` 跳过后续任务。

### 6.2 报告声明（可选）

新脚本无需向报告工具注册名称。只有 `metrics` 字典时，报告自动遍历嵌套字典和
列表，以 JSON Pointer 路径展示前 30 个数值或 null 指标；CSV 保留所有数值和
null 叶子。字符串、布尔值和原始逐请求文本保留在 JSON 中，不进入自动数值摘要。

若需要指定核心指标、Case 表格和图表，在 `process()` 返回的 `metadata.report`
中增加声明。例如：

```python
from luban_meter.result.report_spec import line, table


def process(raw_result):
    return {
        "metrics": {
            "cases": [
                {
                    "batch_size": 4,
                    "input_length": 128,
                    "throughput": {"value": 120, "unit": "token/s"},
                }
            ]
        },
        "metadata": {
            "report": {
                "tables": [
                    table(
                        "吞吐摘要",
                        "/cases",
                        {
                            "/batch_size": "Batch Size",
                            "/input_length": "输入长度",
                            "/throughput": "吞吐量",
                        },
                        charts=[
                            line(
                                "/batch_size",
                                "/throughput",
                                ["/input_length"],
                            )
                        ],
                    )
                ]
            }
        },
    }
```

声明是普通 JSON 数据，随 `result.json` 保存。导出历史 v2 文件无需加载原脚本。

| 字段 | 约定 |
|---|---|
| `tables` | 按显示顺序排列的表格列表 |
| `table.title` | 表格标题 |
| `table.path` | 相对 `metrics` 的 JSON Pointer；空字符串表示整个字典 |
| `table.columns` | `{"path": "/...", "label": "..."}` 列表，路径相对每行记录 |
| `table.mapping` | 为 true 时将字典转成 `{"key": 名称, "value": 指标}` 行，适合学科得分 |
| `table.charts` | 可选图表列表，图表紧跟对应表格 |
| `chart.type` | `line` 或 `bar`，分别由 `line()`、`bar()` 构造 |
| `chart.x`、`chart.y` | 相对记录的指标路径；折线图必须有数值 x，柱状图可省略 x |
| `chart.group_by` | 固定条件的路径列表，每组单独绘图，避免连接不同条件的 Case |

路径中的 `/`、`~` 分别编码为 `~1`、`~0`；例如键名 `a/b` 使用 `/a~1b`。
表格路径可以指向单个字典或字典列表；统计值可直接选择 `/latency/p99`。
表格和图表保留 `table.path` 上级的单位，子字段自己的 `unit`（包括空字符串）
可以覆盖继承单位；`count` 按计数展示。`mapping` 表格中的名称列不继承指标单位，
字符串 `unit` 字段作为单位元数据，不单独生成一行指标。
缺失值在表格显示 `—`、CSV 留空，折线保留缺口；0 是正常数值。
没有可用数值时不生成图表。声明格式错误时回退到自动摘要并记录原因，完整结果
仍保留在 JSON 中。报告不推断不同运行之间的对应关系，也不计算差值或百分比。

### 6.3 效果评测的条件摘要

`module=inference` 的报告在硬件总览之后、指标表之前展示“评测条件”，控制台
同步展示。该摘要读取通用字段，不按 Benchmark 名称注册，`metrics` 协议不变。

- `metadata` 优先提供实际模型、数据集、`split`、`sample_count`、`eval_mode`、
  `prompt_format`、`prompt_version`、`few_shot`、`scorer_version`、解码参数等；
- 模型名称缺少元数据记录时读取 `model.name`；模型版本可由
  `metadata.model_version` 或 `model.version` 提供。推理引擎及版本读取
  `metadata` 或 `environment` 中的 `engine`、`engine_version`；
- 其余同名字段可回退到 `parameters`，回退值标注“配置”，不当作额外测量结果；
- 样本选择显示 `max_samples`、`shuffle`、`seed`；若已有 `dataset_sha256` 或
  `few_shot_path`，同时展示。Few-shot 数量表示保存的设置，不推断实际示例数；
- 必要字段完全缺失时显示“未记录”；显式 `null`、空列表、0、false 按原值
  展示，不能因其为空而改用配置值。例如 ppl 结果中的 `max_tokens=null` 和
  `stop=[]` 保持原样；
- 仅提取上述摘要，不展开 Prompt 全文，不查询硬件或服务，不推测版本和取值。

CSV 继续只导出 `metrics` 中的数值与 null 指标；完整参数、环境和元数据保留
在 JSON 中。自定义脚本不提供这些可选条件时仍可导出报告，缺项按上述规则展示。

## 7. Generate Benchmark 指南

生成性能脚本应明确观察边界：

- 在线客户端：包含 HTTP、API Server、排队、Prefill、Decode 和流式传输；
- Engine 内部：由稳定的引擎时间戳定义，不包含完整在线链路。

改变输入长度、输出长度、Request Rate、Batch Size、缓存或精度时，应建立独立
Case，禁止将不同条件的样本混合统计。

在线服务优先使用标准 HTTP 协议，使同一 Benchmark 可在不同硬件环境的兼容服务上
直接复用。只有必须访问引擎内部字段的场景才建立引擎专属 Benchmark。

## 8. Inference Benchmark 指南

`inference` 通过在线推理服务评测模型任务效果，建议每个 Benchmark 封装一类任务
协议或数据集族，例如 `ceval`、`gsm8k`、`summarization`。当前已端到端实现
`ceval`、`cmmlu` 和 `gsm8k`，协议细节参见
[Inference 评测指标说明](inference.md)。

一次运行通常包含：

```text
读取固定数据集
→ 构造确定性 Prompt
→ 调用在线推理服务
→ 保存逐样本输入、原始输出和参考答案
→ 解析答案
→ 计算 Accuracy/F1/EM/ROUGE 等指标
```

必须记录：

- 数据集名称、版本、Split 和样本数量；
- Prompt 模板及版本；
- 解码参数；
- 原始模型输出与解析结果；
- 无法解析、服务失败和超时样本；
- 评分规则及指标实现版本。

禁止只保存聚合分数而丢失逐样本审计信息。

多个数据集共用的在线服务调用、数据集加载、Prompt 渲染、答案解析和指标计算
逻辑放在 `benchmark/inference/common/`；数据集官方格式到本地 jsonl 的转换脚本
放在 `benchmark/inference/scripts/`，Benchmark 运行时只读取本地数据集文件。
样例数据集随包内置在 `benchmark/inference/data/`；`common/dataset.py` 的
`resolve_data_path()` 对相对路径先按 CWD 解析，未命中时回退到包内置目录，使
默认配置无需额外准备即可从任意工作目录运行。

## 9. Suite

多个 Benchmark 顺序执行时，在以下目录添加 YAML：

```text
src/luban_meter/suite/definitions/generation-basic.yaml
```

```yaml
name: generation-basic
tasks:
  - name: online
    module: generate
    benchmark: serving-online
    config: configs/serving-online.yaml

  - name: engine
    module: generate
    benchmark: vllm-engine-offline
    config: configs/vllm-engine-offline.yaml

  - name: metrics
    module: generate
    benchmark: vllm_metrics
    config: configs/vllm-metrics.yaml
```

Suite 不声明硬件环境。所有任务使用启动命令时的当前环境，每个任务仍通过
`CoreEngine` 独立生成 `result.json`。启用 `--monitor-url` 时，采集结果写入
`result.json` 的 `environment.device_monitoring`。

`SuiteRunner` 写入 `luban-meter.suite-result/v2`：`tasks` 按 YAML 中的任务顺序
排列，每项保留任务名、模块、脚本、状态、run_id、结果路径，并在 `output` 内嵌
完整的 `luban-meter.result/v2`。被 fail-fast 跳过的任务 `output` 为 null。
因此 Suite 报告可只用 `suite_result.json` 生成，指标提取不依赖各任务文件存在。

CLI 的 `run` 和 `suite` 在结果保存后自动生成报告；直接调用 `CoreEngine` 或
`SuiteRunner` 的 Python 接口只写结果，可再通过 `luban-meter report` 导出。
多份 Suite 输入各自产生独立报告，任务参数不同也不做匹配或比较。

报告在每个 Benchmark 区块开头，将 `environment.hardware_environment` 和
`environment.device_monitoring` 合并绘制为一张硬件总览图。环境字段按记录
展示，监控摘要保留已存数值；`timeseries` 中同一设备缺失的采样不会补零。
只有环境信息也可生成图，不要求 Benchmark 额外声明硬件报告布局。

## 10. 测试要求

提交 Benchmark 前至少验证：

- [ ] 目录为 `benchmark/<module>/<benchmark>/`；
- [ ] 模块为 `generate` 或 `inference`；
- [ ] `benchmark.py` 接收 `--request` 和 `--output`；
- [ ] `result.py` 定义 `process(raw_result)`；
- [ ] 成功和失败均符合 `luban-meter.raw/v1`；
- [ ] 配置字段有类型、范围和边界校验；
- [ ] 指标单位、样本数和观察边界明确；
- [ ] `result.json` 中的模块、脚本、模型和参数正确；
- [ ] 成功和失败的最终结果符合 v2，`metrics` 为字典；
- [ ] 使用 `luban-meter report --input <result.json>` 验证指标、单位和可选图表；
- [ ] 不按硬件品牌复制脚本或引入路由分支；
- [ ] `ruff check src tests` 通过；
- [ ] `pytest -q` 通过；
- [ ] 在目标运行环境完成真实冒烟测试。
