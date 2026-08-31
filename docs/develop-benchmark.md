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
    benchmark: vllm-metrics
    config: configs/vllm-metrics.yaml
```

Suite 不声明硬件环境。所有任务使用启动命令时的当前环境，每个任务仍通过
`CoreEngine` 独立生成 `result.json`。

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
- [ ] 不按硬件品牌复制脚本或引入路由分支；
- [ ] `ruff check src tests` 通过；
- [ ] `pytest -q` 通过；
- [ ] 在目标运行环境完成真实冒烟测试。
