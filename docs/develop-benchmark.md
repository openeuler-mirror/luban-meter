# LuBan-Meter Benchmark 脚本开发指南

## 1. 开发目标

一个 Benchmark 脚本由以下内容组成：

```text
benchmark.py
result.py
config.example.yaml
```

三者分别负责：

| 文件 | 职责 |
|---|---|
| `benchmark.py` | 调用厂商技术栈、执行测试、输出原始 JSON |
| `result.py` | 将原始数据整理成最终指标 |
| `config.example.yaml` | 告诉用户该脚本可以设置哪些参数 |

一次测试通过以下参数选择脚本：

```text
module + vendor + benchmark
```

用户通过 `--config` 提供本次测试参数。

## 2. 选择功能模块

当前功能模块：

| 模块 | 适合的测试 |
|---|---|
| `generate` | TTFT、TPOT、吞吐量、并发生成等大模型生成测试 |
| `inference` | 模型端到端推理性能测试 |
| `operation` | MatMul、Attention、通信算子等算子测试 |

例如，开发 Ascend TTFT 脚本时使用：

```text
module = generate
vendor = ascend
benchmark = ttft
```

## 3. 创建脚本目录

目录格式：

```text
src/luban_meter/vendors/<vendor>/benchmark/<module>/<benchmark>/
├── benchmark.py
├── result.py
└── config.example.yaml
```

Ascend TTFT：

```text
src/luban_meter/vendors/ascend/benchmark/generate/ttft/
├── benchmark.py
├── result.py
└── config.example.yaml
```

NVIDIA TTFT：

```text
src/luban_meter/vendors/nvidia/benchmark/generate/ttft/
├── benchmark.py
├── result.py
└── config.example.yaml
```

`vendor` 和 `benchmark` 名称使用以下字符：

```text
小写字母、数字、下划线、连字符
```

合法名称示例：

```text
ascend
nvidia
metax
ttft
online-serve
matmul_fp16
```

厂商目录中存在 `benchmark/<module>/<benchmark>/benchmark.py` 和 `result.py` 后，
框架会自动发现该脚本。

`benchmark` 应按测试场景命名，例如 `serving-online`，不建议为能从同一批数据
计算的 TTFT、TPOT、ITL 和 E2EL 分别创建脚本。场景共享代码可放在：

```text
benchmark/<module>/common/
```

`common/` 中不放置 `benchmark.py` 和 `result.py`，因此不会被发现为可执行场景。

查看发现结果：

```bash
luban-meter benchmarks list
```

输出示例：

```text
generate    ascend/ttft,nvidia/ttft    Large-model generation benchmarks
inference   -                           Model inference benchmarks
operation   -                           Operator performance benchmarks
```

## 4. 编写参数模板

脚本开发者决定该 Benchmark 需要哪些参数，并在 `config.example.yaml` 中提供
可直接复制的示例。

TTFT 示例：

```yaml
# 测试请求数
rounds: 100

# 正式测试前的预热请求数
warmup: 10

# 并发请求数
concurrency: 1

# 输入和输出 Token 长度
input_length: 1024
output_length: 128

# 已启动的推理服务地址
service_url: http://127.0.0.1:8000

# 单请求超时
request_timeout: 120
```

用户复制模板：

```bash
cp \
  src/luban_meter/vendors/ascend/benchmark/generate/ttft/config.example.yaml \
  configs/benchmarks/ascend-ttft.yaml
```

框架会检查：

1. 配置文件存在；
2. YAML 顶层是键值映射。

字段类型、必填字段、取值范围和字段组合由 `benchmark.py` 校验。

Ascend 与 NVIDIA 可以根据各自调用方式定义不同的配置字段。

## 5. benchmark.py 的执行协议

框架使用启动 `luban-meter` 的当前 Python 执行：

```text
<current-python> benchmark.py
  --request <run_dir>/request.json
  --output <run_dir>/raw/raw_result.json
```

因此 `benchmark.py` 必须接收两个参数：

```text
--request
--output
```

### 5.1 request.json

框架生成的请求文件包含两部分：

```json
{
  "request": {
    "run_id": "generate-20260731T120000Z-abcd1234",
    "module": "generate",
    "vendor": "ascend",
    "benchmark": "ttft",
    "config": "configs/benchmarks/ascend-ttft.yaml",
    "model_path": "/data/models/Qwen3-8B",
    "model_name": "Qwen3-8B",
    "output_dir": "runs",
    "timeout": 3600
  },
  "parameters": {
    "rounds": 100,
    "warmup": 10,
    "concurrency": 1,
    "input_length": 1024,
    "output_length": 128,
    "service_url": "http://127.0.0.1:8000",
    "request_timeout": 120
  }
}
```

脚本从：

- `request` 读取模型、数据集和本次 Run 的公共信息；
- `parameters` 读取开发者在配置模板中定义的测试参数。

### 5.2 raw_result.json

执行成功时写入：

```json
{
  "schema_version": "luban-meter.raw/v1",
  "status": "success",
  "metrics": {
    "ttft_samples_ms": [81.2, 79.8, 83.4]
  },
  "metadata": {
    "successful_requests": 3,
    "failed_requests": 0
  },
  "artifacts": {}
}
```

结构化失败结果：

```json
{
  "schema_version": "luban-meter.raw/v1",
  "status": "failed",
  "metrics": {},
  "metadata": {},
  "artifacts": {},
  "error": {
    "type": "ConfigurationError",
    "message": "rounds must be greater than 0"
  }
}
```

`schema_version` 固定为：

```text
luban-meter.raw/v1
```

## 6. benchmark.py 开发模板

下面的模板完成参数读取、校验、厂商调用入口和标准结果输出：

```python
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_request(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    request = payload.get("request")
    parameters = payload.get("parameters")
    if not isinstance(request, Mapping):
        raise ValueError("request must be an object")
    if not isinstance(parameters, Mapping):
        raise ValueError("parameters must be an object")
    return dict(request), dict(parameters)


def positive_integer(parameters: dict[str, Any], name: str) -> int:
    value = parameters.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def non_negative_integer(parameters: dict[str, Any], name: str) -> int:
    value = parameters.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def run_vendor_benchmark(
    request: dict[str, Any],
    parameters: dict[str, Any],
) -> dict[str, Any]:
    rounds = positive_integer(parameters, "rounds")
    warmup = non_negative_integer(parameters, "warmup")
    service_url = parameters.get("service_url")
    if not isinstance(service_url, str) or not service_url:
        raise ValueError("service_url must be a non-empty string")

    # 在此处实现当前厂商的 vLLM 调用和 TTFT 计时。
    # 返回值应来自真实测试过程。
    ttft_samples_ms = call_vendor_vllm(
        service_url=service_url,
        model_name=request.get("model_name"),
        model_path=request.get("model_path"),
        rounds=rounds,
        warmup=warmup,
        parameters=parameters,
    )

    return {
        "schema_version": "luban-meter.raw/v1",
        "status": "success",
        "metrics": {
            "ttft_samples_ms": ttft_samples_ms,
        },
        "metadata": {
            "successful_requests": len(ttft_samples_ms),
            "vendor": request.get("vendor"),
        },
        "artifacts": {},
    }


def call_vendor_vllm(
    *,
    service_url: str,
    model_name: Any,
    model_path: Any,
    rounds: int,
    warmup: int,
    parameters: dict[str, Any],
) -> list[float]:
    """由脚本开发者实现厂商专属请求和首 Token 计时。"""
    raise NotImplementedError


def write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    try:
        request, parameters = load_request(args.request)
        raw_result = run_vendor_benchmark(request, parameters)
    except Exception as error:
        raw_result = {
            "schema_version": "luban-meter.raw/v1",
            "status": "failed",
            "metrics": {},
            "metadata": {},
            "artifacts": {},
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }

    write_result(args.output, raw_result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

厂商差异集中在 `call_vendor_vllm()` 中。例如：

- Ascend 脚本使用 Ascend 环境中的 vLLM 接口；
- NVIDIA 脚本使用 NVIDIA 环境中的 vLLM 接口；
- 算子脚本可以直接调用 PyTorch、厂商算子库或测试命令。

当脚本写出结构化失败 JSON 并返回 `0` 时，ResultManager 会保留其中的
`error`。进程启动或运行阶段直接退出时，框架会记录退出码，并将标准错误保存
到 `stderr.log`。

## 7. TTFT 计时原则

TTFT 表示从请求发出到收到第一个输出 Token 的时间：

```text
TTFT = 第一个 Token 到达时间 - 请求发出时间
```

推荐使用单调时钟：

```python
from time import perf_counter

started = perf_counter()

for event in streaming_response:
    token = parse_first_token(event)
    if token is not None:
        ttft_ms = (perf_counter() - started) * 1000
        break
```

开发脚本时应明确：

- 请求是否使用流式返回；
- 计时起点位于 HTTP 请求发送前还是 SDK 调用前；
- 第一个有效 Token 的判定方法；
- 预热请求是否排除在正式样本之外；
- 请求失败、空响应和超时如何计数；
- 输入 Token 长度如何构造和验证。

同一指标在不同厂商脚本中建议使用相同单位和名称，例如：

```text
ttft_samples_ms
ttft_mean_ms
ttft_p50_ms
ttft_p90_ms
ttft_p99_ms
```

## 8. result.py 开发协议

`result.py` 必须定义：

```python
def process(raw_result):
    ...
```

输入是 `raw_result.json` 解析后的字典，返回值是一个 Mapping。

TTFT 示例：

```python
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def percentile(samples: list[float], ratio: float) -> float:
    ordered = sorted(samples)
    index = round((len(ordered) - 1) * ratio)
    return ordered[index]


def process(raw_result: Mapping[str, Any]) -> dict[str, Any]:
    metrics = raw_result.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("raw metrics must be an object")

    samples = metrics.get("ttft_samples_ms")
    if not isinstance(samples, list) or not samples:
        raise ValueError("ttft_samples_ms must be a non-empty list")

    values = [float(value) for value in samples]
    return {
        "metrics": {
            "ttft_mean_ms": sum(values) / len(values),
            "ttft_p50_ms": percentile(values, 0.50),
            "ttft_p90_ms": percentile(values, 0.90),
            "ttft_p99_ms": percentile(values, 0.99),
        },
        "metadata": {
            "sample_count": len(values),
        },
    }
```

执行关系：

```text
厂商环境中的 Python
└── benchmark.py
    └── raw_result.json

LuBan-Meter 主进程
└── result.py::process()
    └── result.json
```

因此 `result.py` 适合使用 Python 标准库进行纯数据处理。厂商 SDK 和硬件调用
集中在 `benchmark.py`。

ResultManager 会：

1. 检查原始结果是 JSON Object；
2. 检查 `schema_version`；
3. 检查 `status`；
4. 对成功结果加载同目录的 `result.py`；
5. 调用 `process(raw_result)`；
6. 合并原始和处理后的 `metadata`；
7. 对结构化失败结果保留 `error`；
8. 写出最终 `result.json`。

## 9. 准备运行环境

框架不读取环境配置文件，也不主动设置环境变量。开发和运行厂商脚本前：

1. 激活已经安装厂商 SDK、vLLM 和依赖的 Python 环境；
2. 在 `~/.bashrc` 中配置必要变量，或在当前 Shell 中执行 `export`；
3. 在同一个 Shell 中启动 `luban-meter`。

Benchmark 子进程使用当前 Python，并继承当前工作目录和环境变量。

## 10. 执行脚本

准备用户配置后执行：

```bash
luban-meter run \
  --module generate \
  --vendor ascend \
  --benchmark ttft \
  --config configs/benchmarks/ascend-ttft.yaml \
  --model-path /data/models/Qwen3-8B \
  --model-name Qwen3-8B \
  --output runs
```

框架生成：

```text
runs/<run_id>/
├── request.json
├── raw/
│   ├── raw_result.json
│   ├── stdout.log
│   ├── stderr.log
│   └── artifacts/
└── result.json
```

排查执行问题时，依次查看：

```text
result.json
raw/raw_result.json
raw/stderr.log
raw/stdout.log
```

## 11. 脚本测试

### 11.1 测试参数校验

至少覆盖：

- 完整配置；
- 缺少必填字段；
- 错误字段类型；
- 数值超出范围；
- 空响应和请求超时。

### 11.2 测试结果处理器

为 `result.py` 准备固定的原始数据：

```json
{
  "schema_version": "luban-meter.raw/v1",
  "status": "success",
  "metrics": {
    "ttft_samples_ms": [10.0, 12.0, 14.0]
  },
  "metadata": {},
  "artifacts": {}
}
```

验证平均值、分位数、单位和样本数量。

### 11.3 工程测试

```bash
pytest -q
ruff check src tests
luban-meter benchmarks list
```

### 11.4 端到端验收

端到端测试至少确认：

1. `benchmarks list` 能发现 `vendor/benchmark`；
2. Benchmark 使用启动框架的当前 Python；
3. `export` 设置的环境变量能被 Benchmark 子进程读取；
4. 用户 YAML 参数出现在 `request.json` 的 `parameters` 中；
5. `raw_result.json` 使用 `luban-meter.raw/v1`；
6. `result.py` 能生成预期指标；
7. `result.json` 中的模块、厂商、脚本、模型和参数正确；
8. `stdout.log` 和 `stderr.log` 可以定位失败原因。

## 12. 将多个脚本编排为 Suite

同一厂商下需要顺序执行多个 Benchmark 时，在厂商的 `suites` 目录添加 YAML：

```text
src/luban_meter/vendors/ascend/suites/generation-basic.yaml
```

示例：

```yaml
name: generation-basic
tasks:
  - name: ttft
    module: generate
    benchmark: ttft
    config: configs/ttft.yaml
  - name: throughput
    module: generate
    benchmark: throughput
    config: configs/throughput.yaml
    timeout: 1800
```

`config` 相对于 Suite YAML 所在目录解析。Suite 本身不再声明厂商，厂商由
`--vendor` 和所在目录共同确定。每个任务仍调用现有单任务执行链，并生成独立的
`result.json`；Suite 只负责顺序调度和生成 `suite_result.json` 索引。

运行：

```bash
luban-meter suite \
  --vendor ascend \
  --suite generation-basic \
  --model-path /data/models/Qwen3-8B \
  --model-name Qwen3-8B \
  --output runs
```

## 13. 开发检查清单

提交厂商 Benchmark 脚本前检查：

- [ ] 目录为 `vendors/<vendor>/benchmark/<module>/<benchmark>/`
- [ ] 厂商名和脚本名符合命名规则
- [ ] `benchmark.py` 接收 `--request` 和 `--output`
- [ ] `config.example.yaml` 包含注释、默认值和单位
- [ ] 配置参数在 `benchmark.py` 中完成校验
- [ ] 正式样本与预热样本分开
- [ ] 时间、吞吐量、显存等指标标注单位
- [ ] 原始结果使用 `luban-meter.raw/v1`
- [ ] `result.py` 定义 `process(raw_result)`
- [ ] `result.py` 返回 `metrics` 和 `metadata`
- [ ] 如需保留部分失败指标，返回可选 `status` 和 `error`
- [ ] 执行失败可以在 JSON 或日志中定位
- [ ] `benchmarks list` 能发现新脚本
- [ ] 单元测试、静态检查和端到端测试通过
