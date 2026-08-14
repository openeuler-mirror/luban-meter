# LuBan-Meter 使用说明

## 1. 安装

要求 Python 3.12 或更高版本：

```bash
cd /path/to/LuBan-Meter
python3.12 -m pip install -e .
```

厂商驱动、Python 环境、模型服务、模型和数据集由用户提前准备。

## 2. 目录约定

每个厂商拥有自己的脚本和 Suite：

```text
src/luban_meter/vendors/<vendor>/
├── benchmark/<module>/<benchmark>/
│   ├── benchmark.py
│   ├── result.py
│   └── config.example.yaml
└── suites/
    └── <suite>.yaml
```

例如：

```text
vendors/ascend/benchmark/generate/ttft/
vendors/ascend/benchmark/generate/throughput/
vendors/ascend/benchmark/operation/matmul/
vendors/ascend/suites/full-benchmark.yaml
```

## 3. 准备厂商环境

LuBan-Meter 不读取环境配置文件，也不设置厂商环境变量。运行前进入需要测试
的厂商环境：

```bash
source /opt/ascend-vllm/bin/activate
export ASCEND_RT_VISIBLE_DEVICES=0
```

框架使用当前 Python 执行脚本，并继承当前 Shell 环境。

## 4. 运行单个任务

```bash
luban-meter run \
  --vendor ascend \
  --module generate \
  --benchmark ttft \
  --config configs/benchmarks/ascend-ttft.yaml \
  --model-path /data/models/Qwen3-8B \
  --model-name Qwen3-8B \
  --output runs/ascend
```

框架解析：

```text
vendors/ascend/benchmark/generate/ttft/benchmark.py
vendors/ascend/benchmark/generate/ttft/result.py
```

单任务参数：

| 参数 | 必填 | 说明 |
|---|---:|---|
| `--vendor` | 是 | 厂商目录名 |
| `--module` | 是 | `generate`、`inference` 或 `operation` |
| `--benchmark` | 是 | 厂商脚本逻辑名称 |
| `--config` | 是 | 本次测试参数 YAML |
| `--model-path` | 否 | 模型路径 |
| `--model-name` | 否 | 模型名称 |
| `--output` | 否 | 输出根目录，默认 `runs` |
| `--timeout` | 否 | 超时秒数，默认 3600 |

## 5. 定义 Suite

Suite 文件必须位于对应厂商的 `suites/` 目录。

`src/luban_meter/vendors/ascend/suites/full-benchmark.yaml`：

```yaml
name: ascend-full-benchmark

tasks:
  - name: ttft
    module: generate
    benchmark: ttft
    config: configs/ttft.yaml

  - name: throughput
    module: generate
    benchmark: throughput
    config: configs/throughput.yaml

  - name: matmul
    module: operation
    benchmark: matmul
    config: configs/matmul.yaml
    timeout: 600
```

每个任务必须设置：

| 字段 | 说明 |
|---|---|
| `name` | Suite 内唯一任务名称 |
| `module` | 功能分类 |
| `benchmark` | 当前厂商的脚本名称 |
| `config` | Benchmark 参数文件 |
| `timeout` | 可选，覆盖 Suite 默认超时 |

相对 `config` 路径以 Suite YAML 所在目录为基准。也可以填写绝对路径。

Suite 不填写 `vendor`。运行时指定的厂商决定所有脚本都从哪个厂商目录加载。

## 6. 运行 Suite

```bash
luban-meter suite \
  --vendor ascend \
  --suite full-benchmark \
  --model-path /data/models/Qwen3-8B \
  --model-name Qwen3-8B \
  --output runs/ascend
```

执行顺序与 Suite YAML 中的任务顺序一致。默认情况下任务失败后继续执行。

遇到失败立即停止：

```bash
luban-meter suite \
  --vendor ascend \
  --suite full-benchmark \
  --fail-fast
```

`--fail-fast` 停止后，尚未执行的任务在 Suite 结果中标记为 `skipped`。

Suite 参数：

| 参数 | 必填 | 说明 |
|---|---:|---|
| `--vendor` | 是 | 厂商目录名 |
| `--suite` | 是 | 厂商 `suites/` 下的逻辑名称 |
| `--model-path` | 否 | 所有任务共享的模型路径 |
| `--model-name` | 否 | 所有任务共享的模型名称 |
| `--output` | 否 | 输出根目录，默认 `runs` |
| `--timeout` | 否 | 默认任务超时，默认 3600 秒 |
| `--fail-fast` | 否 | 第一个失败后停止 |

## 7. 查看已发现脚本

```bash
luban-meter benchmarks list
```

示例输出：

```text
generate    ascend/ttft,nvidia/ttft    Large-model generation benchmarks
inference   -                           Model inference benchmarks
operation   ascend/matmul               Operator performance benchmarks
```

## 8. 输出目录

单任务：

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

Suite：

```text
runs/<suite_id>/
├── suite_request.json
├── suite_result.json
└── tasks/
    ├── <task-run-id>/
    │   ├── request.json
    │   ├── raw/
    │   └── result.json
    └── ...
```

`suite_result.json` 示例：

```json
{
  "schema_version": "luban-meter.suite-result/v1",
  "suite_id": "ascend-full-benchmark-...",
  "name": "ascend-full-benchmark",
  "vendor": "ascend",
  "status": "partial_failed",
  "tasks": [
    {
      "name": "ttft",
      "module": "generate",
      "benchmark": "ttft",
      "status": "success",
      "run_id": "ttft-...",
      "result": "runs/.../tasks/ttft-.../result.json"
    }
  ]
}
```

Suite 只输出任务状态和结果路径，不比较或合并业务指标。

## 9. 常见错误

### Suite 不存在

检查：

```text
vendors/<vendor>/suites/<suite>.yaml
```

### Benchmark 脚本不存在

检查：

```text
vendors/<vendor>/benchmark/<module>/<benchmark>/
├── benchmark.py
└── result.py
```

### Benchmark 配置不存在

单任务的 `--config` 相对于当前工作目录解析；Suite 任务中的 `config` 相对于
Suite YAML 所在目录解析。

### 厂商技术栈不可用

确认启动 LuBan-Meter 前已进入正确的 Python 环境，并设置必要的厂商环境变量。

## 10. 开发验证

```bash
pytest -q
ruff check src tests
```

## 11. NVIDIA vLLM 生成式推理

在线场景使用已经启动的 OpenAI-compatible 服务，通过 `/tokenize` 构造精确
Token 长度，再按固定 Request Rate 调用 `/v1/completions` 流式接口，同时计算
TTFT、ITL、TPOT、E2EL、Token 吞吐和并发指标：

```bash
luban-meter run \
  --vendor nvidia \
  --module generate \
  --benchmark serving-online \
  --config \
    src/luban_meter/vendors/nvidia/benchmark/generate/serving-online/nvidia_vllm_serving_online.yaml \
  --model-name <served-model-name>
```

如果不传 `--model-name`，脚本会从服务的 `/v1/models` 读取第一个模型。
服务还必须支持 Token ID Prompt、`min_tokens`、`ignore_eos` 和流式 usage。

vLLM 引擎阶段测试直接加载本地引擎，在一轮测试中采集 Prefill、Decode、
内部 TTFT 和引擎执行时延，并记录 KV Cache 静态容量环境：

```bash
CUDA_VISIBLE_DEVICES=0 luban-meter run \
  --vendor nvidia \
  --module generate \
  --benchmark vllm-engine-stage \
  --config \
    src/luban_meter/vendors/nvidia/benchmark/generate/vllm-engine-stage/nvidia_vllm_engine_stage.yaml \
  --model-path /data/models/<model>
```

在线场景通过 `input_lengths`、`output_lengths` 和 `request_rates` 定义测试矩阵，
每个 Case 独立执行 `rounds` 个正式请求，`max_concurrency` 是客户端安全上限。
引擎阶段场景通过 `input_lengths`、`output_lengths` 和
`request_batch_sizes` 定义测试矩阵。

架构细节见 [架构说明](architecture.md)，脚本协议见
[Benchmark 脚本开发指南](develop-benchmark.md)。
完整指标语义和结果字段见 [生成式推理指标说明](metrics.md)。
