# LuBan-Meter

面向多硬件厂商的模块化 AI Benchmark 工具集。

## 文档

- [架构说明](docs/architecture.md)
- [使用说明](docs/usage.md)
- [Benchmark 脚本开发指南](docs/develop-benchmark.md)
- [生成式推理指标说明](docs/metrics.md)

## 当前范围

LuBan-Meter 当前只在宿主机直接执行 Benchmark：

- 用户提前准备可用的厂商 Python、vLLM、驱动和运行时环境；
- 运行前由用户激活对应 Python 环境；
- 必需的厂商环境变量通过 `~/.bashrc` 或 `export` 设置；
- 框架负责选择脚本、加载配置、执行测试和输出结果。

框架使用启动 `luban-meter` 的当前 Python 执行 Benchmark，并继承当前
Shell 环境，不再读取单独的环境配置文件。

要求：

```text
Python >= 3.12
```

## 核心设计

单任务由四个核心参数确定：

```text
module + vendor + benchmark + config
```

执行流程：

```text
CLI
→ Core Engine
→ 按 module/vendor/benchmark 选择厂商脚本
→ Host Execution
→ raw_result.json
→ ResultManager
→ result.json
```

厂商 Benchmark 脚本和 Suite 使用固定目录：

```text
src/luban_meter/vendors/<vendor>/
├── benchmark/<module>/<benchmark>/
│   ├── benchmark.py
│   ├── result.py
│   └── config.example.yaml
└── suites/
    └── <suite>.yaml
```

例如，NVIDIA 生成式推理按测试场景组织：

```text
src/luban_meter/vendors/nvidia/
├── benchmark/generate/
│   ├── common/             # 流式解析、token 校验和统计聚合
│   ├── serving-online/     # 完整在线服务压测
│   │   ├── benchmark.py
│   │   ├── result.py
│   │   └── nvidia_vllm_serving_online.yaml
│   └── vllm-engine-stage/  # vLLM Prefill/Decode 引擎阶段测试
│       ├── benchmark.py
│       ├── result.py
│       └── nvidia_vllm_engine_stage.yaml
└── suites/
```

在线场景遍历精确输入长度、输出长度和固定请求速率矩阵，每个 Case 一次采集
完整请求时间线，同时计算用户视角和服务视角指标；
引擎阶段场景一次加载模型并遍历输入、输出和请求批量矩阵，计算 Prefill、
Decode 与内部 TTFT 指标，并在指标前记录 KV Cache 静态容量环境。
不同厂商可以使用不同的调用方式和配置字段。框架按照目录发现脚本，
新增目录后即可被模块发现。

## 测试参数

框架不设置 Case。脚本开发者在 `config.example.yaml` 中说明需要的参数，
并在 `benchmark.py` 中完成字段校验。

用户复制模板并填写自己的测试场景，例如：

```yaml
rounds: 100
warmup: 10
concurrency: 1
input_length: 1024
output_length: 128
service_url: http://127.0.0.1:8000
```

## CLI 示例

```bash
luban-meter run \
  --module generate \
  --vendor ascend \
  --benchmark ttft \
  --config configs/benchmarks/ascend-ttft.yaml \
  --model-path /data/models/Qwen3-8B \
  --model-name Qwen3-8B
```

`--benchmark` 是受厂商目录管理的逻辑脚本名，不是任意 Python 文件路径。

同一厂商环境中顺序执行多个脚本：

```bash
luban-meter suite \
  --vendor ascend \
  --suite full-benchmark \
  --model-path /data/models/Qwen3-8B \
  --model-name Qwen3-8B
```

Suite 文件位于 `vendors/ascend/suites/full-benchmark.yaml`。Suite 中的每个
任务只能调用 `vendors/ascend/benchmark/` 下的脚本，每个任务独立输出
`result.json`，框架不比较任务结果。

查看模块和已发现的厂商脚本：

```bash
luban-meter benchmarks list
```

## 安装与验证

```bash
python3.12 -m pip install -e ".[dev]"
luban-meter benchmarks list
pytest -q
ruff check src tests
```

真实厂商 Benchmark 脚本和 Suite 可按照上述目录约定继续添加。
