# LuBan-Meter 使用说明

## 1. 安装与运行环境

```bash
python3.12 -m pip install -e ".[dev]"
```

模型、驱动、Python 依赖、推理引擎和在线服务由用户提前准备。LuBan-Meter 使用
当前 Python 和 Shell 环境，不读取或切换额外环境配置。

## 2. 查看 Benchmark

```bash
luban-meter benchmarks list
```

当前模块：

```text
generate    serving-online,vllm-engine-offline,vllm-metrics  Large-model generation benchmarks
inference   ceval,cmmlu,gsm8k,humaneval         Online-service model evaluation benchmarks
```

`generate` 测量生成式推理性能；`inference` 用于基于在线推理服务的模型效果评测。
当前不提供算子层模块。

## 3. 运行单个 Benchmark

```bash
luban-meter run \
  --module <module> \
  --benchmark <benchmark> \
  --config <config.yaml> \
  [--model-path <path>] \
  [--model-name <name>] \
  [--output runs] \
  [--timeout 3600]
```

参数：

| 参数 | 必填 | 说明 |
|---|---:|---|
| `--module` | 是 | `generate` 或 `inference` |
| `--benchmark` | 是 | `benchmark/<module>/` 下的目录名称 |
| `--config` | 是 | 本次测试参数 YAML |
| `--model-path` | 否 | 本地模型路径 |
| `--model-name` | 否 | 逻辑模型名或在线服务模型名 |
| `--output` | 否 | 结果根目录，默认 `runs` |
| `--timeout` | 否 | 执行超时秒数 |
| `--name` | 否 | 保存到结果中的易读运行名称 |
| `--format` | 否 | 控制台输出 `text`（默认）或 `json` |
| `--monitor-url` | 否 | Prometheus exporter 地址，启用硬件监控 |
| `--monitor-interval` | 否 | 硬件监控采样间隔秒数，默认 `1.0` |

框架按以下路径解析脚本：

```text
src/luban_meter/benchmark/<module>/<benchmark>/benchmark.py
src/luban_meter/benchmark/<module>/<benchmark>/result.py
```

### 硬件监控（可选）

通过 `--monitor-url` 指定 Prometheus exporter 地址后，框架在 Benchmark 运行
期间通过 HTTP GET `/metrics` 端点周期性采集硬件指标（GPU 利用率、功耗、温度、
显存 + CPU 利用率、内存）。不指定该参数时监控完全跳过，零开销。

支持的 exporter：

- **DCGM exporter**（NVIDIA GPU）— 默认端口 9400，提供 `DCGM_FI_DEV_*` 系列指标
- **node_exporter**（CPU/内存）— 默认端口 9100，提供 `node_cpu_*`、`node_memory_*` 指标

通常 DCGM exporter 已包含 node_exporter 的 CPU/内存指标，只需指定一个 URL。

```bash
luban-meter run \
  --module generate \
  --benchmark serving-online \
  --config config.yaml \
  --monitor-url http://43.138.110.236:9400 \
  --monitor-interval 1.0
```

监控数据注入 `raw_result.json` 和 `result.json` 的 `environment.device_monitoring`
字段，包含：

- `hardware_environment`：静态硬件环境（GPU 型号、CPU、内存总量）
- `devices[]`：每张卡的 avg/p50/p90/p99 统计
- `timeseries[]`：每采样周期的完整快照
- `charts`：6 张折线图 PNG（GPU 利用率/功耗/温度/显存 + CPU 利用率/内存）

## 4. 在线生成服务测试

先启动兼容 OpenAI Completions API 的流式服务，然后执行：

```bash
luban-meter run \
  --module generate \
  --benchmark serving-online \
  --config src/luban_meter/benchmark/generate/serving-online/serving_online.yaml \
  --model-name <served-model-name>
```

配置矩阵：

```yaml
service_url: http://127.0.0.1:8000
request_timeout: 120
warmup: 2
rounds: 10
input_lengths: [128, 1024]
output_lengths: [1, 128]
request_rates: [1.0, 5.0]
max_concurrency: 32
seed_prompt: "Explain the benchmark methodology."
temperature: 0.0
ignore_eos: true
seed: 0
```

每个 `input_length × output_length × request_rate` 组合形成独立 Case。脚本通过
`/tokenize` 构造精确长度输入，通过 `min_tokens = max_tokens` 固定输出长度，并以
开放式固定 Request Rate 调度请求。

## 5. vLLM 离线引擎测试

进入已安装 vLLM 的 Python 环境后执行：

```bash
CUDA_VISIBLE_DEVICES=0 luban-meter run \
  --module generate \
  --benchmark vllm-engine-offline \
  --config src/luban_meter/benchmark/generate/vllm-engine-offline/vllm_engine_offline.yaml \
  --model-path /data/models/<model>
```

该 Benchmark 直接调用 vLLM Engine，因此属于带引擎约束的实现；它不代表某个
硬件厂商，能否运行由当前环境是否支持 vLLM 决定。

如需判定满足 Engine 内部时延目标的有效吞吐，可在配置中增加 `engine_slo`。该
结果只覆盖 vLLM Engine 内部调度至 Token 生成的时间窗口，不包含 HTTP、网络和
客户端排队，不能与 `serving-online` 的服务 Goodput 直接比较。字段定义和计算公式
见 [生成式推理指标说明](metrics.md#55-engine-内部-slo-与-goodput)。

## 6. vLLM 服务端指标采集

对正在运行的 vLLM 推理服务，通过 `/metrics` 端点采集 Prometheus 格式指标，
用于瓶颈定位与服务容量评估：

```bash
luban-meter run \
  --module generate \
  --benchmark vllm-metrics \
  --config src/luban_meter/benchmark/generate/vllm_metrics/vllm_metrics.yaml \
  --model-name <served-model-name>
```

配置参数：

```yaml
service_url: http://127.0.0.1:8000
api_key: ""
request_timeout: 10.0
collect_interval: 1.0
collect_duration: 60.0
```

- `collect_interval`：每次 HTTP 请求 `/metrics` 的间隔（秒）
- `collect_duration`：总采集时长（秒），到达后停止采集并聚合

该 Benchmark 不发起推理请求，只读取服务端已有指标，适合接入已运行的 vLLM 服务。

## 8. Suite

Suite YAML 位于：

```text
src/luban_meter/suite/definitions/<suite>.yaml
```

当前内置的标准模型效果 Suite：

```yaml
name: inference-standard
tasks:
  - name: ceval
    module: inference
    benchmark: ceval
  - name: cmmlu
    module: inference
    benchmark: cmmlu
  - name: gsm8k
    module: inference
    benchmark: gsm8k
  - name: humaneval
    module: inference
    benchmark: humaneval
```

相对 `config` 路径以 Suite YAML 所在目录为基准。运行命令：

```bash
luban-meter suite \
  --suite inference-standard \
  --model-name <served-model-name> \
  --output runs
```

部署环境需要替换某个任务的 Benchmark YAML 时，使用可重复的
`--task-config TASK=PATH`，覆盖会写入 `suite_request.json`。例如让 HumanEval
连接专用 Docker daemon：

```bash
luban-meter suite \
  --suite inference-standard \
  --task-config humaneval=/data/luban-meter-config/humaneval-full.yaml \
  --model-name <served-model-name> \
  --output runs
```

失败后立即停止后续任务：

```bash
luban-meter suite --suite inference-standard --fail-fast
```

`suite_result.json` 使用 `luban-meter.suite-result/v2`。每个已执行任务的
`output` 保存完整的 `luban-meter.result/v2` 结果，包括参数、环境、元数据和
指标；跳过的任务 `output` 为 null。各任务也在 `tasks/<run-id>/result.json`
中独立保存结果。仅有 Suite JSON 时仍能生成指标报告；原始产物链接和监控
图片需要一起保留对应文件。

Suite 参数：

| 参数 | 必填 | 说明 |
|---|---:|---|
| `--suite` | 是 | `suite/definitions/` 下的逻辑名称 |
| `--model-path` | 否 | 所有任务共享的模型路径 |
| `--model-name` | 否 | 所有任务共享的模型名称 |
| `--output` | 否 | Suite 输出根目录 |
| `--timeout` | 否 | 单任务默认超时 |
| `--fail-fast` | 否 | 首个失败后停止调度 |
| `--name` | 否 | 保存到结果中的易读运行名称 |
| `--format` | 否 | 控制台输出 `text`（默认）或 `json` |
| `--task-config TASK=PATH` | 否 | 替换指定 Suite 任务的配置 YAML，可重复使用 |
| `--monitor-url` | 否 | Prometheus exporter 地址，启用硬件监控 |
| `--monitor-interval` | 否 | 硬件监控采样间隔秒数，默认 `1.0` |

## 9. 结果目录

单任务：

```text
runs/<run-id>/
├── request.json
├── raw/
│   ├── raw_result.json
│   ├── stdout.log
│   ├── stderr.log
│   └── artifacts/
├── result.json
└── report/
    ├── report.md
    ├── results.csv
    └── figures/          # 有图表声明或监控图片时生成
```

Suite：

```text
runs/<suite-id>/
├── suite_request.json
├── suite_result.json
├── report/               # Suite 总报告，含 MD、CSV 和对应图表
└── tasks/<run-id>/
    ├── result.json
    ├── raw/...
    └── report/           # 每个已执行任务的独立报告
```

`suite_result.json` 内嵌所有已执行任务的完整标准结果；各任务文件也独立保留。
Suite 报告按任务顺序拼接摘要，不对不同语义的指标求平均。

运行请求和最终结果不包含硬件厂商路由字段。硬件、驱动和引擎版本等事实后续统一
写入 `environment`。

## 10. 常见问题

### Benchmark 不存在

确认目录中同时存在：

```text
benchmark/<module>/<benchmark>/benchmark.py
benchmark/<module>/<benchmark>/result.py
```

### 运行依赖不可用

确认启动 LuBan-Meter 前已激活包含所需引擎与依赖的 Python 环境，并设置必要的
设备可见性等环境变量。

### 结果失败

依次检查 `result.json.error`、`raw/stderr.log`、`raw/stdout.log` 和
`raw/raw_result.json`。

## 11. 开发验证

```bash
ruff check src tests
pytest -q
python -m luban_meter benchmarks list
```

## 12. inference 模型任务效果测试

`inference` Benchmark 基于本地数据集调用在线推理服务。安装包内置了 `ceval`、
`cmmlu`、`gsm8k` 和 `humaneval` 的标准评测数据，位于
`src/luban_meter/benchmark/inference/data/`，开箱即用。若需替换数据版本，
可用离线准备脚本将外部官方格式转换为本地 jsonl（运行时不下载数据）：

```bash
python src/luban_meter/benchmark/inference/scripts/prepare_ceval.py \
  --source /path/to/ceval --out data/ceval
python src/luban_meter/benchmark/inference/scripts/prepare_cmmlu.py \
  --source /path/to/cmmlu --out data/cmmlu
python src/luban_meter/benchmark/inference/scripts/prepare_gsm8k.py \
  --source /path/to/gsm8k --out data/gsm8k
python src/luban_meter/benchmark/inference/scripts/prepare_humaneval.py \
  --source /path/to/HumanEval.jsonl.gz \
  --out data/humaneval/HumanEval.jsonl
```

配置中的 `dataset_path` 为相对路径时按以下顺序解析：先相对当前工作目录
（CWD），未命中时回退到包内置的 `benchmark/inference/data/` 目录。因此默认
默认相对 `dataset_path` 会回退到包内置数据，从 `/tmp` 等任意目录运行也可
正常加载。

运行示例（C-Eval 选择题 Accuracy，ppl 模式）：

```bash
luban-meter run \
  --module inference \
  --benchmark ceval \
  --config src/luban_meter/benchmark/inference/ceval/ceval.yaml \
  --model-name <name>
```

HumanEval 首版固定为 completion-only Pass@1。执行模型生成的代码前必须先构建
专用 Docker 沙箱镜像：

```bash
docker build \
  -f src/luban_meter/benchmark/inference/humaneval/Containerfile \
  -t luban-meter-humaneval-sandbox:v1 \
  src/luban_meter/benchmark/inference/humaneval
luban-meter run \
  --module inference \
  --benchmark humaneval \
  --config src/luban_meter/benchmark/inference/humaneval/humaneval.yaml \
  --model-name <name>
```

沙箱不可用时评测直接失败，不会退回宿主机执行。服务器部署时应在配置中把
`docker_host` 指向评测专用 Docker daemon 的 Unix socket。详细安全约束、数据
协议和状态定义见 [HumanEval 协议说明](humaneval-protocol.md)。

当前可用 Benchmark：`ceval`、`cmmlu`（选择题 Accuracy，支持 ppl/gen 两种评测
模式）、`gsm8k`（数学题 Exact Match，gen 模式）和 `humaneval`（代码补全
Pass@1，base completions 模式）。其中 ppl / loss 模式依赖
`/v1/completions` 的 `echo + logprobs` 回显，且仅允许 `prompt_format=base`
（对话格式层会注入特殊 Token 破坏 ppl 续写打分，组合 ppl + chat 会被配置校验
拒绝）；gen 模式可使用 chat 或 base 传输。配置字段、评测模式和指标口径参见
[Inference 评测指标说明](inference.md)。

## 13. v2 结果与报告

所有 Benchmark 的最终 `result.json` 统一为 `luban-meter.result/v2`，其
`metrics` 为字典，内部结构由脚本定义。Suite 使用 `luban-meter.suite-result/v2`。
本版本直接切换到 v2，报告不读取旧 v1 结果；采集文件 `raw_result.json` 的协议
仍为 `luban-meter.raw/v1`，不能将它作为报告输入。

`run` 和 `suite` 执行完成后自动输出控制台摘要，并在结果目录的 `report/`
中生成 Markdown、CSV 和适用的静态 PNG。用 `--name "第一次测试"` 保存易读
名称；用 `--format json` 让标准输出保持纯 JSON，执行日志及报告路径走标准错误。
报告生成失败会提示错误，已保存的评测结果和评测退出状态保留不变。

### 单次 Benchmark 和 Suite

单次 Benchmark 报告展示该脚本的核心指标。单次 Suite 报告先列出所有任务的
执行状态，再按任务顺序展示各脚本的指标表格和图表。失败原因和跳过状态也保留。

现有脚本的摘要范围：

| 脚本 | Markdown 核心指标 | 静态图 |
|---|---|---|
| `serving-online` | Case 条件、成功/失败请求、请求和输出吞吐、TTFT/TPOT P50/P99 | 同输入输出长度下，吞吐和 TTFT 随请求速率变化 |
| `vllm-engine-offline` | Case 条件、Engine Prefill/Decode 吞吐、内部 TTFT、Decode Step 和 Engine E2E 延迟 | 同输入输出长度下，吞吐和内部 TTFT 随 Batch Size 变化 |
| `vllm_metrics` | 服务状态、请求/Token 吞吐、TTFT/TPOT 分位数 | 聚合指标不生成时间曲线 |
| `ceval`、`cmmlu` | 总体和分学科 Accuracy、评分及失败样本数 | 总体和分学科得分柱状图 |
| `gsm8k` | Exact Match、评分及失败样本数 | 得分柱状图 |
| `humaneval` | Pass@1、任务数、通过数、解析/服务/沙箱失败及超时数 | Pass@1 柱状图 |

评测图表紧跟对应的指标表格。每个 Benchmark 区块开头先展示一张**硬件环境与
监控总览图**，将已记录的硬件环境、监控摘要和曲线集中在同一张 PNG 中：

- 上方展示设备型号、厂商、CPU、内存容量等环境信息，以及已保存的监控摘要；
- 下方按两列排列设备利用率、功耗、温度、设备内存、CPU 利用率和系统内存曲线；
- 多卡保留各自的曲线，缺失采样保持空缺，不补零，不进行优劣或稳定性判断；
- 只有环境信息或聚合摘要时仍可生成总览图，不推造监控曲线；
- 优先使用 JSON 中的 `timeseries` 绘图，无需附带原 PNG。没有可绘制的采样时，
  可将结果引用的现存监控 PNG 排入同一张总览图；
- 环境字段来自结果中保存的信息，报告导出不查询本机或远端硬件，也不补填
  未采集的设备规格；完全没有硬件信息时省略该图。

单次报告和 Suite 内的各任务均采用“硬件总览 → 评测指标 → 评测曲线”的顺序。
原始数据和独立监控产物继续保留，报告只引用一张 `hardware-<任务序号>-overview.png`。

效果评测在硬件总览与指标表之间另有精简的“评测条件”：展示模型、数据集与
样本范围、评分模式、Prompt 版本、few-shot 和解码参数，控制台同步展示。
优先使用结果中已记录的实际条件，回退到原配置的值标注“配置”；必要字段缺失时
显示“未记录”。已记录的 `null`、空列表、零值保持原样，不推测缺项，不展开
Prompt 全文。Suite 中每个任务分别展示自己的条件。

错误的可选表格或图表声明会让该任务回退到通用指标摘要，并说明原因，其他任务
正常展示。表格、CSV 与图表保持指标单位一致，子字段单位可覆盖父级单位。

已保存的 v2 结果可重新导出，`--input` 接受 JSON 文件或包含唯一结果文件的目录：

```bash
luban-meter report --input runs/<run-id>/result.json
luban-meter report --input runs/<suite-id>/suite_result.json
luban-meter report --input runs/<run-id> --output reports/single --name "测试 A"
```

没有 `--output` 时写入源结果旁的 `report/`；指定时直接写入该目录。

### 多次 Suite

重复 `--input`，每份输入独立生成报告：

```bash
luban-meter report \
  --input runs/<suite-id-a>/suite_result.json \
  --input runs/<suite-id-b>/suite_result.json \
  --output reports
```

输出为 `reports/suite-<来源路径摘要>/report.md` 等独立目录。即使两份结果使用
相同 Suite ID，也不会因来自不同路径而覆盖。输入也可混合单次 Benchmark 结果。
不指定 `--output` 则各自写到源结果旁的 `report/suite-<来源路径摘要>/`，多份
JSON 存在同一个目录时也互不覆盖。一份输入无效时提示错误并继续处理其余输入。
多个输入不能共同使用 `--name`，各报告沿用结果中保存的名称。

多次运行之间不做基线选择、参数一致性检查、Case 匹配、差值、百分比或优劣判断。

### CSV 和新增脚本

`results.csv` 使用通用长表：每行是某任务的一个数值或 null 叶子，列为
`report_id,name,task,module,benchmark,run_id,status,metric,value,unit`。
`metric` 为相对 `metrics` 的 JSON Pointer，例如
`/cases/0/request_view/ttft/p99`；列表下标用于定位原结果，不跨运行匹配。
普通数值也可导出，不强制采用 `{value, unit}` 结构；没有单位则留空。
CSV 使用 UTF-8 BOM，零值保留为 0，缺失或非有限数值留空，空指标任务保留状态行。

Markdown 只展示声明的核心字段；CSV 遍历全部数值指标；JSON 保留完整结构及
文本信息。新增脚本未声明摘要规则时，自动展示前 30 个数值/null 指标，不会
自动猜测曲线含义。可选表格与图表声明参见
[Benchmark 脚本开发指南](develop-benchmark.md#62-报告声明可选)。
