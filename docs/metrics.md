# LuBan-Meter 生成式推理指标说明

本文定义 LuBan-Meter 生成式推理 Benchmark 的指标语义，覆盖：

- 在线服务测试：从客户端和整个服务测试窗口观察性能；
- 离线引擎测试：直接读取推理引擎内部时间戳，观察 Prefill、Decode 和
  Engine 执行性能。

指标必须与测试视角、计时边界、负载参数和单位一起解释。名称相似但视角不同
的指标不能直接等同或混合比较。

## 1. 指标组织原则

LuBan-Meter 按“测试场景和采集边界”组织 Benchmark，不按最终指标拆分脚本。

一次在线测试完整读取流式响应，保存同一批请求的时间线和 Token 数，再从这些
原始数据计算 TTFT、ITL、TPOT、E2EL 和吞吐量。一次 Engine 测试读取同一批
`RequestOutput.metrics` 时间戳，再计算内部 TTFT、Prefill、Decode 和 Engine
吞吐量。

因此：

- `benchmark.py` 负责执行负载并采集原始事实；
- `result.py` 负责校验原始数据并计算全部适用指标；
- 用户通过配置选择测试场景、输入/输出长度、请求数量和并发度；
- 用户不需要逐项选择 Mean、P50、P90 或 P99；
- 只有采集代价较高的服务端监控、设备监控和逐事件明细才需要独立开关。

同一次运行得到多个指标，不代表一次运行可以覆盖全部负载场景。低并发延迟、
高并发吞吐和不同输入/输出长度仍应分别测试。

## 2. 两类测试的边界

| 测试 | 当前 Benchmark | 观察位置 | 包含的主要开销 |
|---|---|---|---|
| 在线服务 | `serving-online` | OpenAI-compatible API 客户端 | HTTP、API Server、排队、调度、Prefill、Decode、流式传输 |
| 离线引擎 | `vllm-engine-offline` | vLLM Engine 内部 | Engine 排队/调度、Prefill、Decode；不包含 HTTP 和网络 |

在线和 Engine 测试的结果用于回答不同问题：

- 在线测试回答“用户实际等待多久、服务整体能处理多少负载”；
- Engine 测试回答“引擎内部各阶段耗时和 Token 处理能力如何”。

除非测试条件和时间边界完全一致，否则在线 TTFT 与 Engine 内部 TTFT 只能用于
定位额外开销，不能视为同一个指标的两种实现。

## 3. 通用统计字段

样本型指标统一输出以下描述统计：

| 字段 | 含义 |
|---|---|
| `unit` | 单位，例如 `ms`、`ms/token`、`token/s` |
| `count` | 有效样本数 |
| `mean` | 算术平均值 |
| `median` | 中位数，与当前 `p50` 相同 |
| `p50` | 第 50 百分位数 |
| `p90` | 第 90 百分位数 |
| `p99` | 第 99 百分位数 |
| `min` | 最小值 |
| `max` | 最大值 |
| `stddev` | 总体标准差 |

LuBan-Meter 当前使用线性插值计算分位数。比较不同运行时，必须同时检查样本数；
样本很少时，特别是 P99，统计稳定性不足。

## 4. 在线服务指标

在线测试通过 `/tokenize` 构造精确长度的 Token ID Prompt，再通过流式
`/v1/completions` 采集完整请求时间线。脚本对 `input_lengths ×
output_lengths × request_rates` 做笛卡尔积，每个组合形成一个独立 Case，分别
预热、计时和聚合，不能把不同负载条件的样本混合统计。

服务必须接受 Token ID Prompt，并支持 `min_tokens`、`ignore_eos` 和流式
`stream_options.include_usage`。脚本强制 `temperature=0`、`ignore_eos=true`、
`min_tokens=max_tokens=output_length`，请求返回的实际输入和输出 Token 数必须与
Case 完全一致，否则该请求记为失败。

### 4.1 Case 维度与固定请求速率

每个在线 Case 由以下三个维度唯一确定：

```text
input_length + output_length + request_rate
```

`request_rate` 是客户端计划每秒发起的请求数。第 `i` 个请求的计划时刻为：

```text
scheduled_time[i] = case_start_time + i / request_rate
```

这是开放式固定速率负载。`max_concurrency` 只作为客户端安全上限；当在途请求
达到上限时，线程池会延迟后续请求的实际启动。结果使用 `dispatch_delay` 和
`achieved_request_start_rate` 暴露这种偏差，不能只看配置的 Request Rate。

### 4.2 原始采集数据

每个请求至少记录：

| 字段 | 含义 |
|---|---|
| `scheduled_offset_ms` | 请求计划提交时刻相对 Case 起点的偏移 |
| `start_offset_ms` | 请求开始时刻相对正式测试起点的偏移 |
| `dispatch_delay_ms` | 实际开始时间减计划提交时间 |
| `end_offset_ms` | 完整流式响应结束时刻相对测试起点的偏移 |
| `duration_ms` | 请求完整持续时间 |
| `ttft_ms` | 请求开始至第一个非空生成事件到达的时间 |
| `e2el_ms` | 请求开始至完整流式响应结束的时间 |
| `itl_samples_ms` | 相邻非空生成事件的到达时间间隔 |
| `input_tokens` | API `usage.prompt_tokens` 报告的输入 Token 数 |
| `output_tokens` | API `usage.completion_tokens` 报告的输出 Token 数 |
| `status` | 请求成功或失败 |

输入 Prompt 正文和生成正文不写入结果，只保留 Token 数与时间线。

### 4.3 Request View：单请求用户体验

#### TTFT

Time To First Token，客户端从开始发送请求到收到第一个非空生成事件的时间：

```text
TTFT = first_output_event_time - request_start_time
```

- 输出名：`metrics.cases[].request_view.ttft`
- 单位：`ms`
- 包含：客户端 HTTP 等待、服务排队、调度、Prefill、首 Token 生成与传输；
- 不包含：第一个输出事件之后的 Decode。

当前实现以“第一个非空文本 SSE 事件”为首个输出事件。如果服务发送空事件、
心跳或只有 usage 的事件，这些事件不计为首 Token。

#### ITL

Inter-Token Latency，在线流中相邻非空生成事件的到达间隔：

```text
ITL[i] = output_event_time[i] - output_event_time[i - 1]
```

- 输出名：`metrics.cases[].request_view.itl`
- 单位：`ms`
- 汇总范围：所有成功请求的全部相邻事件间隔。

当前采集器记录的是 SSE 生成事件，而不是由客户端 tokenizer 逐 Token 解码得到
的时间戳。只有服务保证一个非空事件对应一个 Token 时，该值才是严格的逐 Token
ITL；如果一个事件包含多个 Token，它表示事件间隔。

#### TPOT

Time Per Output Token，首个输出之后，平均生成一个后续 Token 的时间：

```text
decode_duration = E2EL - TTFT
TPOT = decode_duration / (output_tokens - 1)
```

- 输出名：`metrics.cases[].request_view.tpot`
- 单位：`ms/token`
- 仅在 `output_tokens > 1` 且 Decode 时间大于 0 时产生样本。

首 Token 由 TTFT 表示，因此分母使用 `output_tokens - 1`。输出长度为 1 的纯
TTFT 测试不能计算 TPOT。

#### E2EL

End-to-End Latency，客户端从请求开始到完整流式响应结束的时间：

```text
E2EL = request_end_time - request_start_time
```

- 输出名：`metrics.cases[].request_view.e2el`
- 单位：`ms`

近似关系为：

```text
E2EL ~= TTFT + TPOT * (output_tokens - 1)
```

该关系受网络抖动、流式事件聚合和结束事件传输开销影响，不要求逐样本完全相等。

#### 单请求输出 Token 吞吐量

```text
request_output_token_throughput = output_tokens / E2EL
```

- 输出名：`metrics.cases[].request_view.output_token_throughput`
- 单位：`token/s`

它把 TTFT 也计入请求总耗时，描述单个用户从发起请求到完成期间观察到的平均
输出速率。

#### 单请求 Decode Token 吞吐量

```text
request_decode_token_throughput = 1 / TPOT
```

- 输出名：`metrics.cases[].request_view.decode_token_throughput`
- 单位：`token/s`

它排除首 Token 前的等待，更接近用户看到首 Token 后的平均生成速度。

#### Dispatch Delay

```text
dispatch_delay = actual_request_start_time - scheduled_time
```

- 输出名：`metrics.cases[].request_view.dispatch_delay`
- 单位：`ms`

该值反映客户端是否按配置速率及时发起请求。持续增大通常表示
`max_concurrency` 太小、客户端资源不足或服务处理速度低于施加负载的速度。

### 4.4 Service View：整个 Case 窗口的服务能力

这里的 Service View 仍由客户端测试窗口推导，不是 vLLM 内部监控指标。

#### Offered 与 Achieved Request Start Rate

`offered_request_rate` 是 Case 配置的固定请求速率；
`achieved_request_start_rate` 根据第一个和最后一个请求的实际启动时间计算：

```text
achieved_request_start_rate =
    (total_requests - 1) / (last_start_time - first_start_time)
```

两者应结合 `dispatch_delay` 判断负载生成是否符合计划。只有一个正式请求时，
无法形成启动时间窗口，Achieved Request Start Rate 输出 0。

#### 请求吞吐量

```text
request_throughput = successful_requests / benchmark_duration
```

- 输出名：`metrics.cases[].service_view.request_throughput`
- 单位：`req/s`

失败请求不计入成功吞吐量，但会记录在 `failed_requests` 中。

#### 输入 Token 吞吐量

```text
input_token_throughput = total_input_tokens / benchmark_duration
```

- 输出名：`metrics.cases[].service_view.input_token_throughput`
- 单位：`token/s`

#### 输出 Token 吞吐量

```text
output_token_throughput = total_output_tokens / benchmark_duration
```

- 输出名：`metrics.cases[].service_view.output_token_throughput`
- 单位：`token/s`

这是评价在线生成服务整体产出能力的主要吞吐量指标，不应与单请求
`request_view.output_token_throughput` 混淆。

#### 总 Token 吞吐量

```text
total_token_throughput =
    (total_input_tokens + total_output_tokens) / benchmark_duration
```

- 输出名：`metrics.cases[].service_view.total_token_throughput`
- 单位：`token/s`

该指标混合了 Prefill 输入 Token 和 Decode 输出 Token。两类 Token 的计算特征
不同，因此必须同时报告输入和输出吞吐量，不能只报告总 Token 吞吐量。

#### 并发度

| 指标 | 输出名 | 含义 |
|---|---|---|
| 配置并发上限 | `maximum_request_concurrency` | 固定速率调度的客户端安全上限 |
| 峰值并发 | `peak_concurrent_requests` | 测试实际观察到的最大 in-flight 请求数 |
| 平均并发 | `average_concurrency` | 全部请求持续时间之和除以测试窗口，符合 Little's Law 的时间加权并发 |

平均并发计算式：

```text
average_concurrency = sum(all_request_duration) / benchmark_duration
```

失败请求占用服务和客户端资源，因此当前平均并发包含失败请求持续时间。

### 4.5 成功与失败状态

| 请求结果 | Benchmark 状态 |
|---|---|
| 全部成功 | `success` |
| 部分失败 | `partial_failed` |
| 全部失败 | `failed` |

每个 Case 记录自己的 `request_outcome`、`total_requests`、`successful_requests` 和
`failed_requests`。比较吞吐量时必须检查失败数量，不能以降低成功率换取更高的
表面吞吐量。顶层 Benchmark 状态根据全部 Case 的请求共同确定。

### 4.6 SLO 与 Goodput

SLO（Service Level Objective）是服务质量的量化边界。LuBan-Meter 在在线服务
Benchmark 中支持可选的 SLO 配置块，用于两个目的：

1. **熔断**：执行期 Case 级 P99 E2EL 超过阈值时，停止后续 Case；
2. **Goodput**：区分“有量无质”和“有效产出”，只统计满足 SLO 的成功请求吞吐。

#### SLO 配置

SLO 块在 `serving_online.yaml` 中以 `slo` 字段配置，可选：

```yaml
slo:
  p99_ms: 2000    # 熔断阈值：Case 级 P99 E2EL
  ttft_ms: 500    # Goodput 维度：首 Token 延迟
  tpot_ms: 50     # Goodput 维度：每 Token 生成时间
  e2el_ms: 8000   # Goodput 维度：端到端延迟
```

| 字段 | 用途 | 单位 |
|---|---|---|
| `p99_ms` | 熔断判定 | `ms` |
| `ttft_ms` | Goodput 判定 | `ms` |
| `tpot_ms` | Goodput 判定 | `ms/token` |
| `e2el_ms` | Goodput 判定 | `ms` |

SLO 块整体可选。省略时既不触发熔断，也不计算 Goodput。配置了 SLO 块时至少
需要一个阈值字段。

#### 熔断机制

熔断在 `benchmark.py` 的 Case 循环中执行：

1. 每个 Case 完成后，从成功请求的 E2EL 样本计算 P99；
2. 若 P99 超过 `p99_ms` 阈值，设置 `circuit_breaker` 并跳过后续全部 Case；
3. 当前 Case 的结果正常输出，不中断；
4. 后续被跳过的 Case 不执行，只记录维度信息。

P99 计算复用 `common/statistics.py` 的 `percentile(samples, 0.99)` 函数，与
现有统计字段保持一致。成功样本数少于 10 时不计算 P99，避免小样本统计不稳定。

熔断结果记录在 `metadata.circuit_breaker`：

| 字段 | 含义 |
|---|---|
| `triggered` | 是否触发 |
| `threshold_p99_ms` | 配置的阈值 |
| `actual_p99_ms` | 触发时的实际 P99 |
| `triggered_at_case` | 触发位置（input/output/rate） |
| `remaining_cases_skipped` | 被跳过的 Case 数 |

被跳过的 Case 列表记录在 `metadata.skipped_cases`。

#### Goodput 计算

Goodput 在 `result.py` 的 `process_case()` 中计算。SLO 配置从
`metadata.slo_config` 读取，逐请求判定。

**判定逻辑**（AND 关系）：

一个成功请求被判定为 SLO-satisfied，当且仅当所有已配置维度均满足阈值：

- `ttft_ms` 配置时：`TTFT <= ttft_ms`；
- `tpot_ms` 配置时且 `output_tokens > 1` 时：`TPOT <= tpot_ms`；
- `e2el_ms` 配置时：`E2EL <= e2el_ms`。

任一维度违反，该请求记为 SLO-violated。TPOT 仅在 `output_tokens > 1` 时
判定，因为单 Token 输出没有 Decode 阶段，TPOT 未定义。

**输出字段**：

Goodput 结果输出在 `metrics.cases[].service_view.goodput`：

| 字段 | 含义 | 单位 |
|---|---|---|
| `status` | 适用性状态：`applicable` 或 `not_applicable` | — |
| `slo_config` | 配置的 SLO 阈值（含 None 维度） | — |
| `applicable_dimensions` | 实际生效的维度列表 | — |
| `not_applicable_dimensions` | 不适用维度列表（如全部单 Token 时 TPOT 不适用） | — |
| `slo_satisfied_count` | 满足 SLO 的成功请求数 | `request` |
| `slo_violated_count` | 违反 SLO 的成功请求数 | `request` |
| `slo_satisfied_rate` | 满足率 | `ratio` |
| `goodput_request_throughput` | 满足 SLO 的请求吞吐 | `req/s` |
| `goodput_output_token_throughput` | 满足 SLO 的输出 Token 吞吐 | `token/s` |

当 `status` 为 `not_applicable` 时，仅输出 `status`、`reason`、`slo_config`、
`applicable_dimensions` 和 `not_applicable_dimensions`，不包含统计字段。

```text
goodput_request_throughput = slo_satisfied_count / benchmark_duration
goodput_output_token_throughput = sum(satisfied output_tokens) / benchmark_duration
```

Goodput 与 `request_throughput` 的区别：

- `request_throughput` 统计全部成功请求；
- `goodput_request_throughput` 只统计满足 SLO 的成功请求。

当所有请求均满足 SLO 时两者相等；否则 Goodput 低于 request throughput，
反映出“有量无质”的差距。

#### 向后兼容性

SLO 块省略时：

- `metadata` 中不出现 `slo_config` 和 `circuit_breaker`；
- `service_view` 中不出现 `goodput` 字段；
- 所有 Case 正常执行；
- 结果与未引入 SLO 前完全一致。

## 5. 离线 Engine 指标

Engine 测试直接创建 vLLM `LLM`，对输入长度、输出长度和请求 Batch 大小做
笛卡尔积测试。每个 Case 独立预热，然后执行多个正式 Round。

当前实现固定以下条件以保持可比性：

- `temperature = 0.0`；
- `ignore_eos = true`；
- `seed = 0`；
- `enable_prefix_caching = false`；
- `enable_chunked_prefill = false`；
- `detokenize = false`；
- `disable_log_stats = false`，确保 vLLM 输出内部 Metrics。

### 5.1 Case 维度

每个结果 Case 由以下三个维度唯一确定：

```text
input_length + output_length + request_batch_size
```

其中：

- `input_length` 是每个请求的精确 Prompt Token 数；
- `output_length` 是每个请求被强制生成的精确 Token 数；
- `request_batch_size` 是一次 `engine.generate()` 同时提交的请求数，不等同于
  在线服务持续压测中的并发连接数。

### 5.2 Engine 原始时间戳

每个 `RequestOutput` 记录：

| 字段 | 来源/含义 |
|---|---|
| `internal_ttft_seconds` | `RequestOutput.metrics.first_token_latency` |
| `scheduled_ts` | 请求被 Engine 调度执行的时间戳 |
| `first_token_ts` | 首个输出 Token 的 Engine 时间戳 |
| `last_token_ts` | 最后一个输出 Token 的 Engine 时间戳 |
| `actual_prompt_tokens` | vLLM 返回的实际输入 Token 数 |
| `actual_output_tokens` | vLLM 返回的实际输出 Token 数 |

要求时间线满足：

```text
scheduled_ts <= first_token_ts <= last_token_ts
```

### 5.3 Request Metrics：单请求 Engine 指标

#### Engine 内部 TTFT

- 输出名：`request_metrics.internal_ttft`
- 来源：vLLM `metrics.first_token_latency`
- 单位：`ms`

该值由 vLLM 定义，通常描述请求进入 Engine 后到首 Token 产生的时间。它可能
包含调度前等待，因此不应直接当作纯 Prefill Kernel 时间。

#### Prefill Latency

```text
prefill_latency = first_token_ts - scheduled_ts
```

- 输出名：`request_metrics.prefill_latency`
- 单位：`ms`

这是当前实现对“被调度后到首 Token”的阶段定义。它包含该时间窗口中的 Engine
调度和执行开销，不等同于由 GPU profiler 单独测得的 Prefill Kernel 时间。

#### 单请求 Prefill Token Throughput

```text
prefill_token_throughput = prompt_tokens / prefill_latency
```

- 输出名：`request_metrics.prefill_token_throughput`
- 单位：`token/s`

该指标按每个请求分别计算后再做描述统计。

#### Decode Latency

```text
decode_latency = last_token_ts - first_token_ts
```

- 输出名：`request_metrics.decode_latency`
- 单位：`ms`
- 仅适用于 `output_length > 1`。

#### Mean Decode Step Latency

```text
mean_decode_step_latency =
    decode_latency / (output_tokens - 1)
```

- 输出名：`request_metrics.mean_decode_step_latency`
- 单位：`ms/token`

它是整个 Decode 阶段的平均步延迟，不是每一步 Decode 的独立样本分布，也不应
写成 Engine 的逐 Token ITL。

#### Per-sequence Decode Rate

```text
per_sequence_decode_rate =
    (output_tokens - 1) / decode_latency
```

- 输出名：`request_metrics.per_sequence_decode_rate`
- 单位：`token/s/sequence`

它描述单条 Sequence 的平均 Decode 速率。

#### Engine Execution Latency

```text
engine_execution_latency = last_token_ts - scheduled_ts
```

- 输出名：`request_metrics.engine_execution_latency`
- 单位：`ms`

该时间覆盖从调度到最后一个 Token 的 Engine 执行窗口，不包含 HTTP 和客户端
传输，也不保证包含请求进入 Engine 后、首次调度前的排队时间。

### 5.4 Batch Metrics：批量 Engine 能力

#### Aggregate Prefill Token Throughput

当前只在 `output_length == 1` 的纯 Prefill Case 中计算：

```text
prefill_window = max(first_token_ts) - min(scheduled_ts)
aggregate_prefill_token_throughput =
    sum(prompt_tokens) / prefill_window
```

- 输出名：`batch_metrics.aggregate_prefill_token_throughput`
- 单位：`token/s`

限定输出长度为 1，是为了避免后续 Decode 与 Prefill 时间窗口重叠后改变指标语义。

#### Aggregate Decode Token Throughput

当前在 `output_length > 1` 的 Case 中计算：

```text
decode_window = max(last_token_ts) - min(first_token_ts)
aggregate_decode_token_throughput =
    sum(output_tokens - 1) / decode_window
```

- 输出名：`batch_metrics.aggregate_decode_token_throughput`
- 单位：`token/s`

这是 Batch 在 Decode 窗口内的总产出速率，不应与单 Sequence 的 Decode Rate
混淆。

### 5.5 Engine 内部 SLO 与 Goodput

`vllm-engine-offline` 支持可选的 `engine_slo` 配置，用于判断固定矩阵 Case 中满足
Engine 内部时延目标的有效处理能力。这里的 SLO 只描述 vLLM Engine 内部边界，
不包含 HTTP、API Server、网络和客户端排队，不能当作在线服务 SLO。

```yaml
engine_slo:
  internal_ttft_ms: 500
  prefill_latency_ms: 500
  mean_decode_step_latency_ms: 50
  engine_execution_latency_ms: 8000
```

支持的阈值均为请求级最大允许值：

| 字段 | 判定数据 | 适用范围 |
|---|---|---|
| `internal_ttft_ms` | `RequestOutput.metrics.first_token_latency` | 全部请求 |
| `prefill_latency_ms` | `first_token_ts - scheduled_ts` | 全部请求 |
| `mean_decode_step_latency_ms` | `(last_token_ts - first_token_ts) / (output_tokens - 1)` | `output_length > 1` |
| `engine_execution_latency_ms` | `last_token_ts - scheduled_ts` | 全部请求 |

请求只有在全部适用阈值均满足时才记为 Engine SLO-satisfied。输出长度为 1 时，
`mean_decode_step_latency_ms` 标记为不适用；如果它是唯一配置的维度，该 Case 的
Engine Goodput 状态为 `not_applicable`。

Engine Goodput 的时间分母使用正式 Round 的 Engine 活跃窗口之和：

```text
round_engine_window = max(last_token_ts) - min(scheduled_ts)
engine_active_duration = sum(round_engine_window)

engine_goodput_request_throughput =
    engine_slo_satisfied_count / engine_active_duration

engine_goodput_output_token_throughput =
    sum(satisfied output_tokens) / engine_active_duration
```

按 Round 求和可以排除两次同步 `LLM.generate()` 调用之间的 Python 间隔，并避免
把不同 Round 的绝对时间戳跨度误当成 Engine 执行时间。该时间边界仍不同于在线
Benchmark 使用的完整 Case 持续时间。

结果输出在 `metrics.cases[].engine_goodput`：

| 字段 | 含义 | 单位 |
|---|---|---|
| `measurement_boundary` | 固定为 `vllm_engine_internal` | — |
| `duration_basis` | 固定为 `sum_of_formal_round_engine_windows` | — |
| `engine_slo_config` | Engine 内部阈值 | — |
| `applicable_dimensions` | 当前 Case 参与判定的维度 | — |
| `not_applicable_dimensions` | 当前 Case 不适用的维度 | — |
| `engine_active_duration` | 正式 Round Engine 活跃窗口总和 | `s` |
| `evaluated_request_count` | 参与判定的请求数 | `request` |
| `engine_slo_satisfied_count` | 满足全部适用阈值的请求数 | `request` |
| `engine_slo_violated_count` | 至少违反一个阈值的请求数 | `request` |
| `engine_slo_satisfied_rate` | 满足率 | `ratio` |
| `engine_goodput_request_throughput` | 满足阈值的请求吞吐 | `req/s` |
| `engine_goodput_output_token_throughput` | 满足阈值请求的输出 Token 吞吐 | `token/s` |

省略 `engine_slo` 时，原始结果中不写入 `metadata.engine_slo_config`，最终 Case
中也不出现 `engine_goodput`，保持与旧配置兼容。本功能不包含 P99 全局熔断；一个
矩阵 Case 未达目标不会跳过其他输入长度、输出长度或 Batch Size。

### 5.6 KV Cache 环境信息

Engine 报告同时保存以下运行环境信息：

| 字段 | 含义 |
|---|---|
| `num_gpu_blocks` | vLLM 分配的 GPU KV Cache Block 数 |
| `block_size` | 每个 KV Cache Block 容纳的 Token 数 |
| `kv_cache_size_tokens` | KV Cache 总 Token 容量 |
| `kv_cache_max_concurrency` | vLLM 根据容量估算的最大并发倍数 |

这些字段属于 `environment.kv_cache`，用于解释测试条件，不是本次负载实际达到的
KV Cache 使用率或并发度。

## 6. 在线与 Engine 指标对应关系

| 在线指标 | Engine 相关指标 | 能否直接比较 | 原因 |
|---|---|---|---|
| Request TTFT | `internal_ttft` | 否 | 在线包含 HTTP/API/网络；Engine 边界由 vLLM 定义 |
| Request TTFT | `prefill_latency` | 否 | Prefill 从 `scheduled_ts` 开始，不包含在线链路和可能的排队 |
| Request TPOT | `mean_decode_step_latency` | 仅可趋势对照 | 在线包含流式传输，Engine 只看内部时间戳 |
| Request Decode Throughput | `per_sequence_decode_rate` | 仅可趋势对照 | 时间边界不同 |
| Service Output Throughput | Aggregate Decode Throughput | 否 | 在线按完整测试窗口，Engine 按 Decode 窗口 |
| E2EL | Engine Execution Latency | 否 | E2EL 覆盖完整客户端链路 |

可以在相同模型、输入/输出长度和负载条件下计算差值辅助定位开销，例如：

```text
online_extra_latency ~= online_TTFT - engine_internal_TTFT
```

但该差值只是诊断性近似。两个 Benchmark 如果不是同一时刻、同一请求和同一负载，
它还包含运行波动，不能精确拆分为网络或 API Server 开销。

## 7. 数据复用关系

### 7.1 在线测试内可复用的数据

同一批完整流式请求可同时计算：

- TTFT；
- ITL 事件间隔；
- TPOT；
- E2EL；
- 单请求输出/Decode 速率；
- 请求、输入 Token、输出 Token 和总 Token 吞吐量；
- 成功/失败数量；
- 平均、峰值和配置并发度。

因此不需要为这些指标分别发送一轮请求。

### 7.2 Engine 测试内可复用的数据

同一批 Engine 时间戳可同时计算：

- Internal TTFT；
- Prefill Latency 和单请求 Prefill Rate；
- Decode Latency、Mean Decode Step Latency 和 Per-sequence Decode Rate；
- Engine Execution Latency；
- Aggregate Prefill/Decode Throughput。

因此也不需要按每个 Engine 指标分别加载一次模型。

### 7.3 不能直接复用的情况

以下测试条件改变了工作负载，必须作为不同 Case 或不同运行：

- 不同输入长度；
- 不同输出长度；
- 不同 Batch Size 或在线并发度；
- 低并发延迟与高并发吞吐；
- 流式与非流式请求；
- 开启和关闭 Prefix Cache、Chunked Prefill；
- 不同量化、精度、Tensor Parallel 或模型版本；
- 在线 API 服务与直接 Engine 调用。

## 8. 结果解释要求

发布或比较指标时至少同时记录：

- Benchmark 名称和指标完整路径；
- 模型名称、版本和精度；
- 推理引擎及版本；
- 硬件型号、数量和 Tensor Parallel Size；
- 输入长度、输出长度；
- 请求数量、Batch Size 或在线并发度；
- 预热次数和正式采样次数；
- Prefix Cache、Chunked Prefill、CUDA Graph/Eager 等关键开关；
- 成功请求数和失败请求数；
- 指标单位和样本数。

禁止只写“TTFT = 10 ms”而不说明它是在线客户端 TTFT、vLLM Internal TTFT，
还是 `scheduled_ts` 到 `first_token_ts` 的 Prefill Latency。

## 9. 当前实现范围与规划边界

当前代码已经实现：

- `serving-online` 的精确输入/输出长度、固定请求速率、Request View 与客户端
  推导的 Service View；
- `vllm-engine-offline` 的 Engine Request/Batch Metrics；
- `vllm-metrics` 的 vLLM 服务端 /metrics 指标采集与聚合，包括 KV Cache 使用率、
  请求排队数、TTFT/TPOT/E2EL 延迟分解、Prefix Cache 命中率及瓶颈推断；
- 通用 Mean、P50、P90、P99、Min、Max 和 Stddev；
- Engine KV Cache 容量环境信息；
- SLO 配置与熔断机制（Case 级 P99 E2EL 超阈值时跳过后续 Case）；
- Goodput 计算（满足 SLO 的有效请求吞吐，区分有量无质与有效产出）。

当前尚未实现或不应从现有字段推断：

- Prefill/Decode GPU Kernel 级时间；
- 每个 Engine Decode Step 的独立延迟分布；
- 跨节点网络和通信指标；
- ROUGE、准确率等生成质量指标；
- MFU。

这些能力需要设备监控或 profiler 等额外 Collector。新增后应放在独立的
`device` 或 `quality` 结果分组中，不能用客户端推导值冒充服务端内部测量值。
