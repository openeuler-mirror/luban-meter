# 在线性能与思考阶段指标说明

本文说明`online_serving`、`multi_turn_serving`的整体在线性能指标，
以及显式开启后的思考与正文分阶段指标。配置、运行及其他场景见
[README](README.md)；本文不以客户端时间替代引擎内部时间。
图文计算示例及其他在线观测指标见[HTML指标说明](METRICS.html)。

## 1. 观察位置与原始事实

在线性能使用客户端单调时钟。一次请求的时间符号如下，均以客户端观察为准：

| 符号 | 含义 | 原始响应字段 |
|---|---|---|
| `t0` | HTTP请求构造完成后、实际发送前的计时起点 | 各延迟与事件偏移的共同起点 |
| `t1` | 首个有效输出事件 | `output_event_offsets_ms[0]`，相对`t0` |
| `tn` | 最后有效输出事件 | `last_output_latency_ms`，相对`t0` |
| `te` | 流式采集结束 | `e2el_ms`，相对`t0` |
| `r1`、`rn` | 首个、最后一个思考事件 | `reasoning_event_offsets_ms`首尾，相对`t0` |
| `c1`、`cn` | 首个、最后一个正文事件 | `content_event_offsets_ms`首尾，相对`t0` |
| 工具事件 | 有效工具调用片段 | `tool_call_event_offsets_ms`，相对`t0`；不属于正文或思考通道 |

Completions使用非空`choices[0].text`；Chat使用独立的
`delta.reasoning`、`delta.reasoning_content`和`delta.content`，以及有效
`delta.tool_calls`。工具调用列表中任一`function.name`或`function.arguments`
为非空字符串即有效；只有`id`、`type`、`index`或空字段不计时。空格和换行字符串
属于非空输出。整体有效输出包括思考、正文和工具调用；同一事件共用时间戳，只记一次。
空角色事件、空文本、仅usage或Token ID的事件、心跳及`[DONE]`不算有效输出。
工具事件只保存时间，不拼入`generated_text`或`reasoning_text`；无正文时首正文延迟
仍为`null`。仅工具输出的性能请求可成功，仍沿用usage及流完整性校验。
`finish_reason=tool_calls`可进入整体指标，但分阶段仍按`other_finish`排除。
多轮无正文时记录请求性能后以`tool_calls_only`结束会话，不执行工具或自动续轮。

TTFT从`t0`计时，包含HTTP发送、服务排队及首个有效输出前的客户端可见等待。
Chat在已有有效输出后，可收到`[DONE]`或正常EOF结束采集；
Completions必须收到`[DONE]`，即使已收到正文、正常stop和完整usage，
缺少该标记仍会使请求失败，不产生成功响应的E2EL。
`te`在采集器成功返回后记录；之后HTTP响应排空与关闭不再增加E2EL，
但仍可能增加Case窗口及请求执行槽占用。Chat在EOF结束时也不会自动补造
`finish_reason=stop`，分阶段资格仍按实际结束原因判断。
因此`tn`与`te`不能混用，客户端TTFT也不能直接当作服务端Prefill或引擎内部TTFT。

## 2. 整体在线指标

令`N`为单请求的`usage.completion_tokens`，`I`为`usage.prompt_tokens`。
应确认服务的输出总数覆盖思考、正文与工具调用；工具不再把思考Token数累加到`N`，
也不按工具调用片段数量估算缺失Token数。

| 指标 | 正常采集的单请求计算 | 单位 | Case汇总 |
|---|---|---|---|
| TTFT | `t1−t0` | ms | 请求等权 |
| TPOT | `(tn−t0)/N`，包含首输出等待 | ms/token | 按`N`加权 |
| ITL | `(tn−t1)/(N−1)`，要求`N>1` | ms/token | 按`N−1`加权 |
| E2EL | `te−t0` | ms | 请求等权 |
| Stream Event ITL | 同一请求内相邻有效流事件的时间差 | ms | 每个事件间隔等权 |
| Request Throughput | 成功请求数/Case正式窗口 | req/s | Case级速率 |
| Input/Output Token Throughput | 成功请求对应输入/输出Token合计/Case正式窗口 | token/s | Case级速率 |
| Total Token Throughput | 成功请求输入与输出Token合计之和/Case正式窗口 | token/s | Case级速率 |

请求级`output_token_throughput`是`1000×N/E2EL_ms`，
`decode_token_throughput`是`1000/ITL_ms`，各自按有效请求形成分布；
它们不等于Case级输出Token总数除以Case窗口。

Case正式窗口以保存的实际窗口为准，不用请求延迟均值代替；预热不进入正式样本。
单轮和多轮共用`streaming_performance`。多轮每轮是一次请求，
请求指标的样本数是实际保存的正式请求数；会话耗时包含轮间等待，属于独立视图。
失败或部分完成的会话中，已结束的请求仍按其自身状态统计，未发送轮次不计请求失败。

整体延迟分布纳入成功请求，不要求`finish_reason=stop`；
HTTP、解析或超时等失败计入请求结果统计，不贡献成功请求吞吐量和Token。
成功但截断的请求可进入整体指标，不能套用下文分阶段的正常stop过滤规则。
整个`usage`缺失或为null时，只要已有有效输出并满足对应接口的结束条件，
请求仍可成功并保留TTFT/E2EL；输入、输出Token计数均缺失，不能提供TPOT/ITL样本。
若`usage`对象存在，`prompt_tokens`与`completion_tokens`必须同时为非负整数
（不接受布尔值）。任一字段缺失、为null或类型/取值非法，均沿用采集校验令请求失败，
不保留成功响应的TTFT/E2EL，也不进入成功延迟分布；有有效输出却报告
`completion_tokens=0`时也失败，`prompt_tokens=0`允许。
这与下文“仅缺失思考计数`reasoning_tokens`”的补充规则不同。
不按字符或事件数估算缺失计数，读取结果时同时检查`count`、单位和缺失原因。

当前整体计算器对缺少`last_output_latency_ms`的事实保留`E2EL/N`的TPOT回退，
此时Token ITL不可用；正常流式采集会保存最后有效输出时间。
缺少必要计时或非法计数仍沿用既有请求/指标校验，不补造时间。

### 2.1 均值、分位数与样本数

- TTFT、E2EL及流事件间隔采用线性插值分位数：排序后位置为`(样本数−1)×p`。
- 整体TPOT、ITL采用Token权重计算均值、总体标准差和加权CDF分位数；
  分位数取累计权重首次达到`p×总权重`的请求值。
- `count`为有效请求数或有效事件间隔数，取决于指标样本单位；
  `weight_sum`为Token数或Token间隔数，不能当作请求数。
- 一个SSE事件可能含多个Token。事件间隔分布与请求平均Token ITL不能互相替代。

`--goodput tpot:50`沿用已有SLO含义：约束排除首输出等待的Token ITL，
不是上表包含首输出等待的TPOT。多个SLO需同时满足；缺少必要观测时结果可为未判定，
应结合Goodput结果中的状态、分母和未判定数量读取。

## 3. 开启分阶段指标

整组分阶段指标默认关闭，包括思考指标、正文输出时长和首正文延迟。
在现有Chat性能任务中加入：

```json
{
  "thinking_metrics": true
}
```

这是客户端统计开关，不控制模型是否思考，也不自动添加服务专有请求参数。
服务必须实际返回可识别的独立思考/正文通道；正文中的`<think>`文本标签
不用于拆分时间通道。普通Completions接口不能提供这些分阶段统计。

开启后默认指标列表包含`streaming_phase_timing`；关闭时不生成
单请求`response.phase_timing`、分阶段Metric汇总或终端/HTML分阶段展示，
原始通道事实、服务usage及原有校验继续保留。

CLI对应`--thinking-metrics`和`--no-thinking-metrics`，任务JSON显式字段优先于
CLI基础参数。显式`metrics`列表仍覆盖默认列表；只开启开关不会覆盖自选指标。
显式选择`streaming_phase_timing`时也必须开启开关，否则配置检查报错。

## 4. 思考与正文指标

令`R`为单次请求的思考Token数，即请求记录的`reasoning_tokens`。

| 指标键 | 计算 | 单位 | 汇总 |
|---|---|---|---|
| `reasoning_ttft` | `r1−t0` | ms | 有效请求等权 |
| `reasoning_duration` | `rn−r1` | ms | 有效请求等权 |
| `reasoning_tpot` | `(rn−t0)/R`，要求`R>0` | ms/token | 按`R`加权，仅均值 |
| `reasoning_itl` | `(rn−r1)/(R−1)`，要求`R>1` | ms/token | 按`R−1`加权，仅均值 |
| `reasoning_stream_event_itl` | 请求内相邻思考事件的时间差 | ms | 每个事件间隔等权 |
| `content_duration` | `cn−c1` | ms | 有效请求等权 |
| `first_content_latency` | `c1−t0` | ms | 有效请求等权 |

思考TTFT和TPOT包含首思考事件之前的等待，思考ITL排除这段等待。
首正文延迟包含思考及正文出现前的全部客户端可见等待，不等于正文生成耗时。
正文、思考跨度可能重叠，不能要求二者相加等于E2EL。

思考TPOT/ITL只保存和展示均值、有效请求数`count`及权重`weight_sum`，
不生成分位数、极值或标准差。其他时间指标及流事件间隔保留各自的统计结构。
一个事件内含多个Token时，不为它们虚构逐Token时间戳。

### 4.1 加权示例

以下事件时间均为相对本轮请求开始的毫秒偏移：

| 请求 | 首思考事件 | 末思考事件 | R | 思考TPOT | 思考ITL |
|---|---:|---:|---:|---:|---:|
| A | 100 | 500 | 2 | 250 | 400 |
| B | 200 | 800 | 4 | 200 | 200 |

- 思考TTFT均值为`(100+200)/2=150 ms`。
- 思考TPOT均值为`(500+800)/(2+4)=216.667 ms/token`，N=2，权重=6。
- 思考ITL均值为`((500−100)+(800−200))/((2−1)+(4−1))=250 ms/token`，
  N=2，权重=4。

多轮按实际请求采用同样的汇总方式，不先计算各会话均值再等权平均，
轮间等待不加入某一轮请求的思考计时。

### 4.2 样本排除与缺失

分阶段主要汇总只纳入成功且`finish_reason=stop`的请求：

| 情况 | 分阶段处理 |
|---|---|
| 请求失败 | 整个请求排除，记为`failed` |
| `finish_reason=length` | 整个请求排除，记为`truncated`；即使思考已结束也不单独保留 |
| 缺少结束原因 | 整个请求排除，记为`missing_finish` |
| 其他结束原因 | 整个请求排除，记为`other_finish` |
| 缺思考或正文通道 | 相应指标为null，不进该指标分母，不补0 |
| R缺失/null或为0 | 思考TPOT/ITL不可用；已有思考时间指标仍可计算 |
| R=1 | 思考TPOT可用，Token ITL不可用 |
| 某阶段只有一个非空事件 | 该阶段观察跨度为0；没有事件间隔样本 |

单事件时若R>1，平均Token ITL的观察值可为0；这不表示引擎内部各Token生成耗时为0。
缺失阶段不证明模型没有思考。原始单请求记录可保留被排除请求的观测，
不应绕过统计资格自行混入主要分布。

## 5. 思考Token数的来源

原生计数的对应字段是`usage.completion_tokens_details.reasoning_tokens`，
不是整体的`completion_tokens`或`total_tokens`。

| 服务返回 | 处理与配置 |
|---|---|
| 合法非负整数，包括0 | 优先使用原生值；只需开启`thinking_metrics`即可用于思考统计 |
| 缺失或null | 如需思考TPOT/ITL，可显式请求Token ID并提供匹配tokenizer补充计数 |
| 非法负值、布尔值、小数、字符串，或非null且非对象的`completion_tokens_details` | 沿用既有解析校验令请求失败，不降级为缺失，不用ID覆盖非法值 |

合法0不触发补充计数。非法服务字段的校验不由思考指标开关控制。

### 5.1 用curl确认原生计数

不了解服务能力时，建议使用与压测一致的模型、流式接口、鉴权和思考参数，
先发一次小样本请求。例如：

```bash
curl -sS -N http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "served-model",
    "messages": [{"role": "user", "content": "What is 1 + 1? Give only the final number."}],
    "stream": true,
    "stream_options": {"include_usage": true},
    "max_tokens": 1024
  }'
```

确认有独立思考事件、正文及正常stop，再检查末尾带usage的事件。
usage可能与文本分属不同SSE事件。以下响应摘录表示原生上报100个思考Token：

```json
{
  "usage": {
    "completion_tokens": 120,
    "completion_tokens_details": {"reasoning_tokens": 100}
  }
}
```

检查不是每次运行的强制步骤，工具不会自动探测。因预算不足截断时应调整后再确认；
单次缺失只能说明该次响应没有可用计数，服务版本、模型或思考配置改变后需重新核实。

### 5.2 Token ID补充计数

服务原生计数缺失/null，且服务支持返回原始ID时，在现有任务中设置：

```json
{
  "thinking_metrics": true,
  "request_params": {"return_token_ids": true},
  "tokenizer": "/models/matching-model"
}
```

- `return_token_ids`必须在请求发送前设置，不能结束后补取。
  它是服务扩展；不支持的服务应省略，HTTP拒绝仍按原失败规则处理。
- tokenizer必须匹配模型并可由测试进程访问；优先复用已加载对象，
  否则按显式`tokenizer`准备，补充计数不会仅凭服务`model`别名下载tokenizer。
  输入生成、上下文检查等功能对tokenizer的要求独立保留。
- 读取`prompt_token_ids`及各`choices[0].token_ids`；生成ID总数必须与
  `usage.completion_tokens`一致，提供了提示词ID时也要与`usage.prompt_tokens`一致。
- 首版仅支持各自为单Token的`<think>`、`</think>`边界和一个完整非嵌套思考块；
  支持生成起始标记，或提示词末尾已有未闭合起始标记及可选空白。边界标记不计入R。
- ID、计数或边界无法可靠确认时保留null及原因，不按字符、事件数或重新分词估算，
  不因此新增请求失败。异常ID的Unicode等诊断采用ASCII JSON转义文本保存。
- 原生计数与ID同时存在仍采用合法原生计数。补充计数在原请求结束时间固定后执行，
  不改写原始usage或整体输入/输出Token数；额外传输、保存和计数仍会增加客户端工作，
  实际Case窗口包含这些工作，不能声称开关或ID传输没有性能开销。

## 6. 在哪里查看

终端结果摘要及HTML的“Case分析”按保存字段展示指标，关闭时隐藏整组分阶段内容。
JSON中的位置如下：

```text
tasks[].cases[].metrics.streaming_performance.values
  request_view                  整体延迟分布
  service_view                  Case吞吐、并发及请求数

tasks[].cases[].metrics.streaming_phase_timing
  values.request_view           思考及正文分阶段指标
  values.phase_counts           有效、截断、失败、排除及缺失通道数量
  metadata.reasoning_token_sources
  metadata.reasoning_token_missing_reasons
```

计数来源为`service_usage`或`token_ids`，不可用时原始请求另有
`reasoning_tokens_reason`，例如`token_ids_not_requested`、
`tokenizer_not_configured`、`missing_output_token_ids`、`incomplete_thinking_block`。
来源汇总只针对符合正常stop条件的请求，不替代各指标自己的有效N。
原始请求文件中的`response.phase_timing`只在开启时保存，
其TPOT/ITL字段虽以`_ms`结尾，数值单位仍为ms/token。

## 7. 源码与回归入口

| 内容 | 源码/回归 |
|---|---|
| 整体请求时间与公式 | [online_serving.py](src/luban_meter/benchmarking/generation_performance/online_serving.py) |
| SSE通道与事件采集 | [streaming.py](src/luban_meter/benchmarking/generation_performance/common/streaming.py) |
| 分阶段计算、资格、权重 | [phase_timing.py](src/luban_meter/benchmarking/generation_performance/common/phase_timing.py)、[test_phase_timing.py](tests/test_phase_timing.py) |
| 原生计数校验 | [thinking.py](src/luban_meter/runtime/thinking.py) |
| Token ID补充与持久化 | [reasoning_tokens.py](src/luban_meter/benchmarking/generation_performance/common/reasoning_tokens.py)、[test_reasoning_token_ids.py](tests/test_reasoning_token_ids.py) |
| 整体统计与分位数 | [statistics.py](src/luban_meter/benchmarking/generation_performance/common/statistics.py) |

跨工具比较前，应同时对齐有效事件、首末时间、Token来源、成功/截断规则、
样本权重和负载轨迹；名称相同不代表口径相同。
