# LuBan-Meter

统一执行生成式推理性能与模型服务质量评测，保存原始事实、计算指标，
提供终端结果摘要与离线HTML报告。

- 生成性能：在线OpenAI兼容流式服务、多轮会话在线服务、离线vLLM Engine。
- 模型服务质量：C-Eval、CMMLU、GSM8K、HumanEval、LCSTS、SQuAD、WikiText。
- 同一份JSON编排一个或多个任务，每个任务可以展开多个Case。

- 场景、指标和负载Profile通过类与装饰器注册，内置实现由`__init__.py`显式导入。
- 性能和质量共用本地数据加载与实际请求调度；Prometheus及硬件监控按需开启。

## 安装

```bash
python3.12 -m pip install -e '.[dev,generation]'
# LCSTS的中文ROUGE计算需要：
python3.12 -m pip install -e '.[quality]'
```

Python版本要求≥3.12。模型、在线服务、GPU驱动和vLLM由用户提前准备。
HTTPX作为基础依赖随工具安装，在线性能与质量评测使用同步HTTP/1.1连接池。
HumanEval使用Docker沙箱执行候选代码；先按
[沙箱说明](src/luban_meter/benchmarking/model_service_quality/common/Containerfile)
构建镜像，禁止回退到宿主机直接执行。数据下载脚本保留在
`src/luban_meter/benchmarking/model_service_quality/scripts/`，默认输出到
仓库根目录的[dataset/](dataset/)。性能和质量评测共用该目录，正式评测只读取本地数据。
数据集不内嵌到Python安装包；仅安装wheel时需自行准备数据并设置`--dataset-path`。

## 单任务

```bash
luban-meter benchmarks list
luban-meter metrics list

luban-meter run \
  --scenario online_serving \
  --backend openai_http \
  --base-url http://127.0.0.1:8000 \
  --endpoint /v1/completions \
  --model served-model --tokenizer /models/your-model \
  --data random_token \
  --input-tokens 1024 --output-tokens 128 \
  --profile rate --request-rate 10 --max-concurrency 32 \
  --arrival-process poisson \
  --num-requests 1000 --num-warmups 32 \
  --goodput ttft:1000 tpot:50 e2el:10000 \
  --result-dir results
```

`random_token`通过Faker生成随机文本，经本地Tokenizer编码、
按目标长度截取并解码。不从词表直接随机采样，也不再循环重复种子文本。
在线发送文本，支持Completions和Chat接口，无需服务提供`/tokenize`；
`--tokenizer`可指定本地目录或模型标识，未指定时使用`--model`。
如果服务模型名只是别名，应显式指定Tokenizer。Tokenizer不会自动启用远程代码。
安装`generation`可选依赖即可使用Faker和Transformers，离线还需自行安装vLLM。


```bash
# 固定输入长度与输出预算
--input-tokens 1024 --output-tokens 128

# 输入正态采样：均值1024，标准差128；输出预算固定
--input-tokens 1024 --input-tokens-std 128 --output-tokens 128

# 输入在闭区间内均匀采样；输出预算固定
--input-tokens-min 512 --input-tokens-max 1536 --output-tokens 128
```

输出预算也支持`--output-tokens-std/min/max`。标准差缺省或为0时，范围相等即固定，
否则整数均匀采样；标准差大于0时进行正态采样，再取整并裁剪。
未提供的边界默认由`目标值±5×标准差`计算，下限至少为1；无标准差时缺省边界为目标值。
均匀采样建议同时提供min/max，正态采样的均值必须位于上下限内。

`--seed`为基础种子；`--data-seed`可单独覆盖数据种子。文本、输入长度、输出预算、
预热和请求调度使用独立随机序列，修改速率、并发或预热次数不改变同序号正式输入。
合成输入在正式计时前准备并保存在内存中；内存用量随请求数和输入长度增长。
预热使用独立文本。可复现比较需保持Tokenizer文件、Faker等依赖版本一致，随机文本不保证零缓存命中。

默认`--length-policy observe`记录目标长度、实际长度和偏差；`strict`检查服务端实际输入
是否等于每条请求的目标，不符合时单独标记`nonconforming`，不改写请求成功状态。
两种模式均按实际Token数统计性能，并报告长度分布摘要、符合/不符合/未知数量与符合率。
服务缺失usage时长度条件为未知，Token总量和吞吐等无法完整计算的值为null，保留请求延迟。

每条请求先采样输出预算。默认`ignore_eos=true`设置`min_tokens=max_tokens=预算`；
`--no-ignore-eos`仅设置输出上限，允许自然结束。预算偏差独立记录，不判为HTTP请求失败。
不支持这些生成参数的服务将按真实HTTP错误记录，不静默降级。

在线可通过`--max-model-len`指定上下文上限；离线读取Engine上限。
已知上限时提前校验采样范围，Chat输入还计入本地聊天模板；超限不自动缩短。
在线未指定时不猜测服务上限，也不把Tokenizer的占位极大值视为可信上限。

逐请求负载提供以下明确入口，CLI与任务JSON使用相同参数：

| Profile | CLI参数示例 | 行为 |
| --- | --- | --- |
| `constant` | `--request-rate 10 --max-concurrency 32` | 从时间0开始，每0.1秒计划一次请求 |
| `poisson` | `--request-rate 10 --max-concurrency 32` | 首次计划在0，后续指数间隔，平均每秒10次 |
| `rate` | `--request-rate 10 --arrival-process gamma` | 兼容入口，按到达过程安排请求，详见下文 |
| `concurrent` | `--max-concurrency 8` | 最多8个请求并发，样本完成后补充 |
| `throughput` | `--max-concurrency 32` | 在并发容量内持续尽快发送，无到达速率等待 |
| `replay` | `--replay-speed 2 --max-concurrency 32` | 按轨迹时间的1/2安排请求，详见轨迹回放 |
| `synchronous` | 无需额外参数 | 请求逐个执行，并发容量固定为1 |

例如，将在线命令的负载参数改为`--profile throughput --max-concurrency 32`，
即可执行有界尽力发送；改为`--profile synchronous`即可串行执行。
`constant`和`poisson`默认速率为1、并发上限为32；速率必须有限且为正。
`concurrent`默认并发上限为8，`throughput`为32；并发上限均须为正整数。
`synchronous`只允许省略`max_concurrency`或设为1，其他值提前报错。
输入准备、评分和客户端调度会影响实际请求间隔及并发，不能保证始终满并发。
相同容量下，`concurrent`与`throughput`共享即时准入行为；后者不会自动搜索最佳并发。

现有`--profile rate --arrival-process constant/poisson/gamma`继续可用。
Gamma通过`--burstiness`控制间隔分布形状：小于1更易扎堆，等于1与指数分布相同，
大于1更均匀；平均间隔仍为`1/request_rate`。例如：

```bash
luban-meter run --scenario online_serving --model served-model \
  --profile rate --request-rate 10 --arrival-process gamma --burstiness 0.5
```

旧的`--profile rate --request-rate inf`继续可用，仍受`max_concurrency`限制，
所有请求的计划时间保持为Case起点。新增`throughput`的计划时间取实际准入时间，
因此两者的`scheduling_delay_s`含义不同，不应直接对比；请求延迟和吞吐计算方式不变。
`batch`继续用于同步离线Engine：

```bash
luban-meter run --scenario offline_vllm_engine \
  --model /models/your-model --profile batch --batch-size 8 \
  --input-tokens 1024 --output-tokens 128 --num-requests 100
```

最后一批不足8条时保存实际批大小。Engine内部时间和客户端时间分别计算，不混为同一种TTFT。

本地真实输入可选择`json`、`jsonl`、`csv`或`sharegpt`，例如：

```bash
luban-meter run --scenario online_serving --model served-model \
  --data sharegpt --dataset-path /data/sharegpt.json \
  --output-tokens 128 --num-requests 100 --profile concurrent
```

ShareGPT发送首个有效human消息，保留原文（包括首尾空白），不裁剪或补齐。
可用`--input-tokens-min/max`验收服务端实际输入长度范围；越界只标记负载条件不符合，
不筛除样本或改写请求状态。真实文本不使用`input_tokens`作为逐条目标，也不支持输入标准差采样。
输出预算仍可固定或按分布采样。

默认只读取选定数据一遍；性能任务可用`--repeat-data`循环输入。
JSON/JSONL/CSV记录的输入字段默认是`prompt`，可通过`--prompt-field`调整。
CSV（包括BurstGPT轨迹）支持带或不带文件头BOM的UTF-8编码。
主数据集和few-shot通过同一公共`DatasetLoader`读取，保留原始记录；
Prompt构造、答案解析与指标计算独立执行。Python调用方式见[架构指南](ARCHITECTURE.md)。

质量任务示例：

```bash
luban-meter run --scenario ceval --model served-model \
  --base-url http://127.0.0.1:8000 \
  --dataset-path /data/ceval/val.jsonl \
  --eval-mode gen --endpoint /v1/chat/completions \
  --few-shot 0 --max-samples 100 --max-concurrency 8
```

选择题支持`gen`和`ppl`；`ppl`要求completions接口支持echo/logprobs。
CMMLU与C-Eval的默认`gen`模板采用`cmmlu-v3`/`ceval-v3`，要求只输出一行
`答案：D`，不输出思考、分析或解释；few-shot示例使用相同格式。
v2模板保留“最终答案单独写在最后一行”的原有指令，便于复现历史Prompt。
`ppl`保留v1提示词及原有logprob评分方式。原始事实中的`prompt_version`
记录实际模板版本，新旧模板的效果分数应分别比较。

选择题生成解析先排除完整的`<think>...</think>`块，再匹配独立的A/B/C/D
或完整答案行，例如`答案：D`、`答案是D`、`最终答案：D`、`Answer: D`。
完整答案行允许使用成对的Markdown粗体标记，例如`**答案：D**`。
支持独立的`\boxed{A}`、`答案：\boxed{A}`及其`$...$`、`$$...$$`、
`\(...\)`、`\[...\]`数学包装；显示数学包装支持换行。boxed内容仅限单个
A/B/C/D，与普通答案一起检查冲突，不从解释或一般数学表达式中猜选项。
不会从解释或选项列表中搜索任意字母。思考标签未闭合或结构异常、没有明确答案、
出现冲突的答案行时，记录`parse_failed`；原始输出不被改写。
解析失败按错误进入准确率分母，服务失败及未完成样本沿用既有规则。
选择题默认`stop=[]`，避免思考标签后的换行截断；显式配置的停止符仍生效。
这不会关闭模型思考或增加输出预算，思考模式需预留足够的输出和上下文长度。

在线性能与质量任务通过`request_params`向模型推理请求追加自定义JSON字段。
默认`{}`，不自动发送思考开关，也不根据模型选择协议。支持嵌套对象、数组、
字符串、布尔、有限数字及null；CLI使用`--request-params 'JSON对象'`。

例如，服务支持模板参数时，在任务JSON中配置：

```json
"request_params": {
  "chat_template_kwargs": {"enable_thinking": false}
}
```

服务支持DeepSeek思考控制字段时，可以改为：

```json
"request_params": {
  "thinking": {"type": "disabled"}
}
```

自定义字段原样追加到chat、completions及logprobs推理请求的顶层；
不会发送到健康检查、metrics或tokenize接口。参数是否生效由被测服务决定，
客户端不转换思考协议或对未知服务字段进行自动重试。
`model`、`messages`、`prompt`、`stream`、`stream_options`、Token预算、`stop`、
`temperature`、`seed`、`n`、`best_of`及`echo/logprobs`等评测核心字段禁止覆盖，
静态预检会报错；运行前构造请求时也检查同名冲突。使用已有专用参数修改这些条件。
任务中的`request_params`整体替换CLI基础对象，不做隐式深层合并；显式`{}`可清空它。

原有`thinking_protocol`和`enable_thinking`任务参数及CLI选项已移除，旧配置会报错。
迁移时删除旧字段，将需要的请求字段写入`request_params`。
完整示例见[examples/tasks-thinking.json](examples/tasks-thinking.json)。

```bash
luban-meter run --scenario online_serving --model served-model \
  --base-url http://127.0.0.1:8000 --endpoint /v1/chat/completions \
  --data jsonl --dataset-path dataset/prompts.jsonl \
  --request-params '{"chat_template_kwargs":{"enable_thinking":false}}'
```

在线性能保持`stream=true`以采集TTFT/ITL等指标，质量评测保持`stream=false`。
HTTP生成请求默认显式发送`stop: []`，不注入换行停止符；显式停止列表仍生效。
PPL/logprobs评分必需字段保持原有行为，也可通过`request_params`追加服务扩展。
空停止列表不会禁用EOS或绕过输出预算，也不能保证清除服务端自行配置的停止规则。

chat性能采集同时识别`delta.reasoning`、`delta.reasoning_content`和`delta.content`。
TTFT、最后输出时间及流事件间隔覆盖思考和正文；同一个事件包含两个通道时只计时一次。
原始请求事实分别保存`reasoning_text`和`generated_text`，并记录：

- `output_observation=reasoning_and_content`：标记本次输出观察口径；
- `output_event_offsets_ms`、`reasoning_event_offsets_ms`、`content_event_offsets_ms`：
  相对请求开始的事件时间，单位毫秒；
- `first_content_latency_ms`：首个非空正文延迟，无正文时为`null`；
- `has_content`、`finish_reason`及服务返回的`usage`；
- `reasoning_tokens`：优先保存服务`usage.completion_tokens_details.reasoning_tokens`；
  开启思考指标后，服务计数缺失/null时可按下述Token ID规则补充，否则为`null`。

只有思考的性能响应仍视为成功，并标记`has_content=false`；完全没有有效输出仍失败。
输出Token总数沿用服务`usage.completion_tokens`，不再额外累加思考Token，
也不按字符或事件估算缺失计数；需确认被测服务的总数包含思考，才能按全部输出比较吞吐。

### thinking与正式回答的分阶段计时

单轮及多轮在线Chat性能的整组分阶段指标默认关闭，包括正式回答输出时长和
首个正式回答延迟。显式设置`thinking_metrics=true`后，默认指标列表才包含
独立指标`streaming_phase_timing`；CLI对应`--thinking-metrics`和
`--no-thinking-metrics`。关闭时不生成单请求的`response.phase_timing`，
不计算或保存整组分阶段汇总，终端和HTML也不展示这些指标及其样本计数。
原始思考文本、事件时间、服务用量及原有校验保持不变。
该指标复用原始流事件时间，不增加模型调用；普通Completions接口返回不可用。
显式设置`metrics`仍覆盖默认列表；显式选择`streaming_phase_timing`时也必须
开启`thinking_metrics`，否则在配置检查时报告错误。
该开关仅控制客户端统计，不控制模型是否思考，不自动添加服务请求参数，
也不覆盖显式`metrics`选择。例如，任务JSON中可设置：

```json
{
  "name": "thinking_test",
  "scenario": "online_serving",
  "model": "served-model",
  "thinking_metrics": true
}
```

下表中全部分阶段指标及下文对应统计规则仅在开启该开关时生效。

| 指标（`values.request_view`） | 计算方法 | 单位 |
|---|---|---|
| `reasoning_duration` | 最后一个thinking事件时间减第一个thinking事件时间 | ms |
| `content_duration` | 最后一个正文事件时间减第一个正文事件时间 | ms |
| `first_content_latency` | 第一个正文事件时间减请求开始时间 | ms |
| `reasoning_ttft` | 第一个thinking事件时间减请求开始时间 | ms |
| `reasoning_tpot` | `(最后一个thinking事件时间−请求开始时间)/R`，R>0 | ms/token |
| `reasoning_itl` | `(最后一个thinking事件时间−第一个thinking事件时间)/(R−1)`，R>1 | ms/token |
| `reasoning_stream_event_itl` | 同一请求内相邻thinking流事件的时间差 | ms |

`R`为请求记录中的`reasoning_tokens`，优先使用服务上报值，缺失/null时才尝试
Token ID补充计数；不使用事件数、字符数、正文Token数或对文本重新分词估算。
thinking TTFT和TPOT包含首个thinking事件前的等待；thinking ITL排除这段等待。
在请求成功且符合分阶段统计条件的前提下，补充计数后`reasoning_tokens`仍缺失、为`null`
或为0时，thinking TPOT/ITL不可用，时间类补充指标仍可计算；R=1时只有ITL不可用。
服务返回负值、布尔值、小数或字符串等非法`reasoning_tokens`，或非null且非对象的
`completion_tokens_details`时，沿用既有解析校验：整个请求失败，全部分阶段指标
被排除，也不进入整体指标的成功请求分布；不将非法值转换为缺失计数。
只有一个thinking事件时，没有流事件间隔样本；若R>1，Token ITL的观察值为0，
但不能解释为引擎内部各Token生成耗时为0。
这些thinking指标仅作补充，整体`streaming_performance`的TTFT、TPOT、ITL、
E2EL等仍按原公式和原样本范围计算，覆盖thinking与正文；不新增正文TPOT/ITL。

**如何判断是否需要Token ID补充计数**

原生返回`usage.completion_tokens_details.reasoning_tokens`的服务，只需开启
`thinking_metrics`即可使用其原生思考Token计数，无需为此配置`return_token_ids`
和`tokenizer`。输入生成、上下文检查等其他功能对tokenizer的要求仍按原规则处理。
`completion_tokens`和`total_tokens`是整体计数，不能代替思考Token数。

| 服务返回情况 | 思考指标配置 |
|---|---|
| `reasoning_tokens`为合法非负整数 | 只需`thinking_metrics=true`；合法0也采用服务值，不触发补充计数 |
| `reasoning_tokens`缺失或为`null` | 如需思考TPOT、ITL均值，还需显式设置`request_params.return_token_ids=true`并配置匹配模型的`tokenizer`；服务须支持返回原始Token ID，格式须满足下述计数条件 |
| 没有原生计数，也无法可靠使用Token ID补充 | 只开启`thinking_metrics`仍可按独立思考事件计算思考TTFT、时长等时间类指标；思考TPOT、ITL保留缺失及原因 |

尚不清楚服务能力时，建议先用curl发一次小样本流式请求；这不是每次压测前的必需
步骤，LuBan-Meter也不会自动探测。地址、模型、鉴权、思考模式及其他影响响应的
参数应与压测配置一致。`thinking_metrics`只控制指标计算，不开启模型的思考模式。
以下为原生计数检查示例；如服务要求鉴权或额外的思考参数，需按实际配置补充：

```bash
curl -sS -N http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "served-model",
    "messages": [
      {"role": "user", "content": "请计算1+1，简短回答。"}
    ],
    "stream": true,
    "stream_options": {"include_usage": true},
    "max_tokens": 1024
  }'
```

检查末尾携带`usage`的SSE事件，计数可能与文本分属不同事件。例如以下响应摘录
表示服务原生上报了100个思考Token，无需请求Token ID：

```json
{
  "usage": {
    "completion_tokens": 120,
    "completion_tokens_details": {"reasoning_tokens": 100}
  }
}
```

检查时应确认模型实际输出了独立的`delta.reasoning`或`delta.reasoning_content`
事件，且请求以`finish_reason=stop`正常结束；因长度限制截断时，应调整输出上限
后再检查。单次响应缺少计数，只能说明该次请求没有可用原生计数，不能据此断言
服务在所有配置下都不支持。服务或模型版本、思考配置变化后应重新确认。

当上述字段缺失或为`null`，且服务支持返回原始Token ID时，可在任务中增加以下
配置。服务同时返回合法原生计数时，仍优先采用原生值：

```json
{
  "thinking_metrics": true,
  "request_params": {"return_token_ids": true},
  "tokenizer": "/models/matching-model"
}
```

`tokenizer`必须与被测模型匹配，并且可由运行LuBan-Meter的进程访问。
工具优先复用输入准备时已加载的tokenizer；否则仅按显式`tokenizer`加载，
不会把服务端模型别名当作补充计数的下载地址。资源每个Case准备一次，加载失败
仅使补充计数不可用，原有输入生成或上下文检查必需的tokenizer错误处理保持不变。
`return_token_ids`必须在原请求发送前设置，不自动添加、不探测或重发模型请求；
不支持此字段的服务应省略它，服务拒绝请求时仍按原有HTTP失败规则处理。

首版读取响应首块的`prompt_token_ids`和各`choices[0].token_ids`，拼接生成ID，
要求ID有效且总数等于服务`usage.completion_tokens`；提供了提示词ID时，其数量
也须等于`usage.prompt_tokens`。仅支持tokenizer中各自为一个Token的
`<think>`、`</think>`以及一个完整、非嵌套的思考块，边界标记本身不计入R。
支持生成内容以`<think>`开始（之前只允许空白），也支持提示词以未闭合的
`<think>`加可选空白结束。提示词中的ID不计入生成Token数。
未知标记、多块/嵌套、未闭合、ID缺失或数量不一致等情况保留缺失原因，不猜测数量，
不因此新增请求失败。合法服务计数包括0，均优先使用；非法服务计数仍令请求失败。

补充计数在原有请求结束时间固定后执行。只有ID、没有文本的SSE事件不增加输出
事件或时间戳；同事件内多个Token不拆成虚构的Token时间线，ITL仅计算平均值。
该功能不从正文`<think>`文本拆分时间通道，时间指标仍依赖独立reasoning通道。
原始`usage`和整体输入/输出Token计数不被改写；额外ID传输、保存和计数会增加
客户端开销，Case窗口仍包含实际发生的工作，实际性能数值可能随之变化。

开启开关后，请求记录另保存以下诊断字段：

- `reasoning_tokens_source`：`service_usage`、`token_ids`或`null`；
- `reasoning_tokens_reason`：不可用原因，例如`token_ids_not_requested`、
  `tokenizer_not_configured`、`missing_output_token_ids`、
  `output_token_ids_count_mismatch`、`thinking_start_unavailable`、
  `incomplete_thinking_block`；成功取得计数时为`null`；
- `token_id_observation`：实际收到的提示词ID、原始生成ID块和冲突诊断；
- `reasoning_token_counting`：补充计数使用的tokenizer、边界ID和初始思考状态。

异常Unicode、NaN/Infinity等非法ID诊断采用ASCII JSON转义文本保存，不参与计数。
正常整数ID数组保持原值；原始请求JSON及包含请求明细的完整HTML仍可正常读取。

Case的`streaming_phase_timing.metadata`按符合正常stop条件的请求汇总计数来源和
缺失原因；该数量不替代TPOT、ITL各自的有效样本数。统一日志记录资源准备、来源、
缺失原因及异常堆栈，不输出提示词、回答或Token ID列表。

这些是客户端观察跨度，包含输出期间的客户端可见等待，不是引擎内部纯计算耗时。
两个输出跨度不包含各自首事件之前的等待；首个正式回答延迟则包含请求开始到正文
出现的全部等待。两个跨度不要求相加等于E2EL，通道交错时甚至可能重叠。
同一流事件含两个通道时共用时间戳；只有一个事件的阶段跨度为0，不表示计算耗时为0。
缺失阶段为`null`，不计入该阶段分母，不把缺失当成0，也不据此断言模型没有思考。
仅依据独立`reasoning`/`reasoning_content`与`content`字段识别通道，正文中的
`<think>`标签不拆分；此时正文通道跨度包含标签内文本，不应解读为纯正式回答耗时。

开启`thinking_metrics`时，每个Chat请求的`response.phase_timing`保存时长、首正文延迟，
thinking TTFT/TPOT/ITL对应的`_ms`字段（TPOT/ITL单位实际为ms/token），以及
`reasoning_stream_event_itl_samples_ms`间隔列表、
`reasoning_event_count`、`content_event_count`、`interleaved`和统计资格`status`。
`interleaved`表示首个正文事件早于最后一个thinking事件；两者同一时间不算交错。
关闭时不生成该对象。
原始`finish_reason`和请求成功失败状态不变，统计资格如下：

| `phase_timing.status` | 含义 | 进入主要分布 |
|---|---|---|
| `eligible` | 成功、`finish_reason=stop`且有通道时间记录 | 存在的阶段参与 |
| `truncated` | `finish_reason=length` | 否，保留观察时长 |
| `missing_finish` / `other_finish` | 结束原因缺失或其他结束原因 | 否，保留观察时长 |
| `failed` | 请求失败或超时 | 否，阶段值为`null` |

排除作用于整个请求，即使thinking已经结束而正文才截断，也同时排除两个阶段。
成功返回但没有结束原因的EOF不作为正常结束。失败流不额外保存部分输出。
除thinking TPOT/ITL只计算平均值外，每项分布独立保存`count`、均值、中位数、
P50/P90/P95/P99/P999、最小值、最大值和标准差；无有效样本时`count=0`，统计值为`null`。
thinking TTFT和已有时长按请求等权，分位数线性插值。
thinking TPOT按R加权，ITL按R−1加权，仅保存平均值，不计算或展示分位数、
中位数、最小值、最大值或标准差；
其`count`为有效请求数，`weight_sum`分别为thinking Token数及Token间隔数。
Case平均TPOT为有效请求的末思考事件延迟之和除以这些请求的R之和；
平均ITL为有效请求的思考跨度之和除以这些请求的R−1之和。
两项分别筛选有效样本，不对单请求均值再作简单平均，也不代表逐Token实测间隔。
仅计算平均值仍需要思考Token数量；计数缺失时不按字符或事件数估算。
thinking流事件间隔合并符合条件请求内部的间隔样本，每个间隔等权，分位数线性
插值，`count`为事件间隔数；不跨请求相减，也不把它解释为Token间隔。
`values.phase_counts`保存总请求、符合条件、排除总数及各排除原因的请求数，
另保存交错请求数（全部可观察请求）以及单事件和缺失通道数（仅符合统计条件请求）。
多轮按实际请求汇总，不按会话数计数，也不包含轮间人为等待。

HTML的Case摘要在各补充指标均值旁分别显示有效样本数N，并同时显示每个Case的截断
和排除请求数；这些数量独立于整体完成/失败数，真实0保留为0。
thinking TPOT/ITL还显示权重总数；流事件间隔的N标注为有效间隔数。
Case分析按保存内容提供统计值和样本、权重数量；终端简要显示均值、P95和有效样本数，
其中thinking TPOT/ITL只显示均值、有效请求数和权重总数，事件指标注明计数单位。
HTML以“思考首Token延迟（TTFT）”“思考每Token耗时（TPOT）”和
“思考Token间延迟（ITL）”显示相应指标；中文显示名不改变原始指标键。
详细统计用`report --format text --verbose`查看。分阶段统计和展示仅支持当前字段结构，
不提供旧结果适配；需要这些指标时，应使用当前版本重新运行测试。
统一实例日志按任务、Case和metrics阶段记录有效样本及排除数量，不逐事件记录日志。

### 质量评测的最终输出处理

质量任务保留独立的`reasoning_text`，不从该字段提取答案。生成式能力评测
（CMMLU/C-Eval、GSM8K、SQuAD、LCSTS、HumanEval）在解析/评分前统一过滤
回答中内嵌的`<think>...</think>`，不区分标签大小写：

- `raw_output`保留服务返回的正文原文，包括其中的思考标签；
- `final_output`保存去除思考后的文本，保留答案所需空白；只用该文本进入内建或
  自定义答案解析器；
- 只有思考、无最终答案、标签未闭合/结构异常时记`parse_failed`，安全提取失败时
  `final_output=null`；原文、Token计数和请求延迟仍保留；
- HumanEval只剥离代码开始前的完整思考块前缀，支持连续多块；代码字符串、注释、
  缩进及既有代码围栏处理保持原样，解析失败不进入沙箱执行；
- LCSTS解析失败按零分计入ROUGE分母，不把空预测改写为“空”再评分；
  服务失败仍沿用原有处理规则。SQuAD复算校验原文与`final_output`一致，
  缺少该字段的旧结果保持原有回放行为。

质量请求固定`stream=false`，过滤发生在完整响应返回后；PPL/logprobs不做上述
生成文本过滤。新旧版本含思考响应的性能TTFT口径不同，应分别比较。

WikiText使用滑动窗口与Token logprobs。HumanEval只支持每题生成一次和pass@1。
LCSTS保留中文分词ROUGE，GSM8K保留数值答案抽取，SQuAD保留答案规范化与EM/F1。
这些处理位于场景的共享模板或领域函数中，数据引擎不会把参考答案和测试代码丢弃。

## 一份配置编排任务

[examples/tasks.json](examples/tasks.json)展示在线性能→C-Eval→HumanEval。
替换其中的模型、服务与数据路径后运行：

```bash
luban-meter plan --config examples/tasks.json
luban-meter run --config examples/tasks.json --result-dir results
# 只执行指定任务；可以重复传入--task：
luban-meter run --config examples/tasks.json --task online --task ceval
```

配置格式为`{"tasks": [{...}, {...}]}`。每个对象独立覆盖同一份显式CLI基础参数，
不继承前一个对象。优先级为：场景默认值＜CLI＜任务对象＜Case覆盖。
例如`--model served-model`可作为所有任务的基础参数；任务内`model`可以覆盖它。
基础CLI只放所有任务都支持的参数，不适用的字段会报错。

CLI使用`--base-url`，JSON使用`base_url`；每个参数对应一个选项。
列表整体替换，`stop`等普通列表不会自动展开。Case需显式声明：

```json
{
  "tasks": [{
    "name": "online",
    "scenario": "online_serving",
    "model": "served-model",
    "cases": {
      "matrix": {
        "input_tokens": [128, 1024],
        "request_rate": [1, 10]
      }
    }
  }]
}
```

这会生成4个Case。也可使用`"cases":{"items":[{"input_tokens":128},{"input_tokens":1024}]}`。
只有声明允许按Case变化的参数才能进入矩阵；配置路径相对于JSON所在目录，CLI路径相对于工作目录。
所有选中任务先完成静态预检，再启动执行；预检不代表服务、GPU或Docker已可用。

## HTTP连接复用

在线生成和模型服务质量评测保留线程池及原有调度，每个Case创建一个共享HTTPX客户端。
预热与正式请求共用连接池，Case结束后关闭；不同Case不共享连接。Prometheus采样独立运行。

| CLI参数 | 任务JSON字段 | 默认值与约束 |
|---|---|---|
| `--http-max-connections` | `http_max_connections` | 未指定或JSON为`null`时自动采用实际请求容量；显式值必须为正整数且不低于该容量 |
| `--http-keepalive-expiry` | `http_keepalive_expiry` | 空闲连接保留秒数，默认5，必须为有限正数 |

两项参数均可按Case配置。空闲连接上限等于连接总上限；实际请求容量由Profile与策略共同决定，
不一定等于`max_concurrency`。连接池大小不会提高调度并发，128也不是程序硬上限。
服务端关闭连接或连接空闲过期后会重新建连；顺序预热不保证所有并发连接都已建立。

固定使用HTTP/1.1，不新增自动重试，不跨请求保留Cookie，使用`Accept-Encoding: identity`。
证书信任沿用系统SSL上下文，代理继续按环境发现。常规重定向继续处理，POST的307/308不自动重发。
客户端版本及有效连接配置保存在Case原始事实的`metadata.http_client`中。

请求构造完成后开始计时，TTFT包含实际发生的建连和连接池等待。有效流式采集结束时固定E2EL，
随后消费剩余HTTP响应以归还连接；该收尾不改变输出事件或Token计数。
收尾失败会关闭连接并记录请求级`connection_cleanup_error`，不改变已有效生成的成功状态。
收尾仍占用请求执行槽并计入Case统计窗口，不从吞吐量分母中扣除。
原始请求结构和指标公式保持兼容；底层网络错误的`error.type`可能改为HTTPX异常类型。
连接复用会改变真实客户端开销，新旧结果数值不要求相同，比较时应注明客户端与连接条件。

Completions流式响应在原始请求的`usage`中保存`[DONE]`之前最后一个非null
usage对象，包括`prompt_tokens_details.cached_tokens`和服务返回的其他字段。
usage-only尾事件不计为输出事件；重复usage整体替换，不累加，也不从较早事件补字段。
未收到`[DONE]`便结束、读取中断或收到服务端错误事件时，请求失败，不将部分usage
作为成功请求事实。此完整性检查仅针对Completions；Chat处理保持不变。
收到`[DONE]`但没有usage时，仍保留延迟，Token计数为null。

生成性能公共解析器`common.cache_usage.parse_cache_usage`返回`CacheUsage`：
`state`为`valid`、`missing`或`invalid`，另含`prompt_tokens`、
`cached_tokens`及诊断`reason`。只读取嵌套的
`usage.prompt_tokens_details.cached_tokens`；缺失/null不等于零。
两个计数必须为非负整数（不接受bool或浮点数），且缓存数不得大于输入数；
非法字段优先于缺失字段判定。不完整或非法计数对的`cached_tokens`为null，
不估算、不截断；原始usage保留证据。0/0是合法计数对，但不定义命中率。
解析器不计算命中率、不判断请求成功状态，也未接入多轮统计；非法缓存明细本身
不改变已有HTTP请求成功状态。输入/输出基础Token计数沿用既有严格校验。

## Prefix Cache 受控 Token ID 输入

`prefix_cache` 按固定长度准备 Completions 输入。规划示例可离线执行：

```bash
luban-meter plan --config examples/tasks-prefix-cache.json
```

运行时用 `--tokenizer /path/to/local/tokenizer` 指定与服务一致的本地快照。
示例中输入 1024 Token，目标 0.5，块长 16，5 组共 100 条。每组重复
512 个前缀 ID，另有 512 个独立后缀 ID。序号从 0 开始按组轮转。
请求的 `prompt` 是整数列表，`add_special_tokens=false`。每条样本的
`source` 保留输入列表及 `prefix_id`。Case 的 `metadata.prefix_cache`
记录目标、理论共享比例、两者之差（比例单位）和前后缀长度。

设完整长度为 T、块长为 B、目标为 r。默认 `floor` 模式取
`L = B * floor(min(r*T, T-1)/B)`，`strict` 要求 L 精确等于 r*T。
正目标若量化为零，配置会被拒绝。100% 目标受最后一个 Token 和整块复用
限制。0% 使用空前缀，各请求的首块（短输入则整条）互异。
`prefix_count` 取 1 到正式请求数之间的整数，`kv_cache_block_size`
应填写服务实配值。`prefix_cache_target` 可作为 Case 矩阵变量。

种子、Tokenizer 词表和输入配置固定时，调度参数不影响正式输入。
生成器从基础词表的有效非特殊 ID 中抽样，为每组预留不同首块，
为每条后缀预留不同首块（短后缀则预留全部 ID）。同一位置的冲突
最多尝试 64 次，仍冲突时准备失败。内存用量随请求数与输入长度的乘积增长。
本阶段输出的是输入构造条件，真实复用情况需要服务端逐请求缓存计数。

Prefix Cache 的缓存预热在普通 `num_warmups` 完成后执行。运行器调用
`after_warmup(context)`，随后才启动正式观察窗口和正式调度器。
普通场景的该钩子默认为空。

缓存预热按“全部种入 → 全部探测 → 完整验证轮”串行执行，每条请求生成
1 个 Token 并等待 SSE 完成。种入使用共享前缀本身，探测和验证使用独立
后缀，首块与正式后缀互异。所有响应都要求有效缓存计数、准确输入长度、
成功结束状态及 1 个输出 Token。种入允许合法的实际命中数，探测和验证
要求 `cached_tokens == shared_prefix_tokens`。任一失败终止 Case。
缓存准备允许空可见文本，通过完成状态和 usage 判断请求是否完成。

每个 Case 使用安全随机 Salt，种入、探测、验证和正式请求共用同一个值，
普通预热使用第二个值。0% 使用第三个 Salt 执行一次独立能力检查，要求
显式零缓存计数，并记录 `not_applicable`。Salt 仅保存在内存与请求体中，
响应回显会在写入请求事实之前脱敏。服务端自身的请求体日志仍需负责人管理。
Salt 隔离缓存复用关系，物理 KV 容量和其他流量干扰仍需控制。

缓存预热标记为 `cache_warmup`，正式请求配额和正式吞吐窗口单独启动。
成功后的摘要位于 `metadata.cache_warmup`。失败 Case 的
`setup_diagnostic` 保留阶段、前缀编号、完成数量、预期及有效实际计数、
原因类别。进程被终止时，恢复逻辑读取逐请求前写入的白名单检查点。
缓存诊断仅保存结构化字段。缺失计数保持缺失，禁止自动重试补造命中。
取消会在发请求前、收到流式行时及请求完成后检查。等待网络数据时由
`request_timeout` 限定读取等待，任务总时限继续由任务子进程管理。

Prefix Cache 场景默认组合 `streaming_performance` 和独立的
`prefix_cache_hit_rate` Metric。缓存 Metric 只读取正式
`facts.requests[*].response.usage`，单请求前缀编号与派生值存入
`response.prefix_cache`。每个 Case 的结果沿用
`metrics.prefix_cache_hit_rate` JSON 结构，由现有任务和运行结果携带。

集合 R 为实际提交的正式请求，分母取正式调度器的 admitted_requests。
S 为 R 中成功完成的请求，M 为 S 中具有合法且一致的缓存计数对的请求。
计划请求数单独记录为 planned_requests。已完成请求数包含成功和失败，
incomplete_requests 为已提交但缺少终态的数量。
仅当采集完成、请求身份明确且已提交请求都有终态、S 非空且 M=S=R、
输入 Token 总分母大于零时，提供 actual_hit_rate。
observed_subset_hit_rate 始终按 M 的缓存 Token 总数除以输入 Token 总数。
失败请求不进入 S，缺失和非法计数不进入 M，显式零计数进入 M。
旧事实缺少请求级前缀编号时，各组诊断归入 unknown。

request_coverage=|M|/|R|，success_rate=|S|/|R|。
token_coverage 以完整正式请求输入 Token 为分母；存在失败请求或未闭合记录时，
完整 Token 分母无法保证，返回 null，不使用配置输入长度补造分母。
successful_request_coverage 与 successful_token_coverage 保留成功集合 S 的
请求和 Token 覆盖率。输入长度符合率以 S 为分母，任一输入计数未知时为 null。
total_prompt_tokens 覆盖 S，total_cached_tokens 和 measured_prompt_tokens
覆盖 M。失败请求缺少计数时，actual_hit_rate=null、目标状态 unknown；
observed_subset_hit_rate 保留已测子集的真实比例，测量为 partial 或 unavailable。
旧结果的 request_coverage/token_coverage 仍按历史 S 口径读取，范围见 metadata。


构造偏差=理论共享比例−目标，运行偏差=actual_hit_rate−理论共享比例，
总体目标偏差=actual_hit_rate−目标，均采用 ratio 单位。
actual_hit_rate 缺失时，后两项为 null。Token 加权比例保留计算精度。

`prefix_cache_runtime_tolerance` 默认 0，允许 0～1，比较实际比例与理论
比例的绝对差。target_attainment.status 在运行超差时为 failed，
在构造精确且运行容差内时为 passed，构造有量化误差但运行容差内时为
construction_limited，缺少实际比例时为 unknown。
exact_target_reached 单独表示是否精确达到原始目标，因此 100% 的块级
上限命中可得到 construction_limited，而 exact_target_reached=false。

缓存结论分别保存在 measurement 和 target_attainment。measurement 使用
valid/partial/unavailable/invalid。预热事实保存在 warmup。旧配置的
prefix_cache_environment 仍可读取，仅作历史注记。报告读取旧结果时
从 experimental_conditions 提取预热事实，不使用其中的评分。

正式命中不足、超出理论数量、零命中和各前缀组统计保存在 diagnostics。
预热通过后的下降可标为 suspected，缓存淘汰确认仍需要服务端直接证据。
指标只计算已有事实，缺失、低命中和高命中结果全部保留。
阶段 0 的真实服务缓存明细契约与最终 GPU 验收仍需独立完成。

## 多轮会话性能

`multi_turn_serving`按会话评测在线Chat服务：一个样本是一整个会话，会话内
轮次严格顺序执行，每轮请求携带此前轮次的**实际模型回答**作为历史。
不同会话并行，同一会话同时最多一个在途请求。

`online_serving`和`multi_turn_serving`保留各自的执行方式，默认都使用
`streaming_performance`计算整体指标。两种场景都需设置`thinking_metrics=true`
才默认启用`streaming_phase_timing`，计算、保存和展示思考及正文分阶段指标。
是否执行多轮由场景决定，不由Metric名称决定。
旧任务若显式配置`"metrics": ["multi_turn_performance"]`，需改为
`"metrics": ["streaming_performance"]`；旧名称已移除，不保留配置别名。
未指定`metrics`时直接使用场景默认值。已保存结果中的历史
`multi_turn_performance`指标键仍可由当前报告读取。

```bash
luban-meter run --scenario multi_turn_serving \
  --model served-model --base-url http://127.0.0.1:8000 \
  --data sharegpt --dataset-path /data/sharegpt.json \
  --multi-num-sessions 100 --multi-max-turns 8 \
  --profile concurrent --max-concurrency 4 \
  --shuffle --data-seed 42
```

统一任务JSON示例见[examples/tasks-multi-turn.json](examples/tasks-multi-turn.json)；
其中第二个任务用`cases.matrix`按`multi_max_turns`展开3个Case，并开启轮间等待。

多轮专用参数统一使用`multi_`前缀，CLI对应`--multi-`前缀：

| CLI参数 | 任务JSON字段 | 默认值与含义 |
|---|---|---|
| `--multi-num-sessions` | `multi_num_sessions` | 100；要执行的会话数量，不是模型请求次数 |
| `--multi-max-turns` | `multi_max_turns` | 8；每个会话最多执行的轮数**上限**，不是最低入选轮数 |
| `--multi-think-time` | `multi_think_time` | 0；轮间平均空闲秒数，0表示背靠背发送 |
| `--multi-think-time-std` | `multi_think_time_std` | 0；轮间空闲的标准差，需要正的`multi_think_time` |
| `--multi-system-prompt` | `multi_system_prompt` | 空；仅在记录没有system消息时前置 |
| `--multi-session-timeout` | `multi_session_timeout` | 空；单会话墙钟上限，超时提前结束 |
| `--multi-save-response-text` | `multi_save_response_text` | false；额外保存整段`messages`数组 |

数据只支持`--data sharegpt`，端点只支持`/v1/chat/completions`。轮次按参考
assistant消息切分：参考回答只作分隔符，**从不回放**，历史一律使用被测模型的
实际输出；每轮之前出现的system等消息保留在该轮内。因此数据集自带的system
指令不会被静默改写，`--multi-system-prompt`只在记录没有system时生效，
两种来源的实际使用数量记录在Case元数据的`system_prompt_source`中。

`multi_max_turns`是上限而不是筛选条件：原始轮数少于上限的会话正常参与并
自然结束，超过上限的会话只执行前`multi_max_turns`轮。每个会话的
`planned_turns=min(有效轮数, multi_max_turns)`；准备阶段记录源轮次分布、
计划轮次分布、被截断会话数和`planned_requests_total`（一次完整遍历所需的
模型请求数），使固定深度筛选与轮数上限不会被混淆。

会话选择、轮次截断与请求准入是三个独立环节：

1. **会话选择**由`DatasetLoader`完成，`--max-samples`限制的是候选记录数量，
   在轮次处理**之前**应用；`--shuffle`与`--data-seed`以整个会话为单位打乱，
   会话内部轮次保持原顺序。可用会话不足`multi_num_sessions`且未开启
   `--repeat-data`时，准备阶段直接报错并给出可用数量，不静默重复补足。
2. **轮次截断**对每条已选会话独立应用，不为凑齐统一深度拼接其他记录。
3. **请求准入**沿用现有Profile和`RequestScheduler`。`--num-requests`按
   **模型调用**计数，一个会话不算一次请求；不要求预算覆盖全部会话的所有轮次。

多轮只接受请求级Profile（`constant`、`poisson`、`concurrent`、`throughput`、
`synchronous`），`batch`和`replay`会在预检阶段被拒绝，无需任何专有并发字段。
会话样本工作线程数取有效请求容量，因此`--profile concurrent
--max-concurrency 4`表示同时最多4个会话、4个在途请求；会话结束或失败后再
领取下一个。请求准备、轮间等待和数据尾部可能使实际并发低于上限。

输出预算逐轮独立采样，沿用`--output-tokens`及`-std/-min/-max`；
`--ignore-eos`、`--length-policy`、`--stop`、`--request-params`、
`--goodput`与在线服务含义相同。指定`--max-model-len`时，每轮在**发送前**
用本地聊天模板计算已组装历史的实际长度，`输入Token数＋本轮输出预算`超限即
停止该会话并记录`abort_reason=context_overflow`与实测长度，不作为请求失败，
也不在请求完成后用连续user消息重新估算。

## 停止与失败

- 任务顺序执行，失败默认继续；`--fail-fast`跳过后续任务。
- `--num-requests`统计实际模型请求。一个C-Eval PPL样本可能产生多个请求，不能当作一个请求限速。
- 多轮会话中，请求预算或时长到限后不再准入下一轮，排空已准入请求；已结束的轮次
  保留，会话记为部分完成，会话事实保存停止原因（`num_requests`/`max_duration`/
  `user_interrupt`）与已完成、未开始轮数，未准入的轮次不计入请求失败。某轮HTTP失败
  或超时即结束该会话，该轮计入请求失败，后续轮记为未开始，其他会话继续执行，不做
  轮级重试；`abort_reason`取值`turn_failed`、`context_overflow`、`session_timeout`、
  `num_requests`、`max_duration`、`user_interrupt`、`reasoning_only`、`no_output`；
  其中`turn_failed`、`reasoning_only`、`no_output`发生在请求发出后，该轮计入
  `sent_failed_turns`，`session_view`的未开始轮次不再包含它。
  `--max-error-rate`的分母是本Case已结束的全部请求（失败计入），未准入的轮次不算
  服务失败；`--circuit-breaker`只用各会话首轮成功请求的P99 E2EL，因为后续轮prefill
  更长，同一阈值含义不同。
- 数据耗尽、请求数或`--max-duration`任一条件达到即停止新请求，等待在途请求结束。
- `--num-warmups`按样本预热，不纳入正式请求预算和指标。
- 首次Ctrl+C停止整个运行的新请求，等待在途工作；第二次取消工作进程并清理本次资源。
- `--request-timeout`约束HTTP连接、读、写及连接池等待，收尾沿用读超时；不是请求总时限。
  持续输出的长请求可以超过此时长。`--task-timeout`约束整个子进程任务。
- 结果分别记录完整、失败、不完整和未开始样本。多请求样本中途停止后，不进入要求完整答案的正确率分母。
- `--circuit-breaker`为在线Case的P99 E2EL毫秒阈值，至少10个成功样本才判断；超限跳过该任务后续Case。
  `--max-error-rate`同样在Case完成后判断。它们不改变逐请求Goodput阈值。

取消本地等待不保证远端模型立即停止计算。父进程被强制杀死或机器掉电时，无法保证自动报告完成；
首次Ctrl+C停止新请求，默认等待60秒，可用`--interrupt-grace`调整；再次Ctrl+C立即终止工作进程。
未正常完成采集的Case会删除请求、样本和指标，仅保留配置与丢弃原因，不参与统计。
完整采集已落盘的Case保留，不因后续指标或报告中断而删除；结果整理不续跑、不重发请求。

## 附加观测与HTML报告

```bash
luban-meter run --scenario online_serving --model served-model \
  --observe prometheus --observe-url http://127.0.0.1:8000/metrics \
  --observe-interval 1 \
  --device-exporter-url http://127.0.0.1:9400/metrics \
  --host-exporter-url http://127.0.0.1:9100/metrics \
  --host-sample-interval 1
```

服务端Prometheus采样只适用于HTTP任务。硬件exporter地址单独指定；不提供时不启动硬件监控。
观测故障独立记录，不让推理任务因为指标接口故障而失败。报告保留实际采样时间和缺失信息。

服务端观测聚合保存在Case的`observation.metrics`。其中`latency_decomposition.queue_time_window`（s）
是等待队列中平均每个请求的等待时间（分母为窗口内全部完成请求，含等待约等于0的），
为服务端指标（`boundary=server`）：按观测窗口对vLLM的`vllm:request_queue_time_seconds`
histogram差分，分子为各标签序列累计等待秒数增量之和，分母为窗口内上报完成请求数
的增量之和。计数器重置通过相邻快照逐对检测：累计值只增不减，任何回退即判定重置，
该序列整体剔除并记录`counter reset between snapshots`，即使末值重新超过起点也不再用
首末差分；末快照消失的序列记录`series missing from the last snapshot`。均值与增量
按原始精度落盘，展示层才格式化，微秒级正等待不会被保存成0。结果保存首末快照时间戳
（`window.started_at/ended_at`，unix秒）、每条纳入序列的标签与增量
（`series[].labels/delta_sum/delta_count`）、合计分子分母与序列数，全部为已落盘事实，
可从原始快照复算。首末快照都没有该指标时字段不出现；窗口内没有完成请求时`mean`
留空并给出`reason`，不补0；有效观测的零等待显示为真实0。该窗口是Prometheus观测
采样区间，与正式Case窗口边界略有差异；共享服务的其他流量会计入分子分母，不能承诺
仅包含本次压测请求。vLLM在请求结束后才上报该histogram，数值不代表当前仍在队列中
请求已等待的时长。

另一口径来自逐请求指标：vLLM v0.25.0起服务端加`--enable-per-request-metrics`后，
响应（流式为最后usage chunk）顶层`metrics.queue_time_ms`给出该请求在调度队列中的
等待毫秒数。LuBan-Meter将其保存为请求事实`server_queue_time_ms`（原始精度），并在
`streaming_performance.values`中计算客户要求的"只算等待过请求"口径：
`request_view.queue_time_ms`为等待时间大于0请求的分布（`count`即等待过的请求数，
保留6位小数以维持微秒级精度），`service_view.queue_time_waited_mean`为这些请求的
平均等待（`ms`）。读取与聚合两层都拒绝非数值、负值和非有限值（无穷、NaN），非法
计时视为未上报，不影响TTFT等其余指标的汇总。计时覆盖率保存为
`service_view.queue_time_reported_requests`/`queue_time_missing_requests`：全部缺失
（未开启参数、`n>1`等无法归属）给出`reason`；部分上报时保留已上报子集的结果并
以`condition`标注"partial timing coverage"，缺失请求不会被解释为零等待；已上报
请求全部为零且无缺失是完整真实观测。逐请求口径按请求过滤、不受共享流量影响，
与上文的Prometheus全请求口径互补。不能同时启用`--disable-log-stats`；高并发下该
参数可能增加服务端CPU开销。多轮会话每轮复用同一计算，轮次事实保存在各轮请求记录中。

`--host-exporter-url`接入被测Linux服务器的node_exporter，独立采集CPU、内存、
文件系统、块设备I/O和网络接口；可与GPU监控同时启用，也可单独启用，不依赖GPU识别。
两个Exporter需预先启动，配置地址不会安装或启动服务器组件。`127.0.0.1`指LuBan
所在网络环境；远程运行时填写被测服务器可达地址。主机采样不使用环境HTTP代理。

主机原始计数器、采样起止时间和派生速率保存在Case的`host_observation`，GPU继续
使用`device_observation`。CPU忙碌率按相邻`node_cpu_seconds_total`增量计算
`100 × (1 - Δidle / Δtotal)`，包含iowait，排除guest重复计数，并单列user、system、
iowait。内存已用量为总量减MemAvailable；文件系统已用量为size减free，可用量
另取avail，保留保留空间差异。磁盘和网络按原始计数器增量除以实际单调时钟间隔，
字节单位保留bytes，HTML换算GiB/MiB/s。第一点、采集失败后的首点、计数器回退
或设备身份变化不生成速率；不补零、不外推。速率汇总仅使用完整落在正式Case窗口
内的有效区间，均值按区间时长加权，并保存覆盖时长；容量采用有效采样均值。
两个数据源独立后台采样，接口故障分别记录；原始记录保留基线与收尾采样。

输出目录：

```text
runs/<run_id>/               # 可用 --result-dir 修改根目录
  plan.json                  # 有效配置，凭据字段脱敏
  result.json                # 运行汇总
  report.html                # 可离线打开的单文件报告
  log/
    luban-meter-<run_id>-<YYYY-MM-DD>.log  # 主程序、任务和Case共用
  tasks/<task_name>/
    result.json
    raw/events.jsonl         # 请求、样本和观测事件
    raw/case_1.json          # Case原始事实
    raw/case_1.result.json
    raw/manifest.json
    artifacts/
```

日志接口和格式参考[luban-sculpt](https://gitcode.com/openeuler/luban-sculpt/blob/master/luban_sculpt/log/logging.py)，
配置与格式统一放在`src/luban_meter/log/`，各模块通过`get_logger()`获取日志对象。
`run`默认启用INFO级别的终端与文件日志，记录任务/Case、准备、预热、正式采集、指标、
结果落盘、报告生成和资源清理；超时、中断、失败及捕获到的异常堆栈也会保存。
格式为`时间 | 模块 | 函数:行号 | 级别 | 上下文 | 消息`；上下文包含PID、任务、
Case和阶段标识，请求错误另带样本/请求编号。主程序和任务子进程共用同一个实例日志，不另建stdout/stderr文件。
交互终端中，DEBUG为青色、INFO为绿色、WARNING为橘色、ERROR为红色。
实例文件和重定向输出保持纯文本；设置`NO_COLOR=1`可关闭终端颜色。
日志日期采用实例开始时的UTC日期。CLI解析、最终生效的Case参数、启动的命令
默认以INFO记录；可恢复异常以WARNING记录，执行失败以ERROR记录并附堆栈，
正常迭代结束等控制流使用DEBUG。默认不逐条记录成功请求、提示词或模型回答。
密钥字段和已声明的密钥值脱敏，服务端返回的错误消息仍可能含业务数据。

所有CLI子命令支持`--log-level debug|info|warn|error`及`--log-dir`。
目录优先级为`--log-dir`、环境变量`LUBAN_LOG_DIR`、默认目录。
`run`默认写到实例目录的`log/`；`plan`、`report`等单独执行时默认写到
当前目录的`log/`，使用独立调用ID。参数解析失败也保留日志。自定义目录下
每次执行使用独立文件；UTF-8追加写入，不自动轮转或清理。

关键过程提示（计划摘要、阶段状态、结果和报告保存位置等）通过统一日志
处理器写入`stderr`与实例文件，受`--log-level`控制。HTML报告的路径及
完整报告说明也通过日志输出，不再单独写入`stdout`。
完整计划、最终生效的Case参数和准备信息JSON以INFO仅写入实例文件，
不在终端展开；终端保留简短阶段提示、进度及完成记录。这些详细记录
同样受`--log-level`过滤。
交互终端的阶段日志与动态进度分行显示；异常中的已声明敏感值及其转义形式也会脱敏，
同时保留异常类型与堆栈。
计划JSON、Benchmark/Metric列表和文本报告正文保留在`stdout`，不受日志
级别过滤，可通过管道或重定向使用；指定文本报告`--output`时正文写入
指定文件。数据准备及示例生成脚本的下载、转换、跳过文件及完成提示
同样写入日志。

```bash
tail -f runs/<run_id>/log/luban-meter-*.log

luban-meter run --config tasks.json --log-level debug
luban-meter run --config tasks.json --log-dir /work/custom-log
LUBAN_LOG_DIR=/work/custom-log luban-meter run --config tasks.json
```

HTML按模块分页展示，支持报告总结、任务与Case筛选、摘要表格排序、Case指标分析、
观测时间线以及覆盖率与错误详情。
图表每Case最多展示500个观测点，原始记录保留完整数据；报告不依赖CDN或在线服务。
可重新导出：

```bash
luban-meter report --input results/<run_id>/result.json --output report.html
```

新结果不带协议版本标记。旧配置、文件对发现机制和旧报告读取器已移除，
已有用户运行文件不会被自动删除。

## 指标口径

在线性能从客户端观察，Engine性能使用引擎提供的内部时间戳。
令首个有效输出时刻为t1、最后有效输出时刻为tn、请求开始为t0、输出Token数为N：

| 在线指标 | 定义 | 单位 |
|---|---|---|
| TTFT | t1−t0 | ms |
| TPOT | (tn−t0)/N，包含首Token等待 | ms/token |
| ITL | (tn−t1)/(N−1)，N>1时有效 | ms/token |
| E2EL | 响应收集结束−t0 | ms |
| Stream Event ITL | 相邻有效流式事件的时间间隔 | ms |
| Request/Token Throughput | 对应成功请求或Token数除以统计窗口 | req/s、token/s |
| Total Token Throughput | 成功请求的输入与输出Token总数之和除以统计窗口 | token/s |

TPOT按输出Token数加权，ITL按后续Token数加权；流式事件可能包含多个Token，
因此Stream Event ITL单独统计。`--goodput tpot:50`沿用已有SLO含义，
约束的是排除首Token等待的ITL，不是报告中的TPOT；多个SLO条件需同时满足。
迁移保留这些公式，不将CLI风格与其他工具的指标含义混同。

`streaming_performance`从正式采集的请求事实计算一次共有请求级与服务级指标，
单轮和多轮使用同一套在线公式。多轮中每一轮就是一次普通流式请求，分母是本Case
已保存的正式请求，不是会话数或计划轮次数；预热不参与。失败或部分完成的会话中，
已经结束的请求仍参与请求统计，未发送轮次不计为请求失败。失败请求进入请求分母，
不贡献Token；缺少usage时保留可用计时与请求计数，对应Token统计留空并说明原因。
缺少必要响应计时仍会产生可诊断的指标计算错误，不补造时间。
正式请求带有轮次事实时，统一Metric额外保存以下多轮视图：

| 视图 | 内容 | 口径 |
|---|---|---|
| `request_view`/`service_view` | TTFT、TPOT、ITL、E2EL、吞吐量 | 与在线服务完全相同 |
| `turn_view` | 各轮TTFT、请求级平均ITL、Stream Event ITL、E2EL、输入与输出Token分布，及每轮计划参与/成功/已发送失败/未开始会话数（`turn_participation`） | 分布按`turn_index`分组，每组带自己的样本数；`itl_ms_by_turn`为请求级平均ITL（`(最后输出时间−首输出时间)/(输出Token数−1)`，按输出Token数−1加权，与整体`request_view.itl`同口径），`stream_event_itl_ms_by_turn`为相邻流事件间隔（每间隔等权，一个事件可含多个Token），两者独立保存不可互相替代；缺输出Token计数、输出不足2Token或缺首末输出时间时请求级ITL留空，事件间隔仍可统计；参与台账来自会话事实，与`session_view`对账，未开始轮次未向服务发送、不计入失败 |
| `session_view` | 会话耗时、轮间等待合计、计划/完成/未开始/发送失败轮次、已结束会话数、会话吞吐量、中止原因、会话覆盖率 | 样本是会话；`sessions_per_second`分母是Case窗口 |
| `caching_view` | 首轮与后续轮TTFT均值及比值、每轮和每千Token的TTFT增长 | 只描述延迟随上下文的增长，不作为前缀缓存命中的证据 |
| `caching_view.prefix_cache` | 客户端可复用前缀Token与占比、服务端上报的命中Token、命中率、上报轮次数及上报轮次输入与前缀Token合计（`reporting_prompt_tokens_total`/`reporting_eligible_tokens_total`） | 客户端估算与服务端实测分开；服务端命中率与eligible比率分母仅含上报了缓存Token的轮次（真实上报0纳入分母），缺失轮次不解释为0命中，两分母可与`server_cached_tokens`复算展示比率；服务未上报时留空并给出`reason`，不补0，也不推断缓存关闭或驱逐原因 |
| `pressure_view` | 单轮上下文Token分布、Prefill Token合计与速率、单轮上下文峰值 | 只统计成功轮次，需结合设备观测判断瓶颈 |

后续轮次只包含延续到该轮的会话，各轮样本数必须与参与台账一起阅读：样本减少本身
不是性能变化，需区分会话失败与从未开始的轮次。会话耗时包含轮间等待，不能与单请求
延迟混用。

### SLO阈值、达标率与Goodput吞吐的展示

配置`--goodput`后，`service_view.goodput`保存SLO判定与有效吞吐。达标率分母为
已判定请求数（满足+违反+失败），失败请求计入分母、单独计数、不并入违反；
任一已配置维度缺必要观测的请求记为无法判定，不进分母；`tpot_ms`维度沿用GuideLLM口径，约束排除首Token
等待的ITL。Goodput请求吞吐量为满足请求数除以Case窗口，输出Token吞吐量为其
Token总数除以窗口，有达标请求缺输出Token计数时该吞吐为空。
HTML报告将SLO相关内容集中在结果摘要下方的独立章节“SLO 与 Goodput”，不进入
总览卡区与Case摘要表，避免与Peak Throughput等输出吞吐展示重复：章节内含
“Best SLO Attainment”与“Best Goodput Output Throughput”两张卡（取已保存值的
最佳Case，来源处注明SLO阈值），及按Case的明细表（并发、SLO阈值、达标率、
满足/违反/失败/无法判定计数、Goodput请求与输出Token吞吐量），达标率悬停显示
该Case阈值、无法判定请求数与达标率分母。终端简洁摘要在指标下方输出SLO阈值、
达标率及满足/违反/失败/无法判定计数、两种Goodput吞吐；不可判定字段为N/A并
注明原因。未配置`--goodput`时该数据块不存在，章节显示未配置说明，总览与摘要
不出现相应内容；`--verbose`保留完整字段表。

质量指标包括选择题Accuracy、GSM8K数值Exact Match、SQuAD规范化EM/F1、
HumanEval单次候选pass@1、LCSTS中文ROUGE-1/2/L、WikiText困惑度与Bits-per-Byte。
完整样本中的服务失败、解析失败、沙箱失败沿用各指标既有分母规则；
多请求样本因停止条件而不完整时另报覆盖率，不混入完整样本的评分。


## 开发

参见[架构与扩展指南](ARCHITECTURE.md)、[领域术语](CONTEXT.md)和[团队开发约定](AGENTS.md)。
A方案支持仓库内继承、注册与显式导入；独立安装的用户插件不在当前范围。

```bash
pytest -q
ruff check --select E,W,F --line-length 79 src tests
git diff --check
```

模拟HTTP/Engine测试验证调度、结果和公式衔接，不能替代真实模型性能、GPU兼容性或Docker沙箱验证。


## BurstGPT长度采样与轨迹回放

CSV使用`timestamp,input_token,output_token`列，时间单位为秒，长度必须为正整数。
默认遇到非法记录立即报错，包含文件、行号、字段和值；`--skip-invalid`允许跳过，
准备摘要中保留跳过数量和原因。跳过不改变其他记录的原始时间戳。

长度采样以整条记录为单位保留输入输出长度配对，使用共享合成文本生成器构造请求，
发送时间由rate、concurrent或batch决定。可用`--shuffle --data-seed 42`确定性打乱。
请求数量超过选中可用记录时报错，不自动循环。

```bash
luban-meter run --scenario online_serving --model served-model \
  --base-url http://127.0.0.1:8000 --tokenizer /models/your-model \
  --data burstgpt --dataset-path /data/trace.csv --num-requests 100 \
  --profile rate --request-rate 10 --max-concurrency 32
```

轨迹回放使用`--profile replay`，稳定排序后取连续片段，首条为时间零点，禁止shuffle。
`--trace-start`指定从第几条开始（从0计），`--num-requests`指定请求数。
BurstGPT的`--max-samples`限制候选记录数；若小于`--num-requests`，规划阶段直接报错，
不加载Tokenizer或Engine。实际可用记录不足仍在读取轨迹后报错。
`--replay-speed 2`表示间隔除以2；时间与输入输出长度始终绑定。
并发满时排队，保留原计划时间并记录实际准入、提交及调度延迟。达到`--max-duration`后
不再发送排队请求，等待在途请求结束；这属于正常停止，保留Case及未发送数量。

```bash
luban-meter run --scenario online_serving --model served-model \
  --tokenizer /models/your-model --data burstgpt \
  --dataset-path /data/trace.csv --num-requests 100 \
  --profile replay --replay-speed 2 --max-concurrency 32

luban-meter run --scenario offline_vllm_async --model /models/your-model \
  --data burstgpt --dataset-path /data/trace.csv --num-requests 100 \
  --profile replay --replay-speed 2 --max-concurrency 32
```

异步Engine需要目标环境安装vLLM，逐条提交请求，复用同一调度策略。
提交侧时延与Engine内部时间分别记录，内部时间不可用时留空，不互相替代。
同步`offline_vllm_engine`继续用于batch吞吐测试及长度采样，不接受replay。

## 输入模板与答案解析

常规质量场景使用`--input-template`与`--answer-parser`选择已注册实现，默认行为保持不变。
内置输入模板有choice、numeric_qa、summarization、reading和format_string；
解析器有choice、number、text和reading。类型不兼容的组合在请求发出前拒绝。
例如GSM8K自定义输入文字，同时沿用数值解析及数值准确率：

```bash
luban-meter run --scenario gsm8k --model served-model \
  --dataset-path dataset/gsm8k/test.jsonl --max-samples 10 \
  --input-template format_string --input-format '回答问题：{q}' \
  --field-mapping q=question --answer-parser number
```

在线真实文本可选format_string输入模板，默认prompt_record按prompt字段或ShareGPT用户消息取文本。
WikiText、HumanEval及选择题ppl保留特殊流程，不强行套用通用答案解析器。
扩展基类、字段契约、策略模式及示例见[架构指南](ARCHITECTURE.md)。


## 运行中的终端进度

`run` 默认在 stderr 显示当前 Case 的请求/样本进度。
每个任务打印一次“当前任务”标题；交互终端只原地刷新当前 Case 一行。
行内显示任务名、Case 序号/总数、进度条、样本与请求计数以及失败、未完成、
未开始计数。结束后保留完整行及最终状态，下一 Case 从新行继续。
重定向输出只在阶段变化时打印进度，完成行同样保留。
准备信息 JSON 不在终端展开；准备确认流程和原始文件保留。
运行前只显示任务数、Case 数及任务列表，不展开 Case 参数 JSON。
完整有效参数可用同样配置执行 luban-meter plan --config tasks.json 查看，
或读取运行目录的 plan.json。

```bash
luban-meter run --config tasks.json
luban-meter run --config tasks.json --no-progress
luban-meter run --config tasks.json --no-progress --no-summary
```

- `--no-progress` 仅关闭运行进度；`--no-summary` 仅关闭结束后的摘要。
  进度开关不进入 Benchmark 参数或结果 JSON。
- Case 序号/总数取当前任务计划，序号不代表成功数量。
  每个 Case 按任务与 ID 去重打印，成功、失败、跳过、中断/丢弃按实际状态显示。
  进度条仅表示样本处理进度，填满不代表全部成功；无已知总量不推算百分比。
- 当前 Case 的样本数使用已保存事件的 `weight`，请求数使用
  `logical_requests`，两者分别显示。正式进度排除预热，不等同于评分正确率。
  未完成和未开始样本单列，不算作已完成样本。
- 样本总量取 `case_ready.total_samples`，请求总量取配置的请求数量上限。
  无已知总量时显示“未知”及累计数，不推算百分比或 ETA；达到
  时间限制、数据耗尽或中断时，不强行将当前 Case 的计数补满。
  多轮场景的样本是会话、请求是轮次，样本总量为 `multi_num_sessions`，
  两个计数分别显示，不把一个会话当作一次请求。
- 读取已有事件和 Case 提交结果，约每 0.25 秒刷新，每次最多读取 1 MiB
  事件；最终 Case 状态以保存结果为准。高吞吐或大记录下实时计数可能滞后，
  不截断原始文件。无法解析的记录不补造计数，最终样本数以覆盖率校准。
- 准备摘要输出前清除进度行；结束或中断后换行。进度读取或输出失败会关闭
  进度显示，不改变评测执行、指标、退出码或结果保存。
- 每个 Case 结束后保留任务名、Case 序号、进度条、最终计数及状态。
  后续进度在下方刷新，不清除历史完成记录；无采集证据的计数显示“未知”。
  重复读取相同结果不重复打印；最终状态发生变化时追加“更正”记录。

## 服务器终端结果摘要

每个性能评测 Case 的 total_token_throughput 以正式成功请求的实际输入和输出
Token 总数除以该 Case 的正式测试窗口，单位为 tokens/s。在线性能优先读取
已保存的 service_view.total_token_throughput。效果评测不计算、保存或展示吞吐量指标及 Case Token 总量；原始响应事实保留。
性能评测总体吞吐量仅使用 generation_performance 类别 Case 的上述 Token 数之和
除以其正式测试窗口时长之和，不纳入效果评测的 Token 和时长。
预热 Token 和预热耗时均不参与吞吐量计算。另列整轮墙钟时长（duration，秒），
使用单调时钟计算经过时长，包含准备、预热及任务切换耗时，不受系统时间校准影响。
started_at 和 finished_at 保留 Unix 时间戳，不用于计算 duration。
Case 筛选和默认展示条数不改变整轮统计范围。
指标在执行与汇总阶段计算并写入结果 JSON，终端仅展示保存值。
Case 指标保存在原有性能指标的 values.service_view 中，例如
metrics.streaming_performance.values.service_view，包含 total_tokens、
benchmark_duration 和 total_token_throughput 三个带单位的统计节点。
整轮 result.json 的 metrics.token_throughput 保存性能 Case 汇总，
这是运行级汇总字段，不是独立注册的 Metric 类；顶层 duration_s 保存整轮耗时。
原有性能指标保留请求成功/失败数；缺失 Token 用量时吞吐量不可用。
整轮 metadata 逐项保存纳入、跳过、不可用的任务/Case 身份、统计依据和原因。
不可用 Case 不会被静默排除后生成局部整轮吞吐量。
终端与 HTML 总览均展示整轮已保存值、单位和不可用原因，详细区展示统计依据。
吞吐量计算在原有性能 Metric 内完成，复用公共计算函数，不新增 Metric 类，
也不在执行阶段自动追加指标。
Case 原始 .result.json、任务结果及整轮结果均保留 Case 指标。
旧结果缺少字段时显示 N/A；重新导出终端报告不读取原始文件或补算指标。

`run` 完成后默认打印简洁摘要；`run --no-summary` 关闭摘要。结果保存、HTML
生成和运行退出码保持原行为，摘要显示故障不改变 Benchmark 的执行结果。

```bash
luban-meter report --input results/<run>/result.json --format text
luban-meter report --input results/<run>/result.json --format text --verbose
luban-meter report --input results/<run>/result.json --format text --all
luban-meter report --input results/<run>/result.json --format text \
  --task task_1 --case case_1 --verbose
luban-meter report --input results/<run>/result.json --format text \
  --case task_2/case_1 --output report.txt
luban-meter report --input results/<run>/result.json --format text | less
```

默认展示运行和 Case 状态、模型/场景/数据源、并发/预热、覆盖率、窗口时长、
实际保存的性能或质量统计及必要错误。`--verbose` 增加绝对时间戳、目标长度、
aggregation、percentile_method、weight_sum、统计口径和已保存的诊断元数据。
`--all` 只控制 Case 数量，不代替 `--verbose`。

- 默认显示前 5 个匹配 Case；状态汇总包含全部匹配 Case，并提示省略数量。
  `--task`、`--case` 可重复使用，各自取并集，两类条件取交集。
  同名 Case 用 `TASK/CASE` 区分；未知筛选条件报错。
- 表格从现有统计节点动态生成，不按场景名推测分数，也不补齐固定性能指标。
  默认优先显示已保存的 TTFT、ITL、TPOT、E2EL、输入/输出 Token、请求与输出
  Token 吞吐量；其他性能项提示数量并在 `--verbose` 中完整展示。若没有这些
  优先项，则直接展示已有统计，未知指标不会被丢弃。
  新增 F1、pass@k、自定义统计节点无需修改终端指标白名单；原字段名未配置
  展示别名时直接保留。默认隐藏明确的学科/子类别分组（如
  accuracy_by_subject），提示隐藏项数；任意嵌套扩展指标不会因此被隐藏。
  使用 --format text --verbose 查看全部明细，--task / --case 可精确筛选；
  原始指标数据、诊断和评分分母保持不变。
- Mean/Value、P50、P95、Unit、N 读取已保存结果。N 不由总样本数推算。
  浮点值显示最多三位小数，微小非零值使用科学计数法，绝不修改 JSON 精度。
  标量质量指标使用 Metric/Value/Unit/N 四列，ratio 保持原比例，不自动转百分比。
  数字列按格式化后的实际长度分配宽度，优先保证数值完整；空间不足时使用
  纵向布局，避免将分位数拆成上下两行。
  不存在的指标不显示；存在但不可用的值显示 N/A，真实 0 保持 0。
  无统计样本时不把占位零当作有效测量；标量的分位数不适用。
- 请求级 Token 数是请求分布，服务级 `total_*_tokens` 是总数，
  `*token_throughput` 是吞吐量；请求级速率和服务级吞吐量分表、分名显示。
  简洁模式把 sample/request/token/s 单位的标量计数或窗口信息压缩为文本行；
  详细模式恢复表格。两种模式均保留这些值和已保存的缺失原因。
- ASCII 边框和中英表头不依赖颜色。76 列及以上使用表格，窄窗口采用分块布局。报告最多占 120 列，
  避免超宽终端或 `COLUMNS` 将 Metric 列无限拉长；按中文显示宽度计算对齐。
  结果路径及查看命令不人为折行，以便复制执行；终端可自然折行，`less -S`
  可横向滚动。输出重定向与 `--output` 都是纯 UTF-8 文本。
- 输出末尾包含真实结果路径及带原筛选条件的详细查看命令。失败 Case 另有
  精确定位命令，以及 `--details errors` 查看已保存请求错误的命令。
  原始目录齐全时也可使用 `report --include-requests` 导出 HTML。
- 文本摘要不读取原始请求、不连接模型、不重算指标。HTML 和终端共用
  `reporting/data.py` 的结果校验、单位及计数继承规则。默认 `report` 仍导出 HTML；
  `--verbose`、`--all`、任务/Case 筛选只适用于文本，`--include-requests`
  只适用于 HTML。`--raw-root` 也可用于文本请求详情。
  文本 `--output` 不允许覆盖输入 JSON，请求详情也禁止覆盖原始 Case 文件。
- 支持当前未标版本的 run/task 结果，包括缺失可选字段的历史结果；
  `schema_version` 等旧协议结构不自动迁移，明确报错并保留原文件。
  未掌握旧字段的统计定义时，不猜测转换为当前口径。

ITL/TPOT 权重核对：当前在线计算器以最后内容事件为 Token 计时终点；
ITL=(最后内容延迟−TTFT)/(输出 Token 数−1)，按 Token 间隔数加权；
TPOT=最后内容延迟/输出 Token 数，包含首 Token 等待，按输出 Token 数加权。
缺失最后内容时间时 TPOT 使用既有 E2EL 回退，ITL 无该样本值。
`count` 为有效请求数，`weight_sum` 为对应权重，分位数采用保存的 weighted CDF。
因此合成示例的 50 条、每条 16 Token 对应 ITL 权重 750、TPOT 权重 800。
本次报告改进没有修改上述公式。

体验与验收样例：

```bash
luban-meter report --input examples/terminal-result.json --format text
luban-meter report --input examples/terminal-result.json --format text --verbose
luban-meter report --input examples/terminal-gsm8k-result.json --format text
python examples/generate_terminal_examples.py
```

`terminal-result.json` 是合成请求经过现有计算器生成的展示数据，并非真实 GPU
性能。`terminal-gsm8k-result.json` 是仓库已有 2026-10-08 GSM8K 运行结果的原样
副本（Qwen2.5-3B，5 条，Exact Match=0.2），不是本次新运行；其中原始文件的
绝对路径只在原环境有效。生成脚本不访问模型，会在 `results/terminal-report-v1/`
保存成功、部分失败、历史质量结果、混合多 Case 的 JSON 及默认/详细/40 列文本。
失败样例来自模拟原始请求和真实计算器；多 Case 是演示组合，不代表一次真实运行。

## SSH 请求详情与慢请求定位

请求详情在已有摘要之上按需读取原始 Case 文件；不会重新执行请求，不修改
JSON 或重算指标。仅支持命令式文本输出，不包含全屏 TUI。

```bash
luban-meter report --input results/<run>/result.json --format text \
  --details errors --case task_1/case_1
luban-meter report --input results/<run>/result.json --format text \
  --details requests --sort e2el --limit 10
luban-meter report --input results/<run>/result.json --format text \
  --details requests --task task_1 --include-body --limit 1 --verbose
luban-meter report --input result.json --format text --details errors \
  --raw-root /path/to/original/run --output errors.txt
luban-meter report --input results/<run>/result.json --format text \
  --details requests --limit 100 | less
```

- `--details errors` 包含已保存的失败、超时、取消、未完成请求，以及关联样本的
  错误诊断；请求失败数和仅有样本的记录数分开显示。样本答错不等于请求失败。
  无关联请求的样本标为 `sample-only`，不伪造请求 ID。未知状态明确提示，
  不自动视为成功；无错误记录时明确说明，缺失原始文件时不声称全部成功。
- `--details requests` 读取所有匹配 Case。`--limit` 是跨 Case 的全局展示上限，
  默认 10，必须是正整数；不受摘要的前 5 个 Case 限制，`--all` 不扩大请求数。
  默认保持 Task/Case 和原始记录顺序，这不一定等于请求开始顺序。
  `--task`、`--case` 沿用摘要的筛选规则，只加载匹配的 Case。
  默认表格直接显示 Request ID，并在表内标明 Task/Case 分组；80 列下用
  Request ID 替代序号列，长标识换行显示。成功请求的 raw_index、sample_id、
  facts.status 及样本事实移到 `--verbose`，避免表后逐条重复定位信息。
  错误、未知状态、原始负值和连接清理问题仍保留必要诊断。
- `--sort` 可选 `e2el`、`ttft`、`itl`、`tpot`、`latency`、`input_tokens`、
  `output_tokens`，仅读取 `response` 中对应的 `*_ms` 或 Token 字段。
  降序排列，相同值保留原序；缺失/无效值放在末尾，明确显示其数量，仍受 limit
  限制。原始负值保留并提示核查。慢请求不自动被标记为性能异常。
  `latency` 专指已保存的 `response.latency_ms`，不冒充 E2EL。
- N/A 表示字段未保存、不适用或不是有效数值；真实 0 保留。
  当前流式原始响应通常没有独立保存 `itl_ms`、`tpot_ms`，因此显示 N/A；
  Case 汇总中的 ITL/TPOT 仍采用现有计算结果，不用均值反填单请求。
  `duration_ms` 原样标注，不用调度起止时间补算 E2EL。批次记录保留逻辑请求数，
  不拆成虚构的单请求耗时。
- 质量评测的耗时、Token 和正文可能只存在于 `samples[].facts`。按 `sample_id`
  关联后在 `--verbose` 中标为样本事实；错误详情也保留相关样本事实。
  不混入请求表格或请求排序；缺少 ID 时不猜测关联。
  不改动现有评分或数据采集。错误类型未保存时显示 N/A，不从错误文本猜类型。
  展示记录中的相同错误消息只打印一次，后续以 E1 等错误编号引用；
  每条记录分别列出 request、response、
  sample[index]、sample[index].facts 的实际来源及各自保存的错误类型；
  不把 sample 的错误类型填给 request。不同消息不因预览截断而合并。
- 默认隐藏 Prompt/模型回答。`--include-body` 必须与 `--details` 配合使用，
  输出敏感信息提醒及实际保存正文的预览；每字段最多 2000 字符、每记录最多
  展示 5 条关联样本，优先显示失败样本，超出部分明确提示查看 raw。
  正文来源标明原始样本索引。普通错误/诊断每字段预览
  最多 500 字符；控制字符转义，保留原文件完整内容。数据集原文与实际请求
  正文分别标注，未保存时说明，不重新生成。错误描述本身也可能含服务返回内容。
- `--verbose` 增加已保存的调度时间字段；请求表格支持 80/100/120 列，窄窗口
  使用分块布局，仍支持重定向、UTF-8 文本文件和 less。
- 复用 HTML 的原始数据读取与目录边界校验。搬迁整次运行目录可自动定位，
  不读取指定根目录之外的文件或符号链接目标。历史未标版本结果若缺少
  requests/samples 中一个数组，展示剩余记录并标注不完整；不支持的原始结构
  显示原因，旧协议 result.json 仍明确拒绝，不猜测迁移。
- 各 Case 顺序读取，排序只保留 limit 条候选；单个 Case 的 JSON 仍需完整解析，
  内存需要容纳最大的原始 Case 文件。limit 限制输出，不会改变保存的请求数。

可复现验收样例（全部为符合原始结构的模拟数据，无模型请求）：

```bash
python examples/generate_terminal_details.py
COLUMNS=80 luban-meter report \
  --input results/terminal-report-v2/partial/result.json --format text \
  --details errors
COLUMNS=80 luban-meter report \
  --input results/terminal-report-v2/partial/result.json --format text \
  --details requests --sort e2el --limit 10
COLUMNS=80 luban-meter report \
  --input results/terminal-report-v2/partial/missing-result.json --format text \
  --details errors
```

生成目录同时保存 40/80/100/120 列、正文和详细模式的文本样例；其中缺失文件
案例是刻意缺失，ITL/TPOT 的 N/A 是未保存请求字段，不代表 Case 统计不可用。

## HTML 结果摘要与详情导航

离线报告仍是一个 HTML 文件，导航按从概括到详细的顺序切换六页：
**报告总览 → 结果摘要 → Case 分析 → 全部指标 → 资源观测 → 运行诊断**。
每次只显示当前页；主题和筛选状态在切页时保留。浏览器前进、后退、刷新和
原有 Case / 请求锚点可定位到所属页面，不需要额外服务器或网络。

页面默认暗色，可切换明暗主题。报告标题位于导航栏上方并左对齐。
报告总览顶部展示保存配置中的模型名（`model`）和服务地址（`base_url`），
按名称与地址的组合去重；多个模型或地址逐项对应展示，缺失字段显示“未采集”。
“报告总览”与“评测总结”合并：先展示 Case 总数、执行成功的 Case 数，以及用
大字号突出的性能和效果最佳指标，其后展示紧凑的概览与趋势。Case 数统计保存
结果中的全部 Case，成功数
仅统计 `status=success`，不等同于正确作答数或请求成功数。

性能默认逐项独立择优，标明各自来源 Case；不同条件、不同并发的最佳数值不
代表同一个同时可达的工作点。总览可选择其他性能 Case，切换后所有性能卡片
展示该 Case 自己的数值，不再择优，并改用较小字号的紧凑卡片。效果评测也支持
独立选择 Case；只有最佳结果使用大字号和状态色突出。
点击具体数据会进入“结果摘要”并定位来源 Case 行，再通过“分析”查看完整指标。

最佳指标卡片之后保留“推理性能 / 模型能力”两个概览标签，使用普通字号，支持
方向键、Home / End、历史记录和刷新定位。性能趋势仍按固定测试条件分组，
与顶部 Case 选择独立，不把跨条件数据连成一条曲线。

下方概览与趋势使用紧凑布局呈现测试条件和核心数值：

- 推理性能按任务及除 `max_concurrency` 外的全部已保存配置分组，模型、
  输入输出长度、负载策略等条件不同的 Case 不混成一条曲线。默认选中本次
  已测最高输出吞吐量所在组，可切换查看其他条件组。
- 峰值取在线性能指标
  `service_view.output_token_throughput`
  （`token/s`），同时展示同一峰值 Case 的请求吞吐量及 TTFT、TPOT、ITL Mean。
  直接读取原有均值、单位和权重口径，缺失值显示“未采集”，不改写指标结果。
  在线服务与多轮会话保存的是同一套视图，两者按各自任务与条件分别成组，
  不混成一条曲线；并发轴仍是配置并发上限。
- 并发—吞吐图的横轴是配置并发上限，不是实际平均并发。所有峰值和趋势仅
  针对已测点；最高已测并发处取得最大值时明确标注，不能据此推断服务极限。
  并列最大值全部保留；只有一个有效并发点时不判断趋势；同一并发多次测量
  保留各点，不合并均值或判定单一趋势。图中数据点可进入对应 Case 分析。
- 只纳入 `status=success`、`collection_complete=true`、指标无失败且覆盖率与
  请求统计未记录异常的 Case，并要求有模型、负载策略、有效并发上限和上述
  吞吐字段。采集完成标记缺失也不纳入。未参与总结的数量与原因类别会说明，
  其他结果（包括 Engine 指标）仍可在摘要、Case 分析和诊断中查看。
  按正常停止条件结束后未开始的计划样本不等于请求失败，不因此排除成功 Case。
- 模型能力按任务及 Case 展示保存的得分和各学科 / 分类指标，CMMLU 学科
  得分也可在模型能力总结标签展开查看。样本 / 计数采用各指标保存的 `sample_count` 或
  `count`，不以采集完成数代替评分分母；PPL 等指标标注数值越低越好。
  本次不接入行业基线或其他模型对照，不生成相对行业强弱结论，也不将学科
  分数自行归并成未定义的能力维度。

首屏以 2 列 × 3 行的深色摘要卡片展示关键结果，窄屏改为单列。六张卡为
Peak Throughput、Best TTFT P99、Best TPOT P99、Best Success Rate、
Max Tested Concurrency 和 Best E2E P99。任务、Case 及样本数量缩为一行元信息。
效果评测最佳得分与性能卡片并列展示，数据、单位、评分样本数和来源 Case 一并
保留，不伪造综合评分或 SLO。

- 峰值吞吐取本次运行中成功且采集完整的性能 Case 中的最大值；最佳 P99 分别取对应
  延迟指标的最小已保存 P99，不合并、平均或重新计算分位数。秒可换算为毫秒，
  TPOT 保留 `ms/token` 的 Token 权重语义；未知单位或无效值显示 Unavailable。
- Best Success Rate 是已完成性能 Case 中最高的单 Case 请求成功率，不是整次
  运行的合计成功率。每个 Case 使用自身成功请求数除以总请求数，失败请求
  进入分母；计数缺失、不一致或分母为零的 Case 不参与择优。来源旁保留分子、
  分母和并列信息。手动选 Case 后仅显示其自身成功率，无效计数显示 Unavailable。
- 服务统计可独立于请求延迟视图存在。缺少请求延迟时，相应延迟显示 Unavailable，
  已保存的吞吐量、请求成功率与趋势仍按各自条件判断；缺失字段不从另一份指标
  结果补齐，也不因已选结果失败而换用其他结果。
- 最大测试并发采用已完成且有请求执行证据的 Case 的配置上限，不等同于
  实际平均或峰值并发。手动查看时改为该 Case 的已保存配置并发上限。
- 能力得分在同一任务、数据集、划分、few-shot、评分模式和样本选择条件内
  择优。准确率、Exact Match、Pass@1、F1、ROUGE 越高越好，PPL、平均损失、
  BPB 越低越好；不跨数据集生成综合排名。多指标时可选择择优指标，未声明
  优化方向的自定义指标不自动判优。WikiText 默认选 PPL；CMMLU / C-Eval、
  GSM8K、HumanEval 分别默认总体准确率、Exact Match、Pass@1；其余采用首个
  已保存指标，并明确显示当前选择。失败、未完成或零评分样本的 Case 不参与择优。
- 只有表示占比的指标使用真实百分比圆环；性能卡中仅请求成功率使用
  圆环，0% 不画进度弧，100% 为满环。吞吐量、延迟、并发数等指标只展示数值。
  是否画圆环由指标的比例语义决定，不根据名称含 Rate 或单位含 `%` 猜测；
  比例缺失或无效时不画圆环。
  每卡保留标题、数值、单位、说明和来源 Case；点击数值可定位到对应 Case 的
  结果摘要。缺失、异常数据不转换为零。
- 通用组件位于 `reporting/summary_cards.js`，接受数值、单位、比例类型、说明、
  状态色和链接，与 Benchmark 的指标选择和分组逻辑分开，可复用到其他结果页。
  明暗主题下卡片都采用深色背景；详细分析继续使用原有曲线、分位数图和表格。

结果摘要先按“推理性能评测 / 模型能力评测”分组，再按场景或数据集分表，每行一个
Case；共同模型、场景和数据来源在表格上方，变化条件与核心指标在行内。
类别使用保存的 `category` 判定，C-Eval、CMMLU 等数据集位于模型能力评测下，
不依赖任务名是否为 `knowledge_accuracy`。运行诊断采用同样的类别层级，
各数据集任务位于“模型能力评测”外层折叠标签中；原有任务、Case、请求锚点保留，
深链可自动展开所有父级。
性能任务默认展示 TTFT Mean、TPOT Mean、窗口输出吞吐量和总 Token 吞吐量；质量任务根据已保存
指标生成列，不强制使用 Accuracy。完成、失败、未完成和未开始样本分别保留，
不能把完成数当作评分分母。数值列可排序，表头固定，窄屏可横向滚动。
“完成 / 失败”和“未完成 / 未开始”组合列按前一项排序。

点击每行的“分析”会定位并选中对应任务和 Case；重名任务或 Case 用各自位置
区分。Case 指标分析在桌面端横向排列带标签的选择框，窄屏自动换行：

- 生成性能任务：任务 → Case → 性能指标。主菜单按 TTFT、TPOT、ITL、E2EL、
  请求吞吐量、输出 Token 吞吐量、总 Token 吞吐量排序，只显示当前 Case 已保存的核心指标。展示全部
  已保存统计量、单位及 count；有分位数时显示分位数图，标量只显示聚合值。
  Median 和 P50 分别保留，不根据数值相等推定统计定义相同。
  多轮 Case 在核心指标下方额外展示“按轮次统计”：先给出每轮的计划参与、成功、
  已发送失败与未开始会话数台账，再逐轮列出 TTFT、ITL（请求级平均，按Token间隔
  加权）、Stream Event ITL（相邻流事件间隔）、E2EL、输入与
  输出 Token 的 Mean 和 P99、各轮已保存样本数、各轮 TTFT Mean 柱状图，以及
  已结束会话、计划与完成轮次、未开始与发送失败轮次和会话中止原因。全部轮次
  失败时延迟统计留空并说明，参与台账与中止原因仍然展示。Case 选择框同时显示
  会话数与轮数上限。
- 质量任务：任务 → Case，隐藏第三个选择框。自动展示全部已保存指标，按
  所属维度和单位分表。CMMLU 主分析区隐藏各学科正确率，保留总体指标；
  学科原始数据仍可在“全部指标”和原始详情查看，其他数据集分类不受影响。GSM8K 的
  Exact Match、HumanEval 的 Pass@1、PPL 等以保存字段为准，不补造 Pass@k。
- 比例单位 `ratio` 显示为百分比；悬停数值可看原值，展开原始精度与路径可看
  统计口径。原始指标说明、权重及评分信息均保留；未知指标使用字段名回退。
- 无指标、空任务、失败或丢弃 Case 显示对应提示；丢弃 Case 不展示有效指标。
  “测试条件、覆盖率与诊断”可进入原有详情，并继续访问完整报告中的请求列表。

Case 分析与“全部指标”页的搜索、筛选相互独立。无需额外服务器或外部脚本。

资源观测按任务和 Case 位置分组，重名任务或 Case 不会合并采样。选择 Case 后，
GPU、主机和服务负载一起切换；顶部显示运行 ID、模型、Case 测试窗口、采样数量与
错误数量，并标出该 Case 在整次运行时间轴中的位置。图表共用时间范围，浅色区域
标记测试窗口，悬停采样点可查看时间和数值，较大采样缺口不连线。

- GPU 区域按卡显示保存的平均利用率、显存、功耗与温度，并用四类多序列折线图
  展示变化，可筛选单卡。单位分别为 `%`、`MiB`、`W`、`°C`；缺少显存总量时
  显示不可用，不推算容量或显存占比。设备观测可能包含其他服务使用的 GPU。
- 主机区域显示CPU、内存、文件系统空间、磁盘读写速度/IOPS/忙碌率和网络收发、
  错误/丢包速率。挂载点、块设备、网卡分别选择，不将不同层次设备直接求和；
  保留lo回环接口以观察同机压测。整机数据包含其他业务及压测客户端的开销。
  旧结果回退读取原有CPU/内存字段，仅有汇总时注明无时间序列；缺失指标显示不可用。
- 服务负载展示已保存的运行 / 排队请求曲线，其他服务序列收在折叠区域中，保留
  原始数值；原始服务序列没有保存单位时明确注明，不猜测量纲。
- GPU、主机、服务采样各最多选取500个时间点用于绘图，原始记录保持完整。均值使用
  已保存聚合，不以图表抽样重算；缺失值不补零，无 Case 起始时间时不为相对
  硬件采样补造时间戳。所有资源图表由内嵌 SVG 本地渲染，无 Grafana 在线依赖。

左侧导航覆盖六个页面，高亮当前页；桌面端位于报告标题下方，与当前页面内容区
顶部对齐，滚动时保持可见；窄屏折叠为顶部菜单，
明暗主题共用样式。相邻任务摘要之间自动使用虚线分隔。页面将 `performance`
显示为“推理性能评测”、`knowledge_accuracy` 显示为“模型能力评测”，
`online_serving` 显示为“在线服务”，`multi_turn_serving` 显示为“多轮在线服务”；
JSON 标识及未知任务名保持原样。

核心性能菜单严格匹配在线服务计算器保存的 `values.request_view` 下的
`ttft/tpot/itl/e2el`，及 `values.service_view` 下的
`request_throughput/output_token_throughput/total_token_throughput`；指标名不参与匹配，因此
当前 `streaming_performance` 与历史结果中的 `multi_turn_performance` 都能命中，
但不会替代为 Engine 或其他来源的同名指标。
默认 TTFT，缺失时选择第一个可用项；切换 Case 保留仍存在的选择，否则回退。
其他性能诊断数据继续保留在“全部指标”及原始详情中。


每个 Case 独立展示，不跨场景合并均值。缺失数据标为“未采集”，实际 Token
长度与输入长度参数分别呈现；失败、丢弃和未完成状态保留，不补算缺失指标。
详情保留指标路径、单位、样本数、分位数及原始指标说明，统计公式与 JSON
结果不变。默认摘要报告不内嵌逐请求输入输出。

需要请求详情时，从已有结果导出，不重新调用模型：

```bash
luban-meter report --input results/<run>/result.json --include-requests
```

默认生成同目录的 `report-full.html`，也可使用 `--output` 指定文件名。
完整报告内嵌原始 Case 文件中保存的请求及样本，可以独立离线转发；请留意其中
包含原始输入输出。支持编号搜索、异常筛选、耗时排序和每页 50 条记录。
从 Case 分析进入详情后，“异常请求/样本”或最慢请求可定位到列表。
长文本、计时和原始字段点击后展开。
分页不会截断内嵌数据，也不会减少 HTML 文件体积，完整报告导出和打开时需要
容纳这些数据的内存。无明确大小上限，不静默删减记录。

请保留 `result.json` 和 `tasks/<任务名>/raw/<Case编号>.json` 的目录关系。
搬迁整个运行目录后仍可导出；若结果 JSON 单独复制，使用
`--raw-root <原始运行目录>`。单任务结果则使用 `<任务目录>/raw/<Case编号>.json`。
仅读取所指定目录以内的文件（包括解析符号链接之后的路径），不读取结果中
任意指向其他位置的文件。缺失、损坏或已丢弃的原始记录在 HTML 中标明不可用，
不影响结果摘要导出；这不代表已成功包含所有请求详情。

请求与样本按 `sample_id` 关联，不根据完成顺序猜测。一个样本可关联多个请求；
无关联请求的已保存样本也会显示。列表不包含预热事件或未保存的请求。样本错误
与请求错误分别保留，异常筛选包括失败、取消、未开始和未完成的已保存记录。
输入输出和时间线仅展示保存的字段，不重新生成或反推。当前在线流式性能记录
不保存输出正文，因此其正文显示“未采集”；质量评测的已保存文本仍可查看。
耗时采用响应中的 `e2el_ms`、`latency_ms` 或 `duration_ms`，不会以调度占位时间
替代响应耗时。Engine 批次和质量样本的专用事实保留在展开详情中，不伪装成
单请求客户端 TTFT。


HTML Case 分析的离线浏览器回归（使用已安装的 Playwright 和 Edge）：

```bash
python tests/test_report_analysis.py results/report-display-review/fixture.html
node tests/report_analysis.browser.cjs <真实报告.html> \
  results/report-display-review/fixture.html results/report-display-review/screenshots
node tests/report_overview.browser.cjs <真实报告.html> \
  results/report-display-review/overview-screenshots
```

浏览器脚本默认使用 Edge，可设置 `LUBAN_BROWSER_CHANNEL=chrome` 使用已安装的
Chrome；`NODE_PATH` 需包含 Playwright。总览回归额外检查分组峰值与对应延迟、
各学科得分及计数、单点 / 并列峰值 / 异常数据、分页前进后退与刷新，以及
最佳指标与单 Case 切换、P99 来源跳转、真实比例、极值及桌面 / 窄屏布局。

此真实数据回归针对包含 12 个性能 Case 和 1 个 CMMLU Case 的报告；额外结构和
异常场景使用明确标注的合成测试数据。只读取保存结果，不调用模型或下载数据。


终端摘要额外展示“最佳性能指标”和“最佳能力得分”。择优范围为 CLI
任务/Case 筛选后的全部结果，不受默认 Case 展示条数限制。性能指标分别取
最大输出/总 Token 吞吐量、最低 TTFT/TPOT/E2E P99、最高请求成功率和
最大已测配置并发，标注来源 Case、并发和并列数量；各项最佳值不代表
同一工作点。请求成功率使用已保存的成功/总请求计数，失败计入分母。
缺失 P99 不使用 P95 替代。吞吐量和延迟仅从成功、采集完整的结果择优。
能力得分按任务、数据集与评分条件分组；准确率等取最大值，
困惑度/损失等取最小值，不跨数据集排名，不展开学科明细。
能力指标支持顶层字段和`values`包装两种保存方式，评分分母读取各指标的
`sample_count`或`count`。总Token吞吐优先读取当前性能指标中的已保存字段，
历史`token_throughput`仅作旧布局回退；当前来源缺值或失败时不从旧来源补值。
所有择优仅读取保存值，不重新计算吞吐量、分位数或准确率。

### Prefix Cache 报告（阶段 5）

终端及离线 HTML 直接展示已保存的 `prefix_cache_hit_rate` 指标。目标、
理论共享比例、实际命中率、请求/Token 覆盖率和成功率显示为百分比；
构造、运行和总体目标偏差显示为百分点（pp）。真实的零显示为 0%，缺失值
显示“不可测”或“未采集”。已测子集命中率单独标记，不能替代完整 Case 的
实际命中率。成功请求输入总数属于 S，缓存命中总数及已测输入总数属于 M，
报告不把不同请求集合的总数相除。

HTML 的结果摘要保留性能指标，并展示预热事实、测量有效性和目标达标状态；100% 目标的 `construction_limited` 单独显示。
单档摘要标题使用“前缀缓存评测 · 50%”中的目标档位，多档任务在各 Case 标出目标。
缓存命中、负载与状态、请求与性能分表展示，窄屏纵向排列。
Case 分析的指标菜单按性能与 Prefix Cache 分组。请求详情使用已有原始记录，
展示前缀编号、长度、缓存计数、单请求命中率、字段与长度状态，不复制 Token ID。
报告导出删除 `cache_salt` 字段及已知值的回显；原始结果文件不被改写。

多命中率对比位于结果摘要，横轴明确为**目标命中率**，纵轴可选择实际命中率、
TTFT、ITL、TPOT、E2EL（延迟使用保存均值）或吞吐量。
同一任务中除目标比例与历史环境注记外的已保存参数必须相同才归入一组；单位不一致时
只展示表格。不声明未保存的服务条件相同，Salt 也不隔离物理 KV Cache。
缓存测量有效、正式请求全部成功完成且性能指标有效时，各指标独立择优。
预热偏差和未达标仍在对应 Case 中展示。

旧的无 Prefix Cache Metric 的无版本号 JSON 行为保持不变；此前已不支持的
带协议版本旧格式仍拒绝读取，不进行隐式迁移。请求详情仍需
`--include-requests`，以及可访问的原始文件。

本地 Mock 示例（不连接服务、不下载模型）：

```bash
HF_HUB_OFFLINE=1 python -m tests.test_prefix_cache_reporting /tmp/prefix-cache-mock
luban-meter report --input /tmp/prefix-cache-mock/result.json --format text --all
luban-meter report --input /tmp/prefix-cache-mock/result.json --include-requests
node tests/prefix_cache_report.unit.cjs /tmp/prefix-cache-mock/result.json
node tests/prefix_cache_report.browser.cjs /tmp/prefix-cache-mock/report.html \
  /tmp/prefix-cache-mock/screenshots
```

浏览器检查复用现有 Playwright/Edge 环境，`NODE_PATH` 和可选的
`LUBAN_BROWSER_CHANNEL` 用法与上述报告测试相同。示例明确标记 Mock；
当前目标 vLLM 尚未返回 `cached_tokens`，真实缓存与最终服务验收仍待验证。

### Prefix Cache 预热策略

可在任务 JSON 中设置 `"prefix_cache_warmup_policy": "observe"`；
默认值为 `observe`，允许合法命中偏差继续正式评测。CLI 对应 `--prefix-cache-warmup-policy`。
任务 JSON 的值仍优先于 CLI 基础参数。两种模式都按全部种入、全部探测、
完整验证轮执行，不补发请求、不重复预热以追求达标，也不剔除低命中样本。

Strict 保持原行为：Probe / Verify 的有效命中计数必须等于共享前缀长度。
Observe 只放宽这项相等条件：合法的零值、低于预期或高于预期均被记录，
随后允许正式请求继续。HTTP 失败、超时、SSE 不完整、输入长度不符、
缓存字段缺失或非法、取消和中断仍为失败，不受 Observe 放宽。
缓存字段仍为必需数据；缺失不能解释为 0。

0% 不种入非空前缀，仍使用独立 Salt 做能力检查。若能力检查合法但非零，
Strict 失败；Observe 记录 `over_hit` 并继续。此时预热状态为
`observed_mismatch`，不能标记为零命中校验通过或 `not_applicable`。

原始 metadata 的 `cache_warmup` 及脱敏恢复诊断新增：

- `warmup_policy`、`warmup_completed`、`warmup_fully_matched`。
- `warmup_under_hit_prefixes`：Probe / Verify 中出现不足的不同前缀数；
  同一前缀多轮不足仅计一次。
- `warmup_summary`：按 `probe`、`verify`（0% 时为 `capability`）保存
  `prefix_id`、`expected_cached_tokens`、`actual_cached_tokens`、
  `deviation_tokens` 和 `validation_state`。
  偏差为实际减预期，状态为 `matched/under_hit/over_hit`；
  数据不可用时记录 `missing/invalid/unavailable` 并终止。

最终缓存 Metric 在 `warmup` 中透传这些字段和预热原因。
Observe 预热不符时，正式数据仍可产生有效的实际命中率，测量可为 `valid`，
预热诊断含 `warmup_cache_mismatch`；
目标达标由正式命中率与既有容差规则独立判定。预热命中不进入正式分子分母。
淘汰仍只可能是疑似或未知，不因命中不足直接确认为淘汰。

终端和 HTML 显示预热模式、完成与符合情况、不足前缀数和逐阶段实测。
预热不符的 Case 保留正式性能和缓存数据，并展示预热偏差。
旧结果没有新字段时仍可读取；其新增诊断显示未采集，策略按历史 Strict 解释。
这些扩展不改变 TTFT、ITL、TPOT、E2EL 或吞吐量公式。

### Prefix Cache 终端摘要

默认终端报告将缓存指标精简为目标命中率、实际命中率、目标偏差（pp）、
预热模式、预热完成及符合情况、请求统计覆盖率和测量与目标两个结论。
新结果覆盖率使用已提交正式请求，例如 `85.7143% (6/7)`；
若正式请求有失败，另外显示成功率，避免误读覆盖率分母。

正常结果不展开理论比例、S/M Token 总量或逐前缀 Probe/Verify。
预热不足、统计缺失/非法、输入长度不符、实际偏离理论、100% 构造上限等仍显示必要诊断。完整 Case 不可测时保留“不可测”，已测子集
命中率仅以明确的子集标签补充，不替代完整实测值。
`--format text --verbose` 展开全部已有缓存字段、阶段记录及完整指标上下文；
`--details requests/errors` 仍用于请求和错误明细，不是新的缓存开关。

Prefix Cache 使用固定长度合成 Token ID，暂不支持 ShareGPT 前缀比例。
ShareGPT 继续沿用原有在线评测路径。

终端进度及结果标题采用“前缀缓存评测（目标 25%）”等数字名称；
任务 ID、CLI 选择器、文件路径、JSON 和本轮 HTML 展示均保持原值。
新名称取目标参数或保存的目标 Metric，不用实测值生成任务标题。
