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
- `reasoning_tokens`：仅保存服务`completion_tokens_details.reasoning_tokens`，
  未提供时为`null`。

只有思考的性能响应仍视为成功，并标记`has_content=false`；完全没有有效输出仍失败。
输出Token总数沿用服务`usage.completion_tokens`，不再额外累加思考Token，
也不按字符或事件估算缺失计数；需确认被测服务的总数包含思考，才能按全部输出比较吞吐。
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

## 多轮会话性能

`multi_turn_serving`按会话评测在线Chat服务：一个样本是一整个会话，会话内
轮次严格顺序执行，每轮请求携带此前轮次的**实际模型回答**作为历史。
不同会话并行，同一会话同时最多一个在途请求。

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
  保留，会话记为部分完成，不为凑完整会话继续发送。某轮HTTP失败或超时即结束该会话，
  该轮计入请求失败，后续轮记为未开始，其他会话继续执行，不做轮级重试；
  `abort_reason`取值`turn_failed`、`context_overflow`、`session_timeout`、
  `reasoning_only`、`no_output`；其中`turn_failed`、`reasoning_only`、
  `no_output`发生在请求发出后，该轮计入`sent_failed_turns`，`session_view`的
  未开始轮次不再包含它。
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
results/<run_id>/
  plan.json                  # 有效配置，凭据字段脱敏
  result.json                # 运行汇总
  report.html                # 可离线打开的单文件报告
  tasks/<task_name>/
    result.json
    raw/events.jsonl         # 请求、样本和观测事件
    raw/case_1.json          # Case原始事实
    raw/case_1.result.json
    raw/manifest.json
    logs/stdout.log
    logs/stderr.log
    artifacts/
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

TPOT按输出Token数加权，ITL按后续Token数加权；流式事件可能包含多个Token，
因此Stream Event ITL单独统计。`--goodput tpot:50`沿用已有SLO含义，
约束的是排除首Token等待的ITL，不是报告中的TPOT；多个SLO条件需同时满足。
迁移保留这些公式，不将CLI风格与其他工具的指标含义混同。

多轮性能（`multi_turn_performance`）同样从客户端观察，每一轮就是一次普通流式
请求，请求级与服务级视图直接复用在线公式，不另写一套。分母是本Case已保存的
轮次请求，不是会话数或计划轮次数。此外保存：

| 视图 | 内容 | 口径 |
|---|---|---|
| `request_view`/`service_view` | TTFT、TPOT、ITL、E2EL、吞吐量 | 与在线服务完全相同 |
| `turn_view` | 各轮TTFT、E2EL、输入与输出Token分布 | 按`turn_index`分组，每组带自己的样本数 |
| `session_view` | 会话耗时、轮间等待合计、计划/完成/未开始/发送失败轮次、已结束会话数、会话吞吐量、中止原因、会话覆盖率 | 样本是会话；`sessions_per_second`分母是Case窗口 |
| `caching_view` | 首轮与后续轮TTFT均值及比值、每轮和每千Token的TTFT增长 | 只描述延迟随上下文的增长，不作为前缀缓存命中的证据 |
| `caching_view.prefix_cache` | 客户端可复用前缀Token与占比、服务端上报的命中Token、命中率及上报轮次数 | 客户端估算与服务端实测分开；服务未上报时留空并给出`reason`，不补0，也不推断缓存关闭或驱逐原因 |
| `pressure_view` | 单轮上下文Token分布、Prefill Token合计与速率、单轮上下文峰值 | 只统计成功轮次，需结合设备观测判断瓶颈 |

后续轮次只包含延续到该轮的会话，各轮样本数必须与指标一起阅读：样本减少本身
不是性能变化。会话耗时包含轮间等待，不能与单请求延迟混用。

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

`run` 默认在 stderr 显示整轮 Case 进度与当前 Case 的请求/样本进度。
交互终端使用两行原位刷新，显示任务、Case、当前阶段和已用时间；窄终端会
缩短行宽。重定向或非交互终端仅输出阶段和 Case 状态变化，不逐请求刷屏。

```bash
luban-meter run --config tasks.json
luban-meter run --config tasks.json --no-progress
luban-meter run --config tasks.json --no-progress --no-summary
```

- `--no-progress` 仅关闭运行进度；`--no-summary` 仅关闭结束后的摘要。
  进度开关不进入 Benchmark 参数或结果 JSON。
- 整轮分母取计划 Case 数，“已结束”包含成功、失败、跳过、中断/丢弃，
  这些状态分别计数；100% 结束不代表全部成功。
- 当前 Case 的样本数使用已保存事件的 `weight`，请求数使用
  `logical_requests`，两者分别显示。正式进度排除预热，不等同于评分正确率。
  未完成和未开始样本单列，不算作已完成样本。
- 样本总量取 `case_ready.total_samples`，请求总量取配置的请求数量上限。
  无已知总量时显示“未知”及累计数、已用时间，不推算百分比或 ETA；达到
  时间限制、数据耗尽或中断时，不强行将当前 Case 的计数补满。
  多轮场景的样本是会话、请求是轮次，样本总量为 `multi_num_sessions`，
  两个计数分别显示，不把一个会话当作一次请求。
- 读取已有事件和 Case 提交结果，约每 0.25 秒刷新，每次最多读取 1 MiB
  事件；最终 Case 状态以保存结果为准。高吞吐或大记录下实时计数可能滞后，
  不截断原始文件。无法解析的记录不补造计数，最终样本数以覆盖率校准。
- 准备摘要输出前清除进度行；结束或中断后换行。进度读取或输出失败会关闭
  进度显示，不改变评测执行、指标、退出码或结果保存。

## 服务器终端结果摘要

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
  展示别名时直接保留。CEVAL/CMMLU 的按学科分数也按保存结构显示。
- Mean/Value、P50、P95、Unit、N 读取已保存结果。N 不由总样本数推算。
  浮点值显示最多三位小数，微小非零值使用科学计数法，绝不修改 JSON 精度。
  标量质量指标使用 Metric/Value/Unit/N 四列，ratio 保持原比例，不自动转百分比。
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
性能任务默认展示 TTFT Mean、TPOT Mean 和窗口输出吞吐量；质量任务根据已保存
指标生成列，不强制使用 Accuracy。完成、失败、未完成和未开始样本分别保留，
不能把完成数当作评分分母。数值列可排序，表头固定，窄屏可横向滚动。
“完成 / 失败”和“未完成 / 未开始”组合列按前一项排序。

点击每行的“分析”会定位并选中对应任务和 Case；重名任务或 Case 用各自位置
区分。Case 指标分析在桌面端横向排列带标签的选择框，窄屏自动换行：

- 生成性能任务：任务 → Case → 性能指标。主菜单按 TTFT、TPOT、ITL、E2EL、
  请求吞吐量、输出 Token 吞吐量排序，只显示当前 Case 已保存的核心指标。展示全部
  已保存统计量、单位及 count；有分位数时显示分位数图，标量只显示聚合值。
  Median 和 P50 分别保留，不根据数值相等推定统计定义相同。
  多轮 Case 在核心指标下方额外展示“按轮次统计”：逐轮列出 TTFT、E2EL、输入与
  输出 Token 的 Mean 和 P99、各轮已保存样本数、各轮 TTFT Mean 柱状图，以及
  已结束会话、计划与完成轮次、未开始与发送失败轮次和会话中止原因。Case 选择框同时显示
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
`request_throughput/output_token_throughput`；指标名不参与匹配，因此
`streaming_performance` 与复用同一套视图的 `multi_turn_performance` 都能命中，
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
