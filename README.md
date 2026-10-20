# LuBan-Meter

统一执行生成式推理性能与模型服务质量评测，保存原始事实、计算指标并导出离线HTML报告。

- 生成性能：在线OpenAI兼容流式服务、离线vLLM Engine。
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

## 停止与失败

- 任务顺序执行，失败默认继续；`--fail-fast`跳过后续任务。
- `--num-requests`统计实际模型请求。一个C-Eval PPL样本可能产生多个请求，不能当作一个请求限速。
- 数据耗尽、请求数或`--max-duration`任一条件达到即停止新请求，等待在途请求结束。
- `--num-warmups`按样本预热，不纳入正式请求预算和指标。
- 首次Ctrl+C停止整个运行的新请求，等待在途工作；第二次取消工作进程并清理本次资源。
- `--request-timeout`约束HTTP请求，`--task-timeout`约束整个子进程任务。
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
  --device-exporter-url http://127.0.0.1:9400/metrics
```

服务端Prometheus采样只适用于HTTP任务。硬件exporter地址单独指定；不提供时不启动硬件监控。
观测故障独立记录，不让推理任务因为指标接口故障而失败。报告保留实际采样时间和缺失信息。

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

HTML支持任务与Case筛选、指标排序、同类Case比较、观测时间线以及覆盖率与错误详情。
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


## HTML 结果摘要与详情导航

离线单文件报告先展示运行状态与每个 Case 的关键测试条件、覆盖率和核心指标。
点击摘要指标可定位到对应指标行并自动展开任务、Case 和指标详情；点击失败数
可查看 Case 错误与指标诊断，详情提供“返回摘要”。无需额外服务器或外部脚本。

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
包含原始输入输出。支持编号搜索、异常筛选、耗时排序和每页 50 条记录；点击
摘要的失败数或最慢请求可定位到列表。长文本、计时和原始字段点击后展开。
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
