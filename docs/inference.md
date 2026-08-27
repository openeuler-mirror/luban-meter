# LuBan-Meter Inference 评测指标说明

本文定义 LuBan-Meter `inference` 模块的 Benchmark 组织、评测协议、Prompt
管理和逐数据集指标计算方式。该模块与 `generate` 的分工：

- `generate` 测量“模型生成得快不快、稳不稳”：TTFT、TPOT、吞吐量等，与回答
  内容无关；
- `inference` 测量“模型答得好不好”：用标准数据集样本调用在线推理服务，把
  输出与参考答案对比，计算正确性指标。

指标必须与数据集、评测模式、Prompt 版本、解码参数和单位一起解释。不同数据
集、不同评测模式得到的结果不能混合或直接比较。

## 1. 指标组织原则

LuBan-Meter 按“数据集族和任务协议”组织 inference Benchmark，不按指标自由
组合。“什么叫答对”由数据集和任务类型决定：选择题比对字母，数学题比对数字，
代码题执行单元测试，摘要题比较文本重叠。同名 `Accuracy` 在不同数据集上的
判定逻辑可以完全不同。

因此：

- 一个数据集对应一个 Benchmark 目录，自带评测模式、Prompt 模板和指标声明；
- `benchmark.py` 负责渲染 Prompt、调用在线服务、记录逐样本原始事实；
- `result.py` 负责校验原始数据并计算该数据集的指标；
- 用户通过 Benchmark 目录内的 YAML 配置选择数据集、评测模式、样本数量和解码
  参数；
- 结果元数据记录数据集、split、样本数、Prompt 版本、评测模式、解码参数和
  评分器版本，保证分数可复现、可审计。

这一组织方式与 OpenCompass 对齐：指标由数据集配置声明（`eval_cfg`），Prompt
模板属于数据集层，答案后处理和评分按数据集协议执行。概念对照如下：

| OpenCompass 概念 | LuBan-Meter 对应 |
|---|---|
| 数据集配置（split、样本数、few-shot） | Benchmark 目录内 YAML 配置 |
| prompt_template | `inference/common/prompts.py` + `prompt_version` |
| PPL / Gen inferencer | 配置字段 `eval_mode` |
| pred_postprocessor | `inference/common/parsers.py` |
| evaluator（Acc/EM/F1/ROUGE/Pass@k） | `inference/common/metrics.py` + 各 Benchmark `result.py` |

## 2. Benchmark 总览

| 优先级 | 能力类型 | 数据集 | 评测模式 | 核心指标 | 一期定位 |
|---|---|---|---|---|---|
| P0 | 知识问答 | C-Eval | PPL/LogLikelihood | Accuracy | 选择题判分基准，验证在线 logprobs 链路；Gen 模式保留作回退 |
| P0 | 知识问答 | CMMLU | PPL/LogLikelihood | Accuracy | 从 C-Eval 协议泛化到另一选择题族 |
| P0 | 推理与数学 | GSM8K | Generation | Exact Match / Accuracy | 验证长生成、答案抽取和判分逻辑 |
| P0 | 代码 | HumanEval | Generation + 执行 | Pass@1 | 验证代码补全、沙箱执行和多次采样统计 |
| P1 | 开放问答 | SQuAD | Generation | Exact Match、Token F1 | 验证文本归一化与 Span 匹配指标 |
| P1 | 中文摘要 | LCSTS | Generation | ROUGE-1/2/L | 验证中文文本重叠指标 |
| P1 | 语言建模 | WikiText | Token Loss | Perplexity、平均 Token Loss | 验证在线 logprobs 链路，缺失时不替代 |

## 3. 执行流程与目录框架

### 3.1 执行流程

```text
luban-meter run --module inference --benchmark ceval \
    --config src/luban_meter/benchmark/inference/ceval/ceval.yaml
  → CLI 构造 RunRequest
  → BenchmarkRegistry 发现 inference/ceval/benchmark.py 和 result.py
  → CoreEngine 创建 ExecutionSession
  → 宿主机 Python 执行 benchmark.py --request request.json --output raw_result.json
      1. 校验配置参数；
      2. 加载本地数据集，按配置选择样本；
      3. prompts.py 渲染任务内容层 Prompt；
      4. client.py 调用在线服务：
         - gen 模式  → /v1/chat/completions 或 /v1/completions；
         - ppl 模式  → /v1/completions（echo=true、max_tokens=1、logprobs），仅允许
           prompt_format=base；
         - loss 模式 → /v1/completions（echo + logprobs）；
      5. gen 模式输出经 parsers.py 提取答案；
      6. 逐样本原始记录写入 raw_result.json；
  → result.py process(raw_result) 计算指标
  → result.json
```

### 3.2 目录框架

```text
src/luban_meter/benchmark/inference/
├── common/
│   ├── client.py       # OpenAI-compatible 调用：chat、completion logprobs、tokenize
│   ├── dataset.py      # 本地 json/jsonl 加载、确定性采样、resolve_data_path 路径解析
│   ├── parameters.py   # 配置参数校验
│   ├── prompts.py      # Prompt 模板注册与渲染
│   ├── parsers.py      # 答案提取与归一化
│   ├── metrics.py      # Accuracy/EM/F1/ROUGE/Pass@k/PPL 计算
│   ├── choice.py       # 四选一题目通用采集流程（ppl/gen）
│   └── choice_result.py # 四选一题目通用指标聚合
├── scripts/            # 数据集离线准备脚本：prepare_ceval/prepare_cmmlu/prepare_gsm8k
├── data/               # 随包内置样例数据集 jsonl（ceval/cmmlu/gsm8k），相对路径未命中时回退到此
├── ceval/            benchmark.py + result.py + ceval.yaml      （已实现）
├── cmmlu/            benchmark.py + result.py + cmmlu.yaml      （已实现）
├── gsm8k/            benchmark.py + result.py + gsm8k.yaml      （已实现）
├── humaneval/        benchmark.py + result.py + humaneval.yaml  （规划）
├── squad/            benchmark.py + result.py + squad.yaml      （规划）
├── summarization/    benchmark.py + result.py + summarization.yaml（规划）
└── wikitext/         benchmark.py + result.py + wikitext.yaml   （规划）
```

多个数据集共用的逻辑放在 `inference/common/`；只服务单一场景的逻辑保留在对应
Benchmark 目录（例如 HumanEval 的沙箱执行）。这与 `generate/common/` 的划分
原则一致。数据集官方格式到本地 jsonl 的转换是一次性离线工作，由
`inference/scripts/` 下的准备脚本完成；Benchmark 运行时只读取本地数据集文件，
不下载数据。

## 4. 配置协议

配置与 `generate/serving-online/serving_online.yaml` 保持同一种风格：扁平
YAML 放在 Benchmark 目录内，文件名与 Benchmark 名称一致，运行时通过 `--config`
指向该文件，参数在 `benchmark.py` 内做类型、范围和组合校验。

以 `ceval.yaml` 为例：

```yaml
# OpenAI-compatible inference service.
service_url: http://127.0.0.1:8000
api_key: ""
request_timeout: 60

# Dataset loading and deterministic sampling.
# Relative paths resolve against CWD first, then bundled package data
# (luban_meter/benchmark/inference/data/), so the shipped defaults work
# from both a source checkout and a wheel install.
dataset_path: data/ceval/val.jsonl
split: val
max_samples: 200
shuffle: false
seed: 42
few_shot_path: data/ceval/dev.jsonl
few_shot: 5

# Evaluation protocol. eval_mode chooses how answers are obtained;
# prompt_format chooses chat or plain-text transport. ppl requires
# prompt_format=base: chat-template logprob scoring is not implemented.
eval_mode: ppl          # ppl | gen
prompt_format: base     # base | chat (ppl+chat is rejected)
prompt_version: ceval-v1

# Generation parameters for gen mode. Ignored by ppl mode.
max_tokens: 8
stop: ["\n"]

# Client-side safety cap for simultaneous requests.
max_concurrency: 8

# Fixed decoding semantics. Changing these values is rejected.
temperature: 0.0
```

| 字段 | 类型 | 含义 |
|---|---|---|
| `service_url` | str | 在线推理服务地址 |
| `api_key` | str | 可选 API Key |
| `request_timeout` | number | 单请求超时秒数 |
| `dataset_path` | str | 本地数据集文件（json/jsonl），不依赖第三方数据加载库；相对路径先按 CWD 解析，再回退到包内置数据目录 |
| `split` | str | 数据集分区，仅记录，不改变加载行为 |
| `max_samples` | int | 样本上限，确定性截断 |
| `shuffle` | bool | 是否先按 `seed` 打乱再截断，默认否 |
| `seed` | int | 采样随机种子 |
| `few_shot_path` | str | 示例来源文件（通常为 dev 分区），可选 |
| `few_shot` | int | few-shot 示例数量，0 表示 zero-shot |
| `eval_mode` | str | `ppl`、`gen` 或 `loss`（WikiText 专用） |
| `prompt_format` | str | `chat` 走 `/v1/chat/completions`；`base` 走 `/v1/completions` 纯文本。`ppl`/`loss` 仅允许 `base` |
| `prompt_version` | str | Prompt 模板版本号，写入结果元数据 |
| `max_tokens` | int | gen 模式最大生成 Token 数 |
| `stop` | list | gen 模式停止序列 |
| `max_concurrency` | int | 客户端并发安全上限 |
| `temperature` | float | 固定 0.0，保证可复现；改值会被拒绝 |

`eval_mode` 为 `ppl` 或 `loss` 时，服务必须支持 `/v1/completions` 的 `echo` 与
`logprobs`。脚本在采集前探测该能力，不支持时直接写出带原因的 `failed` 原始结果，
不做静默降级。

## 5. Prompt 模板与版本化

Prompt 分为两层：

| 层 | 内容 | 归属 |
|---|---|---|
| 任务内容层 | 指令、few-shot 示例、题干、选项、答案引导 | Benchmark 配置与 `common/prompts.py`，版本化管理 |
| 对话格式层 | `<\|start_header_id\|>`、`` 等模型特殊 Token | 由推理服务端按部署模型应用 |

Benchmark 客户端只发送结构化消息（chat 模式）或纯文本（base、ppl、loss 模式），
不感知任何模型特殊 Token；切换模型只需更换服务地址和模型名。

任务内容层的规则：

- 每个数据集模板有唯一版本号（如 `ceval-v1`），模板文本固化在代码中，不允许运行
  时改写；
- few-shot 示例来自固定文件、固定顺序，保证可复现；
- ppl 模式模板以答案引导结尾（如“答案：”），选项内容作为待评分续写文本；
- gen 模式模板必须显式约束答案格式（如“只回答 A/B/C/D”）；
- 渲染后的 Prompt、模板版本、few-shot 数、评测模式和解码参数记录在原始结果中，
  汇总字段进入结果元数据。

C-Eval chat 模式渲染示例：

```text
以下是中国关于计算机网络的单项选择题，请选出其中的正确答案。
问题：……
A. ……
B. ……
C. ……
D. ……
答案：
```

gen 模式答案提取规则（`common/parsers.py`）：

- 选择题：优先匹配“答案是 X”“选 X”等模式，否则取文本中首个 `A-D`；匹配不到记
  `parse_failed` 并判该样本错误；
- GSM8K：优先正则 `####\s*(数字)`，回退取文本最后一个数字，去千分位后数值比较；
- SQuAD：小写、去标点、去冠词、压缩空白；
- HumanEval：剥离 ```` ``` ```` 代码块。

## 6. 逐数据集计算流程与指标

### 6.1 C-Eval（选择题，P0）

样本字段：`{id, question, choices[4], answer, subject}`。

1. 渲染：指令 + dev 分区同学科 few-shot 示例 + 题目 + 选项 + “答案：”；
2. ppl 模式：对每个选项 X，拼接 `text = prompt + 选项续写文本`，调用
   `/v1/completions` 且 `echo=true`、`max_tokens=1`；用 `/tokenize` 分别得到
   `prompt` 与 `prompt + 续写` 的 Token 数 `p` 与 `f`，取返回 logprobs 序列中
   下标 `[p:f)` 的续写 Token 对数概率（明确排除末尾的 1 个生成 Token）求和
   并除以 `f - p`，得到该选项的平均对数概率 `s_X`；
3. 判定：`prediction = argmax s_X`，长度归一化避免偏向长选项；
4. gen 模式（回退）：生成后经 `parsers.extract_choice` 提取字母；
5. 逐样本：`correct = (prediction == answer)`。

指标：

```text
Accuracy = sum(correct) / N    # N 为参与判定的样本数
```

- 输出名：`metrics.task_view.ceval.accuracy`，单位 `ratio`，附 `count`；
- 同时按 `subject` 分组输出 `accuracy_by_subject`；
- gen 模式无法解析的输出不从分母剔除，单独计 `parse_failed` 并判错；
- 服务失败样本计 `service_failed`，不进入准确率分母，单独计数。

### 6.2 CMMLU（选择题，P0）

协议与 C-Eval 完全一致，仅数据集文件和学科集合不同，输出同样为
`accuracy` 与按学科分组结果。

### 6.3 GSM8K（数学推理，P0）

样本字段：`{question, answer}`，参考回答以 `#### 数字` 结尾。

1. 渲染：带完整推理链、以 `#### 数字` 结尾的标准 few-shot 示例 + 题目；
2. gen 模式生成，`max_tokens` 配置化；
3. 按第 5 节规则提取最终数字；
4. 逐样本：`correct = 提取值与参考数值相等`。

```text
Exact Match = sum(correct) / N    # 此处与 Accuracy 等价
```

输出名：`metrics.task_view.gsm8k.exact_match`，单位 `ratio`；`parse_failed`
单独计数。

### 6.4 HumanEval（代码生成，P0）

样本字段：`{task_id, prompt(函数前缀), test, entry_point}`。

1. 渲染：chat 模式将函数前缀放入用户消息；base 模式直接以前缀续写；
2. 生成并配置 `stop`，防止模型续写超出函数范围；
3. 剥离代码块后拼接 `prompt + completion + test + check(entry_point)`；
4. 在独立子进程中执行，超时或异常判失败并记录异常类型；
5. 每题采样 n 次（配置 `samples_per_task`），c 为通过次数。

```text
pass@k = 1 - C(n - c, k) / C(n, k)    # C 为组合数
```

- 一期主报 `Pass@1`（n=1 时退化为通过率均值），输出名
  `metrics.task_view.humaneval.pass_at_1`，单位 `ratio`；
- n > 1 时同时输出 `pass_at_k`、n 和 k；
- 沙箱逻辑只服务本 Benchmark，不下沉到 `common/`。

### 6.5 SQuAD（开放问答，P1）

样本字段：`{id, context, question, answers[]}`。

1. 渲染：指令 + 背景（超长截断）+ 问题 + “答案：”；
2. gen 模式短生成；
3. 归一化：小写、去标点、去冠词 a/an/the、压缩空白；
4. 指标：

```text
EM = 1（norm(pred) 等于任一 norm(ref)），否则 0
F1 = max over refs  2PR / (P + R)
     # P、R 为归一化后按空白分词的 Token 重叠精度与召回
```

输出名：`metrics.task_view.squad.exact_match`、`metrics.task_view.squad.token_f1`，
单位 `ratio`，全样本取均值。

### 6.6 LCSTS（中文摘要，P1）

样本字段：`{article, summary}`。

1. 渲染：摘要指令 + 正文；
2. gen 模式生成；
3. 分词：中文无自然空格，统一按字符切分（确定性基线，口径记录在元数据）；
4. 指标：

```text
ROUGE-N: n-gram 用 Counter 计数
         P = |overlap| / |pred_ngrams|
         R = |overlap| / |ref_ngrams|
         F1 = 2PR / (P + R)
ROUGE-L: 以 pred/ref 字符序列的 LCS 长度代替重叠数，同式计算 P、R、F1
```

输出名：`metrics.task_view.summarization.rouge_1`、`rouge_2`、`rouge_l`，单位
`ratio`，全样本取均值。

### 6.7 WikiText（语言建模，P1）

1. 语料按滑窗切片，窗口长度和步长可配置；
2. 每块调用 `/v1/completions` 且 `echo + logprobs`，只累计步长之后 Token 的对数
   概率（前面的 Token 仅作为上下文）；
3. 指标：

```text
mean_loss = -sum(log p_i) / N    # 单位 nats/token
Perplexity = exp(mean_loss)
```

输出名：`metrics.task_view.wikitext.mean_loss`（单位 `nats/token`）、
`metrics.task_view.wikitext.perplexity`（单位 `ppl`），附参与计分的 Token 数
`count`。

必须依赖在线 logprobs 链路；服务缺失时写出带原因的 `failed` 结果，不用客户端
估算值替代。

## 7. 原始结果与最终结果协议

### 7.1 原始结果（raw_result.json）

沿用 `luban-meter.raw/v1` 结构：`schema_version`、`status`、`metrics`、
`metadata`、`artifacts`、`error`。`metrics.samples[]` 逐样本记录：

| 字段 | 含义 |
|---|---|
| `id` / `subject` | 样本标识与学科/题型 |
| `mode` / `prompt_version` / `prompt_format` | 评测模式与 Prompt 上下文 |
| `prompt` | 任务内容层渲染结果 |
| `choice_scores` | ppl 模式下各选项平均对数概率 |
| `raw_output` / `parsed_output` | gen 模式原始输出与解析结果 |
| `prediction` / `reference` / `correct` | 预测值、参考答案与判定 |
| `latency_ms` / `input_tokens` / `output_tokens` | 请求耗时与 Token 数 |
| `status` | `success`、`parse_failed`、`service_failed` |

`metrics.counts` 汇总 total、correct、parse_failed、service_failed。
`metadata` 记录 `measurement`、数据集名称、split、样本数、Prompt 版本、
few-shot 数、评测模式、`prompt_format`、解码参数和评分器版本。

### 7.2 最终结果（result.json）

`result.py` 校验原始结果后按第 6 节公式聚合，输出组织在
`metrics.task_view.<benchmark>` 下，每个指标带 `value`、`unit`、`count`；
`metadata` 继承原始元数据并补充聚合计数。全部服务失败时结果状态为 `failed`，
部分失败为 `partial_failed`，并携带 `error` 信息。

## 8. 评测模式边界

| 模式 | 接口 | 服务能力要求 | 适用任务 | 说明 |
|---|---|---|---|---|
| ppl | `/v1/completions` + `echo` + `logprobs` | 必须支持 prompt logprob 回显 | 选择题（C-Eval、CMMLU） | 按选项续写 Token 对数概率取均值归一化；仅允许 `prompt_format=base` |
| gen | `/v1/chat/completions` 或 `/v1/completions` | 生成即可 | GSM8K、HumanEval、SQuAD、LCSTS | 依赖答案提取规则 |
| loss | `/v1/completions` + `echo` + `logprobs` | 必须支持 prompt logprob 回显 | WikiText | 不能用生成模式替代 |

`eval_mode` 与 `prompt_format`（chat/base）按配置组合；不同组合得到的分数不能
混合比较。其中 ppl 与 loss 模式基于 `echo` 回显的整段 `prompt+continuation`
Token 序列做按偏移切片打分，对话格式层会注入特殊 Token 并改变 Token 边界，
因此 ppl / loss 仅允许 `prompt_format=base`，组合 ppl + chat（或 loss + chat）
会被 `validate_choice_parameters` 校验拒绝。gen 模式不依赖 Token 切片，可自由
选择 chat 或 base。对话格式层的特殊 Token 由推理服务端按部署模型应用，
Benchmark 客户端不感知。

## 9. 结果解释要求

发布或比较 inference 指标时至少同时记录：

- Benchmark 名称、数据集名称/版本、split、样本数；
- 模型名称和版本、推理引擎及版本；
- `eval_mode`、`prompt_format`、Prompt 模板版本、few-shot 数；
- 解码参数（temperature、max_tokens、stop）；
- correct、parse_failed、service_failed 计数；
- 指标路径、单位和样本数。

禁止只写“C-Eval 得分 0.8”而不说明它是 ppl 模式还是 gen 抽取模式、few-shot
数量、Prompt 版本和样本数。

## 10. 当前实现范围与规划边界

当前已实现：

1. `inference/common/` 公共层（client、dataset、parameters、prompts、parsers、
   metrics、choice、choice_result）与单元测试；
2. `ceval` 端到端（ppl + gen 双路径）；
3. `cmmlu`、`gsm8k` 端到端；
4. 数据集离线准备脚本 `inference/scripts/`（prepare_ceval、prepare_cmmlu、
   prepare_gsm8k）；
5. 随包内置样例数据集 `inference/data/`（ceval/cmmlu/gsm8k 的 jsonl），并由
   `dataset.resolve_data_path()` 提供 CWD → 包内置的相对路径回退解析，使默认
   配置从任意工作目录开箱即用。

尚未实现，后续按以下顺序建设：

1. `humaneval`（Pass@1 + 沙箱执行，完成 P0）；
2. `squad`、`summarization`、`wikitext`（P1）；
3. Suite 编排与跨运行汇总报告。

其中 Token F1、ROUGE、Pass@k、Perplexity 的指标计算能力已在 `common/metrics.py`
中具备，对应数据集评测待接入。

当前尚未实现或不应从现有字段推断：

- 在线数据集加载、HF datasets 依赖和自动下载；
- Judge 模型打分与语义相似度指标；
- 跨数据集混合指标；
- logprobs 缺失时的客户端估算替代；
- 幻觉、事实一致性、安全拒答和 Prompt Injection 等后续阶段评测。



