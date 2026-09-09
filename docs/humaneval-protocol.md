# HumanEval pass@1 协议

本文档定义 LuBan-Meter 第一版 HumanEval 评测协议。任何可能改变评测分数的
变更都必须引入新的协议版本。

## 范围与标识符

- 基准测试：`inference/humaneval`
- 评测协议：`humaneval-completion-v1`
- 沙箱协议：`humaneval-sandbox-v1`
- 评分器：`humaneval-scorer-v1`
- 传输接口：兼容 OpenAI 的 `/v1/completions`
- 指标：`pass_at_1`，单位为 `ratio`

第一版有意限制为每个任务仅生成一个确定性补全结果。Chat 提示词和多样本
pass@k 将作为独立的后续协议实现。

## 数据集约定

基准测试只读取本地 JSON 或 JSONL 文件。官方包含 164 个任务的 HumanEval
数据集内置于
`benchmark/inference/data/humaneval/HumanEval.jsonl`。该数据集来源于
OpenAI human-eval 仓库，采用 MIT 许可证。内置 JSONL 文件的 SHA-256 为
`b2adeeae8b383b6f0c615746a397f25c8abe44da4afb29cd10e6c4bbf0463fd3`。

每条记录必须包含：

```json
{
  "task_id": "HumanEval/0",
  "prompt": "函数签名和文档字符串",
  "canonical_solution": "参考补全代码",
  "test": "定义 check(candidate) 的测试程序",
  "entry_point": "函数名称"
}
```

`canonical_solution` 在运行时是可选字段，并且绝不会发送给模型。
`test` 和 `entry_point` 同样不会发送给模型。

只有在替换内置数据集时，才需要使用离线转换脚本：

```bash
python src/luban_meter/benchmark/inference/scripts/prepare_humaneval.py \
  --source /path/to/HumanEval.jsonl.gz \
  --out data/humaneval/HumanEval.jsonl
```

结果元数据会记录所选数据集文件的 SHA-256 摘要。

## 提示词与输出约定

数据集中的原始 `prompt` 会被完整发送到 `/v1/completions` 接口。模型必须
只返回位于该提示词之后的续写内容。解析器执行以下处理：

1. 统一换行符格式；
2. 删除模型输出中完全重复的提示词；
3. 删除开头的空行，同时保留代码缩进；
4. 保留 Markdown 代码围栏及其他所有生成内容，使沙箱能够判断模型真实生成的
   基础补全结果；
5. 不尝试猜测如何将另一个完整函数改写为函数体。

最终可执行程序按以下方式构造：

```python
program = prompt + completion + test + f"check({entry_point})"
```

## 仅允许 Docker 执行

禁止在宿主机上执行模型生成的代码。基准测试在调用模型前执行 Docker 预检查。
如果 Docker、seccomp、cgroups 或沙箱镜像不可用，则评测按安全失败原则立即
终止。

每个候选答案都会使用一个全新的容器，并满足以下限制：

- 禁止网络访问；
- 不挂载宿主机目录，也不挂载 Docker socket；
- 根文件系统只读，并使用容量受限的临时文件系统；
- 删除全部 Linux capabilities，并启用 `no-new-privileges`；
- 使用非 root 用户 UID/GID 65534；
- 限制 CPU、内存、PID、文件大小和墙钟执行时间；
- 限制 stdout 和 stderr 的输出大小；
- 禁止自动拉取镜像。

Docker host 可以通过配置指定，因此生产服务器能够使用位于 `/data` 目录下的
专用 Docker daemon，而不需要修改基准测试代码。

构建固定版本的沙箱镜像：

```bash
docker -H unix:///path/to/docker.sock build \
  --pull=false \
  -f src/luban_meter/benchmark/inference/humaneval/Containerfile \
  -t luban-meter-humaneval-sandbox:v1 \
  src/luban_meter/benchmark/inference/humaneval
```

## 候选答案状态

以下模型输出结果会被计为失败候选：

- `parse_failed`
- `failed_test`
- `syntax_error`
- `runtime_error`
- `timeout`

以下基础设施异常同样会被计为失败候选，并使整个运行状态变为
`partial_failed` 或 `failed`：

- `service_failed`
- `sandbox_failed`

任何失败都不会被静默地从 pass@1 的分母中移除。

## 指标计算

对于每个任务，`n=1`，`c` 的取值只能为 0 或 1：

```text
pass@1 = 1 - C(n-c, 1) / C(n, 1)
```

最终报告的分数是所有选中任务 ID 对应分数的平均值。结果写入以下字段：

```text
metrics.task_view.humaneval.pass_at_1
```

原始记录会保留 `sample_id=0`。这样，未来发布多样本版本时，可以按照任务对
候选答案进行分组，而不需要修改评测产物的标识模型。

## 本地测试

```bash
pytest -q tests/test_inference_humaneval.py tests/test_humaneval_executor.py
```

真实 Docker 集成测试默认不执行，需要通过环境变量显式启用：

```bash
LUBAN_METER_RUN_DOCKER_TESTS=1 \
LUBAN_METER_DOCKER_HOST=unix:///path/to/docker.sock \
LUBAN_METER_HUMANEVAL_IMAGE=luban-meter-humaneval-sandbox:v1 \
pytest -q tests/test_humaneval_executor.py
```

## 后续 pass@k 扩展

下一版本协议将增加 `samples_per_task > 1`、明确的采样参数、每个候选答案对应的
稳定 `sample_id`，以及可配置的 `pass_k` 列表。后续版本不得直接修改当前
pass@1 协议的行为。
