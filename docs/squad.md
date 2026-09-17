# SQuAD 2.0 生成式评测

基线源码：用户提供的 upstream/master `f53d1c9`。新增 benchmark 名称为 `squad`。
无需安装 lm-evaluation-harness、OpenCompass 或 Hugging Face datasets。

## 数据准备

从 https://rajpurkar.github.io/SQuAD-explorer/ 下载 Dev Set v2.0，保留数据集许可
CC BY-SA 4.0。本补丁不包含官方数据集。以下命令在仓库根目录执行，需 Python 3.12+：

```bash
python -m pip install -e '.[dev]'
python -m luban_meter.benchmark.inference.scripts.prepare_squad \
  --source /你的路径/dev-v2.0.json --out data/squad/validation.jsonl
```

也可以直接把配置中的 dataset_path 改成官方 dev-v2.0.json 的本地路径。
加载器检查版本、重复 ID、字段类型及无答案标签与参考答案的一致性。
split 是结果标签，不会自动选择文件内分区。

## 运行

复制配置后设置 service_url 和 dataset_path。先设置 max_samples: 20 做冒烟测试，
验证成功后改成 null 跑全部文件。默认配置使用完整数据；不内置数据自动下载。

```bash
cp src/luban_meter/benchmark/inference/squad/squad.yaml squad-local.yaml
# 编辑 squad-local.yaml 中的服务地址、数据路径和样本数量
luban-meter run --module inference --benchmark squad \
  --config squad-local.yaml --model-name 你的服务模型名 --output runs
```

chat 对应 /v1/chat/completions，base 对应 /v1/completions。使用服务支持的模式。
不需要客户端加载模型权重或 tokenizer；硬件相关逻辑继续由原有服务和框架承担。
推理模型应通过服务端设置关闭思考或确保 content 只返回最终答案；本协议不自动
剥离思考文本。全文输入不做客户端截断，服务端也应关闭静默截断。

## 结果解释

raw_result.json 保存原始请求与预测；result.json 保存 EM、Token F1、有答案和
无答案分组得分、失败计数。框架会生成 Markdown/CSV/图表报告。

分数为 0–1（官方脚本报告 0–100，比较时乘以 100）。无答案仅通过精确的
unanswerable 标记转为空字符串；空响应为解析失败，不算正确拒答。
服务错误和解析失败均按 0 分保留在分母，并记录 failure_policy。因此有服务失败
时整体分数包含运行可靠性影响，不能当作无失败的纯模型能力分数比较。

## 参考实现与明确差异

- Harness：lm_eval/tasks/squadv2/task.py。参考其多答案、EM/F1、HasAns/NoAns
  指标组织；不复现 generate_until + loglikelihood 的概率协议。
- OpenCompass：datasets/squad20.py 与 configs/datasets/squad20/
  squad20_gen_1710bc.py。参考 zero-shot 生成和显式拒答；不使用 plausible_answers
  作为 gold，不沿用通用标点截断和自定义 score。
- 官方：https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/evaluate-v2.0.py
  对齐 normalize_answer、compute_exact、compute_f1、get_raw_scores。
- 保留现有 common.normalize_answer 的行为；SQuAD 专用标准化独立实现，避免改变
  其他评测。没有新增第三方运行依赖。

## 验证

```bash
python -m pytest -q tests/test_inference_squad.py
python -m ruff check --select E,W,F,I --line-length 79 \
  src/luban_meter/benchmark/inference/squad \
  src/luban_meter/benchmark/inference/scripts/prepare_squad.py \
  tests/test_inference_squad.py tests/verify_squad_official.py
```

独立官方评分对照（需本地官方脚本及其 numpy 依赖）：

```bash
PYTHONPATH=src python tests/verify_squad_official.py /路径/evaluate-v2.0.py
```

开发时 1,000 组固定 seed 的预测/多参考答案对照全部一致，包含空答案、重复词、
标点、冠词、数字及 Unicode 边界。使用的官方脚本 SHA256：
`710840ce2c2c61716334c056777032f24660b83eb9869a451036e6a24d8103c4`。
真实模型服务与官方全量验证集运行需在用户环境完成；模拟服务测试不代表模型效果。
