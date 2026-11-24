# 性能曲线与稳定性趋势的架构

## 正式请求到离线报告


正式请求包装记录使用 requests[].response 中的毫秒偏移、延迟、状态和 Token 计数；扁平记录直接读取这些字段。外层调度秒偏移不替代客户端操作时间。有原始请求但全部完成时间无法归属时，各窗输出吞吐为 null 并附原因；可确定的空窗吞吐为零。
runtime/evaluation_session.py 将正式采集的 requests、parameters 和 window.duration_s 保存到任务 raw/<case>.json。benchmarking/benchmark_metrics.py 中的 StabilityTrends 声明 client_stream_timing 能力并调用 generation_performance/common/time_windows.py。窗口参数依次取自 parameters、metadata、默认 60。Prefix Cache 沿用自己的默认指标列表。

case.metrics.stability_trends.values 包含 window_seconds 与 windows。每行保存 index、start_s、end_s、span_s、complete；started_requests、completed_requests、successful、failed、timeout_failed、other_failed、average_concurrency、peak_concurrency 和各速率采用带 value/unit 的对象。ttft_ms 保存 mean/p99/count，e2el_ms 保存 p99/count；小样本标志位于对应延迟对象。metadata 包含归属规则、超时集合、分母、预热排除、窗口长度及 unattributed_requests 的计数和最多十个请求索引。

对有效正式时长 D 和窗口 W 建立 ceil(D/W) 个窗口。内部边界属于右侧，恰好结束于 D 的请求计入末窗。有效非负延迟按完成时间选样本；P99 使用排序后 (n−1)×0.99 位置线性插值。延迟和吞吐保留三位小数，比率保留四位。失败请求按 error.type 精确匹配 ConnectTimeout、ReadTimeout、WriteTimeout、PoolTimeout、TimeoutException、TimeoutError，其他失败归 other_failed。无成功产出时吞吐可为零，缺失成功 Token 计数时为 null。无有效延迟返回 count=0 和 null，无完成请求时两项比率为 null。无效正式时长返回 unavailable 和空窗口列表。

reporting/trends.js 直接读取窗口，在 Case 内切换指标。reporting/report_overview.js 使用 servingMetric 选择 Case 指标提供者，分别读取 service_view.output_token_throughput.value、request_view.ttft.mean/p99、request_view.e2el.p99。constant、poisson、gamma 以及已保存的旧 rate 结果使用有限计划速率作为速率轴，其他轴见 README。Case 失败或采集不完整不参与性能峰值。稳定性窗口的失败计数仍按原始正式请求计算。SLO 和 Goodput 保留各自的阈值与分母。

在线指标可以独立选择。各计算器共用正式请求与 Case 窗口；终端和 HTML 按所选指标合并
已保存的 request_view、service_view，并为负载曲线提供相应的时延或吞吐字段。
显式 metrics 列表可同时选择 stability_trends 和 step Profile 的 capacity_search。
需要整轮 Token 吞吐时选择 total_token_throughput 或包含该值的完整性能指标组。

## 显式 Auto 窗口

trend_window_seconds 继续默认 60，正整数为 Fixed，字符串 auto 为显式 Auto。
TrendWindowParameter 在场景参数声明中兼容 CLI 文本和 JSON 整数，
不修改通用规划器、调度器或 max_duration。
Metric 解包正式 response 后直接在内存中选窗，完成有效性判定复用固定归桶逻辑。
候选为 1、2、5、10、15、30、60、120 秒，长时按 120 连续倍增。
仅评估 2–10 个完整窗的候选，完整窗完成数至少为 10；
已存在有效延迟样本的字段，其每个完整窗样本数也至少为 10。
优先 5–10 窗、接近 7 窗、无极短尾窗、最小样本数较大的候选，
最后以窗口秒数确定平局。无合格候选时回退单窗摘要。
字段全缺失或全失败时，仍可按完成数选择结果统计粒度，延迟保持 null。
选定窗口后复用同一计算器计算全部值，P99 达到 10 样本不代表统计可靠性已获证明。
metadata 保存 window_mode、requested_window_seconds 和 window_selection，
包括算法版本、候选完整窗样本数、fallback 和选择原因；values.window_seconds 始终为整数。
旧结果缺模式时按历史 Fixed 展示，Fixed 数值与公式保持原有行为。
运行日志沿用统一 task/case/metrics 上下文，记录实际模式、秒数与选择原因。

离线验收派生结果保留原始 Case.parameters，并使用独立的分析 id。
可选顶层 analysis_provenance 记录 source_run_id、来源文件 SHA256、
代码校验信息及每个 Case 的原始/请求/实际窗口；HTML 仅在存在该对象时
显示来源说明。普通运行和历史结果不需要新增此字段，不改变 Metric 结构。

## 验收时需单独核对的边界

完成偏移恰等于正式终点时计入末窗，越界、无效完成时间及起止倒置按原因计数。metadata.coverage 分别记录完成与并发覆盖；只有完成时间的请求可计入完成统计，并发只统计具有有效起止时间的子集。并发占用采用半开区间，零时长请求不贡献峰值。原始记录保留。旧 JSON 缺覆盖计数时报告显示未知。窗口时间字段按毫秒精度输出，复算使用 raw 中的正式时长。

## 手动运行的任务ID

规划器解析任务级`id`和CLI的`--task-id`，按`name`选择任务，并校验整份配置的ID。
`plan`输出显式ID。`run`对缺少ID的选中任务生成UUID，在启动子进程前把本次任务作为
一个事务登记到结果根目录的`task-index.sqlite3`。`tasks`表以`id`为主键，记录
`run_id`、`task_name`和`run_dir`，以及任务结果、事件与实际实例日志的绝对路径
`result_path`、`events_path`和`log_path`。若存在重复ID，事务回滚，本次任务均不启动。
对只差大小写的任务名，在实际输出文件系统中试建目录，确保各Task产物路径彼此独立。

协调器把同一个`id`写入运行的`plan.json`、结果中的`tasks[]`和每个任务的`result.json`，
并由主进程和任务子进程记录到统一日志。索引给出产物的位置，文件随执行生成。

`delete`按ID读取登记。`runs`表记录手动CLI写完运行产物的时间，兼容旧索引的删除用
整轮`finished_at`和已完成报告或实例日志核对。运行的产物全部消失时，旧登记可标记为
已删除。结果根目录下的`.task-delete-locks/`按Run串行化删除。读取登记和写入
`tasks.deleted_at`使用短SQLite事务，共享报告生成与文件清理在事务之外进行。
出错时可使用相同ID重试。Run级日志由原运行目录或外置日志目录继续保存。
