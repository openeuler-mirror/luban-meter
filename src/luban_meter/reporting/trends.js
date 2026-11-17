// Windowed stability trends read saved metric facts only; nothing is
// recomputed here.  Null values break the curve instead of reading as
// zero, partial windows and small-sample P99 stay visibly marked.
function stabilityMetric(c) {
  const metric = c?.metrics?.stability_trends;
  return metric && typeof metric === 'object' ? metric : null;
}
function stabilityWindows(c) {
  const windows = stabilityMetric(c)?.values?.windows;
  if (!Array.isArray(windows) || !windows.length ||
    windows.some((w,index) => !numeric(w?.start_s) || !numeric(w?.end_s) ||
      w.start_s < 0 || w.start_s >= w.end_s ||
      (index > 0 && w.start_s < windows[index-1].end_s))) return null;
  return windows;
}
function trendNode(source, key) {
  if (!source || typeof source !== 'object')
    return {value: null, count: null, low: false, reason: '窗口字段缺失'};
  const value = key === undefined ? source.value : source[key];
  return {
    value: numeric(value) && value >= 0 ? value : null,
    count: numeric(source.count) ? source.count : null,
    low: source.low_sample === true,
    reason: source.reason || null,
  };
}
const trendSpecs = [
  {key: 'output_token_throughput', label: '输出Token吞吐量',
    unit: 'token/s', zeroBased: true,
    get: w => trendNode(w.output_token_throughput)},
  {key: 'ttft_mean', label: 'TTFT Mean', unit: 'ms',
    get: w => trendNode(w.ttft_ms, 'mean')},
  {key: 'ttft_p99', label: 'TTFT P99', unit: 'ms',
    get: w => trendNode(w.ttft_ms, 'p99')},
  {key: 'e2el_p99', label: 'E2EL P99', unit: 'ms',
    get: w => trendNode(w.e2el_ms, 'p99')},
  {key: 'success_rate', label: '请求成功率', unit: 'ratio', zeroBased: true,
    get: w => trendNode(w.success_rate)},
  {key: 'timeout_rate', label: '请求超时率', unit: 'ratio', zeroBased: true,
    get: w => trendNode(w.timeout_rate)},
];
function trendWarnings(w, spec) {
  const warns = [], p = spec.get(w);
  if (w.complete === false)
    warns.push('partial 窗口（实际 ' + valueText(w.span_s) + 's）');
  if (p.reason) warns.push(p.reason);
  if (p.value === null && !p.reason && w.completed_requests?.value)
    warns.push('指标缺失或不可计算');
  for (const [name, node] of [['TTFT', w.ttft_ms], ['E2EL', w.e2el_ms]]) {
    if (node?.low_sample !== true) continue;
    const warn = name + ' 延迟样本 ' + valueText(node.count) + ' <10';
    if (!warns.includes(warn)) warns.push(warn);
  }
  for (const key of ['success_rate', 'timeout_rate']) {
    const reason = w[key]?.reason;
    if (reason && !warns.includes(reason)) warns.push(reason);
  }
  return warns;
}
function trendChart(parent, windows, spec, meta, finalEnd = windows.at(-1)?.end_s) {
  const box = el('div', undefined, 'chart');
  const points = windows.map((w, i) => ({i, w, p: spec.get(w)}));
  const finite = points.filter(pt => numeric(pt.p.value));
  if (!finite.length) {
    box.append(el('p', '「' + spec.label + '」没有可绘制的保存值：' +
      '全部窗口为 null 或缺失。null 造成断点，不会按 0 连接。',
      'analysis-note'));
    parent.append(box);
    return;
  }
  const width = 760, height = 290, left = 75, right = 28, top = 35, bottom = 60;
  const xmin = windows[0].start_s, xmax = windows.at(-1).end_s;
  const centers = points.map(pt => pt.w.start_s + (pt.w.end_s - pt.w.start_s) / 2);
  const x = v => xmin === xmax ? (left + width - right) / 2 :
    left + (v - xmin) / (xmax - xmin) * (width - left - right);
  const values = finite.map(pt => pt.p.value);
  let lo = Math.min(...values), hi = Math.max(...values);
  if (spec.zeroBased) lo = Math.min(lo, 0);
  if (lo === hi) {
    lo -= 1; hi += 1;
    if (lo === hi) lo = 0;
  }
  const y = v => height - bottom - ((v - lo) / (hi - lo)) * (height - top - bottom);
  const svg = svgElement('svg', {viewBox: '0 0 ' + width + ' ' + height,
    role: 'img', 'aria-label': spec.label + ' 分时趋势；横轴为正式测试经过时间'});
  const title = svgElement('title', {});
  title.textContent = spec.label + '（' + displayUnit(spec.unit) +
    '）：横轴为正式测试经过时间（秒），纵轴为保存的窗口值；null 断点不连接';
  svg.append(title);
  for (let i = 0; i <= 4; i++) {
    const v = lo + (hi - lo) * (i / 4), py = y(v);
    svg.append(svgElement('line', {x1: left, x2: width - right, y1: py,
      y2: py, stroke: 'var(--line)'}));
    const label = svgElement('text', {x: left - 9, y: py + 4,
      'text-anchor': 'end', fill: 'currentColor', 'font-size': 12});
    label.textContent = valueText(v, spec.unit);
    svg.append(label);
  }
  let segment = [];
  const flush = () => {
    if (segment.length > 1) svg.append(svgElement('polyline', {
      points: segment.map(pt => x(centers[pt.i]) + ',' + y(pt.p.value)).join(' '),
      fill: 'none', stroke: 'var(--accent)', 'stroke-width': 2}));
    segment = [];
  };
  points.forEach(pt => {
    if (!numeric(pt.p.value) || pt.w.complete !== true) { flush(); return; }
    segment.push(pt);
  });
  flush();
  points.forEach(pt => {
    if (!numeric(pt.p.value)) return;
    const partial = pt.w.complete === false, low = pt.p.low;
    const cx = x(centers[pt.i]), cy = y(pt.p.value);
    const fill = low ? 'var(--panel)' : 'var(--accent)';
    const mark = partial
      ? svgElement('rect', {x: cx - 4, y: cy - 4, width: 8, height: 8,
        fill, stroke: 'var(--accent)', 'stroke-width': 2})
      : svgElement('circle', {cx, cy, r: 4, fill,
        stroke: 'var(--accent)', 'stroke-width': 2});
    const tip = svgElement('title', {});
    tip.textContent = '窗口 ' + valueText(pt.w.index) + ' [' + pt.w.start_s + ', ' +
      pt.w.end_s + (pt.w.end_s === finalEnd ? ']' : ')') + ' s · ' + spec.label + ' ' +
      valueText(pt.p.value, spec.unit) + ' ' + displayUnit(spec.unit) +
      ' · 完成 ' + valueText(pt.w.completed_requests?.value) +
      ' · 开始 ' + valueText(pt.w.started_requests?.value) +
      (partial ? ' · partial 窗口（实际 ' + valueText(pt.w.span_s) + 's）' : '') +
      (low ? ' · 延迟样本 ' + (pt.p.count ?? '未知') +
        ' <10，P99 仅供参考' : '') +
      (pt.p.reason ? ' · ' + pt.p.reason : '');
    mark.append(tip);
    svg.append(mark);
  });
  [xmin, xmin + (xmax - xmin) / 2, xmax].forEach(t => {
    const label = svgElement('text', {x: x(t), y: height - bottom + 23,
      'text-anchor': 'middle', fill: 'currentColor', 'font-size': 12});
    label.textContent = valueText(t);
    svg.append(label);
  });
  [['正式测试经过时间（s）', (left + width - right) / 2, height - 8,
    'middle'],
   [spec.label + '（' + displayUnit(spec.unit) + '）', left, 18, 'start']]
    .forEach(([text, px, py, anchor]) => {
      const label = svgElement('text', {x: px, y: py, 'text-anchor': anchor,
        fill: 'currentColor', 'font-size': 12});
      label.textContent = text;
      svg.append(label);
    });
  box.append(svg);
  parent.append(box, el('p', '实线只连接存在保存值的完整窗口，null 断开不补零；' +
    'partial 窗口以独立刻度方形散点展示；空心标记为延迟样本少于 ' +
    valueText(meta?.low_sample_threshold ?? 10) + ' 的小样本窗口，其 P99 仅供参考。',
    'analysis-note'));
}
function trendTable(parent, windows, spec) {
  const details = el('details');
  details.append(el('summary', '分时窗口明细与警告'));
  dataTable(details, [
    {title: '窗口', numeric: true, get: r => r.index},
    {title: '开始(s)', numeric: true, get: r => r.start_s},
    {title: '结束(s)', numeric: true, get: r => r.end_s},
    {title: '实际长度(s)', numeric: true, get: r => r.span_s},
    {title: '完整窗口', get: r => r.complete === false ? '否（partial）' :
      r.complete === true ? '是' : '未知'},
    {title: '开始请求', numeric: true, get: r => r.started_requests?.value},
    {title: '完成请求', numeric: true, get: r => r.completed_requests?.value},
    {title: '成功', numeric: true, get: r => r.successful?.value},
    {title: '失败', numeric: true, get: r => r.failed?.value},
    {title: '超时失败', numeric: true, get: r => r.timeout_failed?.value},
    {title: spec.label + '（' + displayUnit(spec.unit) + '）', numeric: true,
      unit: spec.unit, get: r => spec.get(r).value},
    {title: 'TTFT 有效样本', numeric: true, get: r => r.ttft_ms?.count},
    {title: 'E2EL 有效样本', numeric: true, get: r => r.e2el_ms?.count},
    {title: '警告', get: r => trendWarnings(r, spec).join('；') || '无'},
  ], windows);
  parent.append(details);
}
function trendCaliber(parent, metric) {
  const meta = metric.metadata || {};
  const details = el('details');
  details.append(el('summary', '统计口径与窗口定义'));
  [
    ['统计窗口长度', valueText(metric.values?.window_seconds ??
      meta.window_seconds) + ' s（参数 trend_window_seconds）'],
    ['窗口模式', meta.window_mode==='auto'?'Auto（从正式请求重新归桶）':
      meta.window_mode==='fixed'?'Fixed（配置或兼容默认值）':'历史 Fixed（未记录模式）'],
    ['选窗依据', meta.window_selection?.reason || '历史结果未保存选窗依据'],
    ['时间轴', '正式测试经过时间：自 Case 正式采集起点起的偏移秒，' +
      '与请求 start/end 偏移同轴'],
    ['预热请求', meta.warmup_included === false ?
      '不参与统计（仅正式请求）' : meta.warmup_included===true?'参与统计':'未知'],
    ['开始归属', 'started_requests 按 start_offset_ms'],
    ['完成归属', '完成数、成功/失败/超时、TTFT/E2EL 与输出 Token 按 ' +
      'end_offset_ms'],
    ['并发', '平均并发按请求起止区间与窗口重叠积分；峰值并发为扫线峰值'],
    ['吞吐量公式', meta.throughput_definition ||
      '窗口内完成的成功请求输出 Token / 窗口实际持续时间'],
    ['completion-attributed 含义', '吞吐按请求完成时间归属，' +
      '不是逐 Token 瞬时生成速率，不能读作实时生成速度'],
    ['成功率/超时率分母', meta.rate_denominator || '窗口内完成的请求数'],
    ['超时错误类型', Array.isArray(meta.timeout_error_types)?
      meta.timeout_error_types.join('、') || '未保存':'未保存'],
    ['P99 小样本阈值', '延迟样本 < ' + valueText(meta.low_sample_threshold) +
      ' 标记 low_sample，P99 仅供参考'],
    ['窗口完整性', 'complete=false 表示实际窗口长度不足一个统计窗口'],
    ['缺失值', 'null 表示不可计算并造成曲线断点；0 为真实零值'],
  ].forEach(([key, value]) =>
    details.append(el('p', key + '：' + value, 'analysis-note')));
  const un = meta.unattributed_requests;
  if (Number.isSafeInteger(un?.count) && un.count>0) details.append(el('p', '不可归属请求：' + un.count +
    '（' + (un.reason || '') + '）', 'analysis-note'));
  const raw = el('details');
  raw.append(el('summary', '保存的 metadata 原始值'),
    el('pre', JSON.stringify(meta, null, 2)));
  details.append(raw);
  parent.append(details);
}
function renderStabilityCase(parent, c, withTitle = true) {
  const metric = stabilityMetric(c);
  if (!metric) return;
  const section = el('div', undefined, 'analysis-group');
  if (withTitle) section.append(el('h3', '稳定性趋势（分时窗口）'));
  const windows = stabilityWindows(c);
  if (!windows) {
    const message = metric.status === 'unavailable' ?
      '分时统计不可用（' + (metric.reason || '原因未保存') + '）。' :
      'stability_trends 结构异常或缺少窗口数组，请检查窗口起止时间。';
    section.append(el('p', message +
      '不绘制分时曲线；请查看运行诊断中的原始指标。', 'analysis-note'));
    parent.append(section);
    return;
  }
  if (metric.status && metric.status !== 'success')
    section.append(el('p', '指标状态：' + metric.status +
      (metric.reason ? ' · ' + metric.reason : ''), 'analysis-note'));
  const un = metric.metadata?.unattributed_requests;
  const coverage = metric.metadata?.coverage;
  const completeWindows = windows.filter(window => window.complete === true);
  const partialWindows = windows.filter(window => window.complete !== true);
  const summary = el('div', undefined, 'overview-metrics');
  for (const [label, value] of [
    ['当前 Case', c.id || '未知'],
    ['窗口模式', metric.metadata?.window_mode==='auto'?'Auto':
      metric.metadata?.window_mode==='fixed'?'Fixed':'历史 Fixed'],
    ['正式测试时长', valueText(metric.metadata?.formal_duration_s) + ' s'],
    ['统计窗口', valueText(metric.values?.window_seconds ?? metric.metadata?.window_seconds) + ' s'],
    ['完整 / partial 窗口', completeWindows.length + ' / ' +
      windows.filter(window => window.complete === false).length],
  ]) {
    const field = el('div', undefined, 'overview-metric');
    field.append(el('div', label, 'muted'), el('div', value));
    summary.append(field);
  }
  section.append(summary);
  if(metric.metadata?.window_mode==='auto'){
    const selection=metric.metadata.window_selection||{};
    section.append(el('p','Auto 实际采用 '+valueText(metric.values?.window_seconds)+
      ' s；'+(selection.fallback?
        '样本数量或时间分布不足，保留单窗口摘要。':'依据正式请求分布和有效样本数选择。')+
      ' 不同 Case 可能采用不同粒度，请分别核对；P99 样本数不代表可靠性结论。','analysis-note'));
  }
  if (Number.isSafeInteger(coverage?.total_requests) &&
      Number.isSafeInteger(coverage?.completion_eligible) &&
      Number.isSafeInteger(coverage?.concurrency_eligible)) {
    section.append(el('p','完成统计覆盖 '+coverage.completion_eligible+'/'+coverage.total_requests+
      '；并发统计覆盖 '+coverage.concurrency_eligible+'/'+coverage.total_requests+
      (coverage.concurrency_eligible<coverage.total_requests?
        '。并发覆盖警告：平均及峰值仅基于起止区间有效的请求。':'。'),'analysis-note'));
  } else section.append(el('p','完成及并发统计的完整覆盖计数未知。','analysis-note'));
  if (Number.isSafeInteger(un?.count) && un.count > 0)
    section.append(el('p', '统计覆盖警告：' + un.count +
      ' 条请求因时间缺失、矛盾或越界无法归入窗口（' + (un.reason || '') +
      '）；分时指标仅基于可归属请求，成功率与超时率的分母同样只含' +
      '可归属的完成请求。', 'analysis-note'));
  const controls = el('div', undefined, 'toolbar');
  const select = el('select');
  select.setAttribute('aria-label', '稳定性趋势指标');
  trendSpecs.forEach(spec =>
    option(select, spec.key, spec.label + '（' + displayUnit(spec.unit) + '）'));
  controls.append(select);
  const partialLabel = el('label', undefined, 'trend-partial-toggle');
  const showPartial = el('input');
  showPartial.type = 'checkbox';
  showPartial.setAttribute('aria-label', '显示 partial 散点图');
  showPartial.checked = completeWindows.length === 0;
  partialLabel.append(showPartial, el('span', '显示 partial 散点图'));
  if(completeWindows.length && partialWindows.length)controls.append(partialLabel);
  section.append(controls);
  const chart = el('div'), table = el('div');
  section.append(chart, table);
  function draw() {
    chart.replaceChildren();
    table.replaceChildren();
    const spec = trendSpecs.find(s => s.key === select.value) || trendSpecs[0];
    if (completeWindows.length) {
      chart.append(el('h4', '完整窗口趋势'));
      trendChart(chart, completeWindows, spec, metric.metadata, windows.at(-1).end_s);
    } else {
      chart.append(el('p', windows.length===1 && windows[0].complete===false?
        '测试时长不足一个完整统计窗口，无法形成分时趋势。':
        '没有完整窗口，无法形成完整窗口趋势；以下展示保存的窗口数据。', 'analysis-note'));
    }
    if (partialWindows.length) {
      const partial = el('div', undefined, 'analysis-group');
      partial.append(el('h4', 'partial / 完整性未知窗口'));
      partial.append(el('p', '短窗口的 completion-attributed 吞吐可能很高。以下保存值使用原统计口径，独立刻度仅供查看。', 'analysis-note'));
      for (const window of partialWindows) {
        const node = spec.get(window);
        partial.append(el('p', '[' + valueText(window.start_s) + ', ' +
          valueText(window.end_s) + (window.end_s === windows.at(-1).end_s ? ']' : ')') + ' s · ' +
          (window.complete === false ? 'partial 窗口' : '完整性未知') +
          ' · 时长 ' + valueText(window.span_s) + ' s · 完成请求 ' +
          valueText(window.completed_requests?.value) + ' · ' +
          spec.label + ' ' + valueText(node.value, spec.unit) + ' ' +
          displayUnit(spec.unit) + ' · TTFT / E2EL 有效样本 ' +
          valueText(window.ttft_ms?.count) + ' / ' + valueText(window.e2el_ms?.count) +
          ' · ' + trendWarnings(window, spec).join('；'), 'analysis-note'));
      }
      if (showPartial.checked && windows.length>1) {
        partial.append(el('p', '独立刻度散点，与完整窗口主曲线分开展示。', 'analysis-note'));
        trendChart(partial, partialWindows, spec, metric.metadata, windows.at(-1).end_s);
      }
      chart.append(partial);
    }
    trendTable(table, windows, spec);
  }
  select.onchange = draw;
  showPartial.onchange = draw;
  draw();
  trendCaliber(section, metric);
  parent.append(section);
}
