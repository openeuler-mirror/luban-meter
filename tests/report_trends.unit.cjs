// Headless checks for the stability trend module; synthetic windows are
// labeled as test data and never exported as real measurements.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const nodes = [];
const domNode = tag => {
  const node = {
    tag, children: [], textContent: '', title: '', className: '',
    value: '', id: '', hidden: false, disabled: false, open: false,
    attrs: {},
    style: {setProperty() {}}, dataset: {},
    classList: {add() {}, remove() {}},
    append(...items) {
      items.forEach(item => {
        if (item == null) return;
        this.children.push(item);
      });
    },
    replaceChildren() { this.children = []; },
    setAttribute(key, value) { this.attrs[key] = value; },
    getAttribute(key) { return this.attrs[key] ?? null; },
    addEventListener() {}, focus() {}, querySelector() { return null; },
  };
  nodes.push(node);
  return node;
};
const collectText = node => [node.textContent, node.title, node.className,
  ...(node.children || []).map(collectText)].join(' ');
const byId = new Map();
const context = vm.createContext({
  taskDisplayName: name => name,
  names: {success: '成功', partial_failed: '部分失败', failed: '失败',
    interrupted: '已中断', skipped: '未执行', discarded: '已丢弃'},
  document: {
    createElement: tag => domNode(tag),
    createElementNS: (ns, tag) => domNode(tag),
    getElementById: id => {
      if (!byId.has(id)) byId.set(id, domNode('div'));
      return byId.get(id);
    },
  },
});
context.el = (tag, text, cls) => {
  const node = domNode(tag);
  if (text !== undefined) node.textContent = text;
  if (cls) node.className = cls;
  return node;
};
context.svgElement = (tag, attrs) => {
  const node = domNode(tag);
  Object.assign(node.attrs, attrs || {});
  return node;
};
context.option = (parent, value, text) => {
  const node = domNode('option');
  node.value = value;
  node.textContent = text;
  parent.append(node);
};
context.link = (text, target) => {
  const node = domNode('a');
  node.textContent = text;
  node.href = '#' + target;
  return node;
};
const source = path.join(__dirname, '../src/luban_meter/reporting');
const html = process.argv[3] ? fs.readFileSync(process.argv[3], 'utf8') : null;
for (const file of ['case_analysis.js', 'summary_cards.js',
  'report_overview.js', 'trends.js']) {
  const script = fs.readFileSync(path.join(source, file), 'utf8');
  if (html !== null) assert.ok(html.includes(script), 'export contains ' + file);
  vm.runInContext(html === null ? script :
    html.slice(html.indexOf(script), html.indexOf(script) + script.length),
    context);
}
const texts = () => nodes.map(node =>
  [node.textContent, node.title, node.className].join(' ')).join('\n');
const scalar = (value, unit = 'request', precision) => ({value, unit});
const window_ = (index, start, end, extra = {}) => ({
  index, start_s: start, end_s: end, span_s: end - start,
  complete: extra.complete !== false,
  started_requests: scalar(extra.started ?? 0),
  completed_requests: scalar(extra.completed ?? 0),
  successful: scalar(extra.successful ?? 0),
  failed: scalar(extra.failed ?? 0),
  timeout_failed: scalar(extra.timeoutFailed ?? 0),
  other_failed: scalar(extra.otherFailed ?? 0),
  success_rate: {value: extra.successRate ?? null, unit: 'ratio',
    ...(extra.rateReason ? {reason: extra.rateReason} : {})},
  timeout_rate: {value: extra.timeoutRate ?? null, unit: 'ratio',
    ...(extra.rateReason ? {reason: extra.rateReason} : {})},
  output_token_throughput: {
    value: extra.throughput === undefined ? null : extra.throughput,
    unit: 'token/s', condition: 'completion-attributed',
    ...(extra.throughputReason ? {reason: extra.throughputReason} : {})},
  ttft_ms: {count: extra.ttftCount ?? 0, mean: extra.ttftMean ?? null,
    p99: extra.ttftP99 ?? null, ...(extra.low ? {low_sample: true} : {})},
  e2el_ms: {count: extra.e2elCount ?? 0, p99: extra.e2elP99 ?? null,
    ...(extra.low ? {low_sample: true} : {})},
  average_concurrency: scalar(extra.concurrency ?? null),
  peak_concurrency: scalar(extra.peak ?? 0),
});
const trendCase = (windows, meta = {}) => ({
  id: 'case-test', status: 'success', collection_complete: true,
  parameters: {model: 'test-model', profile: 'rate', max_concurrency: 4,
    trend_window_seconds: 60},
  metrics: {stability_trends: {
    status: 'success', observation_boundary: 'client',
    values: {window_seconds: 60, windows},
    metadata: {window_seconds: 60, warmup_included: false,
      low_sample_threshold: 10, timeout_error_types: ['ReadTimeout'],
      unattributed_requests: {count: 0}, ...meta},
  }},
});
context.tasks = [{name: 'perf', category: 'generation_performance',
  status: 'success', cases: []}];

// Call module functions with explicit arguments through context globals.
const parent = domNode('div');
const callRender = (name, c, w, s) => {
  context.PARENT = parent;
  context.C = c;
  context.W = w;
  context.S = s;
  const args = ['PARENT', 'C'];
  if (w !== undefined) args.push('W');
  if (s !== undefined) args.push('S');
  return vm.runInContext(name + '(' + args.join(',') + ')', context);
};
// 1. Normal 60s windows render chart, table and caliber.
nodes.length = 0;
callRender('renderStabilityCase', trendCase([
  window_(0, 0, 60, {started: 2, completed: 2, successful: 2, throughput: 1.5,
    ttftCount: 12, ttftMean: 80, ttftP99: 120, e2elCount: 12, e2elP99: 900,
    successRate: 1, timeoutRate: 0}),
  window_(1, 60, 120, {started: 1, completed: 1, successful: 1,
    throughput: 0.8, ttftCount: 11, ttftMean: 90, ttftP99: 130,
    e2elCount: 11, e2elP99: 950, successRate: 1, timeoutRate: 0}),
]));
let body = texts();
assert.ok(body.includes('稳定性趋势（分时窗口）'));
assert.ok(body.includes('输出Token吞吐量'));
assert.ok(body.includes('请求成功率'));
assert.ok(body.includes('completion-attributed'));
assert.ok(body.includes('不是逐 Token 瞬时生成速率'));
assert.ok(body.includes('分时窗口明细与警告'));
assert.ok(body.includes('统计口径与窗口定义'));

// 2. Partial final window is visibly marked.
nodes.length = 0;
callRender('renderStabilityCase', trendCase([
  window_(0, 0, 60, {completed: 1, successful: 1, throughput: 1}),
  window_(1, 60, 90, {complete: false, completed: 1, successful: 1,
    throughput: 0.5}),
]));
body = texts();
assert.ok(body.includes('partial 窗口'));
assert.ok(body.includes('否（partial）'));

// 3. Empty window keeps zero throughput and null rates; all-failed window
// shows zero rates, never null-as-zero confusion.
nodes.length = 0;
callRender('renderStabilityCase', trendCase([
  window_(0, 0, 60, {throughput: 0, rateReason: 'no completed requests in window'}),
  window_(1, 60, 120, {completed: 2, failed: 2, timeoutFailed: 1,
    otherFailed: 1, throughput: 0, successRate: 0, timeoutRate: 0.5}),
]));
body = texts();
assert.ok(body.includes('no completed requests in window'));

// 4. low_sample P99 keeps the value and warns.
nodes.length = 0;
callRender('renderStabilityCase', trendCase([
  window_(0, 0, 60, {completed: 1, successful: 1, throughput: 1, low: true,
    ttftCount: 3, ttftMean: 70, ttftP99: 71, e2elCount: 3, e2elP99: 500}),
]));
body = texts();
assert.ok(body.includes('TTFT 延迟样本 3 <10'));
assert.ok(body.includes('P99 仅供参考'));

// 5. Null throughput breaks the polyline into segments and draws no mark.
nodes.length = 0;
callRender('renderStabilityCase', trendCase([
  window_(0, 0, 60, {throughput: 1}),
  window_(1, 60, 120, {throughput: null,
    throughputReason: '1 of 2 successful requests lack output token counts'}),
  window_(2, 120, 180, {throughput: 2}),
  window_(3, 180, 240, {throughput: 3}),
]));
const marks = nodes.filter(node =>
  node.tag === 'circle' || node.tag === 'rect');
const polylines = nodes.filter(node => node.tag === 'polyline');
assert.equal(marks.length, 3);
// A lone finite window between nulls keeps its mark but no polyline.
assert.equal(polylines.length, 1);
assert.equal(polylines[0].attrs.points.split(' ').length, 2);
body = texts();
assert.ok(body.includes('lack output token counts'));
assert.ok(body.includes('null 断开不补零'));

// 6. Unknown error types stay ordinary failures in the saved counters.
nodes.length = 0;
callRender('renderStabilityCase', trendCase([
  window_(0, 0, 60, {completed: 2, failed: 2, timeoutFailed: 0,
    otherFailed: 2, throughput: 0, successRate: 0, timeoutRate: 0}),
]));
body = texts();
assert.ok(body.includes('超时错误类型'));
assert.ok(body.includes('ReadTimeout'));

// 7. Unattributed requests raise a coverage warning.
nodes.length = 0;
callRender('renderStabilityCase', trendCase(
  [window_(0, 0, 60, {completed: 1, successful: 1, throughput: 1})],
  {unattributed_requests: {count: 2, reason: 'missing offsets'}}));
body = texts();
assert.ok(body.includes('统计覆盖警告'));
assert.ok(body.includes('不可归属请求：2'));

// 8. Historical results without stability_trends render nothing.
nodes.length = 0;
const legacy = {id: 'legacy', status: 'success',
  metrics: {streaming_performance: {status: 'success', values: {}}}};
const beforeLegacy = parent.children.length;
callRender('renderStabilityCase', legacy);
assert.equal(parent.children.length, beforeLegacy);
context.tasks = [{name: 'perf', category: 'generation_performance',
  status: 'success', cases: [legacy]}];
nodes.length = 0;
context.$=context.document.getElementById;
vm.runInContext("selectAnalysisTab('stability')", context);
assert.equal(byId.get('analysis-stability-panel').hidden,false);
assert.equal(byId.get('analysis-metrics-panel').hidden,true);
vm.runInContext("selectAnalysisTab('metrics')", context);
assert.equal(byId.get('analysis-stability-panel').hidden,true);
assert.equal(byId.get('analysis-metrics-tab').attrs['aria-selected'],'true');

// 9. Multiple cases switch through the selector.
nodes.length = 0;
const first = trendCase([window_(0, 0, 60, {throughput: 1.25})]);
first.id = 'case-a';
const second = trendCase([window_(0, 0, 60, {throughput: 7.5})]);
second.id = 'case-b';
context.tasks = [{name: 'perf', category: 'generation_performance',
  status: 'success', cases: [first, second]}];
const content=domNode('div');
context.PARENT=content;context.C=first;
vm.runInContext('renderStabilityCase(PARENT,C)',context);
assert.ok(collectText(content).includes('1.25'));
content.replaceChildren();context.C=second;
vm.runInContext('renderStabilityCase(PARENT,C)',context);
assert.ok(collectText(content).includes('7.5'));
assert.ok(!collectText(content).includes('1.25'));

// 10. Invalid metric structure is reported, not drawn.
nodes.length = 0;
callRender('renderStabilityCase', {id: 'bad', status: 'success',
  metrics: {stability_trends: {status: 'success', values: {windows: {}}}}});
body = texts();
assert.ok(body.includes('结构异常或缺少窗口数组'));

for (const [start,end] of [[undefined,60],[null,60],[NaN,60],[0,Infinity],
  [60,60],[60,59],[0,null]]) {
  nodes.length=0;
  callRender('renderStabilityCase',trendCase([window_(0,start,end)]));
  assert.ok(!nodes.some(node=>node.tag==='svg'));
  assert.doesNotMatch(texts(),/NaN|Infinity|undefined/);
}
const covered=trendCase([window_(0,0,60,{throughput:1})]);
delete covered.metrics.stability_trends.metadata;
nodes.length=0;
callRender('renderStabilityCase',covered);
assert.match(texts(),/覆盖计数未知/);
assert.doesNotMatch(texts(),/NaN|Infinity|undefined/);
covered.metrics.stability_trends.metadata={
  coverage:{total_requests:2,completion_eligible:1,concurrency_eligible:1},
  unattributed_requests:{count:1}};
nodes.length=0;
callRender('renderStabilityCase',covered);
assert.match(texts(),/并发覆盖警告/);
assert.match(texts(),/完成统计覆盖 1\/2/);
assert.match(texts(),/越界/);
assert.match(texts(),/60\] s/);
nodes.length=0;
const partialFixture=trendCase([
  ...[14.933,17.067,14.933,17.067,14.933].map((throughput,index)=>
    window_(index,index*15,(index+1)*15,{throughput,completed:7})),
  window_(5,75,75.178,{complete:false,throughput:179.737,completed:1}),
]);
callRender('renderStabilityCase',partialFixture);
const main=nodes.find(node=>node.tag==='svg');
assert.equal(main.children.filter(node=>node.tag==='circle').length,5);
assert.equal(main.children.filter(node=>node.tag==='rect').length,0);
assert.deepEqual(main.children.filter(node=>node.tag==='polyline')
  .map(node=>node.attrs.points.split(' ').length),[5]);
assert.ok(!collectText(main).includes('179.737'));
assert.match(texts(),/179.737/);
assert.match(texts(),/完整窗口趋势/);
const toggle=nodes.find(node=>node.type==='checkbox');
assert.equal(toggle.checked,false);
const beforeToggle=nodes.length;
toggle.checked=true;toggle.onchange();
const charts=nodes.slice(beforeToggle).filter(node=>node.tag==='svg');
assert.equal(charts.length,2);
assert.equal(charts[0].children.filter(node=>node.tag==='circle').length,5);
assert.equal(charts[1].children.filter(node=>node.tag==='rect').length,1);
assert.equal(charts[1].children.filter(node=>node.tag==='polyline').length,0);
assert.match(texts(),/独立刻度散点/);
nodes.length=0;
callRender('renderStabilityCase',trendCase([
  window_(0,0,16,{complete:false,throughput:32,completed:16}),
]));
assert.match(texts(),/测试时长不足一个完整统计窗口/);
assert.ok(!nodes.some(node=>node.tag==='svg'));
const completeParent=domNode('div');
context.PARENT=completeParent;context.C=trendCase([window_(0,0,60,{throughput:1})]);
vm.runInContext('renderStabilityCase(PARENT,C)',context);
assert.ok(!collectText(completeParent).includes('显示 partial 散点图'));
nodes.length=0;
context.PARENT=domNode('div');
context.C=trendCase([
  window_(0,0,5,{complete:false,throughput:1}),
  window_(1,5,8,{complete:false,throughput:2}),
]);
vm.runInContext('renderStabilityCase(PARENT,C)',context);
assert.equal(nodes.filter(node=>node.tag==='rect').length,2);
assert.equal(nodes.filter(node=>node.tag==='polyline').length,0);
console.log('report_trends.unit.cjs ok');
