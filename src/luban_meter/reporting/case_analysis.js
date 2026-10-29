// Display adapters only: values and metadata remain in the embedded result.
const metricNames = {
  ttft:'首 Token 延迟（TTFT）', itl:'Token 间延迟（ITL）',
  tpot:'每输出 Token 耗时（TPOT）', e2el:'端到端延迟（E2EL）',
  request_throughput:'请求吞吐量', output_token_throughput:'输出 Token 吞吐量',
  input_token_throughput:'输入 Token 吞吐量', total_token_throughput:'总 Token 吞吐量',
  input_tokens:'实际输入 Token', output_tokens:'实际输出 Token',
  decode_token_throughput:'解码 Token 吞吐量', dispatch_delay:'调度延迟',
  scheduled_latency:'从计划时间起的耗时', stream_event_itl:'流式事件间隔',
  accuracy:'准确率', exact_match:'精确匹配（Exact Match）',
  f1:'F1', token_f1:'Token F1', rouge1:'ROUGE-1', rouge2:'ROUGE-2',
  rougeL:'ROUGE-L', perplexity:'困惑度（PPL）', mean_loss:'平均损失',
  bits_per_byte:'每字节比特数（BPB）', scored_samples:'参与评分样本',
  total_samples:'总样本', parse_failed:'答案解析失败',
  service_failed:'服务失败', sandbox_failed:'沙箱失败',
  execution_timeout:'执行超时', total_tasks:'总题目数', passed:'通过数',
  count:'计数', count_bytes:'字节计数', accuracy_by_subject:'各学科正确率',
  internal_ttft:'Engine 首 Token 延迟', engine_execution_latency:'Engine 执行耗时',
  aggregate_decode_token_throughput:'Engine 解码吞吐量',
  request_view:'客户端请求', service_view:'服务窗口', task_view:'质量评测',
  engine_view:'Engine 内部', token_view:'Token 视角',
  benchmark_duration:'测试窗口时长', average_concurrency:'平均并发',
  maximum_request_concurrency:'并发上限', peak_concurrent_requests:'峰值并发',
  achieved_request_start_rate:'实际请求启动速率', offered_request_rate:'目标请求速率',
  request_rate:'请求速率', successful_requests:'成功请求', failed_requests:'失败请求',
  total_requests:'总请求', total_input_tokens:'总输入 Token',
  total_output_tokens:'总输出 Token', goodput:'有效吞吐量',
  lengths:'长度统计', actual_input_tokens:'实际输入 Token',
  actual_output_tokens:'实际输出 Token', local_input_tokens:'本地输入 Token',
  target_input_tokens:'目标输入 Token', target_output_tokens:'目标输出 Token',
  input_delta:'输入长度偏差', output_delta:'输出长度偏差',
  conformance_rate:'长度符合率', counts:'长度符合数量', conforming:'符合',
  nonconforming:'不符合', unknown:'未知', cmmlu:'CMMLU', ceval:'C-Eval',
  gsm8k:'GSM8K', humaneval:'HumanEval', wikitext:'WikiText'
};
const statNames = {mean:'Mean · 均值', median:'Median · 中位数',
  min:'Min · 最小值', max:'Max · 最大值', stddev:'Std · 标准差',
  std:'Std · 标准差', p999:'P99.9'};
const statisticKey = key => /^(mean|median|min|max|stddev|std|variance|p\d+(?:\.\d+)?)$/.test(key);
const numeric = value => typeof value === 'number' && Number.isFinite(value);
const displayUnit = unit => ({ratio:'%',sample:'样本',token:'token'}[unit] || unit || '未注明单位');
const readable = key => metricNames[key] ||
  (/^pass_at_\d+$/.test(key) ? 'Pass@' + key.slice(8) : key);
const metricCache = new WeakMap();
function metricsFor(c) {
  if (!c || c.status === 'discarded') return [];
  if (metricCache.has(c)) return metricCache.get(c);
  const result = [];
  const skip = new Set(['unit','sample_count','weight_sum','metadata',
    'environment','error','reason','condition','status','aggregation',
    'percentile_method','observation_boundary']);
  function visit(node, parts, unit='', count=null, boundary='') {
    const path = '/' + parts.map(p => p.replace(/~/g,'~0').replace(/\//g,'~1')).join('/');
    if (node && typeof node === 'object' && !Array.isArray(node)) {
      unit = node.unit ?? unit;
      if (Object.hasOwn(node,'sample_count'))
        count = numeric(node.sample_count) ? node.sample_count : null;
      else if (Object.hasOwn(node,'count') &&
        (numeric(node.count) || node.count === null)) count = node.count;
      boundary = node.observation_boundary || boundary;
      const hasValue = Object.hasOwn(node, 'value');
      const isStats = !hasValue && Object.keys(node).some(key=>
        statisticKey(key) && (numeric(node[key]) || node[key] === null));
      if (hasValue || isStats) {
        const stats = isStats ? Object.entries(node).filter(([key,value]) =>
          !skip.has(key) && key !== 'count' &&
          (numeric(value) || value === null)) : [];
        result.push({path, parts, key:parts.at(-1), unit, count, boundary,
          kind:isStats?'statistics':'scalar', value:node.value, stats, raw:node});
        return;
      }
      for (const [key,value] of Object.entries(node)) {
        // Numeric count is inherited metadata; a count object is a real metric
        // (e.g. WikiText's scored token count) and must not disappear.
        if (skip.has(key) || (key === 'count' && (numeric(value) || value === null))) continue;
        visit(value, [...parts,key], unit, count, boundary);
      }
    } else if (Array.isArray(node)) {
      node.forEach((item,i) => visit(item,[...parts,String(i)],unit,count,boundary));
    } else if (numeric(node) || node === null) {
      result.push({path, parts, key:parts.at(-1), unit, count, boundary,
        kind:'scalar', value:node, stats:[], raw:node});
    }
  }
  visit(c.metrics || {}, []);
  metricCache.set(c,result);
  return result;
}
function metricTitle(metric) {
  const parts = metric.parts.filter(p => p !== 'values');
  const view = parts.find(p => p.endsWith('_view'));
  const boundary = metric.boundary ?
    ({client:'客户端',engine:'Engine 内部'}[metric.boundary] || metric.boundary) : '';
  return [readable(metric.key), view && readable(view),
    !view && boundary].filter(Boolean).join(' · ');
}
function valueText(value, unit='', exact=false) {
  if (value == null || (typeof value === 'number' && !Number.isFinite(value)))
    return '未采集';
  if (!numeric(value)) return value === '' ? '未采集' : String(value);
  if (exact) return value === '' ? '未采集' : String(value);
  const scaled = unit === 'ratio' ? value * 100 : value;
  if (scaled !== 0 && Math.abs(scaled) < 0.001) return scaled.toExponential(3);
  return scaled.toLocaleString('zh-CN', {maximumFractionDigits:unit==='ratio'?2:3});
}
function valueCell(value, unit='') {
  const cell = el('td',valueText(value,unit),'num');
  if (value != null) cell.title = '原始值：' + valueText(value,'',true) +
    (unit ? ' ' + unit : '');
  return cell;
}
function dataTable(parent, columns, records, className='') {
  const table = el('table',undefined,className), head = el('thead');
  const heading = el('tr'), body = el('tbody');
  let sortIndex = null, ascendingOrder = true;
  columns.forEach((col,index) => {
    const th = el('th',undefined,col.numeric?'num':'');
    if (col.sortable) {
      const button = el('button');
      const arrow=el('span','↕','sort-arrow');
      arrow.setAttribute('aria-hidden','true');
      button.append(el('span',col.title),arrow);
      button.type = 'button';
      th.setAttribute('aria-sort','none');
      button.onclick = () => {
        ascendingOrder = sortIndex===index ? !ascendingOrder : true;
        sortIndex = index;
        heading.querySelectorAll('[aria-sort]').forEach(n => n.setAttribute('aria-sort','none'));
        th.setAttribute('aria-sort',ascendingOrder?'ascending':'descending');
        draw();
      };
      th.append(button);
    } else th.textContent = col.title;
    heading.append(th);
  });
  function draw() {
    const ordered = [...records];
    if (sortIndex != null) ordered.sort((a,b) => {
      const col = columns[sortIndex], x = col.get(a), y = col.get(b);
      if (x == null) return y == null ? 0 : 1;
      if (y == null) return -1;
      return (numeric(x)&&numeric(y)?x-y:String(x).localeCompare(String(y))) *
        (ascendingOrder?1:-1);
    });
    body.replaceChildren();
    for (const record of ordered) {
      const tr = el('tr');
      if (record.anchor) {tr.id = record.anchor;tr.tabIndex = -1;}
      for (const col of columns) {
        const value = col.get(record);
        const td = col.numeric ? valueCell(value,col.unit) :
          el('td',valueText(value));
        if (col.render) {td.replaceChildren();col.render(td,record);}
        tr.append(td);
      }
      body.append(tr);
    }
    if (!records.length) {
      const td = el('td','没有可展示的记录','empty');
      td.colSpan = columns.length;
      const tr = el('tr');tr.append(td);body.append(tr);
    }
  }
  head.append(heading);table.append(head,body);draw();
  const wrap = el('div',undefined,'table-wrap');wrap.append(table);parent.append(wrap);
  return table;
}
const parameterNames = {model:'模型',data:'数据来源',dataset_path:'数据文件',
  profile:'负载策略',max_concurrency:'并发',input_tokens:'输入长度参数（token）',
  output_tokens:'输出上限（token）',batch_size:'Batch Size',request_rate:'请求速率（req/s）'};
function isDimension(metric) {
  const taskView = metric.parts.indexOf('task_view');
  return metric.parts.slice(0,-1).some(p=>p.includes('_by_')) ||
    (taskView>=0 && metric.parts.length>taskView+3);
}
function summaryMetrics(task) {
  const all = new Map();
  task.cases.forEach(c => metricsFor(c).forEach(m => {
    all.set(JSON.stringify([m.path,m.unit]),m);
  }));
  const available = [...all.values()];
  if (task.category !== 'generation_performance')
    return available.filter(m=>!isDimension(m));
  const core = available.filter(m =>
    /\/request_view\/(ttft|tpot)$|\/service_view\/output_token_throughput$/.test(m.path));
  return core.length ? core.sort((a,b)=>
    ['ttft','tpot','output_token_throughput'].indexOf(a.key)-
    ['ttft','tpot','output_token_throughput'].indexOf(b.key)) : available.slice(0,4);
}
const taskCategoryNames = {generation_performance:'推理性能评测',
  model_service_quality:'模型能力评测'};
function reportTaskGroups(taskList) {
  const groups = new Map();
  taskList.forEach((task,ti) => {
    const category = task.category || 'uncategorized';
    if (!groups.has(category)) groups.set(category, {category,
      title:Object.hasOwn(taskCategoryNames,category)?taskCategoryNames[category]:
        (task.category || '其他评测'),tasks:[]});
    groups.get(category).tasks.push({task,ti});
  });
  return [...groups.values()];
}
function taskSectionTitle(task) {
  const scenario = scenarioDisplayName(task.scenario || task.category) || '未采集场景';
  const name = taskDisplayName(task.name);
  return !name || task.name === task.scenario || name === scenario ||
    name === taskCategoryNames[task.category] ? scenario : scenario + ' · ' + name;
}
function renderSummaries() {
  $('case-summaries').replaceChildren();
  const parents = new Map();
  for (const category of reportTaskGroups(tasks)) {
    const section = el('div',undefined,'summary-category');
    section.id = 'summary-category-' + category.category;
    const heading = el('h3',category.title,'task-heading');
    if (category.category === 'generation_performance') heading.classList.add('task-performance');
    else if (category.category === 'model_service_quality') heading.classList.add('task-quality');
    section.append(heading);$('case-summaries').append(section);
    category.tasks.forEach(({ti})=>parents.set(ti,section));
  }
  tasks.forEach((task,ti) => {
    const group = el('div',undefined,'summary-task');
    const heading=el('h4',taskSectionTitle(task),'summary-task-title');
    group.append(heading);
    const commonKeys = ['model','data','dataset_path','profile'];
    const shared = commonKeys.filter(key=>task.cases.length &&
      task.cases.every(c=>c.parameters?.[key]===task.cases[0].parameters?.[key]));
    const context = shared.map(key=>(parameterNames[key]||key)+'：'+
      valueText(task.cases[0].parameters?.[key]));
    group.append(el('p',context.join(' · '),'muted metric-context'));
    const conditions = new Set(task.category==='generation_performance' ?
      ['max_concurrency','input_tokens','output_tokens'] : []);
    commonKeys.filter(key=>!shared.includes(key)).forEach(key=>conditions.add(key));
    // Varying scalar parameters are meaningful comparison conditions. Full
    // structured configuration remains accessible in the original details.
    const keys = new Set(task.cases.flatMap(c=>Object.keys(c.parameters||{})));
    for (const key of keys) {
      if (key==='api_key') continue;
      const values = task.cases.map(c=>c.parameters?.[key]);
      if (values.every(v=>v==null||['string','number','boolean'].includes(typeof v)) &&
        new Set(values).size>1) conditions.add(key);
    }
    const records = task.cases.map((c,ci)=>({c,ci,anchor:'summary-'+ti+'-'+ci,
      metrics:new Map(metricsFor(c).map(m=>[JSON.stringify([m.path,m.unit]),m]))}));
    const columns = [{title:'Case',get:r=>r.c.id}];
    for (const key of conditions) columns.push({title:parameterNames[key]||key,
      numeric:records.some(r=>numeric(r.c.parameters?.[key])),sortable:true,
      get:r=>r.c.parameters?.[key]});
    for (const metric of summaryMetrics(task)) {
      const key = JSON.stringify([metric.path,metric.unit]);
      columns.push({title:({ttft:'TTFT',tpot:'TPOT'}[metric.key] || readable(metric.key))+(metric.kind==='statistics'?' Mean':'')+
        '（'+displayUnit(metric.unit)+'）',numeric:true,sortable:true,unit:metric.unit,
        get:r=>{const m=r.metrics.get(key);return m?.kind==='statistics'?
          m.stats.find(([k])=>k==='mean')?.[1]:m?.value;}});
    }
    for (const [a,b,title] of [['complete','failed','完成 / 失败'],
      ['incomplete','not_started','未完成 / 未开始']])
      columns.push({title,numeric:true,sortable:true,get:r=>r.c.coverage?.[a],
        render:(cell,r)=>{cell.textContent=valueText(r.c.coverage?.[a])+' / '+
          valueText(r.c.coverage?.[b]);
          cell.title='点击表头按'+title.split(' / ')[0]+'数量排序';}});
    columns.push({title:'状态',get:r=>names[r.c.status]||r.c.status},
      {title:'详情',get:()=>'',render:(cell,r)=>
        cell.append(link('分析','analysis-'+ti+'-'+r.ci))});
    dataTable(group,columns,records,'summary-table');
    if (!task.cases.length) group.append(link('任务诊断','task-'+ti));
    parents.get(ti).append(group);
  });
  if (!tasks.length) $('case-summaries').append(el('p','没有任务结果','empty'));
}
function statLabel(key) {return statNames[key] || (key.startsWith('p')?key.toUpperCase():key);}
function statsFor(metric) {
  const priority = ['mean','median','p50','p90','p95','p99','p999','min','max','stddev','std'];
  // Keep Median and P50 separate: equality of stored values alone does not
  // establish identical definitions for every future metric provider.
  return [...metric.stats].sort(([a],[b])=>{
    const ai=priority.indexOf(a),bi=priority.indexOf(b);
    return (ai<0?100:ai)-(bi<0?100:bi);
  }).map(([key,value])=>({key,value}));
}
function rawMetric(parent,metric) {
  const details = el('details',undefined,'metric-context');
  details.append(el('summary','原始精度、指标路径与统计口径'));
  details.append(el('code',metric.path));
  details.append(el('pre',JSON.stringify(metric.raw,null,2)));
  parent.append(details);
}
function barChart(parent,points,unit,title,horizontal=false) {
  points = points.filter(p=>numeric(p.value));
  if (!points.length) return;
  const isRatio = unit==='ratio';
  const values = points.map(p=>isRatio?p.value*100:p.value);
  const lo=Math.min(0,...values),hi=Math.max(isRatio?100:0,...values);
  const span=hi-lo||1;
  const width=760,height=horizontal?Math.max(150,points.length*32+50):260;
  const padLeft=horizontal?280:65,padRight=75,padTop=20,padBottom=40;
  const svg=svgElement('svg',{viewBox:`0 0 ${width} ${height}`,role:'img','aria-label':title});
  const svgTitle=svgElement('title',{});svgTitle.textContent=title+'（'+displayUnit(unit)+'）';svg.append(svgTitle);
  function text(x,y,value,anchor='start') {
    const t=svgElement('text',{x,y,fill:'currentColor','font-size':12,'text-anchor':anchor});
    t.textContent=value;svg.append(t);
  }
  const x=v=>padLeft+(v-lo)/span*(width-padLeft-padRight);
  const y=v=>height-padBottom-(v-lo)/span*(height-padTop-padBottom);
  for(let i=0;i<=4;i++) {
    const v=lo+span*i/4;
    const attrs=horizontal?{x1:x(v),x2:x(v),y1:padTop,y2:height-padBottom}:
      {x1:padLeft,x2:width-padRight,y1:y(v),y2:y(v)};
    svg.append(svgElement('line',{...attrs,stroke:'var(--line)'}));
    text(horizontal?x(v):padLeft-10,horizontal?height-12:y(v)+4,
      valueText(v),horizontal?'middle':'end');
  }
  points.forEach((p,i)=>{
    const v=values[i],px=padLeft+(i+.5)*(width-padLeft-padRight)/points.length;
    const py=padTop+i*32;
    const attrs=horizontal?{x:Math.min(x(0),x(v)),y:py,width:Math.abs(x(v)-x(0)),height:18}:
      {x:px-20,y:Math.min(y(v),y(0)),width:40,height:Math.abs(y(v)-y(0))};
    const mark=svgElement('rect',{...attrs,rx:3,fill:'var(--accent)'});
    const tip=svgElement('title',{});
    tip.textContent=p.label+': '+valueText(p.value,unit)+' '+displayUnit(unit)+
      (p.count!=null?' · 样本/计数：'+p.count:'');
    mark.append(tip);svg.append(mark);
    text(horizontal?padLeft-10:px,horizontal?py+14:height-12,p.label,horizontal?'end':'middle');
    text(horizontal?x(v)+7:px,horizontal?py+14:y(v)-8,valueText(p.value,unit),horizontal?'start':'middle');
  });
  const box=el('div',undefined,'chart');
  box.append(el('p',title+'（'+displayUnit(unit)+'）','analysis-note'),svg);parent.append(box);
}
function renderPerformance(parent,metric) {
  if (!metric) {parent.append(el('p','当前 Case 没有可展示的指标','empty'));return;}
  parent.append(el('h3',metricTitle(metric)));
  parent.append(el('p','已保存样本数 / count：'+valueText(metric.count)+
    (metric.raw?.weight_sum!=null?' · 权重总和：'+valueText(metric.raw.weight_sum):''),
    'analysis-note'));
  const grid=el('div',undefined,'analysis-grid'),tableBox=el('div');
  const rows=metric.kind==='statistics'?statsFor(metric):[{key:'value',value:metric.value}];
  dataTable(tableBox,[{title:'统计量',get:r=>r.key==='value'?'聚合值':statLabel(r.key)},
    {title:'数值（'+displayUnit(metric.unit)+'）',numeric:true,unit:metric.unit,
      get:r=>r.value}],rows,'stat-table');
  grid.append(tableBox);
  const quantiles=metric.stats.filter(([key])=>/^p\d+(?:\.\d+)?$/.test(key))
    .map(([key,value])=>({value,label:statLabel(key)}));
  if (quantiles.some(p=>numeric(p.value)))
    barChart(grid,quantiles,metric.unit,readable(metric.key)+'已保存分位数');
  else grid.classList.remove('analysis-grid');
  parent.append(grid);
  rawMetric(parent,metric);
}
// Exact persisted paths: never substitute similarly named request/Engine data.
const corePerformancePaths = [
  '/streaming_performance/values/request_view/ttft',
  '/streaming_performance/values/request_view/tpot',
  '/streaming_performance/values/request_view/itl',
  '/streaming_performance/values/request_view/e2el',
  '/streaming_performance/values/service_view/request_throughput',
  '/streaming_performance/values/service_view/output_token_throughput'
];
function corePerformanceMetrics(c) {
  const available=new Map(metricsFor(c).map(m=>[m.path,m]));
  return corePerformancePaths.flatMap(path=>available.has(path)?[available.get(path)]:[]);
}
function isCmmluSubjectAccuracy(metric) {
  const view=metric.parts.indexOf('task_view');
  return view>=0 && metric.parts[view+1]==='cmmlu' &&
    metric.parts[view+2]==='accuracy_by_subject';
}
function renderQuality(parent,metrics) {
  // Presentation-only exclusion; metricsFor and raw result stay complete.
  metrics=metrics.filter(m=>!isCmmluSubjectAccuracy(m));
  if (!metrics.length) {parent.append(el('p','当前 Case 没有可展示的质量指标','empty'));return;}
  const groups=new Map();
  for(const metric of metrics) {
    const key=JSON.stringify([metric.parts.slice(0,-1),metric.unit]);
    if(!groups.has(key))groups.set(key,[]);
    groups.get(key).push(metric);
  }
  const orderedGroups=[...groups.values()].sort((a,b)=>
    Number(isDimension(a[0]))-Number(isDimension(b[0])));
  for(const group of orderedGroups) {
    const first=group[0],section=el('div',undefined,'analysis-group');
    const viewIndex=first.parts.indexOf('task_view');
    const title=first.parts.slice(viewIndex>=0?viewIndex+1:0,-1)
      .filter(p=>p!=='values').map(readable).join(' · ');
    section.append(el('h3',title || '质量指标'));
    const rows=group.flatMap(m=>m.kind==='statistics'?statsFor(m).map(s=>
      ({metric:m,name:readable(m.key)+' / '+statLabel(s.key),value:s.value})):
      [{metric:m,name:readable(m.key),value:m.value}]);
    dataTable(section,[{title:isDimension(first)?'学科 / 分类':'指标',get:r=>r.name,sortable:true},
      {title:'数值（'+displayUnit(first.unit)+'）',numeric:true,sortable:true,unit:first.unit,get:r=>r.value},
      {title:'已保存样本 / 计数',numeric:true,sortable:true,get:r=>r.metric.count}],rows);
    if(isDimension(first) && group.every(m=>m.kind==='scalar')) {
      const chartDetails=el('details');chartDetails.append(el('summary','查看分类指标图'));
      chartDetails.open=true;
      barChart(chartDetails,group.map(m=>({label:readable(m.key),value:m.value,count:m.count})),
        first.unit,title,true);
      section.append(chartDetails);
    }
    const raw=el('details',undefined,'metric-context');
    raw.append(el('summary','原始精度与指标路径'));
    raw.append(el('pre',JSON.stringify(group.map(m=>({path:m.path,data:m.raw})),null,2)));
    section.append(raw);parent.append(section);
  }
}
function renderAnalysis() {
  const ti=Number($('analysis-task').value),ci=Number($('analysis-case').value);
  const task=tasks[ti],c=task?.cases[ci],context=$('analysis-context'),content=$('analysis-content');
  context.replaceChildren();content.replaceChildren();
  if(!c) {content.append(el('p','当前任务没有 Case','empty'));return;}
  const id='case-'+ti+'-'+ci;
  context.append(el('h3',taskDisplayName(task.name)+' / '+c.id+' · '+(names[c.status]||c.status)));
  context.append(el('p',Object.entries(c.coverage||{}).filter(([key])=>
    ['complete','failed','incomplete','not_started'].includes(key)).map(([key,v])=>
      ({complete:'完成样本',failed:'失败样本',incomplete:'未完成',not_started:'未开始'}[key])+': '+valueText(v)).join(' · '),
    'analysis-note'));
  const navigation=el('p');
  navigation.append(link('测试条件、覆盖率与诊断',id),document.createTextNode(' · '),
    link('返回结果摘要','summary-'+ti+'-'+ci));context.append(navigation);
  if(c.status==='discarded') {content.append(el('p','已丢弃，不展示有效指标。请查看诊断。','empty'));return;}
  const metrics=metricsFor(c);
  const failedMetrics=Object.entries(c.metrics||{}).filter(([,m])=>m?.error||m?.status==='failed');
  if(c.status!=='success'||failedMetrics.length)
    context.append(el('p','结果存在异常或未完成状态，请结合指标状态、覆盖率与诊断阅读。','analysis-note'));
  if(task.category==='generation_performance')
    renderPerformance(content,metrics.find(m=>m.path===$('analysis-metric').value));
  else {
    context.append(el('p','完成样本、答案解析和参与评分是不同口径；下表仅展示已保存字段。样本/计数可能按样本、Token 或字节统计，请结合原始口径阅读。','analysis-note'));
    renderQuality(content,metrics);
  }
}
function updateAnalysisMetrics() {
  const task=tasks[Number($('analysis-task').value)];
  const c=task?.cases[Number($('analysis-case').value)];
  const select=$('analysis-metric'),previous=select.value;
  select.replaceChildren();
  const performance=task?.category==='generation_performance';
  $('analysis-metric-field').hidden=!performance;
  $('analysis-toolbar').classList.toggle('quality',!performance);
  if(performance) {
    const metrics=corePerformanceMetrics(c);
    for(const m of metrics)option(select,m.path,readable(m.key));
    select.value=metrics.some(m=>m.path===previous)?previous:(metrics[0]?.path||'');
    select.disabled=!metrics.length;
  }
  renderAnalysis();
}
function updateAnalysisCases() {
  const task=tasks[Number($('analysis-task').value)],select=$('analysis-case');
  select.replaceChildren();
  for(const [ci,c] of (task?.cases||[]).entries()) {
    const p=c.parameters||{},conditions=[];
    if(p.max_concurrency!=null)conditions.push('并发 '+p.max_concurrency);
    if(p.input_tokens!=null)conditions.push('输入 '+p.input_tokens);
    if(p.output_tokens!=null)conditions.push('输出上限 '+p.output_tokens);
    option(select,String(ci),[c.id,...conditions,names[c.status]||c.status].join(' · '));
  }
  select.disabled=!task?.cases.length;
  updateAnalysisMetrics();
}
function ensureAnalysisTarget(target) {
  const match=/^analysis-(\d+)-(\d+)(?:-(ttft|tpot|e2el|output_token_throughput))?$/.exec(target);
  if(!match||!tasks[Number(match[1])]?.cases[Number(match[2])])return target;
  $('analysis-task').value=match[1];updateAnalysisCases();
  $('analysis-case').value=match[2];updateAnalysisMetrics();
  if(match[3]) {
    const path=corePerformancePaths.find(p=>p.endsWith('/'+match[3]));
    if([...$('analysis-metric').options].some(o=>o.value===path)) {
      $('analysis-metric').value=path;renderAnalysis();
    }
  }
  return 'case-analysis';
}
function initAnalysis() {
  tasks.forEach((task,ti)=>option($('analysis-task'),String(ti),
    taskDisplayName(task.name)+' · '+scenarioDisplayName(task.scenario||task.category)));
  $('analysis-task').disabled=!tasks.length;
  $('analysis-task').onchange=updateAnalysisCases;
  $('analysis-case').onchange=updateAnalysisMetrics;
  $('analysis-metric').onchange=renderAnalysis;
  updateAnalysisCases();
}

function initReportNavigation() {
  const sidebar=document.querySelector('.sidebar'),toggle=$('nav-toggle');
  const links=[...$('report-nav').querySelectorAll('a')];
  function closeMenu() {
    sidebar.classList.remove('open');toggle.setAttribute('aria-expanded','false');
  }
  toggle.onclick=()=>{
    const open=sidebar.classList.toggle('open');
    toggle.setAttribute('aria-expanded',String(open));
  };
  sidebar.addEventListener('keydown',event=>{
    if(event.key==='Escape'){closeMenu();toggle.focus();}
  });
  links.forEach(a=>a.addEventListener('click',closeMenu));
  window.addEventListener('hashchange',closeMenu);
}
