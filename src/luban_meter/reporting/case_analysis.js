// Display adapters only: values and metadata remain in the embedded result.
const metricNames = {
  ttft:'首 Token 延迟（TTFT）', itl:'Token 间延迟（ITL）',
  tpot:'每输出 Token 耗时（TPOT）', e2el:'端到端延迟（E2EL）',
  reasoning_duration:'思考输出时长', content_duration:'正式回答输出时长',
  reasoning_ttft:'思考首Token延迟（TTFT）', reasoning_tpot:'思考每Token耗时（TPOT）',
  reasoning_itl:'思考Token间延迟（ITL）', reasoning_stream_event_itl:'思考流事件间隔',
  first_content_latency:'首个正式回答延迟', phase_counts:'分阶段样本',
  eligible_requests:'符合统计条件的请求', excluded_requests:'排除请求',
  truncated_requests:'输出截断请求', missing_finish_requests:'缺少结束原因',
  other_finish_requests:'其他结束原因',
  interleaved_requests:'通道交错请求（全部已观察请求）',
  reasoning_single_event_requests:'思考单事件样本（符合统计条件）',
  content_single_event_requests:'正文单事件样本（符合统计条件）',
  missing_reasoning_requests:'缺少思考通道（符合统计条件）',
  missing_content_requests:'缺少正文通道（符合统计条件）',
  request_throughput:'请求吞吐量', output_token_throughput:'输出 Token 吞吐量',
  input_token_throughput:'输入 Token 吞吐量', total_token_throughput:'总 Token 吞吐量',
  input_tokens:'实际输入 Token', output_tokens:'实际输出 Token',
  decode_token_throughput:'解码 Token 吞吐量', dispatch_delay:'调度延迟',
  scheduled_latency:'从计划时间起的耗时', stream_event_itl:'流式事件间隔',
  queue_time_ms:'服务端排队等待时间（等待过的请求）',
  queue_time_waited_mean:'服务端平均排队等待（仅等待过的请求）',
  queue_time_reported_requests:'排队计时上报请求数',
  queue_time_missing_requests:'排队计时缺失请求数',
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
  gsm8k:'GSM8K', humaneval:'HumanEval', wikitext:'WikiText',
  turn_view:'按轮次', session_view:'会话', caching_view:'上下文增长',
  pressure_view:'Prefill 压力', prefix_cache:'前缀缓存',
  turns_observed:'已观测轮次数', session_latency:'会话耗时',
  think_time_total:'轮间等待合计', completed_turns:'完成轮次',
  planned_turns:'计划轮次', not_started_turns:'未开始轮次',
  sent_failed_turns:'发送失败轮次',
  sessions_finished:'已结束会话', sessions_per_second:'会话吞吐量',
  abort_reasons:'会话中止原因', session_coverage:'会话覆盖率',
  first_turn_ttft_mean:'首轮 TTFT 均值', rest_turn_ttft_mean:'后续轮 TTFT 均值',
  first_to_rest_ttft_ratio:'首轮 / 后续轮 TTFT 比',
  ttft_growth_per_turn:'每轮 TTFT 增长', ttft_growth_per_ktok:'每千 Token TTFT 增长',
  measured_turns:'已测轮次', prompt_tokens_total:'输入 Token 合计',
  eligible_prefix_tokens:'可复用前缀 Token', eligible_prefix_ratio:'可复用前缀占比',
  server_cached_tokens:'服务端命中 Token', server_cache_hit_rate:'服务端缓存命中率',
  server_cache_eligible_rate:'服务端命中 / 可复用前缀', reporting_turns:'上报轮次',
  turn_context_tokens:'单轮上下文 Token', total_prefill_tokens:'Prefill Token 合计',
  prefill_token_rate:'Prefill Token 速率', peak_turn_context_tokens:'单轮上下文峰值'
};
const statNames = {mean:'均值', median:'中位数',
  min:'最小值', max:'最大值', stddev:'标准差',
  std:'标准差', p999:'P99.9'};
const reasoningWeightUnits = {reasoning_tpot:'Token',reasoning_itl:'Token间隔'};
const statisticKey = key => /^(mean|median|min|max|stddev|std|variance|p\d+(?:\.\d+)?)$/.test(key);
const numeric = value => typeof value === 'number' && Number.isFinite(value);
const displayUnit = unit => ({ratio:'%',cache_ratio:'%',ratio_pp:'pp',sample:'样本',token:'token'}[unit] || unit || '未注明单位');
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
    if (parts.length===1 && parts[0]==='prefix_cache_hit_rate') {
      visit(node?.values||{}, [...parts,'values']);return;
    }
    const path = '/' + parts.map(p => p.replace(/~/g,'~0').replace(/\//g,'~1')).join('/');
    if (node && typeof node === 'object' && !Array.isArray(node)) {
      unit = node.unit ?? unit;
      if (parts[0]==='prefix_cache_hit_rate' && unit==='ratio')
        unit = parts.at(-1).endsWith('_deviation')?'ratio_pp':'cache_ratio';
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
  const scaled = ['ratio','cache_ratio','ratio_pp'].includes(unit) ? value * 100 : value;
  if (scaled !== 0 && Math.abs(scaled) < 0.001) return scaled.toExponential(3);
  return scaled.toLocaleString('zh-CN', {maximumFractionDigits:['cache_ratio','ratio_pp'].includes(unit)?4:unit==='ratio'?2:3});
}
function valueCell(value, unit='') {
  const cell = el('td',valueText(value,unit),'num');
  if (value != null) cell.title = '原始值：' + valueText(value,'',true) +
    (unit ? ' ' + (['cache_ratio','ratio_pp'].includes(unit)?'ratio':unit) : '');
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
  output_tokens:'输出上限（token）',batch_size:'Batch Size',request_rate:'请求速率（req/s）',
  multi_num_sessions:'会话数',multi_max_turns:'轮数上限',
  multi_think_time:'轮间等待（s）'};
function isDimension(metric) {
  const taskView = metric.parts.indexOf('task_view');
  return metric.parts.slice(0,-1).some(p=>p.includes('_by_')) ||
    (taskView>=0 && metric.parts.length>taskView+3);
}
function summaryMetrics(task) {
  const all = new Map();
  const providers = new Map(task.cases.map(c => [c, calculatorProviderName(c)]));
  task.cases.forEach(c => metricsFor(c).forEach(m => {
    // Run-level throughput metrics saved under other providers must not
    // leak into per-Case summary columns.
    const provider = providers.get(c);
    const phase=phasePerformanceViews.some(([view,key])=>
      m.key===key && m.parts.at(-2)===view) || m.parts.at(-2)==='phase_counts';
    if (provider !== null && m.parts[0] !== provider && !phase && m.parts[0]!=='prefix_cache_hit_rate') return;
    all.set(JSON.stringify([m.path,m.unit]),m);
  }));
  const available = [...all.values()];
  if (task.category !== 'generation_performance')
    return available.filter(m=>!isDimension(m));
  const core = available.filter(m =>
    /\/request_view\/(ttft|tpot|reasoning_duration|content_duration|first_content_latency|reasoning_ttft|reasoning_tpot|reasoning_itl|reasoning_stream_event_itl)$|\/service_view\/(output_token_throughput|total_token_throughput)$|\/phase_counts\/(truncated_requests|excluded_requests)$/.test(m.path));
  const order=['ttft','tpot','output_token_throughput','total_token_throughput','reasoning_ttft',
    'reasoning_tpot','reasoning_itl','reasoning_stream_event_itl','reasoning_duration',
    'content_duration','first_content_latency','truncated_requests','excluded_requests'];
  const cache=available.filter(m=>m.parts[0]==='prefix_cache_hit_rate' && cacheSummaryKeys.includes(m.key));
  return [...cache,...(core.length ? core.sort((a,b)=>
    order.indexOf(a.key)-order.indexOf(b.key)) : available.slice(0,4))];
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
function cacheTarget(c) {
  return cacheValue(c,'target_rate') ?? c.parameters?.prefix_cache_target;
}
function targetHeading(c) {
  const target=cacheTarget(c);
  return numeric(target)?'前缀缓存评测 · '+valueText(target,'cache_ratio')+'%':'前缀缓存评测';
}
function taskSectionTitle(task) {
  if(task.scenario==='prefix_cache') {
    const targets=new Set(task.cases.map(cacheTarget));
    return targets.size===1?targetHeading(task.cases[0]):'前缀缓存评测';
  }
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
    if(task.scenario==='prefix_cache' && task.cases.length) {
      task.cases.forEach((c,ci)=>renderCacheSummary(group,task,c,ti,ci));
      renderCacheComparison(group,task);
      parents.get(ti).append(group);
      return;
    }
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
      unit:key==='prefix_cache_target'?'cache_ratio':'',
      get:r=>r.c.parameters?.[key]});
    for (const metric of summaryMetrics(task)) {
      const key = JSON.stringify([metric.path,metric.unit]);
      columns.push({title:({ttft:'TTFT',tpot:'TPOT'}[metric.key] || readable(metric.key))+(metric.kind==='statistics'?' 均值':'')+
        '（'+displayUnit(metric.unit)+'）',numeric:true,sortable:true,unit:metric.unit,
        get:r=>{const m=r.metrics.get(key);return m?.kind==='statistics'?
          m.stats.find(([k])=>k==='mean')?.[1]:m?.value;}});
      if(task.category==='generation_performance' &&
        phasePerformanceViews.some(([view,name])=>
          metric.key===name && metric.parts.at(-2)===view))
        columns.push({title:readable(metric.key)+
          (metric.raw.sample_unit==='event_interval'?' 有效间隔数（N）':' 有效样本数（N）'),
          numeric:true,sortable:true,get:r=>r.metrics.get(key)?.count});
      if(Object.hasOwn(reasoningWeightUnits,metric.key))
        columns.push({title:readable(metric.key)+' 权重总数（'+
          reasoningWeightUnits[metric.key]+'）',numeric:true,sortable:true,
          get:r=>r.metrics.get(key)?.raw.weight_sum});
    }
    if (task.cases.some(isPrefixCache)) columns.push(...cacheStatusColumns());
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
    renderCacheComparison(group,task);
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
  const isRatio = unit==='ratio'||unit==='cache_ratio';
  const values = points.map(p=>isRatio||unit==='ratio_pp'?p.value*100:p.value);
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
    tip.textContent=(p.detail||p.label)+': '+valueText(p.value,unit)+' '+displayUnit(unit)+
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
  if(metric.count===0 && phasePerformanceViews.some(([,key])=>key===metric.key))
    parent.append(el('p','不可用：没有可计算该指标的有效样本。','empty'));
  const sampleLabel=metric.raw?.sample_unit==='event_interval'?
    '有效事件间隔数（N）：':'已保存样本数（N）：';
  parent.append(el('p',sampleLabel+valueText(metric.count)+
    (metric.raw?.weight_sum!=null?' · 权重总和：'+valueText(metric.raw.weight_sum)+
      (reasoningWeightUnits[metric.key]?' '+reasoningWeightUnits[metric.key]:''):''),
    'analysis-note'));
  const thinkingNotes={
    reasoning_ttft:'从请求开始到首个思考输出；请求等权统计。',
    reasoning_tpot:'从请求开始到最后一个思考输出，除以思考Token数（服务上报优先，缺失时可由Token ID补充）；包含首输出等待，仅按Token数加权计算平均值。缺少有效计数时不可用。',
    reasoning_itl:'每个请求的思考输出跨度除以R−1（R为服务上报或Token ID补充的思考Token数，须大于1）。仅按R−1加权计算平均值，并非逐Token实测间隔。',
    reasoning_stream_event_itl:'同一请求内相邻思考流事件的时间差，每个间隔等权，不跨请求相减。一个事件可能包含多个Token；单事件没有间隔样本。'
  };
  if(thinkingNotes[metric.key])parent.append(el('p',thinkingNotes[metric.key],'analysis-note'));
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
// Exact persisted view/metric pairs of the online serving calculator.
// Session metrics reuse those views unchanged, so the provider name is
// deliberately not part of the match; nothing else may be substituted.
const calculatorViews = [
  ['request_view','ttft'], ['request_view','tpot'],
  ['request_view','itl'], ['request_view','e2el'],
  ['service_view','request_throughput'],
  ['service_view','output_token_throughput']
];
// Menu order; total_token_throughput is also saved by the independent
// run-level token_throughput metric, so its rows are only core when they
// come from the identified serving calculator provider.
const corePerformanceViews = [
  ...calculatorViews, ['service_view','total_token_throughput']
];
const phasePerformanceViews = [
  ['request_view','reasoning_ttft'], ['request_view','reasoning_tpot'],
  ['request_view','reasoning_itl'], ['request_view','reasoning_stream_event_itl'],
  ['request_view','reasoning_duration'], ['request_view','content_duration'],
  ['request_view','first_content_latency']
];
function calculatorProviderName(c) {
  for (const [name, metric] of Object.entries(c?.metrics||{})) {
    const values=metric?.values;
    if (calculatorViews.some(([view,key])=>
        Object.hasOwn(values?.[view]||{},key))) return name;
  }
  return null;
}
function corePerformanceMetrics(c) {
  const available=metricsFor(c);
  const provider=calculatorProviderName(c);
  const core=corePerformanceViews.flatMap(([view,key])=>
    available.filter(m=>m.key===key && m.parts.at(-2)===view &&
      (provider===null || m.parts[0]===provider)));
  return core.concat(phasePerformanceViews.flatMap(([view,key])=>
    available.filter(m=>m.key===key && m.parts.at(-2)===view)));
}
// Saved request/service fields identify the online calculator independently;
// missing distributions do not hide available service statistics.
function servingMetric(c) {
  const name=calculatorProviderName(c);
  return name===null?undefined:c?.metrics?.[name];
}
// Per-turn facts live in turn_view; a metric carrying it is a session metric.
function sessionValues(c) {
  if (!c || c.status === 'discarded') return null;
  for (const metric of Object.values(c.metrics||{}))
    if (metric?.values?.turn_view) return metric.values;
  return null;
}
const turnSeries = [['ttft_ms_by_turn','TTFT'],
  ['e2el_ms_by_turn','E2EL'],['input_tokens_by_turn','输入 Token'],
  ['output_tokens_by_turn','输出 Token']];
function renderSessionTurns(parent,c) {
  const values=sessionValues(c);
  if (!values) return;
  const turn=values.turn_view,session=values.session_view||{};
  const section=el('div',undefined,'analysis-group');
  section.append(el('h3','按轮次统计'));
  const at=(key,index)=>(turn[key]||[]).find(row=>row.turn_index===index);
  const indexes=[...new Set(turnSeries.flatMap(([key])=>
    (turn[key]||[]).map(row=>row.turn_index)))].sort((a,b)=>a-b);
  if (!indexes.length) {
    section.append(el('p','没有已保存的成功轮次。','empty'));
    parent.append(section);
    return;
  }
  section.append(el('p','已观测轮次数：'+valueText(turn.turns_observed)+
    '。第 N 轮只包含延续到该轮的会话，样本数减少本身不是性能变化，'+
    '请与各轮样本数、未开始轮次和中止原因一起阅读。','analysis-note'));
  const records=indexes.map(index=>({index}));
  dataTable(section,[
    {title:'轮次',numeric:true,sortable:true,get:r=>r.index},
    {title:'各轮样本数（TTFT / E2EL / 输入 / 输出）',numeric:true,sortable:true,
      get:r=>at('ttft_ms_by_turn',r.index)?.count,
      render:(cell,r)=>{
        const counts=turnSeries.map(([key])=>at(key,r.index)?.count);
        cell.textContent=counts.map(valueText).join(' / ');
        cell.title='依次为本轮已保存的 '+
          turnSeries.map(([,title])=>title).join('、')+' 样本数';
      }},
    ...turnSeries.flatMap(([key,title])=>[
      {title:title+' 均值',numeric:true,sortable:true,
        unit:at(key,indexes[0])?.unit,get:r=>at(key,r.index)?.mean},
      {title:title+' P99',numeric:true,sortable:true,
        unit:at(key,indexes[0])?.unit,get:r=>at(key,r.index)?.p99}])],
    records,'stat-table');
  barChart(section,indexes.map(index=>({label:'第 '+index+' 轮',
    value:at('ttft_ms_by_turn',index)?.mean,
    count:at('ttft_ms_by_turn',index)?.count})),
    at('ttft_ms_by_turn',indexes[0])?.unit||'ms','各轮TTFT均值');
  const scalarValue=key=>session[key]?.value;
  const statsValue=key=>session[key]?.mean;
  const summary=[['已结束会话',scalarValue('sessions_finished')],
    ['计划轮次均值',statsValue('planned_turns')],
    ['完成轮次均值',statsValue('completed_turns')],
    ['未开始轮次',scalarValue('not_started_turns')],
    ['发送失败轮次',scalarValue('sent_failed_turns')],
    ['会话耗时均值（ms）',statsValue('session_latency')]]
    .filter(([,value])=>value!=null)
    .map(([label,value])=>label+'：'+valueText(value));
  if (summary.length)
    section.append(el('p',summary.join(' · '),'analysis-note'));
  const reasons=Object.entries(session.abort_reasons||{});
  if (reasons.length)
    section.append(el('p','会话中止原因：'+reasons
      .map(([reason,count])=>reason+' '+valueText(count)).join(' · '),
      'analysis-note'));
  parent.append(section);
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
function renderPhaseCounts(parent,c) {
  const counts=metricsFor(c).filter(m=>m.parts.at(-2)==='phase_counts');
  if(!counts.length)return;
  const section=el('div',undefined,'analysis-group');
  section.append(el('h3','分阶段统计口径与样本数量'));
  section.append(el('p','仅成功且 finish_reason=stop 的请求进入分布；缺失阶段不计入该阶段样本数。'+
    '输出时长为客户端收到本通道首尾事件的时间差，单事件为 0；通道跨度可能重叠，相加不等于总耗时。'+
    '截断和其他排除请求的观察时长仅保留在请求明细中。','analysis-note'));
  dataTable(section,[{title:'样本范围',get:m=>readable(m.key)},
    {title:'请求数',numeric:true,get:m=>m.value}],counts,'stat-table');
  parent.append(section);
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
  renderCacheDetails(content,c);
  if(c.status==='discarded') {content.append(el('p','已丢弃，不展示有效指标。请查看诊断。','empty'));return;}
  const metrics=metricsFor(c);
  const failedMetrics=Object.entries(c.metrics||{}).filter(([,m])=>m?.error||m?.status==='failed');
  if(c.status!=='success'||failedMetrics.length)
    context.append(el('p','结果存在异常或未完成状态，请结合指标状态、覆盖率与诊断阅读。','analysis-note'));
  if(task.category==='generation_performance') {
    renderPerformance(content,metrics.find(m=>m.path===$('analysis-metric').value));
    renderPhaseCounts(content,c);
    renderSessionTurns(content,c);
  } else {
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
    const performanceMetrics=corePerformanceMetrics(c);
    const cacheMetrics=metricsFor(c).filter(m=>
      m.parts[0]==='prefix_cache_hit_rate' && cacheAnalysisKeys.includes(m.key));
    const metrics=[...performanceMetrics,...cacheMetrics];
    if (cacheMetrics.length) {
      for (const [label,items] of [['性能指标',performanceMetrics],['Prefix Cache',cacheMetrics]]) {
        const group=el('optgroup');group.label=label;
        items.forEach(m=>option(group,m.path,readable(m.key)));
        select.append(group);
      }
    } else for(const m of metrics)option(select,m.path,readable(m.key));
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
    if(p.prefix_cache_target!=null)
      conditions.push('目标命中率 '+valueText(p.prefix_cache_target,'cache_ratio')+'%');
    if(p.max_concurrency!=null)conditions.push('并发 '+p.max_concurrency);
    if(p.input_tokens!=null)conditions.push('输入 '+p.input_tokens);
    if(p.output_tokens!=null)conditions.push('输出上限 '+p.output_tokens);
    if(p.multi_num_sessions!=null)
      conditions.push('会话 '+p.multi_num_sessions);
    if(p.multi_max_turns!=null)conditions.push('轮数上限 '+p.multi_max_turns);
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
    const path=[...$('analysis-metric').options].map(o=>o.value)
      .find(value=>value.endsWith('/'+match[3]));
    if(path) {$('analysis-metric').value=path;renderAnalysis();}
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

// Prefix Cache presentation reads saved values and judgments only.
const cacheLabels = {
  target_rate:'目标命中率', theoretical_shared_ratio:'理论共享比例',
  actual_hit_rate:'实际缓存命中率', runtime_deviation:'运行偏差',
  construction_deviation:'构造偏差', target_deviation:'总体目标偏差',
  request_coverage:'请求统计覆盖率', token_coverage:'Token 统计覆盖率',
  successful_request_coverage:'成功集合请求覆盖率',
  successful_token_coverage:'成功集合 Token 覆盖率',
  success_rate:'正式请求成功率', input_length_conformance_rate:'输入长度符合率',
  total_prompt_tokens:'成功请求输入 Token 总数（S）',
  measured_prompt_tokens:'已测子集输入 Token 总数（M）',
  total_cached_tokens:'已测子集缓存命中 Token 总数（M）',
  observed_subset_hit_rate:'已测子集命中率（非完整 Case）'
};
Object.assign(metricNames,cacheLabels);
parameterNames.prefix_cache_target='目标命中率（%）';
const cacheSummaryKeys=['target_rate','theoretical_shared_ratio','actual_hit_rate',
  'runtime_deviation','request_coverage','token_coverage'];
const cacheAnalysisKeys=[...cacheSummaryKeys,'target_deviation'];
const cacheMetric=c=>c?.metrics?.prefix_cache_hit_rate;
const isPrefixCache=c=>!!cacheMetric(c) || Object.hasOwn(c?.parameters||{},'prefix_cache_target');
const cacheValue=(c,key)=>c?.status==='discarded'?null:cacheMetric(c)?.values?.[key]?.value;
const cacheStates={observed_mismatch:'已完成但命中不符',matched:'符合',under_hit:'命中不足',over_hit:'高于预期',missing:'缺失',valid:'有效',partial:'部分覆盖',unavailable:'不可测',invalid:'无效',
  passed:'符合',failed:'未通过',unknown:'未知',success:'完成',
  not_applicable:'不适用',construction_limited:'受构造上限限制'};
const cacheState=s=>(cacheStates[s||'unknown']||s)+' / '+(s||'unknown');
function cacheEligible(c) {
  if (!isPrefixCache(c)) return true;
  const m=cacheMetric(c);
  const planned=c.parameters?.num_requests, rate=cacheValue(c,'actual_hit_rate');
  return c.status==='success' && m?.status==='success' && !m.error &&
    m.measurement?.status==='valid' && numeric(rate) && rate>=0 && rate<=1 &&
    Number.isSafeInteger(planned) && planned>0 &&
    ['submitted_requests','successful_requests','reporting_requests'].every(k=>cacheValue(c,k)===planned);
}
const cacheWarmup=c=>c?.setup_diagnostic||cacheMetric(c)?.warmup||{};
const warmupBool=value=>value===true?'是':value===false?'否':'未采集';
function renderCacheSummary(parent,task,c,ti,ci) {
  const panel=el('section',undefined,'cache-summary-case'), p=c.parameters||{};
  panel.id='summary-'+ti+'-'+ci;
  panel.tabIndex=-1;
  const header=el('div',undefined,'cache-summary-header');
  const title=taskSectionTitle(task)==='前缀缓存评测'?
    targetHeading(c)+' · '+c.id:c.id;
  header.append(el('strong',title),el('span',names[c.status]||c.status,'muted'),
    link('查看分析','analysis-'+ti+'-'+ci));
  panel.append(header);
  const grid=el('div',undefined,'cache-summary-grid');
  function quantity(value,unit,suffix) {
    return numeric(value)?valueText(value,unit)+suffix:'未采集';
  }
  function table(title,rows) {
    dataTable(grid,[{title,get:r=>r[0]},{title:'结果',get:r=>r[1]}],rows);
  }
  table('缓存命中',[
    ['目标命中率',quantity(cacheTarget(c),'cache_ratio','%')],
    ...[['实际命中率','actual_hit_rate'],['理论共享比例','theoretical_shared_ratio'],
      ['请求覆盖率','request_coverage'],['Token 覆盖率','token_coverage']].map(
      ([label,key])=>[label,quantity(cacheValue(c,key),'cache_ratio','%')]),
    ['运行偏差',quantity(cacheValue(c,'runtime_deviation'),'ratio_pp',' pp')]
  ]);
  const warmup=cacheWarmup(c), metric=cacheMetric(c);
  const reasons=metric?.measurement?.reasons||[];
  if(reasons.length)panel.append(el('p',
    '测量诊断：'+reasons.map(r=>r==='formal_observation_incomplete'?
      '完整正式请求范围观测不足':r).join('、'),'analysis-note'));
  table('负载与状态',[
    ['并发上限',p.max_concurrency],['输入长度',quantity(p.input_tokens,'',' token')],
    ['输出上限',quantity(p.output_tokens,'',' token')],
    ['预热',[(warmup.warmup_policy||p.prefix_cache_warmup_policy||'strict'),
      cacheState(warmup.warmup_status||warmup.status).split(' / ')[0]].join(' · ')],
    ['测量有效性',cacheState(metric?.measurement?.status).split(' / ')[0]],
    ['目标达标',cacheState(metric?.target_attainment?.status).split(' / ')[0]]
  ]);
  const v=servingMetric(c)?.values||{};
  const rows=[['完成请求',c.coverage?.complete],['失败请求',c.coverage?.failed]];
  for(const [label,node,field] of [
    ['TTFT 均值',v.request_view?.ttft,'mean'],['TPOT 均值',v.request_view?.tpot,'mean'],
    ['输出吞吐',v.service_view?.output_token_throughput,'value'],
    ['总吞吐',v.service_view?.total_token_throughput,'value']])
    rows.push([label,quantity(node?.[field],node?.unit,' '+(node?.unit||''))]);
  table('请求与性能',rows);
  panel.append(grid);parent.append(panel);
}
function cacheStatusColumns() {
  return [
    {title:'预热模式',get:r=>cacheWarmup(r.c).warmup_policy||
      r.c.parameters?.prefix_cache_warmup_policy||'strict'},
    {title:'预热完成 / 完全符合',get:r=>warmupBool(cacheWarmup(r.c).warmup_completed)+
      ' / '+warmupBool(cacheWarmup(r.c).warmup_fully_matched)},
    {title:'预热不足前缀数',get:r=>cacheWarmup(r.c).warmup_under_hit_prefixes},
    {title:'三阶段预热',get:r=>cacheState(r.c.setup_diagnostic?.status ||
      cacheWarmup(r.c).warmup_status)},
    ...[['measurement','测量有效性'],
      ['target_attainment','目标达标']].map(([key,title])=>({
        title,get:r=>cacheState(r.c.status==='discarded'?null:cacheMetric(r.c)?.[key]?.status)}))
  ];
}
function renderCacheDetails(parent,c) {
  if (!isPrefixCache(c) && !c.setup_diagnostic) return;
  const section=el('div',undefined,'analysis-group prefix-cache-details');
  section.append(el('h3','Prefix Cache · 已保存的正式请求统计'));
  const m=c.status==='discarded'?null:cacheMetric(c);
  dataTable(section,[{title:'指标',get:r=>cacheLabels[r.key]},
    {title:'保存值',get:r=>valueText(r.node?.value,r.unit)+
      (numeric(r.node?.value)?' '+displayUnit(r.unit):'')}],
    Object.keys(cacheLabels).map(key=>({key,node:m?.values?.[key],
      unit:key.endsWith('_deviation')?'ratio_pp':m?.values?.[key]?.unit==='ratio'?'cache_ratio':m?.values?.[key]?.unit})));
  dataTable(section,cacheStatusColumns(),[{c}]);
  const warmup=cacheWarmup(c);
  const observations=Object.entries(warmup.warmup_summary||{}).flatMap(
    ([phase,rows])=>rows.map(row=>({phase,...row})));
  if (observations.length) {
    section.append(el('h4','预热 Probe / Verify 与能力检查实测'));
    dataTable(section,[
      {title:'阶段',get:r=>r.phase},
      {title:'prefix_id',numeric:true,get:r=>r.prefix_id},
      {title:'预期命中 Token',numeric:true,get:r=>r.expected_cached_tokens},
      {title:'实际命中 Token',numeric:true,get:r=>r.actual_cached_tokens},
      {title:'偏差 Token',numeric:true,get:r=>r.deviation_tokens},
      {title:'校验状态',get:r=>cacheState(r.validation_state)}
    ],observations);
  }
  const warmupReasons=warmup.reasons||[];
  if (warmupReasons.length)
    section.append(el('p','预热条件异常：'+warmupReasons.join('、'),'analysis-note'));
  section.append(el('p','预热顺序：全部种入 → 全部探测 → 完整验证。'+
    'R=已提交正式请求；S=成功请求；M=成功集合中的有效缓存观测。'+
    '新结果的请求覆盖率使用 R，历史结果的口径见统计范围。'+
    '缺失实测值显示未采集；子集命中率不能代替整 Case 实际命中率。','analysis-note'));
  const diagnostics=el('details');
  diagnostics.append(el('summary','已保存的诊断与统计范围'),
    el('pre',JSON.stringify({measurement:m?.measurement,warmup:m?.warmup,
      target:m?.target_attainment,diagnostics:m?.diagnostics,metadata:m?.metadata,
      setup:c.setup_diagnostic},null,2)));
  section.append(diagnostics);parent.append(section);
}
function cacheComparisonGroups(task) {
  const groups=new Map();
  for (const c of task.cases||[]) {
    if (!isPrefixCache(c)) continue;
    // Conservative: every recorded parameter except the swept target must match.
    const parameters=Object.fromEntries(Object.entries(c.parameters||{})
      .filter(([key])=>key!=='prefix_cache_target' && key!=='prefix_cache_environment'));
    const key=JSON.stringify(stableConfig(parameters));
    if (!groups.has(key)) groups.set(key,{parameters,cases:[]});
    groups.get(key).cases.push(c);
  }
  return [...groups.values()];
}
function renderCacheComparison(parent,task) {
  const groups=cacheComparisonGroups(task);
  if (!groups.length) return;
  const section=el('div',undefined,'analysis-group prefix-cache-comparison');
  section.append(el('h3','不同目标命中率的性能对比'),
    el('p','横轴：目标缓存命中率（%）。其余负载与服务配置相同的 Case 才归为一组。'+
      '预热符合情况与目标达标状态见表格。'+
      '* 表示正式统计不可用；Case 编号见柱形悬浮提示和明细。异常 Case 保留诊断，不参加最佳结果推荐。','analysis-note'));
  const controls=el('div',undefined,'toolbar'),select=el('select'),metric=el('select');
  select.setAttribute('aria-label','缓存对比条件组');
  metric.setAttribute('aria-label','缓存对比指标');
  groups.forEach((g,i)=>option(select,String(i),'条件组 '+(i+1)+' · '+g.cases.length+' Case'));
  const specs=[['actual_hit_rate','实际缓存命中率','cache_ratio'],
    ...corePerformanceViews.map(([view,key])=>[key,readable(key),null,view])];
  specs.forEach(([key,label])=>option(metric,key,label));
  controls.append(select,metric);section.append(controls);
  const context=el('details'),chart=el('div'),table=el('div');
  context.append(el('summary','本组完整测试条件'),el('pre'));
  section.append(context,chart,table);
  function draw() {
    chart.replaceChildren();table.replaceChildren();
    const g=groups[Number(select.value)];
    context.querySelector('pre').textContent=JSON.stringify(g.parameters,null,2);
    const spec=specs.find(([key])=>key===metric.value);
    const rows=g.cases.map(c=>{
      const selected=corePerformanceMetrics(c).find(m=>m.key===spec[0]&&m.parts.at(-2)===spec[3]);
      return {c,target:cacheValue(c,'target_rate')??c.parameters?.prefix_cache_target,
        value:spec[0]==='actual_hit_rate'?cacheValue(c,spec[0]):
          selected?.kind==='statistics'?selected.stats.find(([k])=>k==='mean')?.[1]:selected?.value,
        unit:spec[2]||selected?.unit};
    }).sort((a,b)=>(a.target??Infinity)-(b.target??Infinity));
    const units=[...new Set(rows.filter(r=>numeric(r.value)).map(r=>r.unit))];
    if (units.length<=1) barChart(chart,rows.map(r=>({
      label:valueText(r.target,'cache_ratio')+'%'+(cacheEligible(r.c)?'':' *'),
      detail:r.c.id+(cacheEligible(r.c)?'':' [正式统计不可用]'),value:r.value})),units[0]||spec[2],
      '横轴：目标命中率（%）；纵轴：'+spec[1]+(spec[3]==='request_view'?' Mean':''));
    else chart.append(el('p','保存单位不一致，仅展示明细，不混合绘图。','analysis-note'));
    dataTable(table,[{title:'Case',get:r=>r.c.id},
      ...['target_rate','theoretical_shared_ratio','actual_hit_rate'].map(key=>({
        title:cacheLabels[key]+'（%）',numeric:true,unit:'cache_ratio',get:r=>cacheValue(r.c,key)})),
      ...(spec[0]==='actual_hit_rate'?[]:[{title:spec[1],
        get:r=>valueText(r.value,r.unit)+(numeric(r.value)?' '+displayUnit(r.unit):'')}]),
      ...cacheStatusColumns(),
      {title:'推荐资格',get:r=>usablePerformanceResult(r.c)?'可参与':'排除（统计或执行不完整）'}],rows);
  }
  select.onchange=metric.onchange=draw;draw();parent.append(section);
}
function renderCacheRequest(parent,response) {
  const c=response.prefix_cache;
  if (!c) return;
  parent.append(el('h4','Prefix Cache 请求事实'));
  const rows=[
    ['prefix_id',c.prefix_id],['共享前缀长度',c.shared_prefix_tokens],
    ['完整输入 Token 数（服务端）',c.prompt_tokens],
    ['实际缓存 Token 数',c.cached_tokens],
    ['单请求实际命中率',valueText(c.hit_rate,'cache_ratio')+(numeric(c.hit_rate)?' %':'')],
    ['缓存统计状态',cacheState(c.usage_state)],
    ['输入长度符合状态',response.lengths?.condition||'unknown'],
    ['异常原因',c.reason||'未记录']
  ];
  dataTable(parent,[{title:'字段',get:r=>r[0]},{title:'保存值',get:r=>r[1]}],rows);
}

function reportRowText(row,key) {
  if (row.metric.startsWith('/prefix_cache_hit_rate/values/') && row.unit==='ratio') {
    const unit=row.metric.endsWith('_deviation')?'ratio_pp':'cache_ratio';
    if (key==='value') return valueText(row.value,unit);
    if (key==='unit') return displayUnit(unit);
  }
  return fmt(key==='task'?taskDisplayName(row.task):row[key]);
}
