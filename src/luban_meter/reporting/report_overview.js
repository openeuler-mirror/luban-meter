// Use saved facts; never recompute distributions or infer untested limits.
function stableConfig(value) {
  if (Array.isArray(value)) return value.map(stableConfig);
  if (value && typeof value === 'object') return Object.fromEntries(
    Object.keys(value).sort().map(key=>[key,stableConfig(value[key])]));
  return value;
}
function usablePerformanceResult(c) {
  const metric=c.metrics?.streaming_performance;
  return c.status==='success' && c.collection_complete===true &&
    (!c.metrics_status || c.metrics_status==='complete') &&
    (!metric?.status || metric.status==='success') && !metric?.error &&
    !['failed','incomplete'].some(k=>c.coverage?.[k]>0) &&
    !(metric?.values?.service_view?.failed_requests?.value>0);
}
function performanceSummary(taskList) {
  const groups=new Map();
  let total=0, included=0;
  taskList.forEach((task,ti)=>{
    if(task.category!=='generation_performance')return;
    task.cases.forEach((c,ci)=>{
      total++;
      const p=c.parameters||{},metric=c.metrics?.streaming_performance;
      const service=metric?.values?.service_view;
      const throughput=service?.output_token_throughput;
      if(!usablePerformanceResult(c) ||
        !Number.isInteger(p.max_concurrency) || p.max_concurrency<1 ||
        !p.model || !p.profile || !numeric(throughput?.value) ||
        throughput.value<0 || throughput.unit!=='token/s')return;
      // Keep task/environment and every recorded parameter except concurrency
      // fixed. Object key order is not a change in test conditions.
      const fixed=Object.fromEntries(Object.entries(p)
        .filter(([key])=>key!=='max_concurrency'));
      const key=JSON.stringify([ti,stableConfig(fixed)]);
      if(!groups.has(key))groups.set(key,{task,ti,fixed,points:[]});
      groups.get(key).points.push({c,ci,concurrency:p.max_concurrency,
        value:throughput.value,unit:throughput.unit});
      included++;
    });
  });
  for(const group of groups.values()) {
    group.points.sort((a,b)=>a.concurrency-b.concurrency || a.ci-b.ci);
    group.maximum=Math.max(...group.points.map(p=>p.value));
    group.peaks=group.points.filter(p=>p.value===group.maximum);
  }
  return {groups:[...groups.values()],total,included};
}
function performanceTrend(group) {
  const points=group.points;
  if(points.length<2)return '只有一个有效并发点，无法判断变化趋势。';
  if(new Set(points.map(p=>p.concurrency)).size!==points.length)
    return '同一并发有多次测量，各点分别保留，不合并均值或判定单一趋势。';
  const differences=points.slice(1).map((p,i)=>p.value-points[i].value);
  let trend='吞吐量随并发变化而波动';
  if(differences.every(d=>d===0))trend='已测各并发点的吞吐量相同';
  else if(differences.every(d=>d>=0))trend= differences.some(d=>d===0)?
    '吞吐量整体上升，部分并发点持平':'吞吐量随并发增加持续上升';
  else if(differences.every(d=>d<=0))trend= differences.some(d=>d===0)?
    '吞吐量整体下降，部分并发点持平':'吞吐量随并发增加持续下降';
  else {
    const peak=points.findIndex(p=>p.value===group.maximum);
    if(peak>0 && peak<points.length-1 &&
      differences.slice(0,peak).every(d=>d>=0) &&
      differences.slice(peak).every(d=>d<=0))trend='吞吐量随并发先上升、后回落';
  }
  const first=points[0],last=points.at(-1);
  return '并发上限 '+first.concurrency+' → '+last.concurrency+'：'+trend+
    '，输出吞吐量从 '+valueText(first.value)+' 到 '+valueText(last.value)+' token/s。';
}
function overviewMetric(parent,label,value,unit,count) {
  const item=el('div',undefined,'overview-metric');
  const number=el('div',valueText(value,unit)+
    (numeric(value)&&unit==='ratio'?'%':''),'overview-value');
  if(numeric(value))number.title='原始值：'+String(value)+' '+unit;
  item.append(el('div',label,'muted'),number);
  if(unit!=='ratio')item.append(el('div',displayUnit(unit),'overview-unit'));
  if(count!==undefined)item.append(el('div',
    '已保存样本 / 计数：'+valueText(count),'overview-unit'));
  parent.append(item);
}
function concurrencyChart(parent,group) {
  const points=group.points,width=760,height=290;
  const left=75,right=28,top=35,bottom=60;
  const min=points[0].concurrency,max=points.at(-1).concurrency;
  const high=group.maximum||1;
  const x=p=>min===max?(left+width-right)/2:
    left+(p.concurrency-min)/(max-min)*(width-left-right);
  const y=p=>height-bottom-p.value/high*(height-top-bottom);
  const svg=svgElement('svg',{viewBox:`0 0 ${width} ${height}`,role:'img',
    'aria-label':'并发上限与输出 Token 吞吐量；突出标记已测最大值'});
  const title=svgElement('title',{});
  title.textContent='横轴为配置并发上限，纵轴为输出吞吐量（token/s）。'+performanceTrend(group);
  svg.append(title);
  for(let i=0;i<=4;i++) {
    const value=high*i/4,py=height-bottom-value/high*(height-top-bottom);
    svg.append(svgElement('line',{x1:left,x2:width-right,y1:py,y2:py,stroke:'var(--line)'}));
    const label=svgElement('text',{x:left-9,y:py+4,'text-anchor':'end',
      fill:'currentColor','font-size':12});
    label.textContent=valueText(value);svg.append(label);
  }
  const repeated=new Set(points.map(p=>p.concurrency)).size!==points.length;
  if(!repeated && points.length>1)svg.append(svgElement('polyline',{
    points:points.map(p=>x(p)+','+y(p)).join(' '),fill:'none',
    stroke:'var(--accent)','stroke-width':2}));
  const ticks=new Set();
  points.forEach(p=>{
    const peak=p.value===group.maximum;
    const a=svgElement('a',{href:'#analysis-'+group.ti+'-'+p.ci,
      'aria-label':p.c.id+'，并发上限 '+p.concurrency+'，输出吞吐量 '+
        valueText(p.value)+' token/s'+(peak?'，已测最大值':'')});
    const mark=svgElement('circle',{cx:x(p),cy:y(p),r:peak?7:4,
      fill:peak?'var(--accent)':'var(--panel)',stroke:'var(--accent)',
      'stroke-width':2,'data-case-index':p.ci,'data-peak':String(peak)});
    const tip=svgElement('title',{});tip.textContent=a.getAttribute('aria-label');
    mark.append(tip);a.append(mark);svg.append(a);
    if(!ticks.has(p.concurrency)) {
      const label=svgElement('text',{x:x(p),y:height-bottom+23,
        'text-anchor':'middle',fill:'currentColor','font-size':12});
      label.textContent=p.concurrency;svg.append(label);ticks.add(p.concurrency);
    }
  });
  for(const [text,px,py,anchor] of [['输出吞吐量（token/s）',left,18,'start'],
    ['● 已测最大值',width-right,18,'end'],
    ['配置并发上限',(left+width-right)/2,height-8,'middle']]) {
    const label=svgElement('text',{x:px,y:py,'text-anchor':anchor,
      fill:'currentColor','font-size':12});label.textContent=text;svg.append(label);
  }
  const box=el('div',undefined,'trend-chart');box.append(svg);parent.append(box);
}
function performanceGroupLabel(group,index) {
  const p=group.fixed;
  return ['条件组 '+(index+1),taskDisplayName(group.task.name),p.model,
    '输入 '+valueText(p.input_tokens),'输出上限 '+valueText(p.output_tokens),
    p.profile].join(' · ');
}
function summaryScopeGroups(taskList,summary=performanceSummary(taskList)) {
  const key=(ti,fixed)=>JSON.stringify([ti,stableConfig(fixed)]);
  const available=new Map(summary.groups.map(g=>[key(g.ti,g.fixed),g]));
  const scopes=new Map();
  taskList.forEach((task,ti)=>{
    if(task.category!=='generation_performance')return;
    task.cases.forEach(c=>{
      const p=c.parameters||{};
      if(c.collection_complete!==true || c.status==='discarded' || !p.model ||
        !p.profile || !Number.isSafeInteger(p.max_concurrency) || p.max_concurrency<1)return;
      const fixed=Object.fromEntries(Object.entries(p).filter(([name])=>name!=='max_concurrency'));
      const identity=key(ti,fixed);
      if(!scopes.has(identity))scopes.set(identity,available.get(identity)||
        {task,ti,fixed,points:[],maximum:null,peaks:[]});
    });
  });
  return [...scopes.values()];
}
function summaryGroupRecords(group) {
  if(!group)return [];
  const fixed=JSON.stringify(stableConfig(group.fixed));
  return group.task.cases.map((c,ci)=>({c,ci})).filter(({c})=>
    JSON.stringify(stableConfig(Object.fromEntries(Object.entries(c.parameters||{})
      .filter(([key])=>key!=='max_concurrency'))))===fixed);
}
function savedRequestCounts(c) {
  const service=c.metrics?.streaming_performance?.values?.service_view;
  const count=key=>{
    const m=service?.[key];
    return m?.unit==='request' && Number.isSafeInteger(m.value) && m.value>=0?m.value:null;
  };
  const success=count('successful_requests'),total=count('total_requests');
  const failed=count('failed_requests');
  const hasFailed=Object.hasOwn(service||{},'failed_requests');
  const valid=success!==null && total!==null && success<=total &&
    (!hasFailed || (failed!==null && success+failed===total)) &&
    !c.metrics?.streaming_performance?.error;
  return {success,total,valid};
}
function performanceOverviewCards(records,bestMode=true) {
  const cards=[
    {id:'peak-throughput',title:'Peak Throughput',unit:'tok/s',tone:'cyan',
      description:'higher is better · 已测最大值'},
    {id:'best-ttft-p99',title:'Best TTFT P99',unit:'ms',tone:'purple',
      description:'lower is better · 最小已保存 P99'},
    {id:'best-tpot-p99',title:'Best TPOT P99',unit:'ms/token',tone:'purple',
      description:'lower is better · 原 Token 权重口径'},
    {id:'success-rate',title:'Success Rate',unit:'%',tone:'green',kind:'percentage',
      description:'单 Case 成功率 · 失败计入分母'},
    {id:'max-concurrency',title:'Max Tested Concurrency',unit:'并发',tone:'yellow',
      description:'已测配置上限 · 非实际峰值'},
    {id:'best-e2e-p99',title:'Best E2E P99',unit:'ms',tone:'cyan',
      description:'lower is better · 最小已保存 P99'},
  ].map(card=>({...card,value:null,source:'缺少可用的已保存数据',href:'#summary-section'}));
  if(!bestMode) {
    ['Throughput','TTFT P99','TPOT P99','Success Rate','Configured Concurrency','E2E P99']
      .forEach((title,index)=>{cards[index].title=title;});
    cards.forEach(card=>{card.description='当前 Case 的已保存指标';});
  } else cards[3].title='Best Success Rate';
  if(!records.length)return cards;
  const completed=records.filter(({c})=>c.collection_complete===true &&
    c.status!=='discarded' && (!c.metrics_status || c.metrics_status==='complete'));
  const usable=bestMode?records.filter(({c})=>usablePerformanceResult(c)):
    completed.filter(({c})=>!c.metrics?.streaming_performance?.error);
  function extreme(index,points,minimum=false) {
    if(!points.length)return;
    const best=points.reduce((a,b)=>(minimum?b.value<a.value:b.value>a.value)?b:a);
    const ties=points.filter(p=>p.value===best.value).length;
    cards[index].value=best.value;
    cards[index].source=taskDisplayName(best.task.name)+' / '+best.c.id+
      ' · C'+valueText(best.c.parameters?.max_concurrency)+
      (ties>1?' · '+ties+' 个 Case 并列':'');
    cards[index].href='#summary-'+best.ti+'-'+best.ci;
    if(index===3) {
      const counts=savedRequestCounts(best.c);
      cards[index].source+=' · '+counts.success+' / '+counts.total+' 请求';
    }
  }
  extreme(0,usable.flatMap(record=>{
    const m=record.c.metrics?.streaming_performance?.values?.service_view?.output_token_throughput;
    return m?.unit==='token/s' && numeric(m.value) && m.value>=0?
      [{...record,value:m.value}]:[];
  }));
  for(const [index,key] of [[1,'ttft'],[2,'tpot'],[5,'e2el']]) {
    const units=key==='tpot'?{'ms/token':1,'s/token':1000}:{ms:1,s:1000};
    extreme(index,usable.flatMap(record=>{
      const m=record.c.metrics?.streaming_performance?.values?.request_view?.[key];
      const value=m?.p99*units[m?.unit];
      return numeric(m?.p99) && m.p99>=0 && numeric(value) &&
        (m.count==null || (Number.isSafeInteger(m.count) && m.count>0)) &&
        (m.weight_sum==null || (numeric(m.weight_sum) && m.weight_sum>0))?
        [{...record,value}]:[];
    }),true);
  }
  extreme(3,completed.flatMap(record=>{
    const counts=savedRequestCounts(record.c);
    return counts.valid && counts.total>0?[{...record,value:counts.success/counts.total}]:[];
  }));
  extreme(4,completed.flatMap(record=>{
    const p=record.c.parameters?.max_concurrency;
    const service=record.c.metrics?.streaming_performance?.values?.service_view;
    return Number.isSafeInteger(p) && p>0 && service?.total_requests?.unit==='request' &&
      Number.isSafeInteger(service.total_requests.value) && service.total_requests.value>0 &&
      !record.c.metrics?.streaming_performance?.error?[{...record,value:p}]:[];
  }));
  if(!bestMode) {
    const record=records[0],concurrency=record.c.parameters?.max_concurrency;
    if(Number.isSafeInteger(concurrency) && concurrency>0) {
      cards[4].value=concurrency;cards[4].source=record.c.id+' · 配置并发上限';
      cards[4].href='#summary-'+record.ti+'-'+record.ci;
    }
  }
  return cards;
}
function renderSuccessCounts(parent,group) {
  const details=el('details');details.id='overview-success-counts';details.tabIndex=-1;
  details.append(el('summary','成功率：已完成 Case 的请求计数依据'),
    el('p','仅合计本组已完成 Case 的保存计数，失败请求计入总数。任一 Case 计数缺失、'+
      '不一致，或合计总请求数为零时，成功率显示 Unavailable；不以采集完成数替代请求数。','overview-notice'));
  const records=summaryGroupRecords(group).filter(({c})=>c.collection_complete===true &&
    c.status!=='discarded' && (!c.metrics_status || c.metrics_status==='complete'))
    .map(record=>({...record,counts:savedRequestCounts(record.c)}));
  dataTable(details,[{title:'Case',get:r=>r.c.id},
    {title:'并发上限',numeric:true,get:r=>r.c.parameters.max_concurrency},
    {title:'状态',get:r=>names[r.c.status]||r.c.status},
    {title:'成功请求',numeric:true,get:r=>r.counts.success},
    {title:'总请求',numeric:true,get:r=>r.counts.total},
    {title:'计数校验',get:r=>r.counts.valid?'有效':'缺失或不一致'}],records);
  parent.append(details);
}
function renderPerformanceOverview() {
  const parent=$('performance-overview');parent.append(el('h3','推理性能总结'));
  const summary=performanceSummary(tasks);
  const groups=summaryScopeGroups(tasks,summary);
  if(!groups.length) {
    parent.append(el('p',tasks.some(t=>t.category==='generation_performance')?
      '没有满足总结条件的在线性能 Case。需要成功且采集完整的结果、配置并发上限、'+
      '模型、负载策略以及已保存的服务窗口输出吞吐量（token/s）；其他结果请查看 Case 分析。':
      '本次未包含生成性能评测。','muted'));
    return;
  }
  const label=el('label','选择测试条件组','overview-picker'),select=el('select');
  select.id='summary-performance-group';
  groups.forEach((group,index)=>option(select,String(index),performanceGroupLabel(group,index)));
  label.htmlFor=select.id;label.append(select);parent.append(label);
  const content=el('div');content.id='overview-performance-content';
  content.setAttribute('aria-live','polite');parent.append(content);
  function draw() {
    content.replaceChildren();
    const group=groups[Number(select.value)],point=group.peaks[0];
    if(!point) {
      content.append(el('p','本组没有可用于峰值与趋势的成功性能 Case；'+
        '已保存的成功率、并发配置和可用 P99 仍分别展示，缺失数据不作为零。','overview-notice'));
      renderSuccessCounts(content,group);return;
    }
    const peak=el('div',undefined,'peak');
    peak.append(el('div','本组已测最高输出吞吐量','peak-heading'));
    const value=el('div',valueText(point.value)+' ','peak-value');
    value.title='原始值：'+String(point.value)+' token/s';
    value.append(el('small','token/s'));peak.append(value);
    const where=el('p');
    group.peaks.forEach((p,i)=>{
      if(i)where.append(document.createTextNode('；'));
      where.append(link(p.c.id+' · 并发上限 '+p.concurrency,
        'analysis-'+group.ti+'-'+p.ci));
    });
    if(group.peaks.length>1)where.append(document.createTextNode('（并列最大值）'));
    peak.append(where);
    peak.append(el('p',performanceGroupLabel(group,Number(select.value)),
      'overview-context muted'));
    const values=point.c.metrics.streaming_performance.values;
    const metrics=el('div',undefined,'overview-metrics');
    const throughput=values.service_view.request_throughput;
    overviewMetric(metrics,'请求吞吐量',throughput?.value,throughput?.unit||'');
    for(const key of ['ttft','tpot','itl']) {
      const m=values.request_view?.[key];
      overviewMetric(metrics,key.toUpperCase()+' Mean',m?.mean,
        m?.unit||'',m?.sample_count??m?.count??null);
    }
    peak.append(metrics,el('p','以上吞吐与延迟均来自 '+point.c.id+
      '；Mean 使用原指标保存的均值与权重口径。','overview-notice'));
    content.append(peak,el('p',performanceTrend(group),'trend-text'));
    const boundary=group.peaks.some(p=>p.concurrency===group.points.at(-1).concurrency);
    content.append(el('p',boundary?
      '最大值位于最高已测并发，当前只能确定已测范围内最高，不能据此确定服务极限。':
      '标记的是已测并发点中的最大值，不推断未测试并发的结果。','overview-notice'));
    concurrencyChart(content,group);
    content.append(el('p','可横向滚动查看完整曲线；点击数据点进入对应 Case 分析。','overview-notice'));
    const fixed=el('details');fixed.append(el('summary','本组固定测试条件'),
      el('pre',JSON.stringify(Object.fromEntries(Object.entries(group.fixed)
        .filter(([key])=>key!=='api_key')),null,2)));
    content.append(fixed);
    renderSuccessCounts(content,group);
  }
  const highest=Math.max(...groups.map(g=>g.maximum??-Infinity));
  select.value=String(groups.findIndex(g=>(g.maximum??-Infinity)===highest));
  select.onchange=draw;draw();
  parent.append(el('p','纳入 '+summary.included+' / '+summary.total+
    ' 个性能 Case。异常、采集不完整或缺少必要字段的 Case 不参与峰值与趋势；'+
    '其状态及已保存指标仍保留在结果摘要、Case 分析和运行诊断。','overview-notice'));
}
function qualityScoreMetrics(c) {
  return metricsFor(c).filter(m=>m.kind==='scalar' &&
    !/^(count(?:_|$)|total_|scored_|passed$|parse_failed$|service_failed$|sandbox_failed$|execution_timeout$)/.test(m.key) &&
    !c.metrics[m.parts[0]]?.error &&
    (!c.metrics[m.parts[0]]?.status || c.metrics[m.parts[0]].status==='success'));
}
function overviewCaseLabel(record) {
  const p=record.c.parameters||{};
  return [taskDisplayName(record.task.name),record.c.id,p.model,
    p.max_concurrency!=null?'C'+p.max_concurrency:'',
    p.input_tokens!=null?'输入 '+p.input_tokens:'',
    p.output_tokens!=null?'输出 '+p.output_tokens:'',
    names[record.c.status]||record.c.status].filter(Boolean).join(' · ');
}
function renderPerformanceBest() {
  const records=tasks.flatMap((task,ti)=>task.category==='generation_performance'?
    task.cases.map((c,ci)=>({task,ti,c,ci})):[]);
  const parent=$('overview-scope'),label=el('label','查看性能 Case','overview-picker');
  const select=el('select');select.id='overview-performance-case';label.htmlFor=select.id;
  option(select,'best','各项指标独立择优（默认）');
  records.forEach(record=>option(select,record.ti+':'+record.ci,overviewCaseLabel(record)));
  select.disabled=!records.length;label.append(select);parent.append(label);
  const note=el('p',undefined,'summary-scope-note');parent.append(note);
  function draw() {
    const best=select.value==='best';
    const chosen=best?records:records.filter(r=>r.ti+':'+r.ci===select.value);
    $('cards').dataset.view=best?'best':'case';
    $('overview-performance-heading').textContent=best?'最佳性能指标':'当前 Case 性能概览';
    renderSummaryCards($('cards'),performanceOverviewCards(chosen,best));
    note.textContent=best?'默认独立展示各项已保存最佳值，来源 Case 分别标注；'+
      '不同条件和并发的最优值不代表同一工作点。点击数值查看来源 Case 摘要。':
      overviewCaseLabel(chosen[0])+'。当前卡片仅展示这个 Case 自身的指标。';
  }
  select.onchange=draw;draw();
}
function qualityComparisonGroups(taskList) {
  const keys=['data','dataset_path','split','eval_mode','few_shot','few_shot_path',
    'input_template','input_format','answer_parser','template','field_mapping',
    'max_samples','num_requests','seed','shuffle'];
  const groups=new Map();
  taskList.forEach((task,ti)=>{
    if(task.category!=='model_service_quality')return;
    task.cases.forEach((c,ci)=>{
      const protocol=Object.fromEntries(keys.map(key=>[key,c.parameters?.[key]]));
      const identity=JSON.stringify([ti,stableConfig(protocol)]);
      if(!groups.has(identity))groups.set(identity,{task,ti,protocol,records:[]});
      groups.get(identity).records.push({task,ti,c,ci});
    });
  });
  return [...groups.values()];
}
function qualityDirection(key) {
  if(['accuracy','exact_match','pass_at_1','token_f1','f1','rouge1','rouge2','rougeL'].includes(key))return 1;
  if(['perplexity','mean_loss','bits_per_byte'].includes(key))return -1;
  return null;
}
function qualityMetricFor(record,choice) {
  return choice && qualityScoreMetrics(record.c).find(m=>m.path===choice.path && m.unit===choice.unit);
}
function bestQualityRecord(group,choice) {
  const direction=qualityDirection(choice?.key);
  if(!direction)return null;
  const candidates=group.records.flatMap(record=>{
    const m=qualityMetricFor(record,choice),c=record.c;
    return c.status==='success' && c.collection_complete!==false &&
      (!c.metrics_status || c.metrics_status==='complete') && numeric(m?.value) &&
      m.value>=0 && (m.unit!=='ratio' || m.value<=1) &&
      (m.count==null || (numeric(m.count) && m.count>0))?[{...record,metric:m}]:[];
  });
  return candidates.reduce((best,record)=>!best ||
    (record.metric.value-best.metric.value)*direction>0?record:best,null);
}
function renderQualityBest() {
  const parent=$('overview-quality-best'),heading=el('h3','最佳能力得分');parent.append(heading);
  const groups=qualityComparisonGroups(tasks);
  if(!groups.length) {parent.append(el('p','本次没有可展示的能力评测 Case。','overview-notice'));return;}
  const groupLabel=el('label','数据集与评分条件','overview-picker'),groupSelect=el('select');
  groupSelect.id='overview-quality-group';groupLabel.htmlFor=groupSelect.id;
  groups.forEach((g,i)=>option(groupSelect,String(i),taskDisplayName(g.task.name)+' · '+
    scenarioDisplayName(g.task.scenario)+' · 条件组 '+(i+1)));
  groupLabel.append(groupSelect);groupLabel.hidden=groups.length===1;parent.append(groupLabel);
  const metricLabel=el('label','择优指标','overview-picker'),metricSelect=el('select');
  metricSelect.id='overview-quality-metric';metricLabel.htmlFor=metricSelect.id;
  metricLabel.append(metricSelect);parent.append(metricLabel);
  const caseLabel=el('label','查看能力 Case','overview-picker'),caseSelect=el('select');
  caseSelect.id='overview-quality-case';caseLabel.htmlFor=caseSelect.id;
  caseLabel.append(caseSelect);parent.append(caseLabel);
  const context=el('p',undefined,'summary-scope-note'),cards=el('div',undefined,'summary-cards');
  cards.id='overview-quality-cards';cards.setAttribute('aria-live','polite');parent.append(context,cards);
  let choices=[];
  function draw() {
    const group=groups[Number(groupSelect.value)],choice=choices[Number(metricSelect.value)];
    const automatic=caseSelect.value==='best',winner=bestQualityRecord(group,choice);
    const record=automatic?(winner||group.records[0]):
      group.records.find(r=>String(r.ci)===caseSelect.value);
    const metric=qualityMetricFor(record,choice);
    const usable=numeric(metric?.value) && metric.value>=0 &&
      (metric.unit!=='ratio' || metric.value<=1) &&
      (metric.count==null || (numeric(metric.count) && metric.count>0)) &&
      ['success','partial_failed'].includes(record.c.status) && record.c.collection_complete!==false;
    const direction=qualityDirection(choice?.key);
    cards.dataset.view=automatic&&winner?'best':'case';
    heading.textContent=automatic&&winner?'最佳能力得分':'当前 Case 能力概览';
    context.textContent=scenarioDisplayName(group.task.scenario)+' · '+
      (record.c.parameters?.model||'模型未采集')+' · '+(names[record.c.status]||record.c.status)+' · '+
      (automatic && winner?'所选数据集与指标的最佳已保存得分':
        automatic?'缺少可比较得分或未声明优化方向，展示首个 Case':'当前所选 Case')+
      '；不同数据集不做综合排名。';
    renderSummaryCards(cards,[{id:'quality-score',title:choice?readable(choice.key):'能力得分',
      value:usable?metric.value:null,unit:choice?.unit==='ratio'?'%':choice?.unit||'未注明单位',
      kind:choice?.unit==='ratio'?'percentage':'number',tone:'purple',
      description:direction===1?'higher is better':direction===-1?'lower is better':'未声明优化方向',
      source:taskDisplayName(record.task.name)+' / '+record.c.id+' · 评分样本 / 计数 '+
        valueText(metric?.count),href:'#summary-'+record.ti+'-'+record.ci}]);
  }
  function updateCases() {
    const group=groups[Number(groupSelect.value)];
    caseSelect.replaceChildren();option(caseSelect,'best','自动择优（默认）');
    group.records.forEach(record=>option(caseSelect,String(record.ci),overviewCaseLabel(record)));
    draw();
  }
  function updateGroup() {
    const group=groups[Number(groupSelect.value)],available=new Map();
    group.records.forEach(r=>qualityScoreMetrics(r.c).filter(m=>!isDimension(m))
      .forEach(m=>available.set(JSON.stringify([m.path,m.unit]),m)));
    choices=[...available.values()];metricSelect.replaceChildren();
    choices.forEach((m,i)=>option(metricSelect,String(i),readable(m.key)+'（'+displayUnit(m.unit)+'）'));
    metricSelect.disabled=!choices.length;metricLabel.hidden=choices.length<=1;
    const preferred={cmmlu:'accuracy',ceval:'accuracy',gsm8k:'exact_match',
      humaneval:'pass_at_1',wikitext:'perplexity'}[group.task.scenario];
    const index=choices.findIndex(m=>m.key===preferred);
    if(choices.length)metricSelect.value=String(index>=0?index:0);
    updateCases();
  }
  groupSelect.onchange=updateGroup;metricSelect.onchange=updateCases;caseSelect.onchange=draw;
  updateGroup();
}
function renderQualityOverview() {
  const parent=$('quality-overview');parent.append(el('h3','模型能力总结'));
  const quality=tasks.map((task,ti)=>({task,ti}))
    .filter(({task})=>task.category==='model_service_quality');
  if(!quality.length) {parent.append(el('p','本次未包含模型能力评测。','muted'));return;}
  parent.append(el('p','未接入行业基线或其他模型对照，仅展示本次已测维度得分。'+
    '不同学科的题量和难度不同，评分样本 / 计数随指标展示。','overview-notice'));
  const abilities={cmmlu:'中文多学科知识',ceval:'中文多学科知识',gsm8k:'数学应用题',
    humaneval:'代码生成',squad:'阅读理解',lcsts:'中文摘要',wikitext:'语言建模'};
  for(const {task,ti} of quality) {
    if(!task.cases.length)parent.append(link(taskDisplayName(task.name)+'：没有 Case，查看诊断','task-'+ti));
    task.cases.forEach((c,ci)=>{
      const group=el('div',undefined,'overview-quality');
      group.append(el('h4',[(abilities[task.scenario]||taskDisplayName(task.name)),
        scenarioDisplayName(task.scenario),c.parameters?.model,c.id].filter(Boolean).join(' · ')));
      if(c.coverage)group.append(el('p','采集覆盖：'+
        ['complete','failed','incomplete','not_started'].map(key=>
          ({complete:'完成',failed:'失败',incomplete:'未完成',not_started:'未开始'}[key])+
          ' '+valueText(c.coverage[key])).join(' · '),'overview-notice'));
      const scores=qualityScoreMetrics(c);
      if(!['success','partial_failed'].includes(c.status) || c.collection_complete===false) {
        group.append(el('p',(names[c.status]||c.status)+'：未形成可用的能力总结。','muted'));
      } else {
        if(c.status!=='success')group.append(el('p','结果存在异常，请结合覆盖率和运行诊断阅读。','overview-notice'));
        const main=scores.filter(m=>!isDimension(m));
        const metrics=el('div',undefined,'overview-metrics');
        main.forEach(m=>overviewMetric(metrics,metricTitle(m),m.value,m.unit,m.count));
        if(!main.length)metrics.append(el('p','未采集总体得分。','muted'));
        group.append(metrics);
        if(scores.some(m=>['perplexity','mean_loss','bits_per_byte'].includes(m.key)))
          group.append(el('p','PPL、平均损失和 BPB 的数值越低越好，不作为准确率解释。','overview-notice'));
        const dimensions=new Map();
        for(const m of scores.filter(isDimension)) {
          const key=JSON.stringify([m.parts.slice(0,-1),m.unit]);
          if(!dimensions.has(key))dimensions.set(key,[]);
          dimensions.get(key).push(m);
        }
        for(const entries of dimensions.values()) {
          const first=entries[0],details=el('details');
          const title=first.parts.slice(0,-1).filter(p=>p!=='values'&&p!=='task_view')
            .map(readable).join(' · ');
          details.append(el('summary',title+' · '+entries.length+' 个已测学科 / 分类'));
          dataTable(details,[{title:'学科 / 分类',get:m=>readable(m.key),sortable:true},
            {title:'得分（'+displayUnit(first.unit)+'）',numeric:true,unit:first.unit,
              sortable:true,get:m=>m.value},
            {title:'评分样本 / 计数',numeric:true,sortable:true,get:m=>m.count}],entries);
          group.append(details);
        }
      }
      group.append(link('查看 '+c.id+' 的指标与评分详情','analysis-'+ti+'-'+ci));
      parent.append(group);
    });
  }
}
function renderOverviewModels() {
  const models=new Map();
  for(const task of tasks)for(const c of task.cases) {
    const p=c.parameters||{};
    const model=typeof p.model==='string' && p.model.trim()?p.model:null;
    const address=typeof p.base_url==='string' && p.base_url.trim()?p.base_url:null;
    if(model || address)models.set(JSON.stringify([model,address]),{model,address});
  }
  if(!models.size)models.set('',{model:null,address:null});
  for(const {model,address} of models.values()) {
    const row=el('dl',undefined,'overview-model');
    for(const [label,value] of [['模型名',model],['模型地址',address]]) {
      const field=el('div');
      field.append(el('dt',label),el('dd',value||'未采集'));row.append(field);
    }
    $('overview-models').append(row);
  }
}
function renderOverview() {
  renderOverviewModels();
  renderPerformanceBest();
  renderQualityBest();
  renderPerformanceOverview();
  renderQualityOverview();
  initEvaluationTabs();
}
function selectEvaluationTab(panelId) {
  const tabs=[...document.querySelectorAll('.evaluation-tabs [role="tab"]')];
  if(!tabs.some(tab=>tab.getAttribute('aria-controls')===panelId))return;
  tabs.forEach(tab=>{
    const selected=tab.getAttribute('aria-controls')===panelId;
    tab.setAttribute('aria-selected',String(selected));tab.tabIndex=selected?0:-1;
    $(tab.getAttribute('aria-controls')).hidden=!selected;
  });
}
function initEvaluationTabs() {
  const tabs=[...document.querySelectorAll('.evaluation-tabs [role="tab"]')];
  function activate(tab) {
    const current=tabs.find(t=>t.getAttribute('aria-selected')==='true');
    history.replaceState({...history.state,evaluationTab:current.getAttribute('aria-controls')},'');
    const id=tab.getAttribute('aria-controls');
    if(location.hash!=='#'+id)history.pushState({evaluationTab:id},'','#'+id);
    showTarget();tab.focus({preventScroll:true});
  }
  tabs.forEach((tab,index)=>{
    tab.onclick=()=>activate(tab);
    tab.onkeydown=event=>{
      let next;
      if(event.key==='ArrowRight')next=(index+1)%tabs.length;
      else if(event.key==='ArrowLeft')next=(index+tabs.length-1)%tabs.length;
      else if(event.key==='Home')next=0;
      else if(event.key==='End')next=tabs.length-1;
      else return;
      event.preventDefault();activate(tabs[next]);
    };
  });
}
