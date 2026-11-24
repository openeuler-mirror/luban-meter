// Use saved facts; never recompute distributions or infer untested limits.
function stableConfig(value) {
  if (Array.isArray(value)) return value.map(stableConfig);
  if (value && typeof value === 'object') return Object.fromEntries(
    Object.keys(value).sort().map(key=>[key,stableConfig(value[key])]));
  return value;
}
function usablePerformanceResult(c) {
  const metric=servingMetric(c);
  if(isPrefixCache(c)) {
    const counts=savedRequestCounts(c);
    if(!cacheEligible(c) || metric?.status!=='success' || !counts.valid ||
      counts.total!==counts.success || counts.total!==c.parameters.num_requests ||
      c.coverage?.not_started>0)return false;
  }
  return c.status==='success' && c.collection_complete===true &&
    (!c.metrics_status || c.metrics_status==='complete') &&
    (!metric?.status || metric.status==='success') && !metric?.error &&
    !['failed','incomplete'].some(k=>c.coverage?.[k]>0) &&
    !(metric?.values?.service_view?.failed_requests?.value>0);
}
const loadMetrics = [
  ['输出 Token 吞吐量','token/s','output_token_throughput','value'],
  ['TTFT Mean','ms','ttft','mean'], ['TTFT P99','ms','ttft','p99'],
  ['E2EL P99','ms','e2el','p99'],
];
const loadRelatedMetrics = [
  ...['mean','p50','p95','p99'].map(stat=>
    ['TTFT '+(stat==='mean'?'Mean':stat.toUpperCase()),'ms','ttft',stat,'首 Token 时延']),
  ...['mean','p50','p95','p99'].map(stat=>
    ['E2EL '+(stat==='mean'?'Mean':stat.toUpperCase()),'ms','e2el',stat,'端到端延迟']),
  ...['mean','p99'].map(stat=>
    ['TPOT '+(stat==='mean'?'Mean':'P99'),'ms/token','tpot',stat,'生成延迟']),
  ...['mean','p99'].map(stat=>
    ['ITL '+(stat==='mean'?'Mean':'P99'),'ms/token','itl',stat,'Token 间延迟']),
  ['输出 Token 吞吐量','token/s','output_token_throughput','value','Token 吞吐'],
  ['Request Throughput','req/s','request_throughput','value','请求吞吐'],
  ['Success Rate','ratio','success_rate','value','可靠性'],
  ['Timeout Rate','ratio','timeout_rate','value','可靠性'],
  ['输入 Token 吞吐量','token/s','input_token_throughput','value','Token 吞吐'],
  ['总 Token 吞吐量','token/s','total_token_throughput','value','Token 吞吐'],
];
function selectedLoadFields(records) {
  if(!records.some(({c})=>hasSelectivePerformance(c)))return null;
  const fields=new Set();
  for(const {c} of records) {
    const values=servingMetric(c)?.values;
    for(const key of [...(c.parameters?.metrics||[]),...Object.keys(values?.request_view||{}),
      ...Object.keys(values?.service_view||{})])fields.add(key);
    if(savedRequestCounts(c).valid)fields.add('success_rate');
  }
  return fields;
}
function loadAxis(parameters) {
  return ['rate','constant','poisson','gamma'].includes(parameters.profile) &&
    numeric(parameters.request_rate) && parameters.request_rate>0 ?
    'request_rate':'max_concurrency';
}
function axisLabel(axis) {
  return axis==='request_rate'?'配置请求速率（req/s）':'配置并发上限';
}
function loadConditions(c,axis) {
  return Object.fromEntries(Object.entries(c.parameters||{}).filter(([key])=>
    key!==axis && key!=='trend_window_seconds' &&
    (!isPrefixCache(c)||key!=='prefix_cache_environment')));
}
function validLoad(parameters,axis) {
  return axis==='request_rate' || (Number.isSafeInteger(parameters.max_concurrency) &&
    parameters.max_concurrency>0);
}
function loadMeasurement(c,spec) {
  const [,unit,key,stat]=spec, values=servingMetric(c)?.values;
  if(key==='success_rate'){
    const counts=savedRequestCounts(c);
    return {value:counts.valid && counts.total>0?counts.success/counts.total:null,
      count:counts.total,reason:'成功请求数 / Case 全部请求数（沿用已有成功率展示口径）'};
  }
  if(key==='timeout_rate')return {value:null,count:null,
    reason:'未保存 Case 级超时率；分时超时率不能替代整个 Case 的统计'};
  const node=(stat==='value'?values?.service_view:values?.request_view)?.[key];
  const count=node?.sample_count??node?.count;
  const scale=node?.unit===unit?1:
    ((unit==='ms' && node?.unit==='s') ||
     (unit==='ms/token' && node?.unit==='s/token'))?1000:null;
  const raw=node?.[stat];
  const valid=scale!==null && numeric(raw) && raw>=0 && numeric(raw*scale) &&
    (stat==='value' || count==null || (Number.isSafeInteger(count) && count>0)) &&
    (node?.weight_sum==null || (numeric(node.weight_sum) && node.weight_sum>0));
  return {value:valid?raw*scale:null,count,
    reason:valid?null:node?.reason||'字段缺失、单位不匹配或有效样本不足',
    aggregation:node?.aggregation,weight:node?.weight_sum,
    percentile:node?.percentile_method};
}
function performanceSummary(taskList) {
  const groups=new Map();
  let total=0, included=0;
  taskList.forEach((task,ti)=>{
    if(task.category!=='generation_performance')return;
    task.cases.forEach((c,ci)=>{
      total++;
      const p=c.parameters||{},metric=servingMetric(c);
      const service=metric?.values?.service_view;
      const throughput=service?.output_token_throughput;
      if(!usablePerformanceResult(c) ||
        !validLoad(p,loadAxis(p)) ||
        !p.model || !numeric(throughput?.value) ||
        throughput.value<0 || throughput.unit!=='token/s')return;
      // Keep task/environment and every recorded parameter except concurrency
      // fixed. Object key order is not a change in test conditions.
      const axis=loadAxis(p),fixed=loadConditions(c,axis);
      const unknownProfileCase=p.profile?null:ci;
      const key=JSON.stringify([ti,axis,stableConfig(fixed),unknownProfileCase]);
      if(!groups.has(key))groups.set(key,{task,ti,axis,fixed,unknownProfileCase,points:[]});
      groups.get(key).points.push({c,ci,load:p[axis],concurrency:p.max_concurrency,
        value:throughput.value,unit:throughput.unit});
      included++;
    });
  });
  for(const group of groups.values()) {
    group.points.sort((a,b)=>a.load-b.load || a.ci-b.ci);
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
function curvePoints(group,spec) {
  return summaryGroupRecords(group).filter(({c})=>usablePerformanceResult(c))
    .map(({c,ci})=>({c,ci,load:c.parameters[group.axis],...loadMeasurement(c,spec)}))
    .sort((first,second)=>first.load-second.load || first.ci-second.ci);
}
function loadChartScale(maximum,minimum=0,local=false,ratio=false) {
  if(ratio)return {low:0,high:1,tickCount:4};
  const padding=Math.max((maximum-minimum)*.12,Math.abs(maximum)*.02,1e-6);
  const floor=local&&maximum>0?Math.max(0,minimum-padding):0;
  const limit=local&&maximum>0?maximum+padding:maximum||1;
  const magnitude=10**Math.floor(Math.log10((limit-floor)/4));
  const fraction=(limit-floor)/4/magnitude;
  const step=[1,2,2.5,5,10].find(value=>value>=fraction)*magnitude;
  const high=Math.ceil(limit/step)*step;
  const low=Math.floor(floor/step)*step;
  if(!numeric(high) || !numeric(step) || step<=0)
    return {low:0,high:maximum||1,tickCount:4};
  return {low,high,tickCount:Math.round((high-low)/step)};
}
function concurrencyChart(parent,group,spec,withDetails=true,scaleMode='local') {
  const points=curvePoints(group,spec),width=760,height=290;
  const axis=axisLabel(group.axis),metric=spec[0]+'（'+displayUnit(spec[1])+'）';
  const details=el('details');
  details.append(el('summary','实测点明细（'+points.length+' 个 Case）'));
  const describe=point=>{
    const service=servingMetric(point.c)?.values?.service_view||{};
    const fields=[point.c.id,point.c.parameters.profile||'Profile 未记录',
      axis+' '+point.load,metric+' '+valueText(point.value,spec[1])];
    for(const [key,label] of [['average_concurrency','实际平均并发'],
      ['peak_concurrent_requests','实际峰值并发'],['successful_requests','成功请求'],
      ['failed_requests','失败请求']]) {
      if(numeric(service[key]?.value))fields.push(label+' '+valueText(service[key].value));
    }
    if(spec[3]!=='value')fields.push(point.count==null?'样本数未记录':
      '样本数 '+point.count+(spec[3]==='p99'&&point.count<10?'（小样本）':''));
    if(point.reason)fields.push(point.reason);
    return fields.join(' · ');
  };
  for(const point of points)details.append(el('p',describe(point)));
  if(withDetails)parent.append(details);
  const valid=points.filter(point=>numeric(point.value));
  if(!valid.length){
    parent.append(el('p',metric+' 暂无可绘制数据。'+
      [...new Set(points.map(point=>point.reason).filter(Boolean))].join('；')));
    return;
  }
  const left=75,right=28,top=35,bottom=60;
  const min=points[0].load,max=points.at(-1).load;
  const maximum=Math.max(...valid.map(point=>point.value));
  const minimum=Math.min(...valid.map(point=>point.value));
  const {low,high,tickCount}=loadChartScale(maximum,minimum,scaleMode==='local',spec[1]==='ratio');
  if(low>0)parent.append(el('p','局部缩放：Y 轴从 '+valueText(low,spec[1])+
    ' '+displayUnit(spec[1])+' 开始（非零起点），请结合实际数值判断波动。','analysis-note'));
  if(spec[1]==='ratio')parent.append(el('p','比例指标固定显示 0–100%。','analysis-note'));
  const lowerIsBetter=['ms','ms/token'].includes(spec[1]) || spec[2]==='timeout_rate';
  const extreme=lowerIsBetter?Math.min(...valid.map(point=>point.value)):maximum;
  const extremeLabel=lowerIsBetter?'已测最小值（越低越好）':'已测最大值（越高越好）';
  const x=p=>min===max?(left+width-right)/2:
    left+(p.load-min)/(max-min)*(width-left-right);
  const y=p=>height-bottom-(p.value-low)/(high-low)*(height-top-bottom);
  const svg=svgElement('svg',{viewBox:`0 0 ${width} ${height}`,role:'img',
    'aria-label':axis+'与'+metric});
  const title=svgElement('title',{});
  title.textContent=axis+'与'+metric;
  svg.append(title);
  for(let i=0;i<=tickCount;i++) {
    const value=low+(high-low)*(i/tickCount),py=y({value});
    svg.append(svgElement('line',{x1:left,x2:width-right,y1:py,y2:py,stroke:'var(--line)'}));
    const label=svgElement('text',{x:left-9,y:py+4,'text-anchor':'end',
      fill:'currentColor','font-size':12});
    label.textContent=valueText(value,spec[1]);svg.append(label);
  }
  const repeated=new Set(points.map(p=>p.load)).size!==points.length;
  if(!repeated)for(let index=1;index<points.length;index++){
    const prior=points[index-1],next=points[index];
    if(numeric(prior.value)&&numeric(next.value))svg.append(svgElement('line',{
      x1:x(prior),x2:x(next),y1:y(prior),y2:y(next),stroke:'var(--accent)','stroke-width':2}));
  }
  const ticks=new Set();
  points.forEach(p=>{
    if(!ticks.has(p.load)){
      const label=svgElement('text',{x:x(p),y:height-bottom+23,'text-anchor':'middle',fill:'currentColor'});
      label.textContent=p.load;svg.append(label);ticks.add(p.load);
    }
    if(!numeric(p.value))return;
    const peak=p.value===extreme;
    const a=svgElement('a',{href:'#analysis-'+group.ti+'-'+p.ci,
      'aria-label':describe(p)});
    const mark=svgElement('circle',{cx:x(p),cy:y(p),r:peak?5.5:4,
      fill:peak?'var(--accent)':'var(--panel)',stroke:'var(--accent)',
      'stroke-width':2,'data-case-index':p.ci,'data-peak':String(peak)});
    const tip=svgElement('title',{});tip.textContent=a.getAttribute('aria-label')+
      (peak?' · '+extremeLabel:'');
    mark.append(tip);a.append(mark);svg.append(a);
  });
  for(const [text,px,py,anchor] of [[metric,left,18,'start'],
    ['● '+extremeLabel,width-right,18,'end'],
    [axis,(left+width-right)/2,height-8,'middle']]) {
    const label=svgElement('text',{x:px,y:py,'text-anchor':anchor,
      fill:'currentColor','font-size':12});label.textContent=text;svg.append(label);
  }
  const box=el('div',undefined,'trend-chart');box.append(svg);parent.append(box);
}
function performanceGroupLabel(group,index) {
  const p=group.fixed;
  return ['条件组 '+(index+1),taskDisplayName(group.task.name),p.model,
    '输入 '+valueText(p.input_tokens),'输出上限 '+valueText(p.output_tokens),
    p.profile||'Profile 未记录（单 Case）',group.axis==='request_rate'?
      '固定并发上限 '+valueText(p.max_concurrency):
      p.request_rate!=null?'固定请求速率 '+valueText(p.request_rate):'配置并发上限扫参'].join(' · ');
}
function summaryScopeGroups(taskList,summary=performanceSummary(taskList)) {
  const key=(ti,axis,fixed,single)=>JSON.stringify([ti,axis,stableConfig(fixed),single]);
  const available=new Map(summary.groups.map(g=>[key(g.ti,g.axis,g.fixed,g.unknownProfileCase),g]));
  const scopes=new Map();
  taskList.forEach((task,ti)=>{
    if(task.category!=='generation_performance')return;
    task.cases.forEach((c,ci)=>{
      const p=c.parameters||{};
      if(c.collection_complete!==true || c.status==='discarded' || !p.model ||
        !validLoad(p,loadAxis(p)))return;
      const axis=loadAxis(p),fixed=loadConditions(c,axis);
      const unknownProfileCase=p.profile?null:ci;
      const identity=key(ti,axis,fixed,unknownProfileCase);
      if(!scopes.has(identity))scopes.set(identity,available.get(identity)||
        {task,ti,axis,fixed,unknownProfileCase,points:[],maximum:null,peaks:[]});
    });
  });
  return [...scopes.values()];
}
function summaryGroupRecords(group) {
  if(!group)return [];
  const fixed=JSON.stringify(stableConfig(group.fixed));
  return group.task.cases.map((c,ci)=>({c,ci})).filter(({c,ci})=>
    (group.unknownProfileCase==null || ci===group.unknownProfileCase) &&
    loadAxis(c.parameters||{})===group.axis && validLoad(c.parameters||{},group.axis) &&
    JSON.stringify(stableConfig(loadConditions(c,group.axis)))===fixed);
}
function savedRequestCounts(c) {
  const metric=servingMetric(c),service=metric?.values?.service_view;
  const count=key=>{
    const m=service?.[key];
    return m?.unit==='request' && Number.isSafeInteger(m.value) && m.value>=0?m.value:null;
  };
  const success=count('successful_requests'),total=count('total_requests');
  const failed=count('failed_requests');
  const hasFailed=Object.hasOwn(service||{},'failed_requests');
  const valid=success!==null && total!==null && success<=total &&
    (!hasFailed || (failed!==null && success+failed===total)) && !metric?.error;
  return {success,total,valid};
}
function performanceOverviewCards(records,bestMode=true) {
  // SLO/Goodput cards live in their own section below the summaries,
  // so the overview grid keeps its seven throughput/latency cards and
  // never duplicates the peak output throughput on the same page.
  const cards=[
    {id:'peak-throughput',title:'Peak Throughput',unit:'tok/s',tone:'cyan',
      description:'higher is better · 已测最大值'},
    {id:'peak-total-throughput',title:'Peak Total Throughput',unit:'tok/s',
      tone:'cyan',description:'higher is better · 输入＋输出合计'},
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
    ['Throughput','Total Throughput','TTFT P99','TPOT P99','Success Rate',
      'Configured Concurrency','E2E P99']
      .forEach((title,index)=>{cards[index].title=title;});
    cards.forEach(card=>{card.description='当前 Case 的已保存指标';});
  } else cards[4].title='Best Success Rate';
  if(!records.length)return cards;
  const completed=records.filter(({c})=>c.collection_complete===true &&
    c.status!=='discarded' && (!c.metrics_status || c.metrics_status==='complete'));
  const usable=bestMode?records.filter(({c})=>usablePerformanceResult(c)):
    completed.filter(({c})=>!servingMetric(c)?.error);
  function extreme(index,points,minimum=false) {
    if(!points.length)return;
    const best=points.reduce((a,b)=>(minimum?b.value<a.value:b.value>a.value)?b:a);
    const ties=points.filter(p=>p.value===best.value).length;
    cards[index].value=best.value;
    cards[index].source=taskDisplayName(best.task.name)+' / '+best.c.id+
      ' · C'+valueText(best.c.parameters?.max_concurrency)+
      (ties>1?' · '+ties+' 个 Case 并列':'');
    cards[index].href='#summary-'+best.ti+'-'+best.ci;
    if(index===4) {
      const counts=savedRequestCounts(best.c);
      cards[index].source+=' · '+counts.success+' / '+counts.total+' 请求';
    }
    if(['best-slo-attainment','best-goodput-throughput'].includes(cards[index].id)) {
      const thresholds=sloConfigText(best.c);
      if(thresholds)cards[index].source+=' · SLO '+thresholds;
    }
  }
  extreme(0,usable.flatMap(record=>{
    const m=servingMetric(record.c)?.values?.service_view?.output_token_throughput;
    return m?.unit==='token/s' && numeric(m.value) && m.value>=0?
      [{...record,value:m.value}]:[];
  }));
  extreme(1,usable.flatMap(record=>{
    const m=servingMetric(record.c)?.values?.service_view?.total_token_throughput;
    return m?.unit==='token/s' && numeric(m.value) && m.value>=0?
      [{...record,value:m.value}]:[];
  }));
  for(const [index,key] of [[2,'ttft'],[3,'tpot'],[6,'e2el']]) {
    const units=key==='tpot'?{'ms/token':1,'s/token':1000}:{ms:1,s:1000};
    extreme(index,usable.flatMap(record=>{
      const m=servingMetric(record.c)?.values?.request_view?.[key];
      const value=m?.p99*units[m?.unit];
      return numeric(m?.p99) && m.p99>=0 && numeric(value) &&
        (m.count==null || (Number.isSafeInteger(m.count) && m.count>0)) &&
        (m.weight_sum==null || (numeric(m.weight_sum) && m.weight_sum>0))?
        [{...record,value}]:[];
    }),true);
  }
  const countCandidates=bestMode?completed.filter(({c})=>!isPrefixCache(c)||usablePerformanceResult(c)):completed;
  extreme(4,countCandidates.flatMap(record=>{
    const counts=savedRequestCounts(record.c);
    return counts.valid && counts.total>0?[{...record,value:counts.success/counts.total}]:[];
  }));
  extreme(5,countCandidates.flatMap(record=>{
    const p=record.c.parameters?.max_concurrency,metric=servingMetric(record.c);
    const service=metric?.values?.service_view;
    return Number.isSafeInteger(p) && p>0 && service?.total_requests?.unit==='request' &&
      Number.isSafeInteger(service.total_requests.value) && service.total_requests.value>0 &&
      !metric?.error?[{...record,value:p}]:[];
  }));
  if(!bestMode) {
    const record=records[0],concurrency=record.c.parameters?.max_concurrency;
    if(Number.isSafeInteger(concurrency) && concurrency>0) {
      cards[5].value=concurrency;cards[5].source=record.c.id+' · 配置并发上限';
      cards[5].href='#summary-'+record.ti+'-'+record.ci;
    }
  }
  const selected=new Set(records.flatMap(({c})=>c.parameters?.metrics||[]));
  const selective=records.some(({c})=>hasSelectivePerformance(c));
  if(!selective)return cards;
  for(const [key,title,unit,view,minimum] of [
    ['itl','Best ITL P99','ms/token','request_view',true],
    ['request_throughput','Peak Request Throughput','req/s','service_view',false],
    ['input_token_throughput','Peak Input Throughput','token/s','service_view',false]
  ]) {
    if(!selected.has(key))continue;
    const card={id:'selected-'+key,title,unit,tone:'cyan',value:null,
      source:'缺少可用的已保存数据',href:'#summary-section',
      description:minimum?'lower is better · 已保存P99':'higher is better · 已保存速率'};
    cards.push(card);
    extreme(cards.length-1,usable.flatMap(record=>{
      const node=servingMetric(record.c)?.values?.[view]?.[key];
      const value=minimum?node?.p99:node?.value;
      return node?.unit===unit && numeric(value) && value>=0?
        [{...record,value}]:[];
    }),minimum);
  }
  const fields={
    'peak-throughput':['service_view','output_token_throughput'],
    'peak-total-throughput':['service_view','total_token_throughput'],
    'best-ttft-p99':['request_view','ttft'],
    'best-tpot-p99':['request_view','tpot'],
    'best-e2e-p99':['request_view','e2el']
  };
  return cards.filter(card=>!fields[card.id] || selected.has(fields[card.id][1]) || records.some(({c})=>{
    const [view,key]=fields[card.id];
    return Object.hasOwn(servingMetric(c)?.values?.[view]||{},key);
  }));
}
// The SLO section's own two cards; best values over the saved goodput
// blocks.  Prefix-cache warm-up and observe Cases only join through the
// same usable filter as the overview counts.
function sloOverviewCards(records) {
  const cards=[
    {id:'best-slo-attainment',title:'Best SLO Attainment',unit:'%',tone:'green',
      kind:'percentage',description:'higher is better · 达标请求占比'},
    {id:'best-goodput-throughput',title:'Best Goodput Output Throughput',
      unit:'tok/s',tone:'cyan',
      description:'higher is better · SLO 内输出吞吐量'},
  ].map(card=>({...card,value:null,source:'缺少可用的已保存数据',href:'#slo-section'}));
  if(!records.length)return cards;
  const completed=records.filter(({c})=>c.collection_complete===true &&
    c.status!=='discarded' && (!c.metrics_status || c.metrics_status==='complete'));
  const candidates=completed.filter(({c})=>!isPrefixCache(c)||usablePerformanceResult(c));
  const goodputOf=(c,field)=>{
    const m=servingMetric(c)?.values?.service_view?.goodput?.[field];
    return m && numeric(m.value) && m.value>=0?m:null;
  };
  function extreme(index,points) {
    if(!points.length)return;
    const best=points.reduce((a,b)=>b.value>a.value?b:a);
    const ties=points.filter(p=>p.value===best.value).length;
    cards[index].value=best.value;
    cards[index].source=taskDisplayName(best.task.name)+' / '+best.c.id+
      ' · C'+valueText(best.c.parameters?.max_concurrency)+
      (ties>1?' · '+ties+' 个 Case 并列':'');
    cards[index].href='#summary-'+best.ti+'-'+best.ci;
    const thresholds=sloConfigText(best.c);
    if(thresholds)cards[index].source+=' · SLO '+thresholds;
  }
  extreme(0,candidates.flatMap(record=>{
    const m=goodputOf(record.c,'slo_satisfied_rate');
    return m?.unit==='ratio'?[{...record,value:m.value}]:[];
  }));
  extreme(1,candidates.flatMap(record=>{
    const m=goodputOf(record.c,'goodput_output_token_throughput');
    return m?.unit==='token/s'?[{...record,value:m.value}]:[];
  }));
  const configured=completed.some(({c})=>
    servingMetric(c)?.values?.service_view?.goodput);
  for(const index of [0,1]) {
    if(!configured)cards[index].source='未配置 SLO（goodput 参数）';
    else if(cards[index].value===null)
      cards[index].source='已配置 SLO，暂无可判定的达标数据';
  }
  return cards;
}
// A dedicated page under the result summaries: every SLO/Goodput figure
// lives here instead of the overview grid and the summary columns.
function renderSloSection() {
  const records=[];
  tasks.forEach((task,ti)=>{
    if(task.category!=='generation_performance')return;
    task.cases.forEach((c,ci)=>records.push({task,ti,c,ci}));
  });
  const section=$('slo-section');
  if(!records.length) {
    section.remove();
    document.querySelector('#report-nav a[href="#slo-section"]')?.remove();
    return;
  }
  const goodputOf=c=>servingMetric(c)?.values?.service_view?.goodput;
  const configured=records.filter(({c})=>goodputOf(c));
  if(!configured.length) {
    $('slo-overview').append(el('p',
      '本次运行未配置 SLO（--goodput）：未保存达标率与 Goodput 有效吞吐，'+
      '各 Case 仅保留常规吞吐与延迟指标。需要这些指标时，应在任务中配置'+
      ' goodput 阈值并重新运行。','empty'));
    return;
  }
  renderSummaryCards($('slo-overview'),sloOverviewCards(records));
  // Transposed table: one row per field and one column per Case, so the
  // multi-dimension SLO thresholds get the full row width instead of
  // squeezing into a narrow column.
  const countText=(g,key)=>{
    const value=(g[key]||{}).value;
    return value==null?'N/A':valueText(value);
  };
  const plainText=m=>m?.value==null?'N/A':valueText(m.value);
  // A dedicated class: summary-table's sticky last column targets the
  // horizontal layout and would detach the value column here.
  const table=el('table');
  table.className='slo-field-table';
  const head=el('tr');
  head.append(el('th','指标'));
  configured.forEach(({task,c,ti,ci})=>{
    const th=el('th');
    th.append(link(task.name+' / '+c.id,'analysis-'+ti+'-'+ci));
    head.append(th);
  });
  const thead=el('thead');
  thead.append(head);
  table.append(thead);
  const body=el('tbody');
  const fields=[
    {title:'并发',titleText:'测试期间允许的最大并发请求数',
      get:r=>valueText(r.c.parameters?.max_concurrency)},
    {title:'SLO 阈值',titleText:'仅列出已配置维度；未配置的维度不显示、不虚构',
      get:r=>sloConfigText(r.c)||'N/A'},
    {title:'达标率',dynamicTitle:r=>goodputCellTitle(r.c),
      get:r=>{
        const m=goodputOf(r.c).slo_satisfied_rate;
        return m?.value==null?'N/A':valueText(m.value,'ratio');
      }},
    {title:'满足 / 违反 / 失败 / 无法判定',
      titleText:'依次为满足、违反、失败、无法判定的请求数；'+
        '达标率分母为满足+违反+失败，无法判定不进分母',
      get:r=>{
        const g=goodputOf(r.c);
        return ['slo_satisfied_count','slo_violated_count',
          'failed_count','undetermined_count']
          .map(key=>countText(g,key)).join(' / ');
      }},
    {title:'Goodput 请求吞吐量（req/s）',
      titleText:'满足请求数除以Case窗口',
      get:r=>plainText(goodputOf(r.c).goodput_request_throughput)},
    {title:'Goodput 输出 Token 吞吐量（token/s）',
      titleText:'满足请求的输出Token总数除以Case窗口；'+
        '有达标请求缺输出Token计数时为N/A',
      get:r=>plainText(goodputOf(r.c).goodput_output_token_throughput)},
  ];
  fields.forEach(field=>{
    const tr=el('tr');
    tr.append(el('th',field.title));
    configured.forEach(record=>{
      const value=field.get(record);
      const td=el('td',value);
      const title=field.dynamicTitle?
        field.dynamicTitle(record):field.titleText;
      if(title)td.title=title;
      if(value!=='N/A' && !isNaN(Number(String(value).replace(/,/g,''))))
        td.className='num';
      tr.append(td);
    });
    body.append(tr);
  });
  table.append(body);
  const wrap=el('div',undefined,'table-wrap');
  wrap.append(table);
  $('slo-cases').append(wrap);
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
function hasComparableLoad(group) {
  return new Set(curvePoints(group,loadMetrics[0]).map(point=>point.load)).size>=2;
}
function loadComparisonDescription(group) {
  const points=curvePoints(group,loadMetrics[0]);
  if(!points.length)return '没有满足成功与采集完整条件的可绘制性能数据。';
  const loads=[...new Set(points.map(point=>point.load))];
  const throughput=points.map(point=>point.value).filter(numeric);
  const latency=curvePoints(group,loadMetrics[1]);
  const range=values=>{
    if(!values.length)return '缺失';
    const minimum=Math.min(...values),maximum=Math.max(...values);
    return minimum===maximum?valueText(minimum):valueText(minimum)+'–'+valueText(maximum);
  };
  const selected=selectedLoadFields(summaryGroupRecords(group));
  const text=[points.length+' 个 Case',axisLabel(group.axis)+' '+loads[0]+
    (loads.length>1?' → '+loads.at(-1):'（单点，无法描述负载变化）')];
  if(!selected || selected.has('output_token_throughput'))
    text.push('输出吞吐量 '+range(throughput)+' token/s');
  if(latency.length && (!selected || selected.has('ttft')))text.push('TTFT Mean '+valueText(latency[0].value)+
    (loads.length>1?' → '+valueText(latency.at(-1).value):'')+' ms');
  if(loads.length!==points.length)text.push('重复负载分别保留，不合并均值');
  return text.join(' · ');
}
function renderLoadComparison(parent,group,spec=loadMetrics[1],scaleMode='local') {
  if(!group){
    parent.append(el('p','当前任务没有可用的负载条件或性能数据。','empty'));return;
  }
  parent.append(el('p',loadComparisonDescription(group),'overview-notice'));
  parent.append(el('p',axisLabel(group.axis)+(group.axis==='request_rate'?
    '表示配置到达速率。':'表示客户端配置上限，实际平均/峰值并发见表格。')+
    ' 连线只连接实测点；缺失值断线。','analysis-note'));
  parent.append(el('h3','负载—'+spec[0]));
  concurrencyChart(parent,group,spec,false,scaleMode);
  const measurements=curvePoints(group,spec);
  const definitions=[...new Set(measurements.map(point=>point.aggregation?
    'aggregation='+point.aggregation+'；percentile_method='+(point.percentile||'未记录'):
    point.reason).filter(Boolean))];
  if(definitions.length){
    const explanation=el('details');
    explanation.append(el('summary','统计说明'),el('p',definitions.join('；'),'analysis-note'));
    const records=measurements.filter(point=>point.aggregation);
    records.forEach(point=>explanation.append(el('p',point.c.id+' · 权重合计 '+
      valueText(point.weight)+' · 有效请求样本 '+valueText(point.count),'analysis-note')));
    parent.append(explanation);
  }
  if(['tpot','itl'].includes(spec[2]))
    parent.append(el('p','沿用原指标的 Token 加权统计；具体权重与分位数方法见统计说明。','analysis-note'));
  const records=summaryGroupRecords(group).sort((first,second)=>
    first.c.parameters[group.axis]-second.c.parameters[group.axis] || first.ci-second.ci);
  if(!hasComparableLoad(group))
    parent.append(el('p','只有一个可用负载点，或有效性能数据不足，无法形成负载变化趋势。','analysis-note'));
  parent.append(el('h3','跨 Case 对比'));
  const service=(record,key)=>{
    const node=servingMetric(record.c)?.values?.service_view?.[key];
    return numeric(node?.value)?node.value:null;
  };
  const selected=selectedLoadFields(records);
  const baseMetrics=loadMetrics.filter(metric=>!selected || selected.has(metric[2]));
  dataTable(parent,[
    {title:'Case',get:record=>record.c.id},
    {title:axisLabel(group.axis),numeric:true,get:record=>record.c.parameters[group.axis]},
    {title:'状态',get:record=>names[record.c.status]||record.c.status},
    ...[['平均并发','average_concurrency'],['峰值并发','peak_concurrent_requests']]
      .filter(([,key])=>!selected || selected.has(key))
      .map(([title,key])=>({title,numeric:true,get:record=>service(record,key)})),
    {title:'正式请求数',numeric:true,get:record=>savedRequestCounts(record.c).total},
    {title:'成功请求数',numeric:true,get:record=>savedRequestCounts(record.c).success},
    ...baseMetrics.map(metric=>({title:metric[0]+'（'+metric[1]+'）',
      numeric:true,get:record=>loadMeasurement(record.c,metric).value})),
    ...(!baseMetrics.some(metric=>metric[2]===spec[2] && metric[3]===spec[3] &&
      metric[1]===spec[1])?[{
      title:'当前选择：'+spec[0]+'（'+displayUnit(spec[1])+'）',
      numeric:true,unit:spec[1],get:record=>loadMeasurement(record.c,spec).value,
    }]:[]),
  ],records);
  parent.append(el('p','表格保留每次实验的保存值；仅成功且采集完整的 Case 参与曲线。'+
    '数据点链接进入来源 Case。各条件组独立比较。','analysis-note'));
  const fixed=el('details');
  fixed.append(el('summary','固定测试条件'),el('pre',JSON.stringify(
    Object.fromEntries(Object.entries(group.fixed).filter(([key])=>key!=='api_key')),null,2)));
  parent.append(fixed);
}
function renderPerformanceOverview() {
  const parent=$('performance-overview');
  parent.append(el('h3','推理性能总结'));
  const groups=summaryScopeGroups(tasks);
  if(!groups.length){
    parent.append(el('p','没有可用的性能负载条件，详见 Case 分析。','muted'));return;
  }
  groups.forEach((group,index)=>{
    const summary=el('div',undefined,'overview-quality');
    summary.append(el('h4',performanceGroupLabel(group,index)),
      el('p',loadComparisonDescription(group),'overview-notice'),
      link('查看负载对比','analysis-load-'+group.ti+'-'+index));
    parent.append(summary);
  });
}
function qualityScoreMetrics(c) {
  return metricsFor(c).filter(m=>m.kind==='scalar' &&
    m.parts[0]!=='answer_output_quality' &&
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
    'max_samples','num_requests','seed','shuffle','endpoint',
    'temperature','output_tokens','stop','request_params'];
  const groups=new Map();
  taskList.forEach((task,ti)=>{
    if(task.category!=='model_service_quality')return;
    task.cases.forEach((c,ci)=>{
      const protocol=Object.fromEntries(keys.map(key=>[key,c.parameters?.[key]]));
      const effective=Object.values(c.metrics||{}).map(m=>m.metadata?.evaluation)
        .find(Boolean)||{};
      ['prompt_source','input_template','answer_parser','parser_revision',
        'input_format_sha256','selected_samples_sha256','few_shot_source_sha256']
        .forEach(key=>{if(Object.hasOwn(effective,key))protocol[key]=effective[key];});
      const historical=Object.hasOwn(effective,'prompt_version')?
        effective.prompt_version:c.parameters?.prompt_version;
      if(historical)protocol.prompt_version=historical;
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
      '输入模板 '+(group.protocol.input_template||'未记录')+' · '+
      '来源 '+(group.protocol.prompt_source||'未记录')+' · '+
      '解析器 '+(group.protocol.answer_parser||'未记录/不适用')+' · '+
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
      field.append(el('dt',label),el('dd',value||'未采集',
        label==='模型名'?'model-name':undefined));row.append(field);
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
