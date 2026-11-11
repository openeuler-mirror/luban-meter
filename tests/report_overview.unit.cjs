// Deterministic summary semantics; no DOM, network, or model calls.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const context = vm.createContext({taskDisplayName:name=>name});
// Minimal DOM stubs so renderSessionTurns can be exercised headless:
// created nodes record text/title/class for later assertions.
const domNodes=[];
const domNode=()=>{
  const node={append(){},replaceChildren(){},setAttribute(){},
    addEventListener(){},focus(){},classList:{add(){},remove(){}},
    style:{},dataset:{},textContent:'',className:'',title:'',href:'',
    type:'',tabIndex:0,disabled:false,onclick:null,onkeydown:null};
  domNodes.push(node);return node;
};
context.document={createElement:domNode,createElementNS:domNode};
// el() lives in the template's inline script; stub it here so the
// render functions under test resolve it from the sandbox globals.
context.el=(tag,text,cls)=>{
  const node=domNode();
  if(text!==undefined)node.textContent=text;
  if(cls)node.className=cls;
  return node;
};
// svgElement() is another template-inline helper used by barChart.
context.svgElement=domNode;
const domTexts=()=>domNodes.map(node=>
  [node.textContent,node.title,node.className].join(' ')).join('\n');
const source = path.join(__dirname,'../src/luban_meter/reporting');
const html=process.argv[3]?fs.readFileSync(process.argv[3],'utf8'):null;
for(const file of ['case_analysis.js','summary_cards.js','report_overview.js']) {
  const script=fs.readFileSync(path.join(source,file),'utf8');
  if(html!==null)assert.ok(html.includes(script),'export contains '+file);
  vm.runInContext(html===null?script:
    html.slice(html.indexOf(script),html.indexOf(script)+script.length),context);
}
const call = (name,value,...extra) => {
  context.input=JSON.stringify(value);
  return JSON.parse(vm.runInContext('JSON.stringify('+name+'(JSON.parse(input)'+
    extra.map(v=>','+JSON.stringify(v)).join('')+'))',context));
};
const scalar=(value,unit='token/s')=>({value,unit});
// Phase metrics are selectable even at zero or unavailable; they never
// replace the provider of whole-request throughput and latency metrics.
const phaseKeys=['reasoning_duration','content_duration','first_content_latency'];
const phases={status:'success',values:{request_view:{
  reasoning_duration:{mean:0,p95:0,count:1,unit:'ms'},
  content_duration:{mean:null,p95:null,count:0,unit:'ms'},
  first_content_latency:{mean:12,p95:12,count:1,unit:'ms'}},
  phase_counts:{truncated_requests:scalar(0,'request'),
    excluded_requests:scalar(1,'request')}}};
const phaseCase={status:'success',metrics:{streaming_phase_timing:phases,
  original:{status:'success',values:{request_view:{ttft:{mean:2,unit:'ms'}}}}}};
const phaseBefore=JSON.stringify(phaseCase);
assert.deepEqual(call('corePerformanceMetrics',phaseCase).map(m=>m.key),
  ['ttft',...phaseKeys]);
assert.equal(call('metricsFor',phaseCase).find(m=>m.key==='reasoning_duration').raw.mean,0);
assert.equal(call('metricsFor',phaseCase).find(m=>m.key==='content_duration').count,0);
assert.deepEqual(call('servingMetric',phaseCase),phaseCase.metrics.original);
assert.deepEqual(call('summaryMetrics',{
  category:'generation_performance',cases:[phaseCase]}).map(m=>m.key),
  ['ttft',...phaseKeys,'truncated_requests','excluded_requests']);
assert.equal(JSON.stringify(phaseCase),phaseBefore);
const thinkingCase=JSON.parse(phaseBefore);
Object.assign(thinkingCase.metrics.streaming_phase_timing.values.request_view,{
  reasoning_ttft:{mean:100,count:3,unit:'ms'},
  reasoning_tpot:{mean:200,count:2,weight_sum:5,unit:'ms/token',weight_unit:'token'},
  reasoning_itl:{mean:0,count:1,weight_sum:3,unit:'ms/token',weight_unit:'token_interval'},
  reasoning_stream_event_itl:{mean:10,count:2,unit:'ms',sample_unit:'event_interval'}
});
const thinkingKeys=['reasoning_ttft','reasoning_tpot','reasoning_itl','reasoning_stream_event_itl'];
assert.deepEqual(call('corePerformanceMetrics',thinkingCase).map(m=>m.key),
  ['ttft',...thinkingKeys,...phaseKeys]);
assert.deepEqual(call('summaryMetrics',{
  category:'generation_performance',cases:[thinkingCase]}).map(m=>m.key),
  ['ttft',...thinkingKeys,...phaseKeys,'truncated_requests','excluded_requests']);
assert.deepEqual(call('servingMetric',thinkingCase),phaseCase.metrics.original);
const thinkingItl=call('metricsFor',thinkingCase).find(m=>m.key==='reasoning_itl');
assert.equal(thinkingItl.count,1);
assert.equal(thinkingItl.raw.weight_sum,3);
assert.equal(thinkingItl.stats.some(([key])=>key==='weight_sum'),false);
// Upstream Case throughput and independent thinking metrics coexist;
// unrelated run-level throughput still must not enter the Case views.
const combinedCase=JSON.parse(JSON.stringify(thinkingCase));
combinedCase.metrics.original.values.service_view={
  output_token_throughput:scalar(20),total_token_throughput:scalar(60)};
combinedCase.metrics.token_throughput={values:{service_view:{
  total_token_throughput:scalar(999)}}};
const combinedBefore=JSON.stringify(combinedCase);
for(const candidate of [combinedCase,{...combinedCase,metrics:
  Object.fromEntries(Object.entries(combinedCase.metrics).reverse())}]) {
  for(const selected of [call('corePerformanceMetrics',candidate),
    call('summaryMetrics',{category:'generation_performance',cases:[candidate]})]) {
    assert.equal(selected.filter(m=>m.key==='total_token_throughput').length,1);
    assert.equal(selected.find(m=>m.key==='total_token_throughput').value,60);
    assert.ok(thinkingKeys.every(key=>selected.some(m=>m.key===key)));
    assert.ok(!selected.some(m=>m.parts[0]==='token_throughput'));
  }
  assert.deepEqual(call('servingMetric',candidate),combinedCase.metrics.original);
  const summary=call('summaryMetrics',{category:'generation_performance',cases:[candidate]});
  assert.ok(summary.some(m=>m.key==='truncated_requests' && m.value===0));
  assert.ok(summary.some(m=>m.key==='excluded_requests' && m.value===1));
}
assert.equal(JSON.stringify(combinedCase),combinedBefore);
// Categories, not arbitrary task names, determine summary/diagnostic parents.
const grouped=call('reportTaskGroups',[
  {name:'ceval',category:'model_service_quality'},
  {name:'performance',category:'generation_performance'},
  {name:'cmmlu',category:'model_service_quality'},
  {name:'custom',category:'constructor'},
  {name:'empty'},
]);
assert.deepEqual(grouped.map(g=>[g.title,g.tasks.map(t=>t.ti)]),[
  ['模型能力评测',[0,2]],['推理性能评测',[1]],['constructor',[3]],['其他评测',[4]],
]);
assert.deepEqual(call('reportTaskGroups',[]),[]);
function point(id,concurrency,value,parameters={}) {
  return {id,status:'success',collection_complete:true,metrics_status:'complete',
    parameters:{model:'test-model',profile:'concurrent',input_tokens:128,
      output_tokens:64,max_concurrency:concurrency,...parameters},
    metrics:{streaming_performance:{status:'success',values:{
      request_view:{ttft:{mean:10,unit:'ms'},tpot:{mean:2,unit:'ms/token'},
        itl:{mean:3,unit:'ms/token'}},
      service_view:{output_token_throughput:scalar(value),
        total_token_throughput:scalar(value*2),
        request_throughput:scalar(1,'req/s')}}}}};
}
const task=cases=>({name:'performance',category:'generation_performance',cases});
const summarize=cases=>call('performanceSummary',[task(cases)]);
const peak=(concurrencies,values)=>summarize(values.map((v,i)=>point('c'+i,concurrencies[i],v))).groups[0];

// Sorting must not change peak identity, average repetitions, or mix lengths.
const cases=[point('c16',16,270),point('c4',4,70),point('c8',8,140),
  point('long',16,500,{output_tokens:128}),
  point('different_seed',16,600,{seed:7}),
  point('other_model',16,700,{model:'another-model'}),
  point('other_profile',16,800,{profile:'constant',request_rate:2})];
const before=JSON.stringify(cases),summary=summarize(cases);
assert.equal(summary.groups.length,5);
assert.deepEqual(summary.groups[0].points.map(p=>p.concurrency),[4,8,16]);
assert.equal(summary.groups[0].peaks[0].c.id,'c16');
assert.equal(summary.groups[0].peaks[0].ci,0);
assert.equal(summary.groups[0].maximum,270);
assert.equal(JSON.stringify(cases),before);
assert.equal(call('performanceSummary',[task([cases[0]]),task([cases[0]])]).groups.length,2);
const reordered=point('reordered',32,300);
reordered.parameters=Object.fromEntries(Object.entries(reordered.parameters).reverse());
assert.equal(summarize([cases[0],reordered]).groups.length,1);
const nestedA=point('a',4,10,{generation:{stop:['a','b'],temperature:0}});
const nestedB=point('b',8,20,{generation:{temperature:0,stop:['a','b']}});
assert.equal(summarize([nestedA,nestedB]).groups.length,1);
nestedB.parameters.generation.stop.reverse();
assert.equal(summarize([nestedA,nestedB]).groups.length,2);

// Invalid/incomplete/failed evidence cannot become the reported maximum.
const rejected=[];
for(const status of ['failed','partial_failed','interrupted','discarded','skipped'])
  rejected.push({...point(status,32,999),status});
rejected.push({...point('incomplete',32,999),collection_complete:false});
const noMarker=point('noMarker',32,999);delete noMarker.collection_complete;rejected.push(noMarker);
rejected.push({...point('metric_failed',32,999),metrics_status:'failed'});
for(const metrics_status of ['pending','not_computed'])
  rejected.push({...point('not_computed',32,999),metrics_status});
for(const key of ['failed','incomplete'])
  rejected.push({...point(key,32,999),coverage:{[key]:1}});
const metricError=point('metricError',32,999);
metricError.metrics.streaming_performance.error='invalid';rejected.push(metricError);
const metricStatus=point('metricStatus',32,999);
metricStatus.metrics.streaming_performance.status='failed';rejected.push(metricStatus);
const failedRequests=point('failedRequests',32,999);
failedRequests.metrics.streaming_performance.values.service_view.failed_requests=scalar(1,'request');
rejected.push(failedRequests);
for(const value of [null,-1,'900'])rejected.push(point('bad_value',32,value));
for(const concurrency of [null,0,-1,1.5,'16'])rejected.push(point('bad_concurrency',concurrency,999));
for(const unit of ['req/s','',null]) {
  const c=point('bad_unit',32,999);
  c.metrics.streaming_performance.values.service_view.output_token_throughput.unit=unit;
  rejected.push(c);
}
const missingModel=point('model',32,999,{model:''});rejected.push(missingModel);
const wrongPath=point('wrong_path',32,999);
wrongPath.metrics.streaming_performance.values.service_view={};rejected.push(wrongPath);
const invalidSummary=summarize([point('valid_zero',1,0),...rejected]);
assert.equal(invalidSummary.included,1);
assert.equal(invalidSummary.total,rejected.length+1);
assert.equal(invalidSummary.groups[0].maximum,0);
// A normal duration/sample stop may leave planned samples not started.
const stopped={...point('normal_stop',8,40),coverage:{complete:10,not_started:90}};
assert.equal(summarize([stopped]).included,1);
assert.deepEqual(call('performanceSummary',[]),{groups:[],total:0,included:0});

assert.match(call('performanceTrend',peak([4,8,16],[70,140,270])),/持续上升/);
assert.match(call('performanceTrend',peak([4,8,16],[270,140,70])),/持续下降/);
assert.match(call('performanceTrend',peak([4,8,16],[70,270,140])),/先上升、后回落/);
assert.match(call('performanceTrend',peak([4,8,16],[70,70,70])),/相同/);
assert.match(call('performanceTrend',peak([4,8,16],[70,140,140])),/部分并发点持平/);
assert.match(call('performanceTrend',peak([1,2,3,4],[10,20,15,25])),/波动/);
assert.match(call('performanceTrend',peak([4],[0])),/只有一个/);
assert.match(call('performanceTrend',peak([4,4,8],[70,80,140])),/多次测量/);
assert.equal(peak([4,8,16],[70,140,140]).peaks.length,2);
assert.equal(peak([4,4],[10,10]).points.length,2);

// Subject scores, zero, missing values and independent scoring counts survive.
const quality={status:'success',metrics:{choice_accuracy:{status:'success',values:{
  task_view:{cmmlu:{accuracy:{value:.8,unit:'ratio',sample_count:10},
    accuracy_by_subject:{math:{value:0,unit:'ratio',sample_count:1},
      missing:{value:null,unit:'ratio',sample_count:0}},
    total_samples:scalar(20,'sample'),scored_samples:scalar(10,'sample'),
    parse_failed:scalar(2,'sample'),service_failed:scalar(10,'sample')}}}}}};
const scores=call('qualityScoreMetrics',quality);
assert.deepEqual(scores.map(m=>[m.key,m.value,m.count]),[
  ['accuracy',.8,10],['math',0,1],['missing',null,0]]);
quality.status='discarded';assert.deepEqual(call('qualityScoreMetrics',quality),[]);
quality.status='success';quality.metrics.choice_accuracy.status='failed';
assert.deepEqual(call('qualityScoreMetrics',quality),[]);

// Independent best values retain their exact source Case; manual mode is one Case.
function cardCase(id,concurrency,throughput,success,total,p99) {
  const c=point(id,concurrency,throughput);
  const values=c.metrics.streaming_performance.values;
  values.service_view.successful_requests=scalar(success,'request');
  values.service_view.total_requests=scalar(total,'request');
  values.service_view.failed_requests=scalar(total-success,'request');
  for(const [i,key] of ['ttft','tpot','e2el'].entries())
    values.request_view[key]={p99:p99[i],count:total,unit:key==='tpot'?'ms/token':'ms'};
  return c;
}
const a=cardCase('partially_failed',32,999,8,10,[1,1,1]);a.status='partial_failed';
const b=cardCase('throughput_peak',8,200,100,100,[20,2,45]);
const c=cardCase('best_latency',16,150,50,50,[10,4,25]);
const recordsFor=cases=>cases.map((c,ci)=>({c,ci,ti:0,task:task(cases)}));
const getCards=()=>call('performanceOverviewCards',recordsFor([a,b,c]));
const initial=getCards();
assert.deepEqual(initial.map(m=>m.value),[200,400,10,2,1,32,25]);
assert.equal(initial[0].href,'#summary-0-1');
assert.equal(initial[1].href,'#summary-0-1'); // Total throughput from the same best Case.
assert.equal(initial[2].href,'#summary-0-2');
assert.equal(initial[3].href,'#summary-0-1');
assert.equal(initial[3].unit,'ms/token');
assert.equal(initial[5].href,'#summary-0-0');
assert.equal(initial[4].kind,'percentage');
assert.match(initial[4].source,/100 \/ 100/);
assert.deepEqual(call('performanceOverviewCards',recordsFor([a]),false).map(m=>m.value),
  [999,1998,1,1,.8,32,1]);
c.metrics.streaming_performance.values.service_view.total_requests.value=null;
assert.equal(getCards()[4].value,1); // Explicit best Case, never an aggregate claim.
assert.equal(call('performanceOverviewCards',recordsFor([c]),false)[4].value,null);
c.metrics.streaming_performance.values.service_view.total_requests.value=50;
b.metrics.streaming_performance.values.service_view.output_token_throughput.value=null;
b.metrics.streaming_performance.values.request_view.ttft={p99:.005,unit:'s',count:100};
b.metrics.streaming_performance.values.request_view.tpot={p99:.002,unit:'s/token',count:100};
assert.equal(getCards()[0].value,150);
assert.equal(getCards()[1].value,400); // Output missing does not hide total throughput.
assert.equal(getCards()[2].value,5); // A missing throughput does not hide valid latency.
assert.equal(getCards()[3].value,2);
assert.equal(b.metrics.streaming_performance.values.request_view.ttft.p99,.005);
b.metrics.streaming_performance.values.request_view.ttft={p99:0,unit:'ms',count:100};
assert.equal(getCards()[2].value,0);
b.metrics.streaming_performance.values.request_view.ttft.count=0;
assert.equal(getCards()[2].value,10);
for(const value of [-1,null,'1']) {
  b.metrics.streaming_performance.values.request_view.ttft={p99:value,unit:'ms',count:100};
  assert.equal(getCards()[2].value,10);
}
b.metrics.streaming_performance.values.request_view.tpot.weight_sum=-1;
assert.equal(getCards()[3].value,4);
b.metrics.streaming_performance.values.request_view.ttft={p99:1,unit:'unknown',count:100};
assert.equal(getCards()[2].value,10);
assert.ok(call('performanceOverviewCards',[]).every(card=>card.value===null));
const failed=cardCase('all_failed',4,0,0,10,[null,null,null]);failed.status='failed';
const failedScope=call('summaryScopeGroups',[task([failed])])[0];
assert.equal(failedScope.peaks.length,0);
const failedCards=call('performanceOverviewCards',recordsFor([failed]));
assert.deepEqual(failedCards.map(card=>card.value),
  [null,null,null,null,0,4,null]);
failed.metrics.streaming_performance.values.service_view.total_requests.value=0;
failed.metrics.streaming_performance.values.service_view.failed_requests.value=0;
assert.equal(call('performanceOverviewCards',recordsFor([failed]))[4].value,null);
const badCounts=cardCase('bad_counters',4,10,5,10,[1,1,1]);
badCounts.metrics.streaming_performance.values.service_view.failed_requests.value=-1;
assert.equal(call('performanceOverviewCards',recordsFor([badCounts]))[4].value,null);
// Service statistics remain available when no request distributions were saved.
for(const request of [undefined,null,{}]) {
  const serviceOnly=cardCase('service_only',4,10,1,1,[5,2,20]);
  const metric=serviceOnly.metrics.streaming_performance;
  if(request===undefined)delete metric.values.request_view;
  else metric.values.request_view=request;
  assert.deepEqual(call('performanceOverviewCards',recordsFor([serviceOnly]))
    .map(card=>card.value),[10,20,null,null,1,4,null]);
  assert.equal(summarize([serviceOnly]).included,1);
  // A second provider cannot fill missing distributions in the selected one.
  serviceOnly.metrics.z_later=cardCase('later',4,999,1,1,[1,1,1])
    .metrics.streaming_performance;
  assert.deepEqual(call('performanceOverviewCards',recordsFor([serviceOnly]))
    .map(card=>card.value),[10,20,null,null,1,4,null]);
  metric.error='selected provider failed';
  assert.ok(call('performanceOverviewCards',recordsFor([serviceOnly]))
    .every(card=>card.value===null));
  assert.equal(summarize([serviceOnly]).included,0);
}
const serviceFailure=cardCase('service_failure',4,10,1,4,[5,2,20]);
delete serviceFailure.metrics.streaming_performance.values.request_view;
serviceFailure.status='partial_failed';
serviceFailure.metrics.streaming_performance.status='partial_failed';
assert.deepEqual(call('performanceOverviewCards',recordsFor([serviceFailure]))
  .map(card=>card.value),[null,null,null,null,.25,4,null]);
const engineOnly=point('engine',4,999);
engineOnly.metrics={engine_performance:{status:'success',values:{
  engine_view:{internal_ttft:{p99:1,unit:'ms'}},
  service_view:{total_token_throughput:scalar(999)}}}};
assert.ok(call('performanceOverviewCards',recordsFor([engineOnly]))
  .every(card=>card.value===null));
assert.equal(summarize([engineOnly]).included,0);
const other=cardCase('different_length',16,9000,1,1,[.1,.1,.1]);
other.parameters.input_tokens=256;
const separated=summarize([a,b,c,other]);
assert.equal(separated.groups.length,2);
const crossConditions=call('performanceOverviewCards',recordsFor([a,b,c,other]));
assert.equal(crossConditions[0].value,9000);
assert.equal(crossConditions[0].href,'#summary-0-3');

// The SLO section's own cards read only the saved goodput block; the
// overview grid keeps seven cards and the summary table stays free of
// goodput columns.
function withGoodput(c,fields) {
  c.metrics.streaming_performance.values.service_view.goodput={
    status:'applicable',
    slo_config:{ttft_ms:100,tpot_ms:5,e2el_ms:2000},
    slo_satisfied_count:scalar(8,'request'),
    slo_violated_count:scalar(1,'request'),
    failed_count:scalar(1,'request'),
    undetermined_count:scalar(0,'request'),
    determined_count:scalar(10,'request'),
    slo_satisfied_rate:{value:fields.rate,unit:'ratio'},
    goodput_request_throughput:{value:fields.req,unit:'req/s'},
    goodput_output_token_throughput:{value:fields.out,unit:'token/s'},
  };
  return c;
}
const slo=withGoodput(cardCase('slo_case',16,150,50,50,[10,4,25]),
  {rate:.9,req:5,out:1000});
const sloCards=call('sloOverviewCards',recordsFor([slo]));
assert.equal(sloCards.length,2);
assert.equal(sloCards[0].id,'best-slo-attainment');
assert.equal(sloCards[0].value,.9);
assert.equal(sloCards[0].kind,'percentage');
assert.equal(sloCards[0].href,'#summary-0-0');
assert.match(sloCards[0].source,/SLO TTFT≤100ms TPOT≤5ms E2EL≤2000ms/);
assert.equal(sloCards[1].id,'best-goodput-throughput');
assert.equal(sloCards[1].value,1000);
assert.match(sloCards[1].source,/SLO TTFT≤100ms TPOT≤5ms E2EL≤2000ms/);
// The overview grid never carries SLO/Goodput cards, configured or not.
assert.deepEqual(call('performanceOverviewCards',recordsFor([slo]))
  .map(card=>card.id),['peak-throughput','peak-total-throughput',
  'best-ttft-p99','best-tpot-p99','success-rate','max-concurrency',
  'best-e2e-p99']);
// The summary table stays free of goodput columns even when configured.
assert.ok(!call('summaryMetrics',{
  category:'generation_performance',cases:[slo]}).some(m=>
  m.parts.at(-2)==='goodput'),'summary columns exclude goodput fields');
const hover=call('goodputCellTitle',slo);
assert.ok(hover.includes('TTFT≤100ms') && hover.includes('分母：10'),
  'cell hover text carries thresholds and denominator');
const undeterminedSlo=withGoodput(
  cardCase('undetermined_slo',16,150,50,50,[10,4,25]),
  {rate:null,req:null,out:null});
undeterminedSlo.metrics.streaming_performance.values.service_view
  .goodput.undetermined_count.value=50;
undeterminedSlo.metrics.streaming_performance.values.service_view
  .goodput.determined_count.value=0;
const undeterminedCards=call('sloOverviewCards',
  recordsFor([undeterminedSlo]));
assert.equal(undeterminedCards[0].value,null);
assert.equal(undeterminedCards[0].source,'已配置 SLO，暂无可判定的达标数据');
assert.equal(undeterminedCards[1].value,null);
const undeterminedHover=call('goodputCellTitle',undeterminedSlo);
assert.ok(undeterminedHover.includes('无法判定请求 50 个'),
  'hover text reports undetermined requests');
// Unconfigured runs state the missing configuration on the SLO cards.
const unconfiguredCards=call('sloOverviewCards',recordsFor([a]));
assert.equal(unconfiguredCards[0].source,'未配置 SLO（goodput 参数）');
assert.equal(unconfiguredCards[1].source,'未配置 SLO（goodput 参数）');

function qualityCase(id,key,value,unit='ratio') {
  return {id,status:'success',collection_complete:true,metrics_status:'complete',parameters:{},
    metrics:{quality:{values:{task_view:{scores:{[key]:{value,unit,sample_count:10}}}}}}};
}
const qualityTask=(scenario,cases)=>({name:'ability',scenario,category:'model_service_quality',cases});
const qa=qualityCase('a','accuracy',.2),qb=qualityCase('b','accuracy',.8);
const qf=qualityCase('failed','accuracy',.99);qf.status='failed';
let qualityGroups=call('qualityComparisonGroups',[qualityTask('cmmlu',[qa,qb,qf])]);
const accuracyChoice=call('qualityScoreMetrics',qa)[0];
assert.equal(call('bestQualityRecord',qualityGroups[0],accuracyChoice).c.id,'b');
qb.collection_complete=false;
qualityGroups=call('qualityComparisonGroups',[qualityTask('cmmlu',[qa,qb,qf])]);
assert.equal(call('bestQualityRecord',qualityGroups[0],accuracyChoice).c.id,'a');
qa.metrics.quality.values.task_view.scores.accuracy.sample_count=0;
qualityGroups=call('qualityComparisonGroups',[qualityTask('cmmlu',[qa,qb,qf])]);
assert.equal(call('bestQualityRecord',qualityGroups[0],accuracyChoice),null);
const p1=qualityCase('p1','perplexity',10,'ppl'),p2=qualityCase('p2','perplexity',3,'ppl');
const pplGroup=call('qualityComparisonGroups',[qualityTask('wikitext',[p1,p2])])[0];
assert.equal(call('bestQualityRecord',pplGroup,call('qualityScoreMetrics',p1)[0]).c.id,'p2');
p1.parameters.dataset_path='/dataset/a';p2.parameters.dataset_path='/dataset/b';
assert.equal(call('qualityComparisonGroups',[qualityTask('wikitext',[p1,p2])]).length,2);
const custom=qualityCase('unknown','custom_score',900,'points');
assert.equal(call('bestQualityRecord',call('qualityComparisonGroups',[
  qualityTask('custom',[custom])])[0],call('qualityScoreMetrics',custom)[0]),null);

// Session metrics reuse the serving views under their own provider name, and
// carry per-turn rows whose shrinking sample counts must stay visible.
function sessionPoint(id,concurrency,value,turns) {
  const c=point(id,concurrency,value);
  const values=c.metrics.streaming_performance.values;
  delete c.metrics.streaming_performance;
  values.request_view.e2el={mean:40,unit:'ms'};
  const series=(pick,unit)=>turns.map((turn,index)=>
    ({turn_index:index,unit,count:turn.count,mean:pick(turn),p99:pick(turn)+10}));
  values.turn_view={turns_observed:turns.length,
    ttft_ms_by_turn:series(turn=>turn.ttft,'ms'),
    e2el_ms_by_turn:series(turn=>turn.e2el,'ms'),
    input_tokens_by_turn:series(turn=>turn.input,'token'),
    output_tokens_by_turn:series(turn=>turn.output,'token')};
  values.session_view={sessions_finished:{value:turns[0].count,unit:'session'},
    planned_turns:{unit:'turn',count:turns[0].count,mean:turns.length},
    completed_turns:{unit:'turn',count:turns[0].count,mean:turns.length-1},
    not_started_turns:{value:turns[0].count,unit:'turn'},
    abort_reasons:{turn_failed:1}};
  c.metrics.multi_turn_performance={status:'success',values};
  return c;
}
const sessionCase=sessionPoint('session',4,120,[
  {count:4,ttft:10,e2el:40,input:100,output:50},
  {count:3,ttft:14,e2el:52,input:180,output:50},
  {count:1,ttft:20,e2el:70,input:260,output:50}]);
assert.equal(call('servingMetric',sessionCase).status,'success');
assert.equal(call('sessionValues',sessionCase).turn_view.turns_observed,3);
assert.deepEqual(call('sessionValues',{...sessionCase,status:'discarded'}),null);
assert.deepEqual(call('corePerformanceMetrics',sessionCase).map(m=>m.path),
  ['ttft','tpot','itl','e2el'].map(k=>'/multi_turn_performance/values/request_view/'+k)
    .concat(['request_throughput','output_token_throughput','total_token_throughput']
      .map(k=>'/multi_turn_performance/values/service_view/'+k)));
assert.deepEqual(call('summaryMetrics',task([point('s1',4,100)])).map(m=>m.key),
  ['ttft','tpot','output_token_throughput','total_token_throughput']);
// The independent run-level token_throughput metric shares the
// total_token_throughput key but must never be taken for the serving
// calculator, regardless of metric key order.
const runLevel=point('run',4,100);
runLevel.metrics.token_throughput={status:'success',values:{service_view:{
  total_token_throughput:scalar(999)}}};
assert.deepEqual(call('corePerformanceMetrics',runLevel).map(m=>m.key),
  ['ttft','tpot','itl','request_throughput',
    'output_token_throughput','total_token_throughput']);
assert.equal(call('servingMetric',runLevel).status,'success');
const runLevelReversed=JSON.parse(JSON.stringify(runLevel));
runLevelReversed.metrics=Object.fromEntries(
  Object.entries(runLevelReversed.metrics).reverse());
assert.deepEqual(call('corePerformanceMetrics',runLevelReversed).map(m=>m.path),
  call('corePerformanceMetrics',runLevel).map(m=>m.path));
assert.deepEqual(call('summaryMetrics',task([runLevel])).map(m=>m.key),
  ['ttft','tpot','output_token_throughput','total_token_throughput']);
const sessionSummary=call('performanceSummary',[task([sessionCase])]);
assert.equal(sessionSummary.included,1);
assert.equal(sessionSummary.groups[0].maximum,120);
assert.equal(call('savedRequestCounts',sessionCase).valid,false);
// A metric without per-turn facts is not treated as a session metric.
assert.equal(call('sessionValues',qualityCase('q','accuracy',.5)),null);
// Headless render smoke: the participation ledger, abort reasons and
// summary survive a case with no success turns, while the latency
// statistics are marked empty instead of hiding the whole section.
const failedTurnCase={status:'success',metrics:{streaming_performance:{
  status:'success',values:{
    turn_view:{turns_observed:0,ttft_ms_by_turn:[],itl_ms_by_turn:[],
      turn_participation:[{turn_index:0,planned_sessions:3,succeeded:0,
        sent_failed:3,not_started:0}]},
    session_view:{not_started_turns:{value:0,unit:'turn'},
      sent_failed_turns:{value:3,unit:'turn'},
      sessions_finished:{value:0,unit:'session'},
      abort_reasons:{turn_failed:3}}}}}};
domNodes.length=0;
context.input=JSON.stringify(failedTurnCase);
context.PARENT=domNode();
vm.runInContext('renderSessionTurns(PARENT,JSON.parse(input))',context);
let texts=domTexts();
assert.ok(texts.includes('已发送失败'),'participation columns render');
assert.ok(texts.includes('没有已保存的成功轮次延迟统计'),
  'empty latency statistics are stated, not hidden');
assert.ok(texts.includes('会话中止原因'),'abort reasons survive');
assert.ok(!texts.includes('没有已保存的成功轮次。'),
  'the section is not the generic empty placeholder');
// A successful session still renders the metric table with both ITL
// metrics: request-level ITL and the stream-event interval series.
sessionCase.metrics.multi_turn_performance.values.turn_view={
  turns_observed:1,
  ttft_ms_by_turn:[{turn_index:0,count:2,mean:10,unit:'ms'}],
  itl_ms_by_turn:[{turn_index:0,count:2,mean:3,weight_sum:2,unit:'ms/token'}],
  stream_event_itl_ms_by_turn:[{turn_index:0,count:4,mean:100,unit:'ms'}],
  turn_participation:[{turn_index:0,planned_sessions:1,succeeded:1,
    sent_failed:0,not_started:0}]};
domNodes.length=0;
context.PARENT=domNode();
context.input=JSON.stringify(sessionCase);
vm.runInContext('renderSessionTurns(PARENT,JSON.parse(input))',context);
texts=domTexts();
assert.ok(texts.includes('ITL 均值'),'per-turn ITL renders');
assert.ok(texts.includes('Stream Event ITL 均值'),'stream-event ITL renders');
assert.ok(texts.includes('Stream Event ITL / E2EL'),'sample-count header names it');
assert.ok(!texts.includes('没有已保存的成功轮次延迟统计'),
  'latency statistics are present for a successful session');
console.log('PASS: condition grouping, peak identity/ties, failure exclusions, '+
  'zero/missing data, observed trends, subject scores and scoring denominators; '+
  'independent optimum sources, single Case metrics, P99 units, request ratios; '+
  'quality optimization directions, comparable datasets and summary-row links; '+
  'session metrics resolved by view, per-turn participation, ITL and all-failed '+
  'rendering preserved');


// Provider selection must survive actual sorted JSON persistence.
if(process.argv[2]) {
  const persisted=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
  if(html!==null)assert.deepEqual(JSON.parse(html.split('id="report-data">')[1]
    .split('</script>')[0]).result,persisted);
  for(const task of persisted.tasks) {
    const c=task.cases[0],record={task,c,ti:0,ci:0};
    const provider=Object.keys(c.metrics).find(k=>k!=='token_throughput');
    const hasLatency=Boolean(c.metrics[provider].values.request_view?.ttft);
    const cards=call('performanceOverviewCards',[record]);
    assert.equal(cards[0].value,10);
    // The provider saves no total_token_throughput; the independent
    // run-level token_throughput metric (999) must not leak into the card.
    assert.equal(cards[1].value,null);
    assert.equal(cards[2].value,hasLatency?5:null);
    assert.equal(cards[3].value,hasLatency?2:null);
    assert.equal(cards[4].value,1);
    assert.equal(cards[5].value,4);
    assert.equal(cards[6].value,hasLatency?20:null);
    const groups=call('performanceSummary',[task]);
    assert.equal(groups.groups[0].maximum,10);
    const reordered=JSON.parse(JSON.stringify(c));
    reordered.metrics=Object.fromEntries(Object.entries(c.metrics).reverse());
    const reversed=call('performanceOverviewCards',[
      {...record,c:reordered}
    ]);
    assert.deepEqual(reversed.map(m=>m.value),cards.map(m=>m.value));
    assert.equal(call('servingMetric',c).status,'success');
    // An unavailable selected provider stays unavailable with its own status.
    c.metrics[provider].error='provider failed';
    assert.equal(call('performanceOverviewCards',[record])[0].value,null);
    delete c.metrics[provider];
    assert.ok(call('performanceOverviewCards',[record])
      .every(card=>card.value===null));
    assert.equal(call('performanceSummary',[task]).groups.length,0);
  }
  console.log('PASS: persisted common/online/session/custom provider selection');
}
