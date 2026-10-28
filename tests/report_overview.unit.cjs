// Deterministic summary semantics; no DOM, network, or model calls.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const context = vm.createContext({taskDisplayName:name=>name});
const source = path.join(__dirname,'../src/luban_meter/reporting');
for(const file of ['case_analysis.js','summary_cards.js','report_overview.js'])
  vm.runInContext(fs.readFileSync(path.join(source,file),'utf8'),context);
const call = (name,value,...extra) => {
  context.input=JSON.stringify(value);
  return JSON.parse(vm.runInContext('JSON.stringify('+name+'(JSON.parse(input)'+
    extra.map(v=>','+JSON.stringify(v)).join('')+'))',context));
};
const scalar=(value,unit='token/s')=>({value,unit});
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
assert.deepEqual(initial.map(m=>m.value),[200,10,2,1,32,25]);
assert.equal(initial[0].href,'#summary-0-1');
assert.equal(initial[1].href,'#summary-0-2');
assert.equal(initial[2].href,'#summary-0-1');
assert.equal(initial[2].unit,'ms/token');
assert.equal(initial[4].href,'#summary-0-0');
assert.equal(initial[3].kind,'percentage');
assert.match(initial[3].source,/100 \/ 100/);
assert.deepEqual(call('performanceOverviewCards',recordsFor([a]),false).map(m=>m.value),
  [999,1,1,.8,32,1]);
c.metrics.streaming_performance.values.service_view.total_requests.value=null;
assert.equal(getCards()[3].value,1); // Explicit best Case, never an aggregate claim.
assert.equal(call('performanceOverviewCards',recordsFor([c]),false)[3].value,null);
c.metrics.streaming_performance.values.service_view.total_requests.value=50;
b.metrics.streaming_performance.values.service_view.output_token_throughput.value=null;
b.metrics.streaming_performance.values.request_view.ttft={p99:.005,unit:'s',count:100};
b.metrics.streaming_performance.values.request_view.tpot={p99:.002,unit:'s/token',count:100};
assert.equal(getCards()[0].value,150);
assert.equal(getCards()[1].value,5); // A missing throughput does not hide valid latency.
assert.equal(getCards()[2].value,2);
assert.equal(b.metrics.streaming_performance.values.request_view.ttft.p99,.005);
b.metrics.streaming_performance.values.request_view.ttft={p99:0,unit:'ms',count:100};
assert.equal(getCards()[1].value,0);
b.metrics.streaming_performance.values.request_view.ttft.count=0;
assert.equal(getCards()[1].value,10);
for(const value of [-1,null,'1']) {
  b.metrics.streaming_performance.values.request_view.ttft={p99:value,unit:'ms',count:100};
  assert.equal(getCards()[1].value,10);
}
b.metrics.streaming_performance.values.request_view.tpot.weight_sum=-1;
assert.equal(getCards()[2].value,4);
b.metrics.streaming_performance.values.request_view.ttft={p99:1,unit:'unknown',count:100};
assert.equal(getCards()[1].value,10);
assert.ok(call('performanceOverviewCards',[]).every(card=>card.value===null));
const failed=cardCase('all_failed',4,0,0,10,[null,null,null]);failed.status='failed';
const failedScope=call('summaryScopeGroups',[task([failed])])[0];
assert.equal(failedScope.peaks.length,0);
const failedCards=call('performanceOverviewCards',recordsFor([failed]));
assert.deepEqual(failedCards.map(card=>card.value),[null,null,null,0,4,null]);
failed.metrics.streaming_performance.values.service_view.total_requests.value=0;
failed.metrics.streaming_performance.values.service_view.failed_requests.value=0;
assert.equal(call('performanceOverviewCards',recordsFor([failed]))[3].value,null);
const badCounts=cardCase('bad_counters',4,10,5,10,[1,1,1]);
badCounts.metrics.streaming_performance.values.service_view.failed_requests.value=-1;
assert.equal(call('performanceOverviewCards',recordsFor([badCounts]))[3].value,null);
const other=cardCase('different_length',16,9000,1,1,[.1,.1,.1]);
other.parameters.input_tokens=256;
const separated=summarize([a,b,c,other]);
assert.equal(separated.groups.length,2);
const crossConditions=call('performanceOverviewCards',recordsFor([a,b,c,other]));
assert.equal(crossConditions[0].value,9000);
assert.equal(crossConditions[0].href,'#summary-0-3');

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
console.log('PASS: condition grouping, peak identity/ties, failure exclusions, '+
  'zero/missing data, observed trends, subject scores and scoring denominators; '+
  'independent optimum sources, single Case metrics, P99 units, request ratios; '+
  'quality optimization directions, comparable datasets and summary-row links');
