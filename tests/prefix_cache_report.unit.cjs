// Report-only checks over Mock saved metrics, no network or model calls.
const assert=require('node:assert/strict'),fs=require('node:fs');
const path=require('node:path'),vm=require('node:vm');
const context=vm.createContext({taskDisplayName:x=>x,fmt:x=>String(x??'未采集')});
for (const file of ['case_analysis.js','summary_cards.js','report_overview.js'])
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../src/luban_meter/reporting',file),'utf8'),context);
const call=(name,...args)=>JSON.parse(vm.runInContext(
  'JSON.stringify('+name+'(...'+JSON.stringify(args)+'))',context));
const data=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
const task=data.tasks[0],find=id=>task.cases.find(c=>c.id===id);
assert.equal(call('valueText',0,'ratio'),'0');
assert.equal(call('valueText',null,'ratio'),'未采集');
assert.equal(call('valueText',.984375,'cache_ratio'),'98.4375');
assert.equal(call('valueText',-.015625,'ratio_pp'),'-1.5625');
assert.equal(call('displayUnit','ratio_pp'),'pp');
assert.equal(call('valueText',.123456,'ratio'),'12.35');
assert.equal(call('valueText',.123456,'cache_ratio'),'12.3456');
assert.equal(call('cacheEligible',find('missing')),false);
assert.equal(call('cacheEligible',find('unknown')),true);
assert.equal(call('cacheEligible',find('target-100')),true);
assert.equal(call('cacheEligible',find('below-theory')),true);
assert.equal(call('usablePerformanceResult',find('length-mismatch')),true);
const groups=call('cacheComparisonGroups',task);
assert.equal(groups.length,2); // Only concurrency differs.
assert.equal(groups[0].cases.length,14);
for(const [key,value] of [['model','other'],['input_tokens',512],['output_tokens',64],
  ['profile','rate'],['request_rate',8],['base_url','other-server'],
  ['request_params',{temperature:0}],['kv_cache_block_size',32]]) {
  const a=structuredClone(find('target-50')),b=structuredClone(a);
  b.parameters[key]=value;
  assert.equal(call('cacheComparisonGroups',{cases:[a,b]}).length,2,key);
}
const savedOverride=structuredClone(find('target-50'));
savedOverride.metrics.prefix_cache_hit_rate.values.actual_hit_rate.value=.321;
assert.equal(call('cacheValue',savedOverride,'actual_hit_rate'),.321);
assert.equal(call('metricsFor',savedOverride).find(m=>m.key==='actual_hit_rate').value,.321);
const before=JSON.stringify(find('target-100'));
const metrics=call('metricsFor',find('target-100'));
assert.ok(metrics.every(m=>!m.path.includes('/diagnostics/')));
assert.equal(metrics.find(m=>m.key==='construction_deviation').unit,'ratio_pp');
assert.equal(metrics.find(m=>m.key==='theoretical_shared_ratio').value,.984375);
assert.equal(JSON.stringify(find('target-100')),before);
assert.ok(call('summaryMetrics',task).some(m=>m.key==='actual_hit_rate'));
assert.equal(call('corePerformanceMetrics',find('target-50')).length,7);
assert.equal(call('reportRowText',{metric:'/prefix_cache_hit_rate/values/target_deviation',
  value:-.015625,unit:'ratio'},'value'),'-1.5625');
const subset=call('metricsFor',find('partial'));
assert.equal(subset.find(m=>m.key==='actual_hit_rate').value,null);
assert.equal(subset.find(m=>m.key==='observed_subset_hit_rate').value,.5);
assert.equal(call('cacheComparisonGroups',data.tasks[1]).length,0);
console.log('Prefix Cache report JavaScript checks passed');

const records=task.cases.map((c,ci)=>({task,ti:0,c,ci}));
const eligible=structuredClone(find('unknown'));
const broken=structuredClone(eligible);
broken.parameters.num_requests++;
for(const candidate of [broken,find('missing'),find('request-failure')]) {
  const cards=call('performanceOverviewCards',[{task,ti:0,c:candidate,ci:0}]);
  assert.ok(cards.every(card=>card.value===null));
}
assert.ok(call('performanceOverviewCards',records).every(card=>card.value!==null));

for(const [id,title] of [['target-0','0%'],['target-25','25%'],['target-100','100%']]) {
  assert.equal(call('taskSectionTitle',{scenario:'prefix_cache',cases:[find(id)]}),
    '前缀缓存评测 · '+title);
}
assert.equal(call('taskSectionTitle',task),'前缀缓存评测');

const failedFormal=find('request-failure');
assert.equal(call('cacheValue',failedFormal,'actual_hit_rate'),null);
assert.equal(call('cacheValue',failedFormal,'observed_subset_hit_rate'),.5);
assert.equal(call('cacheValue',failedFormal,'request_coverage'),.5);
assert.equal(call('cacheValue',failedFormal,'token_coverage'),null);
assert.equal(failedFormal.metrics.prefix_cache_hit_rate.target_attainment.status,'unknown');
