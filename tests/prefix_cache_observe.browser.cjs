// Offline Observe UI regression; input must be the explicit Mock fixture.
// python -m tests.test_prefix_cache_observe /tmp/observe.html
// node tests/prefix_cache_observe.browser.cjs /tmp/observe.html [SCREENSHOT]
const assert=require('node:assert/strict'),fs=require('node:fs');
const {chromium}=require('playwright');
(async()=>{
 const browser=await chromium.launch({headless:true,
  channel:process.env.LUBAN_BROWSER_CHANNEL||'msedge'});
 try {
  const page=await browser.newPage({viewport:{width:1440,height:1000}});
  const errors=[],network=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.route('**/*',r=>{network.push(r.request().url());return r.abort();});
  await page.setContent(fs.readFileSync(process.argv[2],'utf8'));
  const data=JSON.parse(await page.locator('#report-data').textContent());
  assert.match(data.result.name,/Mock.*非 GPU/);
  await page.locator('#report-nav a[href="#summary-section"]').click();
  await page.waitForFunction(()=>!document.getElementById('summary-section').hidden);
  const comparison=page.locator('.prefix-cache-comparison');
  assert.match(await comparison.textContent(),/observe.*是 \/ 否.*1.*可参与/s);
  assert.equal(await page.evaluate(()=>usablePerformanceResult(tasks[0].cases[0])),true);
  assert.equal(await page.evaluate(()=>cacheEligible(tasks[0].cases[0])),true);
  await page.locator('#report-nav a[href="#case-analysis"]').click();
  await page.waitForFunction(()=>!document.getElementById('case-analysis').hidden);
  const detail=page.locator('.prefix-cache-details'),text=await detail.textContent();
  assert.match(text,/实际缓存命中率50 %/);
  assert.match(text,/observe/);
  assert.match(text,/是 \/ 否/);
  assert.match(text,/verify05120-512命中不足 \/ under_hit/);
  assert.match(text,/有效 \/ valid/);
  assert.doesNotMatch(text,/实验条件符合性/);
  assert.match(text,/符合 \/ passed/);
  assert.match(text,/warmup_cache_mismatch/);
  const metric=data.result.cases[0].metrics.prefix_cache_hit_rate;
  assert.equal(metric.values.actual_hit_rate.value,.5);
  assert.equal(metric.warmup.warmup_fully_matched,false);
  assert.equal(metric.target_attainment.status,'passed');
  if(process.argv[3])await detail.screenshot({path:process.argv[3]});
  assert.deepEqual(errors,[]);
  assert.deepEqual(network,[]);
  console.log('Observe Mock report UI passed; zero external requests');
 } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exit(1);});
