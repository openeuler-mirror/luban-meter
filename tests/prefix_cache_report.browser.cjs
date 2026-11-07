// Offline Mock UI integration. Requires installed Playwright and Edge/Chrome.
// node tests/prefix_cache_report.browser.cjs MOCK_HTML OUTPUT_DIRECTORY
const assert=require('node:assert/strict'),fs=require('node:fs');
const path=require('node:path'),{chromium}=require('playwright');
(async()=>{
  const browser=await chromium.launch({headless:true,
    channel:process.env.LUBAN_BROWSER_CHANNEL||'msedge'});
  try {
    const page=await browser.newPage({viewport:{width:1440,height:1000}});
    const errors=[],network=[];
    page.on('pageerror',e=>errors.push(e.message));
    await page.route('**/*',route=>{network.push(route.request().url());return route.abort();});
    await page.setContent(fs.readFileSync(process.argv[2],'utf8'));
    const nav=async id=>{
      await page.locator('#report-nav a[href="#'+id+'"]').click();
      await page.waitForFunction(id=>!document.getElementById(id).hidden,id);
    };
    assert.deepEqual(errors,[]);
    assert.match(await page.locator('#subtitle').textContent(),/Mock.*非真实 GPU/);
    assert.equal(await page.locator('#report-nav a').count(),6);
    const report=JSON.parse(await page.locator('#report-data').textContent());
    assert.equal(await page.locator('#analysis-task option').count(),report.result.tasks.length);
    await nav('summary-section');
    assert.equal(await page.locator('.summary-category').count(),2);
    const panel=page.locator('.cache-summary-case').nth(4);
    assert.match(await panel.innerText(),/前缀缓存评测 · 100%/);
    assert.match(await panel.innerText(),/98.4375%/);
    assert.equal(await panel.locator('table').count(),3);
    for(const width of [420,1440]) {
      await page.setViewportSize({width,height:1000});
      assert.ok(await panel.locator('.table-wrap').evaluateAll(
        nodes=>nodes.every(n=>n.scrollWidth<=n.clientWidth+1)));
    }
    const comparison=page.locator('.prefix-cache-comparison');
    assert.equal(await comparison.count(),1);
    assert.equal(await comparison.locator('select').first().locator('option').count(),2);
    const choose=page.getByLabel('缓存对比指标');
    for (const metric of ['actual_hit_rate','ttft','itl','tpot','e2el',
      'output_token_throughput','request_throughput']) {
      await choose.selectOption(metric);
      assert.match(await comparison.textContent(),/横轴：目标/);
      assert.equal(await comparison.locator('svg').count(),1);
    }
    await choose.selectOption('actual_hit_rate');
    const row=id=>comparison.locator('tbody tr').filter({
      has:page.locator('td:first-child').filter({hasText:new RegExp('^'+id+'$')})});
    assert.match(await row('request-failure').textContent(),/未采集.*partial.*unknown.*排除/s);
    assert.match(await row('partial').textContent(),/未采集.*排除/);
    assert.match(await row('target-100').textContent(),/98.4375.*construction_limited/);
    assert.match(await row('below-theory').textContent(),/valid.*failed.*可参与/);
    assert.match(await row('warmup-failed').textContent(),/未通过.*排除/);
    const output=process.argv[3];
    if(output) {
      fs.mkdirSync(output,{recursive:true});
      await comparison.screenshot({path:path.join(output,'comparison.png')});
    }
    await nav('case-analysis');
    await page.locator('#analysis-task').selectOption('0');
    await page.locator('#analysis-case').selectOption('2');
    assert.equal(await page.locator('#analysis-metric optgroup').count(),2);
    assert.equal(await page.locator('#analysis-metric option').count(),14);
    for (const [ci,pattern] of [['0',/实际缓存命中率0 %/],
      ['4',/构造偏差-1.5625 pp/],['5',/实际缓存命中率未采集/],
      ['6',/实际缓存命中率未采集/],['11',/测量有效性.*valid/]]) {
      await page.locator('#analysis-case').selectOption(ci);
      assert.match(await page.locator('.prefix-cache-details').textContent(),pattern);
    }
    await page.locator('#analysis-case').selectOption('5');
    assert.match(await page.locator('.prefix-cache-details').textContent(),
      /已测子集命中率（非完整 Case）50 %/);
    if(output) await page.locator('#case-analysis').screenshot({
      path:path.join(output,'partial-statistics.png')});
    await page.locator('#analysis-task').selectOption('1');
    assert.equal(await page.locator('.prefix-cache-details').count(),0);
    assert.equal(await page.locator('#analysis-metric optgroup').count(),0);
    await page.locator('#analysis-task').selectOption('2');
    assert.equal(await page.locator('#analysis-metric-field').isVisible(),false);
    assert.equal(await page.locator('.prefix-cache-details').count(),0);
    await nav('diagnostics');
    await page.evaluate(()=>{location.hash='#case-0-2-request-0';});
    await page.waitForFunction(()=>!!document.querySelector('#case-0-2-request-0 button[aria-expanded="true"]'));
    const requestText=await page.locator('#case-0-2-request-0 + tr').textContent();
    assert.match(requestText,/Prefix Cache 请求事实/);
    assert.match(requestText,/prefix_id0/);
    assert.match(requestText,/单请求实际命中率50 %/);
    assert.match(requestText,/输入长度符合状态conforming/);
    for (const [ci,expected] of [[0,'0 %'],[6,'未采集']]) {
      await page.evaluate(ci=>{location.hash='#case-0-'+ci+'-request-0';},ci);
      await page.waitForFunction(ci=>!!document.querySelector(
        '#case-0-'+ci+'-request-0 button[aria-expanded="true"]'),ci);
      assert.ok((await page.locator('#case-0-'+ci+'-request-0 + tr').textContent())
        .includes('单请求实际命中率'+expected));
    }
    const savedOnly=await page.evaluate(()=>{
      const parent=document.createElement('div');
      renderCacheRequest(parent,{prefix_cache:{prompt_tokens:1024,
        cached_tokens:512,hit_rate:.123}});
      return parent.textContent;
    });
    assert.match(savedOnly,/单请求实际命中率12.3 %/);
    await nav('summary-section');
    await page.setViewportSize({width:390,height:844});
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
    if(output) await page.screenshot({path:path.join(output,'mobile.png')});
    assert.deepEqual(errors,[]);
    assert.deepEqual(network,[]);
    console.log('Prefix Cache offline browser checks passed; no external requests');
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exit(1);});
