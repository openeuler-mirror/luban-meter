// Offline resource UI regression over explicit synthetic and saved real results.
// node tests/resource_report.browser.cjs FIXTURE_HTML REAL_HTML OUTPUT_DIR
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {pathToFileURL}=require('node:url');
const {chromium}=require('playwright');

(async()=>{
  const browser=await chromium.launch({headless:true,
    channel:process.env.LUBAN_BROWSER_CHANNEL||'msedge'});
  try {
    const page=await browser.newPage({viewport:{width:1560,height:1100}});
    const errors=[],network=[];
    page.on('pageerror',e=>errors.push(e.message));
    await page.route(/^https?:/,r=>{network.push(r.request().url());return r.abort();});
    const open=async file=>{
      await page.goto(pathToFileURL(path.resolve(file)).href+'#observation-section');
      await page.locator('#resource-case').waitFor();
      assert.equal(await page.locator('.report-page:visible').count(),1);
    };
    await open(process.argv[2]);
    assert.equal(await page.locator('#resource-case option').count(),3);
    assert.equal(await page.locator('#resource-case').inputValue(),'0:0');
    assert.match(await page.locator('#resource-context').textContent(),
      /synthetic-resource-run.*fixture-model/);
    assert.equal(await page.locator('.resource-device-table tbody tr').count(),2);
    assert.deepEqual(await page.locator('.resource-device-table tbody tr').first()
      .locator('td').allTextContents(),[
      'GPU 0Synthetic GPU','33.33','2,048','不可用','0','38']);
    assert.equal(await page.locator('#resource-utilization svg circle').count(),6);
    // A five-second hole must not be drawn as a continuous measurement.
    assert.equal(await page.locator('#resource-utilization svg polyline').first()
      .getAttribute('points').then(p=>p.split(' ').length),2);
    for(const [id,unit] of [['utilization','%'],['memory_used','MiB'],
      ['power','W'],['temperature','°C']])
      assert.equal(await page.locator('#resource-'+id+' .resource-unit').textContent(),unit);
    assert.equal(await page.locator('#resource-host-utilization svg circle').count(),3);
    assert.match(await page.locator('#resource-host-content').textContent(),/磁盘不可用.*网络不可用/);
    const ranges=await page.locator('#observation-section svg').evaluateAll(nodes=>
      nodes.map(n=>[n.dataset.start,n.dataset.end]));
    assert.ok(ranges.every(r=>JSON.stringify(r)===JSON.stringify(ranges[0])));
    await page.selectOption('#resource-gpu','0');
    assert.equal(await page.locator('.resource-device-table tbody tr').count(),1);
    assert.equal(await page.locator('#resource-utilization svg circle').count(),3);
    assert.match(await page.locator('#resource-utilization circle title').first().textContent(),
      /GPU 0.*0 %/);
    // Positional identity isolates tasks even when both task and Case names match.
    await page.selectOption('#resource-case','1:0');
    assert.equal(await page.locator('#resource-utilization svg circle').count(),1);
    assert.equal(await page.locator('#resource-utilization svg polyline').count(),0);
    assert.equal(await page.locator('#resource-memory_used svg').count(),0);
    assert.match(await page.locator('#resource-memory_used').textContent(),/不可用/);
    assert.match(await page.locator('#resource-host-utilization').textContent(),/0 %.*仅保存汇总/);
    assert.equal(await page.locator('#resource-service-content svg').count(),0);
    await page.selectOption('#resource-case','0:1');
    assert.equal(await page.locator('#resource-gpu-content svg').count(),0);
    assert.equal(await page.locator('#resource-gpu').isDisabled(),true);
    assert.match(await page.locator('#resource-gpu-content').textContent(),/不可用/);

    const source=fs.readFileSync(process.argv[2],'utf8');
    const match=source.match(/(<script type="application\/json" id="report-data">)([\s\S]*?)(<\/script>)/);
    const sampled=JSON.parse(match[2]);
    sampled.result.tasks[0].cases[0].device_observation.sample_count=1501;
    for(const p of sampled.observations) if(p.task_index===0&&p.case_index===0&&
      p.metric.startsWith('device/')) {
      const start=sampled.result.tasks[0].cases[0].window.started_at;
      p.time=start+({0:0,1:4,6:8}[p.time-start]??0);
    }
    await page.goto('about:blank');
    await page.setContent(source.replace(match[0],match[1]+JSON.stringify(sampled)+match[3]));
    await page.evaluate(()=>{location.hash='observation-section';});
    // Retained four-second samples are continuous after the 500-point cap.
    assert.equal(await page.locator('#resource-utilization polyline').first()
      .getAttribute('points').then(p=>p.split(' ').length),3);

    if(process.argv[3]) {
      await open(process.argv[3]);
      const data=JSON.parse(await page.locator('#report-data').textContent());
      const entries=data.result.tasks.flatMap((task,ti)=>task.cases.map((c,ci)=>({c,ti,ci})));
      assert.equal(await page.locator('#resource-case option').count(),entries.length);
      for(const {c,ti,ci} of entries) {
        await page.selectOption('#resource-case',ti+':'+ci);
        assert.match(await page.locator('#resource-context').textContent(),new RegExp(data.result.id));
        const points=data.observations.filter(p=>p.task_index===ti&&p.case_index===ci&&
          p.metric==='device/utilization'&&Number.isFinite(p.value)&&p.value>=0&&p.value<=100);
        assert.equal(await page.locator('#resource-utilization svg circle').count(),points.length);
        assert.equal(await page.locator('#resource-context a').getAttribute('href'),
          '#summary-'+ti+'-'+ci);
        if(c.device_observation?.devices?.length)
          assert.equal(await page.locator('.resource-device-table tbody tr').count(),
            c.device_observation.devices.length);
      }
      await page.selectOption('#resource-case','0:0');
      assert.match(await page.locator('.resource-device-table tbody tr').first().textContent(),/87,812/);
      assert.match(await page.locator('#resource-host-content').textContent(),/CPU利用率%不可用/);
      await page.locator('#resource-service-content details>summary').click();
      await page.locator('#resource-series').selectOption({index:0});
      assert.ok(await page.locator('#resource-service-content details svg circle').count()>0);
      assert.match(await page.locator('#resource-service-content details').textContent(),/单位未保存/);
      await page.locator('#resource-service-content details>summary').click();
      if(process.argv[4]) {
        fs.mkdirSync(process.argv[4],{recursive:true});
        await page.locator('#observation-section').screenshot({
          path:path.join(process.argv[4],'resources-dark.png')});
        await page.selectOption('#resource-gpu','0');
        await page.locator('#observation-section').screenshot({
          path:path.join(process.argv[4],'resources-gpu0.png')});
        await page.locator('#theme').click();
        await page.locator('#observation-section').screenshot({
          path:path.join(process.argv[4],'resources-light.png')});
        await page.locator('#theme').click();
      }
      await page.setViewportSize({width:390,height:844});
      assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
      assert.equal(await page.locator('.resource-chart-grid').first().evaluate(n=>
        getComputedStyle(n).gridTemplateColumns.split(' ').length),1);
      if(process.argv[4]) await page.screenshot({
        path:path.join(process.argv[4],'resources-mobile.png'),fullPage:true});
    }
    assert.deepEqual(errors,[]);assert.deepEqual(network,[]);
    console.log('Resource report: Case isolation, GPU units, host availability, gaps, '+
      'zero/single/missing samples, real data, themes, narrow layout and offline checks passed');
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
