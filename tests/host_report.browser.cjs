// Offline regression over saved real host/GPU observations.
// node tests/host_report.browser.cjs REAL_HTML OUTPUT_DIRECTORY
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
    await page.goto(pathToFileURL(path.resolve(process.argv[2])).href+'#observation-section');
    const data=JSON.parse(await page.locator('#report-data').textContent());
    const cases=data.result.tasks.flatMap((t,ti)=>t.cases.map((c,ci)=>({c,ti,ci})));
    assert.equal(await page.locator('#resource-case option').count(),cases.length);
    for(const {c,ti,ci} of cases) {
      await page.selectOption('#resource-case',ti+':'+ci);
      assert.equal(c.host_observation.tool,'node_exporter');
      assert.equal(c.host_observation.error_count,0);
      assert.match(await page.locator('#resource-context').textContent(),new RegExp(c.parameters.model.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')));
      assert.equal(await page.locator('#resource-host-content .resource-host-plots>.resource-panel').count(),8);
      const points=data.observations.filter(p=>p.task_index===ti&&p.case_index===ci);
      const cpu=points.filter(p=>p.metric==='host/cpu/utilization'&&Number.isFinite(p.value)&&p.value>=0&&p.value<=100);
      const cpuChart=page.locator('#resource-host-utilization');
      if(cpu.length)assert.ok(await cpuChart.locator('svg circle').count()>=cpu.length);
      for(const id of ['resource-host-memory','resource-host-filesystem-usage',
        'resource-host-disk-throughput','resource-host-network-throughput'])
        assert.ok(await page.locator('#'+id+' svg circle').count()>0,id+' missing real points');
      const ranges=await page.locator('#observation-section svg').evaluateAll(nodes=>
        nodes.map(n=>[n.dataset.start,n.dataset.end]));
      assert.ok(ranges.every(r=>JSON.stringify(r)===JSON.stringify(ranges[0])));
      assert.ok(await page.locator('#resource-host-networks option').count()>0);
      assert.match(await page.locator('#resource-host-networks option:checked').textContent(),/lo/);
      assert.equal(await page.locator('#resource-context a').getAttribute('href'),'#summary-'+ti+'-'+ci);
    }
    await page.selectOption('#resource-case','0:0');
    for(const id of ['resource-host-filesystems','resource-host-disks','resource-host-networks']) {
      const options=page.locator('#'+id+' option');
      await page.selectOption('#'+id,{index:(await options.count())-1});
    }
    await page.selectOption('#resource-host-networks',await page.locator('#resource-host-networks option').evaluateAll(nodes=>nodes.find(n=>n.textContent==='lo').value));
    const output=process.argv[3];
    if(output) {
      fs.mkdirSync(output,{recursive:true});
      await page.locator('#observation-section').screenshot({path:path.join(output,'resources-dark.png')});
      await page.locator('#theme').click();
      await page.locator('#observation-section').screenshot({path:path.join(output,'resources-light.png')});
      await page.locator('#theme').click();
    }
    for(const id of ['overview','summary-section','case-analysis','task-metrics','diagnostics','observation-section']) {
      await page.locator('#report-nav a[href="#'+id+'"]').click();
      await page.waitForFunction(id=>!document.getElementById(id).hidden,id);
      assert.equal(await page.locator('.report-page:visible').count(),1);
    }
    await page.setViewportSize({width:390,height:844});
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
    if(output)await page.screenshot({path:path.join(output,'resources-mobile.png'),fullPage:true});
    assert.deepEqual(network,[]);assert.deepEqual(errors,[]);
    console.log('PASS: '+cases.length+' real Cases, GPU/host data, aligned timelines, '+
      'filesystem/disk/network selectors, lo traffic, themes, six pages, mobile and offline');
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
