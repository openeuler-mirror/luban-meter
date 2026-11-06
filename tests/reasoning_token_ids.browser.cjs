// Run against test_reasoning_token_ids.py's local SSE worker report.
// NODE_PATH must include Playwright. Args: report.html.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const {chromium} = require('playwright');

(async () => {
  const browser = await chromium.launch({headless:true,channel:'chrome'});
  try {
    const page = await browser.newPage({viewport:{width:1440,height:1100}});
    const errors=[], network=[];
    page.on('pageerror',error=>errors.push(error.message));
    await page.route('**/*',route=>{
      network.push(route.request().url());return route.abort();
    });
    await page.setContent(fs.readFileSync(process.argv[2],'utf8'));
    const tables=page.locator('#case-summaries .summary-table');
    for(const [index,count] of [[0,1],[1,2]]) {
      const row=await tables.nth(index).evaluate(table=>{
        const headers=[...table.querySelectorAll('thead th')]
          .map(cell=>cell.textContent.replace('↕','').trim());
        const cells=[...table.querySelector('tbody tr').cells];
        return Object.fromEntries(headers.map((header,i)=>[header,cells[i].textContent]));
      });
      for(const label of ['思考每Token耗时（TPOT）','思考Token间延迟（ITL）']) {
        assert.equal(row[label+' 有效样本数（N）'],String(count));
        assert.ok(Object.hasOwn(row,label+' 均值（ms/token）'));
        assert.notEqual(row[label+' 均值（ms/token）'],'不可用');
        assert.notEqual(row[label+' 均值（ms/token）'],'未采集');
      }
      assert.equal(row['思考每Token耗时（TPOT） 权重总数（Token）'],String(2*count));
      assert.equal(row['思考Token间延迟（ITL） 权重总数（Token间隔）'],String(count));
    }
    assert.ok(!(await tables.nth(2).innerText()).includes('思考'));
    await page.locator('#report-nav a[href="#case-analysis"]').click();
    for(const label of ['思考每Token耗时（TPOT）','思考Token间延迟（ITL）']) {
      await page.locator('#analysis-metric').selectOption({label});
      const content=page.locator('#analysis-content');
      assert.match(await content.innerText(),/Token ID补充/);
      const stats=content.locator('.stat-table').filter({
        has:page.getByRole('columnheader',{name:'统计量',exact:true}),
      });
      assert.equal(await stats.locator('tbody tr').count(),1);
      assert.match(await stats.innerText(),/均值/);
      assert.equal(await content.locator('.chart').count(),0);
    }
    const saved=await page.locator('#report-data').textContent();
    const tasks=JSON.parse(saved).result.tasks;
    for(const [index,count] of [[0,1],[1,2]]) {
      assert.deepEqual(tasks[index].cases[0].metrics.streaming_phase_timing
        .metadata.reasoning_token_sources,{token_ids:count});
    }
    await page.setViewportSize({width:390,height:844});
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
    assert.deepEqual(errors,[]);
    assert.deepEqual(network,[]);
    console.log('PASS: Token ID fallback means, weights, source metadata, disabled task and offline HTML');
  } finally {
    await browser.close();
  }
})().catch(error=>{console.error(error);process.exit(1);});
