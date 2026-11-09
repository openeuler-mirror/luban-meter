// Run against test_phase_timing.py's local-service report; no model/network.
// NODE_PATH must include Playwright. Args: report.html output-directory.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require('playwright');

(async () => {
  const browser = await chromium.launch({headless:true,channel:'chrome'});
  try {
    const page = await browser.newPage({viewport:{width:1440,height:1100}});
    const errors=[], network=[];
    page.on('pageerror',error=>errors.push(error.message));
    await page.route('**/*',route=>{network.push(route.request().url());return route.abort();});
    await page.setContent(fs.readFileSync(process.argv[2],'utf8'));
    async function summaryRow(taskIndex,id) {
      return page.locator('#case-summaries .summary-table').nth(taskIndex)
        .evaluate((table,id)=>{
          const headers=[...table.querySelectorAll('thead th')]
            .map(cell=>cell.textContent.replace('↕','').trim());
          const row=[...table.querySelectorAll('tbody tr')]
            .find(row=>row.cells[0].textContent===id);
          return Object.fromEntries(headers.map((header,index)=>
            [header,row.cells[index].textContent]));
        },id);
    }
    const sampleLabels=['思考输出时长 有效样本数（N）',
      '正式回答输出时长 有效样本数（N）','首个正式回答延迟 有效样本数（N）'];
    const countLabels=['输出截断请求（request）','排除请求（request）'];
    await page.locator('#report-nav a[href="#summary-section"]').click();
    const initial=await summaryRow(0,'case_1');
    for(const label of sampleLabels) assert.equal(initial[label],'3',label);
    assert.equal(initial[countLabels[0]],'1');
    assert.equal(initial[countLabels[1]],'4');
    assert.equal(initial['完成 / 失败'],'7 / 1');
    const totalThroughput='总 Token 吞吐量（token/s）';
    assert.equal(initial[totalThroughput],'未采集');
    const thinkingLabels=[['思考首Token延迟（TTFT）',3],['思考每Token耗时（TPOT）',2],
      ['思考Token间延迟（ITL）',1],['思考流事件间隔',2]];
    for(const [label,count] of thinkingLabels) {
      const unit=label==='思考流事件间隔'?'有效间隔数':'有效样本数';
      assert.equal(initial[label+' '+unit+'（N）'],String(count));
    }
    assert.equal(initial['思考每Token耗时（TPOT） 权重总数（Token）'],'5');
    assert.equal(initial['思考Token间延迟（ITL） 权重总数（Token间隔）'],'3');
    const multi=await summaryRow(1,'case_1');
    assert.ok(Object.hasOwn(multi,totalThroughput));
    assert.notEqual(multi[totalThroughput],'未采集');
    for(const label of sampleLabels) assert.equal(multi[label],'2');
    for(const label of countLabels) assert.equal(multi[label],'0');
    assert.equal(multi['思考每Token耗时（TPOT） 权重总数（Token）'],'8');
    assert.equal(multi['思考Token间延迟（ITL） 权重总数（Token间隔）'],'6');
    const disabled=await summaryRow(2,'case_1');
    assert.ok(!Object.keys(disabled).some(label=>label.includes('思考')));
    for(const label of [...sampleLabels,...countLabels])
      assert.ok(!Object.hasOwn(disabled,label),label);
    await page.locator('#report-nav a[href="#case-analysis"]').click();
    const metric=page.locator('#analysis-metric');
    const options=await metric.locator('option').allTextContents();
    assert.ok(options.includes('总 Token 吞吐量'));
    await metric.selectOption({label:'总 Token 吞吐量'});
    assert.equal(await metric.inputValue(),
      '/streaming_performance/values/service_view/total_token_throughput');
    for(const label of ['思考输出时长','正式回答输出时长','首个正式回答延迟']) {
      assert.ok(options.includes(label));
      await metric.selectOption({label});
      assert.match(await page.locator('#analysis-content').innerText(),/（N）：3/);
    }
    for(const [label,count] of thinkingLabels) {
      assert.ok(options.includes(label));
      await metric.selectOption({label});
      assert.match(await page.locator('#analysis-content').innerText(),
        new RegExp('（N）：'+count));
    }
    await metric.selectOption({label:'思考Token间延迟（ITL）'});
    assert.match(await page.locator('#analysis-content').innerText(),
      /权重总和：3 Token间隔/);
    assert.match(await page.locator('#analysis-content').innerText(),
      /仅按R−1加权计算平均值，并非逐Token实测间隔/);
    for(const label of ['思考每Token耗时（TPOT）','思考Token间延迟（ITL）']) {
      await metric.selectOption({label});
      const rows=await page.locator('#analysis-content .stat-table')
        .filter({has:page.getByRole('columnheader',{name:'统计量',exact:true})})
        .locator('tbody tr').allTextContents();
      assert.equal(rows.length,1);
      assert.match(rows[0],/均值/);
      assert.equal(await page.locator('#analysis-content .chart').count(),0);
    }
    const counts=page.locator('#analysis-content .analysis-group').first();
    assert.match(await counts.innerText(),/排除请求\s+4/);
    assert.match(await counts.innerText(),/输出截断请求\s+1/);
    assert.match(await counts.innerText(),/思考单事件样本（符合统计条件）\s+1/);
    assert.match(await page.locator('#case-summaries').innerText(),/思考输出时长/);
    await page.locator('#analysis-task').selectOption('1');
    await metric.selectOption({label:'思考输出时长'});
    assert.match(await page.locator('#analysis-content').innerText(),/（N）：2/);
    await page.locator('#analysis-task').selectOption('2');
    const disabledOptions=await metric.locator('option').allTextContents();
    assert.ok(!disabledOptions.some(label=>label.includes('思考')));
    for(const label of ['正式回答输出时长','首个正式回答延迟']) {
      assert.ok(!disabledOptions.includes(label));
    }
    const saved=JSON.parse(await page.locator('#report-data').textContent());
    for(const task of saved.result.tasks.slice(2))
      assert.ok(!Object.hasOwn(task.cases[0].metrics,'streaming_phase_timing'));

    // Preserve empty distributions and explicit overall-only metric selection.
    await page.evaluate(() => {
      const original=tasks[0].cases[0];
      const empty=JSON.parse(JSON.stringify(original));
      const view=empty.metrics.streaming_phase_timing.values.request_view;
      for(const node of Object.values(view)) {
        for(const [key,value] of Object.entries(node))
          if(typeof value==='number')node[key]=null;
        node.count=0;
        if(Object.hasOwn(node,'weight_sum'))node.weight_sum=0;
      }
      tasks[0].cases.push({...empty,id:'empty'});
      const overallOnly=JSON.parse(JSON.stringify(original));
      overallOnly.parameters.metrics=['streaming_performance'];
      delete overallOnly.metrics.streaming_phase_timing;
      tasks[0].cases.push({...overallOnly,id:'overall_only'});
      const different=JSON.parse(JSON.stringify(original));
      const phases=different.metrics.streaming_phase_timing.values;
      Object.assign(phases.request_view.reasoning_duration,{mean:0,count:1});
      Object.assign(phases.request_view.content_duration,{mean:8,count:2});
      Object.assign(phases.request_view.first_content_latency,{mean:10,count:2});
      phases.phase_counts.truncated_requests.value=0;
      phases.phase_counts.excluded_requests.value=0;
      tasks[0].cases.push({...different,id:'different_counts'});
      renderSummaries();
    });
    for(const label of sampleLabels)
      assert.equal((await summaryRow(0,'empty'))[label],'0');
    for(const label of [...sampleLabels,...countLabels])
      assert.equal((await summaryRow(0,'overall_only'))[label],'未采集');
    assert.equal((await summaryRow(0,'empty'))['思考每Token耗时（TPOT） 权重总数（Token）'],'0');
    const different=await summaryRow(0,'different_counts');
    assert.deepEqual(sampleLabels.map(label=>different[label]),['1','2','2']);
    assert.equal(different['思考输出时长 均值（ms）'],'0');
    for(const label of countLabels) assert.equal(different[label],'0');
    // The additional columns preserve sorting and Case navigation.
    const summary=page.locator('#case-summaries .summary-table').first();
    await page.locator('#report-nav a[href="#summary-section"]').click();
    await summary.locator('th').filter({hasText:sampleLabels[0]})
      .locator('button').click();
    const populated=await summary.locator('tbody tr').evaluateAll(rows=>
      rows.map(row=>row.cells[0].textContent).filter(id=>
        ['case_1','empty','different_counts'].includes(id)));
    assert.deepEqual(populated,['empty','different_counts','case_1']);
    await summary.locator('tbody tr').filter({hasText:'different_counts'})
      .getByRole('link',{name:'分析',exact:true}).click();
    await page.waitForFunction(()=>document.getElementById('analysis-case').value==='3');
    assert.equal(await page.locator('#analysis-case').inputValue(),'3');
    await page.locator('#analysis-task').selectOption('0');
    await page.locator('#analysis-case').selectOption('1');
    await metric.selectOption({label:'思考输出时长'});
    assert.match(await page.locator('#analysis-content').innerText(),/不可用/);
    await metric.selectOption({label:'思考每Token耗时（TPOT）'});
    assert.match(await page.locator('#analysis-content').innerText(),/不可用/);
    assert.match(await page.locator('#analysis-content').innerText(),/权重总和：0 Token/);
    await page.locator('#analysis-case').selectOption('2');
    const overallOnlyText=await page.locator('#analysis-content').innerText();
    assert.ok(!overallOnlyText.includes('分阶段统计口径与样本数量'));
    assert.ok(!overallOnlyText.includes('未保存分阶段指标'));
    assert.ok(!(await metric.locator('option').allTextContents())
      .includes('思考首Token延迟（TTFT）'));
    await page.locator('#analysis-case').selectOption('0');
    await metric.selectOption({label:'思考Token间延迟（ITL）'});
    fs.mkdirSync(process.argv[3],{recursive:true});
    await page.screenshot({path:path.join(process.argv[3],'phases-desktop.png'),fullPage:true});
    await page.setViewportSize({width:390,height:844});
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
    await page.screenshot({path:path.join(process.argv[3],'phases-mobile.png'),fullPage:true});
    await page.locator('#nav-toggle').click();
    await page.locator('#report-nav a[href="#summary-section"]').click();
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
    await page.screenshot({path:path.join(process.argv[3],'phase-summary-mobile.png'),fullPage:true});
    await page.setViewportSize({width:1440,height:1100});
    await page.screenshot({path:path.join(process.argv[3],'phase-summary-desktop.png'),fullPage:true});

    assert.deepEqual(errors,[]);
    assert.deepEqual(network,[]);
    console.log('PASS: phase selection, summary populations, zero counts, sorting, multi-turn, empty and overall-only metrics, offline rendering');
  } finally {
    await browser.close();
  }
})().catch(error=>{console.error(error);process.exit(1);});
