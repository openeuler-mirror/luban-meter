// Browser regression for the offline Case report. No service/model is called.
// NODE_PATH must include Playwright; LUBAN_BROWSER_CHANNEL defaults to msedge.
// node tests/report_analysis.browser.cjs REAL_HTML FIXTURE_HTML OUTPUT_DIR
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require('playwright');

(async () => {
  const browser = await chromium.launch({headless:true,
    channel:process.env.LUBAN_BROWSER_CHANNEL||'msedge'});
  try {
    const page = await browser.newPage({viewport:{width:1440,height:1000}});
    const errors=[], network=[];
    page.on('pageerror',e=>errors.push(e.message));
    page.on('dialog',()=>{throw Error('Untrusted content executed');});
    await page.route('**/*',route=>{network.push(route.request().url());return route.abort();});
    const open = async file => {
      await page.goto('about:blank');
      await page.setContent(fs.readFileSync(file,'utf8'));
    };
    await open(process.argv[2]);
    const real = JSON.parse(await page.locator('#report-data').textContent()).result;
    const analysis=page.locator('#case-analysis');
    const navigate=async id=>{
      await page.locator('#report-nav a[href="#'+id+'"]').click();
      await page.waitForFunction(id=>!document.getElementById(id).hidden,id);
    };
    assert.deepEqual(await page.locator('#case-summaries .summary-category>h3').allTextContents(),
      ['推理性能评测','模型能力评测']);
    assert.equal(await page.locator('#summary-category-generation_performance h4').textContent(),'在线服务');
    assert.match(await page.locator('#analysis-task').innerText(),/模型能力评测 · CMMLU/);
    assert.match(await page.locator('#rows').innerText(),/推理性能评测/);
    assert.equal(real.tasks[0].name,'performance');
    assert.equal(await page.locator('#report-nav a').count(),6);
    assert.equal(await page.locator('.sidebar').evaluate(n=>getComputedStyle(n).position),'sticky');
    assert.equal(await page.locator('.summary-category').nth(1).evaluate(n=>getComputedStyle(n).borderTopStyle),'dashed');
    assert.deepEqual(await page.locator('#details>.diagnostic-category>summary').allTextContents(),
      ['推理性能评测','模型能力评测']);
    assert.equal(await page.locator('#task-1').evaluate(n=>n.parentElement.id),
      'diagnostic-category-model_service_quality');
    for(const id of ['overview','summary-section','case-analysis','task-metrics','observation-section','diagnostics']) {
      await navigate(id);
      await page.waitForFunction(id=>document.querySelector('#report-nav [aria-current]')?.hash==='#'+id,id,{timeout:5000}).catch(async error=>{
        throw Error(id+' navigation: '+JSON.stringify(await page.evaluate(()=>({active:document.querySelector('#report-nav [aria-current]')?.hash,y:scrollY,sections:[...document.querySelectorAll('main>section')].map(n=>[n.id,n.getBoundingClientRect().top])})))+' '+error.message);
      });
      await page.waitForFunction(id=>{
        const n=document.getElementById(id),target=Math.max(0,Math.min(scrollY+n.getBoundingClientRect().top-parseFloat(getComputedStyle(n).scrollMarginTop),document.documentElement.scrollHeight-innerHeight));
        return Math.abs(scrollY-target)<2;
      },id);
      assert.ok(await page.locator('#'+id).evaluate(n=>n.getBoundingClientRect().top>=0));
      assert.equal(await page.locator('.report-page:visible').count(),1);
    }
    await navigate('summary-section');
    if(process.argv[4]) {
      fs.mkdirSync(process.argv[4],{recursive:true});
      await page.screenshot({path:path.join(process.argv[4],'navigation-desktop.png')});
    }
    assert.equal(await page.locator('#cards a').count(),6);
    assert.equal(await page.locator('#cards .summary-card-value').count(),6);
    assert.match(await page.locator('#run-counts').textContent(),/任务数 2.*Case数 13/);
    assert.equal(await page.locator('#case-summaries .summary-task').count(),2);
    assert.equal(await page.locator('#case-summaries tbody tr').count(),13);
    assert.equal(await page.locator('#analysis-task option').count(),2);
    assert.equal(await page.locator('#analysis-case option').count(),12);
    await navigate('case-analysis');
    const boxes=await page.locator('.analysis-toolbar select').evaluateAll(nodes=>
      nodes.map(n=>({y:n.getBoundingClientRect().y,h:n.getBoundingClientRect().height})));
    assert.ok(boxes.every(b=>b.y===boxes[0].y && b.h===boxes[0].h));
    // Every saved statistic (including p999 and zero) must be present unchanged.
    for(let ci=0;ci<12;ci++) {
      await page.selectOption('#analysis-case',String(ci));
      assert.deepEqual(await page.locator('#analysis-metric option').evaluateAll(nodes=>nodes.map(n=>n.value)),[
        ...['ttft','tpot','itl','e2el'].map(k=>'/streaming_performance/values/request_view/'+k),
        ...['request_throughput','output_token_throughput'].map(k=>'/streaming_performance/values/service_view/'+k)]);
      for(const k of ['request_throughput','output_token_throughput']) {
        await page.selectOption('#analysis-metric','/streaming_performance/values/service_view/'+k);
        const cell=page.locator('#analysis-content .stat-table tbody td').nth(1);
        assert.equal(await cell.getAttribute('title'),'原始值：'+
          real.tasks[0].cases[ci].metrics.streaming_performance.values.service_view[k].value+' '+
          real.tasks[0].cases[ci].metrics.streaming_performance.values.service_view[k].unit);
      }
      for(const name of ['ttft','tpot','itl','e2el']) {
        const key='/streaming_performance/values/request_view/'+name;
        await page.selectOption('#analysis-metric',key);
        const expected=real.tasks[0].cases[ci].metrics.streaming_performance.values.request_view[name];
        const rows=await page.locator('#analysis-content .stat-table tbody tr').evaluateAll(nodes=>
          nodes.map(n=>({name:n.cells[0].textContent,title:n.cells[1].title})));
        const stats=Object.entries(expected).filter(([k,v])=>
          !['count','weight_sum','unit'].includes(k) && (typeof v==='number'||v===null));
        assert.equal(rows.length,stats.length);
        for(const [k,v] of stats) {
          const row=rows.find(r=>r.name.toLowerCase().startsWith(k==='stddev'?'std':k==='p999'?'p99.9':k));
          assert.ok(row,`missing ${ci}/${name}/${k}`);
          assert.ok(row.title.includes(String(v)),`wrong ${ci}/${name}/${k}`);
        }
      }
    }
    await page.selectOption('#analysis-metric','/streaming_performance/values/service_view/output_token_throughput');
    assert.equal(await page.locator('#analysis-content .stat-table tbody tr').count(),1);
    assert.equal(await page.locator('#analysis-content svg').count(),0);
    assert.equal(await page.locator('#analysis-metric option').count(),6);
    assert.match(await page.locator('#rows').innerText(),/offered_request_rate/);
    // Summary link must select the correct task as well as case_1.
    await navigate('summary-section');
    await page.locator('#summary-1-0').getByRole('link',{name:'分析',exact:true}).click();
    await page.waitForFunction(()=>document.activeElement.id==='case-analysis');
    assert.equal(await page.inputValue('#analysis-task'),'1');
    assert.equal(await page.inputValue('#analysis-case'),'0');
    assert.equal(await page.locator('#analysis-case option').count(),1);
    assert.equal(await page.locator('#analysis-metric-field').isVisible(),false);
    const quality=real.tasks[1].cases[0].metrics.choice_accuracy.values.task_view.cmmlu;
    const qualityText=await page.locator('#analysis-content').innerText();
    assert.match(qualityText,/24/);
    assert.match(qualityText,/答案解析失败/);
    assert.match(qualityText,/参与评分样本/);
    assert.equal(await page.locator('#analysis-content svg').count(),0);
    assert.equal(await page.locator('#analysis-content').getByText('查看分类指标图',{exact:true}).count(),0);
    assert.doesNotMatch(qualityText,/accuracy_by_subject|agronomy/);
    const embedded=JSON.parse(await page.locator('#report-data').textContent());
    assert.deepEqual(embedded.result.tasks[1].cases[0].metrics.choice_accuracy.values.task_view.cmmlu.accuracy_by_subject,
      quality.accuracy_by_subject);
    assert.equal(embedded.rows.filter(r=>r.metric.includes('/cmmlu/accuracy_by_subject/')).length,
      Object.keys(quality.accuracy_by_subject).length);
    const qualityBoxes=await page.locator('.analysis-toolbar select:visible').evaluateAll(nodes=>nodes.map(n=>n.getBoundingClientRect().y));
    assert.equal(qualityBoxes.length,2);assert.equal(qualityBoxes[0],qualityBoxes[1]);
    // Independent legacy search must not alter Case analysis.
    await navigate('task-metrics');
    await page.fill('#search','nonexistent');
    assert.equal(await page.inputValue('#analysis-task'),'1');
    assert.equal(await page.locator('#analysis-content tr').count()>0,true);
    await navigate('case-analysis');
    if(process.argv[4]) {
      fs.mkdirSync(process.argv[4],{recursive:true});
      await analysis.screenshot({path:path.join(process.argv[4],'quality.png')});
    }
    await navigate('summary-section');
    await page.locator('#summary-0-1').getByRole('link',{name:'分析',exact:true}).click();
    await page.waitForFunction(()=>document.querySelector('#analysis-task').value==='0');
    assert.equal(await page.inputValue('#analysis-case'),'1');
    assert.equal(await page.locator('#analysis-metric-field').isVisible(),true);
    await page.selectOption('#analysis-case','0');
    await page.selectOption('#analysis-metric','/streaming_performance/values/request_view/ttft');
    if(process.argv[4]) {
      await analysis.screenshot({path:path.join(process.argv[4],'performance.png')});
      await navigate('summary-section');
      await page.locator('#case-summaries .table-wrap').evaluateAll(nodes=>nodes.forEach(n=>{n.scrollLeft=0;}));
      await page.locator('#summary-section').screenshot({path:path.join(process.argv[4],'summary.png')});
      await page.locator('#theme').click();
      await page.locator('#case-summaries .table-wrap').evaluateAll(nodes=>nodes.forEach(n=>{n.scrollLeft=0;}));
      await page.locator('#summary-section').screenshot({path:path.join(process.argv[4],'summary-dark.png')});
    }
    await page.setViewportSize({width:390,height:844});
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth));
    assert.equal(await page.locator('#report-nav').isVisible(),false);
    await page.locator('#nav-toggle').click();
    assert.equal(await page.locator('#report-nav').isVisible(),true);
    await page.locator('#report-nav a[href="#case-analysis"]').click();
    assert.equal(await page.locator('#report-nav').isVisible(),false);
    await page.waitForFunction(()=>document.querySelector('#report-nav [aria-current]')?.hash==='#case-analysis');
    assert.ok(await page.evaluate(()=>document.querySelector('#case-analysis').getBoundingClientRect().top>=
      document.querySelector('.sidebar').getBoundingClientRect().bottom));
    const narrow=await page.locator('.analysis-toolbar select:visible').evaluateAll(nodes=>nodes.map(n=>n.getBoundingClientRect().y));
    assert.ok(narrow[0]<narrow[1]&&narrow[1]<narrow[2]);
    if(process.argv[4])await analysis.screenshot({path:path.join(process.argv[4],'mobile.png')});
    // Sorting keeps links tied to original case identity.
    await page.setViewportSize({width:1440,height:1000});
    await navigate('summary-section');
    const firstTable=page.locator('#case-summaries table').first();
    await firstTable.getByRole('button',{name:/TTFT.*Mean/}).click();
    const expectedFirst=real.tasks[0].cases.map((c,i)=>({i,value:c.metrics.streaming_performance.values.request_view.ttft.mean}))
      .sort((a,b)=>a.value-b.value)[0].i;
    assert.equal(await firstTable.locator('tbody tr').first().getAttribute('id'),'summary-0-'+expectedFirst);

    // Synthetic fixtures cover structures/states absent from the real report.
    console.log('Real report checks passed');
    await open(process.argv[3]);
    await page.setViewportSize({width:1440,height:1000});
    await navigate('case-analysis');
    assert.equal(await page.locator('#analysis-metric option').count(),1);
    assert.match(await page.locator('#analysis-content').innerText(),/未采集/);
    assert.match(await page.locator('#analysis-content').innerText(),/P99.9/);
    assert.match(await page.locator('#analysis-task').innerText(),/duplicate/);
    await page.selectOption('#analysis-case','3');
    assert.deepEqual(await page.locator('#analysis-metric option').evaluateAll(nodes=>nodes.map(n=>n.value)),[
      '/streaming_performance/values/request_view/tpot',
      '/streaming_performance/values/service_view/request_throughput']);
    assert.equal(await page.inputValue('#analysis-metric'),'/streaming_performance/values/request_view/tpot');
    await page.selectOption('#analysis-case','4');
    assert.equal(await page.locator('#analysis-metric').isDisabled(),true);
    assert.equal(await page.locator('#analysis-metric option').count(),0);
    await page.selectOption('#analysis-case','0');
    assert.equal(await page.inputValue('#analysis-metric'),'/streaming_performance/values/request_view/ttft');
    const separators=await page.locator('.summary-category').evaluateAll(nodes=>nodes.map(n=>
      [...n.querySelectorAll('.summary-task')].map(task=>getComputedStyle(task).borderTopStyle)));
    for(const group of separators){assert.equal(group[0],'none');assert.ok(group.slice(1).every(s=>s==='dashed'));}
    const names=['GSM8K','HumanEval','WikiText','C-Eval','Custom'];
    for(let i=1;i<=names.length;i++) {
      await page.selectOption('#analysis-task',String(i));
      assert.equal(await page.locator('#analysis-metric-field').isVisible(),false);
      assert.equal(await page.locator('#analysis-case option').count(),1);
      const text=await page.locator('#analysis-content').innerText();
      if(i===1){assert.match(text,/Exact Match/);assert.doesNotMatch(text,/Accuracy/);}
      if(i===2){assert.match(text,/Pass@1/);assert.match(text,/Pass@5/);}
      if(i===3){assert.match(text,/PPL/);assert.match(text,/1,234/);}
      if(i===4){assert.match(text,/真实零/);assert.match(text,/未知学科/);assert.match(text,/未采集/);}
      if(i===5){
        assert.match(text,/custom_score/);assert.match(text,/P99.9/);
        const maxRow=page.locator('#analysis-content tr').filter({has:page.getByRole('cell',{name:'max',exact:true})});
        assert.equal(await maxRow.locator('td').nth(2).textContent(),'未采集');
        const minRow=page.locator('#analysis-content tr').filter({has:page.getByRole('cell',{name:'min',exact:true})});
        assert.equal(await minRow.locator('td').nth(2).textContent(),'0');
      }
    }
    await page.selectOption('#analysis-task','6');
    assert.equal(await page.locator('#analysis-case').isDisabled(),true);
    assert.match(await page.locator('#analysis-content').innerText(),/没有 Case/);
    await page.selectOption('#analysis-task','0');
    await page.selectOption('#analysis-case','1');
    assert.match(await page.locator('#analysis-content').innerText(),/没有可展示/);
    await page.selectOption('#analysis-case','2');
    assert.match(await page.locator('#analysis-content').innerText(),/已丢弃/);
    assert.equal(await page.locator('#analysis-metric option').count(),0);
    await navigate('summary-section');
    await page.locator('#summary-1-0').getByRole('link',{name:'分析',exact:true}).click();
    await page.waitForFunction(()=>document.querySelector('#analysis-task').value==='1');
    assert.match(await page.locator('#analysis-content').innerText(),/Exact Match/);
    // An empty run is legal. Render the same template with no tasks.
    const empty=fs.readFileSync(process.argv[3],'utf8').replace(
      /(<script type="application\/json" id="report-data">)[\s\S]*?(<\/script>)/,
      '$1'+JSON.stringify({result:{kind:'run',tasks:[],status:'failed'},rows:[],observations:[]})+'$2');
    await page.goto('about:blank');await page.setContent(empty);
    assert.equal(await page.locator('#analysis-task').isDisabled(),true);
    assert.match(await page.locator('#case-summaries').innerText(),/没有任务/);
    assert.deepEqual(errors,[]);assert.deepEqual(network,[]);
    console.log('PASS: real 12 performance Cases + CMMLU; all timing statistics; '+
      'task/Case/metric selection, navigation, sorting, missing/zero, units, '+
      'quality dimensions, unknown metrics, failures, empty results, desktop/mobile, offline');
  } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
