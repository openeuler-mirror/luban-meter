// Offline overview regression over saved results, plus explicit synthetic cases.
// node tests/report_overview.browser.cjs REAL_HTML OUTPUT_DIR
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {pathToFileURL}=require('node:url');
const {chromium}=require('playwright');

(async()=>{
  const browser=await chromium.launch({headless:true,
    channel:process.env.LUBAN_BROWSER_CHANNEL||'msedge'});
  try {
    const page=await browser.newPage({viewport:{width:1440,height:1000}});
    const errors=[],network=[];
    page.on('pageerror',e=>errors.push(e.message));
    page.on('dialog',()=>{throw Error('Untrusted content executed');});
    await page.route(/^https?:/,route=>{network.push(route.request().url());return route.abort();});
    const template=fs.readFileSync(process.argv[2],'utf8'),output=process.argv[3];
    if(output)fs.mkdirSync(output,{recursive:true});
    const shot=async name=>{
      await page.evaluate(()=>scrollTo(0,0));
      if(output)await page.screenshot({path:path.join(output,name+'.png')});
    };
    const navigate=async id=>{
      if(await page.locator('#nav-toggle').isVisible() &&
        !await page.locator('#report-nav').isVisible())await page.locator('#nav-toggle').click();
      await page.locator('#report-nav a[href="#'+id+'"]').click();
      await page.waitForFunction(id=>!document.getElementById(id).hidden,id);
      assert.equal(await page.locator('.report-page:visible').count(),1);
    };
    const card=id=>page.locator('#cards [data-metric="'+id+'"]');
    await page.goto(pathToFileURL(path.resolve(process.argv[2])).href);
    const data=JSON.parse(await page.locator('#report-data').textContent());
    assert.equal(await page.locator('body.dark').count(),1);
    assert.deepEqual(await page.locator('#report-nav a').allTextContents(),[
      '报告总览','结果摘要','Case 分析','全部指标','资源观测','运行诊断']);
    const alignment=await page.evaluate(()=>({
      nav:document.querySelector('.sidebar').getBoundingClientRect(),
      overview:document.querySelector('#overview').getBoundingClientRect(),
      header:document.querySelector('.report-header').getBoundingClientRect()}));
    assert.ok(Math.abs(alignment.nav.top-alignment.overview.top)<1);
    assert.ok(Math.abs(alignment.header.left-alignment.nav.left)<1);
    assert.ok(alignment.header.bottom<alignment.nav.top);
    assert.equal(await page.locator('#overview #performance-overview').count(),1);
    assert.equal(await page.locator('#overview #quality-overview').count(),1);
    assert.equal(await page.locator('#overview-models .overview-model').count(),1);
    assert.deepEqual(await page.locator('#overview-models dd').allTextContents(),[
      data.result.tasks[0].cases[0].parameters.model,
      data.result.tasks[0].cases[0].parameters.base_url]);
    assert.match(await page.locator('#run-counts').textContent(),/Case数 13.*成功Case数 13/);
    assert.equal(await page.locator('#overview-performance-case option').count(),13);
    const cases=data.result.tasks[0].cases;
    const specs=[
      ['peak-throughput','service_view','output_token_throughput','value','tok/s',false],
      ['best-ttft-p99','request_view','ttft','p99','ms',true],
      ['best-tpot-p99','request_view','tpot','p99','ms/token',true],
      ['best-e2e-p99','request_view','e2el','p99','ms',true],
    ];
    async function verify(records) {
      for(const [id,view,key,stat,unit,minimum] of specs) {
        const ranked=[...records].sort((a,b)=>(a.c.metrics.streaming_performance.values[view][key][stat]-
          b.c.metrics.streaming_performance.values[view][key][stat])*(minimum?1:-1));
        const best=ranked[0],value=best.c.metrics.streaming_performance.values[view][key][stat];
        assert.equal(await card(id).locator('.summary-card-value').getAttribute('title'),'原始值：'+value+' '+unit);
        assert.equal(await card(id).getAttribute('href'),'#summary-0-'+best.ci);
        assert.match(await card(id).textContent(),new RegExp(best.c.id));
      }
      assert.equal(await card('success-rate').locator('[role="progressbar"]').getAttribute('aria-valuenow'),'100');
      assert.equal(await page.locator('#cards .summary-ring').count(),1);
      assert.equal(await page.locator('#cards .summary-card:not([data-metric="success-rate"]) svg').count(),0);
    }
    await verify(cases.map((c,ci)=>({c,ci})));
    const hierarchy=await page.evaluate(()=>({
      bestTop:document.querySelector('.overview-result-grid').getBoundingClientRect().top,
      detailsTop:document.querySelector('#evaluation-summary').getBoundingClientRect().top,
      bestSize:parseFloat(getComputedStyle(document.querySelector('#cards .summary-card-value')).fontSize),
      overviewSize:parseFloat(getComputedStyle(document.querySelector('.peak-value')).fontSize)}));
    assert.ok(hierarchy.bestTop<hierarchy.detailsTop);
    assert.ok(hierarchy.bestSize>hierarchy.overviewSize);
    const qualityCard=page.locator('#overview-quality-cards .summary-card');
    assert.equal(await qualityCard.locator('.summary-card-value').textContent(),'24');
    assert.equal(await qualityCard.getAttribute('href'),'#summary-1-0');
    assert.match(await qualityCard.textContent(),/准确率.*case_1.*100/s);
    assert.equal(await qualityCard.locator('[role="progressbar"]').getAttribute('aria-valuenow'),'24');
    await shot('overview-default-dark');
    await page.locator('#theme').click();assert.equal(await page.locator('body.dark').count(),0);
    await shot('overview-light');await page.locator('#theme').click();

    // Every manual Case uses its own values, even when it is not a winner.
    for(const [ci,c] of cases.entries()) {
      await page.selectOption('#overview-performance-case','0:'+ci);
      await verify([{c,ci}]);
      assert.equal(await page.locator('#cards').getAttribute('data-view'),'case');
      assert.ok(await card('peak-throughput').locator('.summary-card-value').evaluate(n=>
        parseFloat(getComputedStyle(n).fontSize)<=28));
      assert.match(await card('peak-throughput').textContent(),/Throughput/);
      assert.doesNotMatch(await card('peak-throughput').textContent(),/Peak Throughput/);
    }
    await page.selectOption('#overview-performance-case','best');
    const bestTpot=cases.map((c,ci)=>({c,ci})).sort((a,b)=>
      a.c.metrics.streaming_performance.values.request_view.tpot.p99-
      b.c.metrics.streaming_performance.values.request_view.tpot.p99)[0];
    await card('best-tpot-p99').locator('.summary-card-value').click();
    await page.waitForFunction(id=>document.activeElement.id===id,'summary-0-'+bestTpot.ci);
    assert.equal(await page.locator('.report-page:visible').getAttribute('id'),'summary-section');
    await navigate('overview');await qualityCard.locator('.summary-card-value').click();
    await page.waitForFunction(()=>document.activeElement.id==='summary-1-0');
    await navigate('overview');
    await page.selectOption('#overview-quality-case','0');
    assert.equal(await qualityCard.locator('.summary-card-value').textContent(),'24');
    assert.equal(await page.locator('#overview-quality-cards').getAttribute('data-view'),'case');
    await page.selectOption('#overview-performance-case','0:0');
    await shot('overview-case-compact');
    await page.selectOption('#overview-performance-case','best');
    await page.selectOption('#overview-quality-case','best');

    // Detailed summaries stay within overview; changing a curve group does not
    // change the selected Case or manufacture an aggregate percentile.
    await page.selectOption('#overview-performance-case','0:1');
    await page.getByRole('tab',{name:'推理性能',exact:true}).click();
    assert.equal(await page.locator('#report-nav [aria-current]').getAttribute('href'),'#overview');
    assert.equal(await page.locator('#summary-performance-group option').count(),4);
    for(let i=0;i<4;i++) {
      await page.selectOption('#summary-performance-group',String(i));
      assert.equal(await page.inputValue('#overview-performance-case'),'0:1');
      assert.match(await page.locator('#overview-performance-content').innerText(),/持续上升/);
    }
    await page.getByRole('tab',{name:'推理性能',exact:true}).press('End');
    assert.equal(await page.locator('#quality-summary-panel').isVisible(),true);
    assert.equal(await page.locator('#performance-summary-panel').isVisible(),false);
    assert.equal(await page.locator('#quality-overview tbody tr').count(),53);
    await page.goBack();await page.waitForFunction(()=>!document.querySelector('#performance-summary-panel').hidden);
    await page.goForward();await page.waitForFunction(()=>!document.querySelector('#quality-summary-panel').hidden);
    await page.reload();
    assert.equal(await page.locator('#quality-summary-panel').isVisible(),true);
    assert.equal(await page.locator('body.dark').count(),1);
    assert.deepEqual(await page.evaluate(()=>D.result),data.result);
    await navigate('overview');await page.setViewportSize({width:390,height:844});
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
    const positions=await page.locator('#cards .summary-card').evaluateAll(nodes=>
      nodes.map(n=>({x:n.getBoundingClientRect().x,y:n.getBoundingClientRect().y})));
    assert.ok(positions.every(p=>p.x===positions[0].x));
    assert.ok(positions[1].y>positions[0].y);
    await shot('overview-mobile');
    await page.getByRole('tab',{name:'模型能力',exact:true}).click();
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));

    async function synthetic(result) {
      const fixture=template.replace(/(<script type="application\/json" id="report-data">)[\s\S]*?(<\/script>)/,
        (_,start,end)=>start+JSON.stringify({result,rows:[],observations:[]}).replace(/</g,'\\u003c')+end);
      await page.goto('about:blank');await page.setContent(fixture);
    }
    const scalar=(value,unit='ratio',sample_count=10)=>({value,unit,sample_count});
    const point=(id,concurrency,throughput,success,total)=>({id,status:success===total?'success':'partial_failed',
      collection_complete:true,parameters:{model:'synthetic',profile:'concurrent',max_concurrency:concurrency},
      metrics:{streaming_performance:{values:{request_view:{},service_view:{
        output_token_throughput:scalar(throughput,'token/s'),successful_requests:scalar(success,'request'),
        total_requests:scalar(total,'request'),failed_requests:scalar(total-success,'request')}}}}});
    const run=cases=>({kind:'run',id:'Synthetic only',status:'success',tasks:[{
      kind:'task',name:'synthetic',category:'generation_performance',status:'success',cases}]});
    const good=point('good',4,10,100,100),bad=point('bad',16,999,0,25);
    const modelRun=run([good,good,good,good]);
    modelRun.tasks[0].cases=modelRun.tasks[0].cases.map((c,i)=>({...c,parameters:{
      ...c.parameters,model:i===2?'model-b':'model-a',
      base_url:i===3?'http://127.0.0.1:8001':'http://127.0.0.1:8000'}}));
    await synthetic(modelRun);
    assert.equal(await page.locator('#overview-models .overview-model').count(),3);
    assert.deepEqual(await page.locator('#overview-models dd').allTextContents(),[
      'model-a','http://127.0.0.1:8000','model-b','http://127.0.0.1:8000',
      'model-a','http://127.0.0.1:8001']);
    await synthetic(run([good,bad]));
    assert.equal(await card('peak-throughput').locator('.summary-card-value').textContent(),'10');
    assert.equal(await card('success-rate').locator('.summary-card-value').textContent(),'100');
    assert.match(await card('success-rate').textContent(),/good.*100 \/ 100/s);
    assert.match(await page.locator('#run-counts').textContent(),/Case数 2.*成功Case数 1/);
    await page.selectOption('#overview-performance-case','0:1');
    assert.equal(await card('peak-throughput').locator('.summary-card-value').textContent(),'999');
    assert.equal(await card('success-rate').locator('.summary-card-value').textContent(),'0');
    assert.equal(await card('success-rate').locator('[role="progressbar"]').getAttribute('aria-valuenow'),'0');
    bad.metrics.streaming_performance.values.service_view.total_requests.value=null;
    await synthetic(run([bad]));
    assert.equal(await card('success-rate').locator('.summary-card-value').textContent(),'Unavailable');
    assert.equal(await card('success-rate').locator('.summary-ring').count(),0);

    const qc=(id,key,value,unit='ratio')=>({id,status:'success',collection_complete:true,parameters:{},
      metrics:{quality:{values:{task_view:{scores:{[key]:scalar(value,unit)}}}}}});
    const qt=(scenario,cases)=>({kind:'task',name:'ability',scenario,
      category:'model_service_quality',status:'success',cases});
    const q1=qc('low','accuracy',.2),q2=qc('high','accuracy',.8),q3=qc('failed','accuracy',.99);
    q3.status='failed';await synthetic(qt('cmmlu',[q1,q2,q3]));
    assert.equal(await qualityCard.locator('.summary-card-value').textContent(),'80');
    assert.equal(await qualityCard.getAttribute('href'),'#summary-0-1');
    await page.selectOption('#overview-quality-case','0');
    assert.equal(await qualityCard.locator('.summary-card-value').textContent(),'20');
    const p1=qc('p1','perplexity',10,'ppl'),p2=qc('p2','perplexity',3,'ppl');
    await synthetic(qt('wikitext',[p1,p2]));
    assert.equal(await qualityCard.locator('.summary-card-value').textContent(),'3');
    assert.match(await qualityCard.textContent(),/lower is better.*p2/s);
    assert.equal(await qualityCard.locator('svg').count(),0);
    const x1=qc('x1','custom_score',10,'points'),x2=qc('x2','custom_score',900,'points');
    await synthetic(qt('custom',[x1,x2]));
    assert.match(await page.locator('#overview-quality-best').innerText(),/未声明优化方向/);
    assert.equal(await qualityCard.locator('.summary-card-value').textContent(),'10');
    await page.selectOption('#overview-quality-case','1');
    assert.equal(await qualityCard.locator('.summary-card-value').textContent(),'900');
    q1.parameters.dataset_path='/a';q2.parameters.dataset_path='/b';
    await synthetic(qt('cmmlu',[q1,q2]));
    assert.equal(await page.locator('#overview-quality-group option').count(),2);
    assert.equal(await qualityCard.locator('.summary-card-value').textContent(),'20');
    await page.selectOption('#overview-quality-group','1');
    assert.equal(await qualityCard.locator('.summary-card-value').textContent(),'80');

    // Dataset tasks share one category even when task names/order differ.
    await synthetic({kind:'run',id:'Synthetic category grouping',status:'success',tasks:[
      {...qt('ceval',[qc('case_1','accuracy',.8)]),name:'ceval'},
      {...run([good]).tasks[0],name:'performance',scenario:'online_serving'},
      {...qt('cmmlu',[qc('case_1','accuracy',.805)]),name:'cmmlu'},
    ]});
    await navigate('summary-section');
    assert.deepEqual(await page.locator('#case-summaries .summary-category>h3').allTextContents(),
      ['模型能力评测','推理性能评测']);
    assert.deepEqual(await page.locator('#summary-category-model_service_quality h4').allTextContents(),
      ['C-Eval','CMMLU']);
    assert.equal(await page.locator('#summary-category-model_service_quality .task-quality').count(),1);
    assert.deepEqual(await page.locator('#details>.diagnostic-category>summary').allTextContents(),
      ['模型能力评测','推理性能评测']);
    assert.deepEqual(await page.locator('#diagnostic-category-model_service_quality>details>summary').allTextContents(),
      ['C-Eval · 成功','CMMLU · 成功']);
    await page.locator('#summary-2-0').getByRole('link',{name:'分析',exact:true}).click();
    await page.waitForFunction(()=>document.activeElement.id==='case-analysis');
    assert.equal(await page.inputValue('#analysis-task'),'2');
    await page.locator('#analysis-context').getByRole('link',{name:'测试条件、覆盖率与诊断'}).click();
    await page.waitForFunction(()=>document.activeElement.id==='case-2-0');
    assert.equal(await page.locator('#diagnostic-category-model_service_quality').evaluate(n=>n.open),true);
    assert.equal(await page.locator('#task-2').evaluate(n=>n.open),true);

    // Generic card rendering: proportions only, no guessed progress or clamping.
    for(const value of [0,.5,.9999999,1,-.1,1.1,null,NaN,Infinity]) {
      await page.evaluate(value=>{
        document.querySelector('#cards').replaceChildren(summaryCard({id:'generic',
          title:'Test proportion',value,unit:'%',kind:'percentage',tone:'green',
          description:'Synthetic only',source:'No model calls',href:'#overview'}));
      },value);
      const valid=typeof value==='number' && Number.isFinite(value) && value>=0 && value<=1;
      assert.equal(await page.locator('#cards .summary-ring').count(),valid?1:0);
      if(valid) {
        assert.equal(await page.locator('#cards [role="progressbar"]').getAttribute('aria-valuenow'),String(value*100));
        assert.equal(await page.locator('#cards .summary-ring-fill').count(),value>0?1:0);
        if(value===.9999999)assert.equal(await page.locator('#cards .summary-card-value').textContent(),'<100');
      } else assert.equal(await page.locator('#cards .summary-card-value').textContent(),'Unavailable');
    }
    await page.evaluate(()=>{
      document.querySelector('#cards').replaceChildren(summaryCard({id:'generic',
        title:'<script>alert(1)</script>',value:1e100,unit:'req/s',tone:'cyan',
        description:'Numeric request rate, not a proportion',source:'Synthetic',href:'#overview'}));
    });
    assert.equal(await page.locator('#cards svg').count(),0);
    assert.equal(await page.locator('#cards .summary-card-value').textContent(),'1.00e+100');
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
    await synthetic({kind:'run',status:'failed',tasks:[]});
    assert.equal(await page.locator('.report-page:visible').count(),1);
    assert.deepEqual(await page.locator('#overview-models dd').allTextContents(),['未采集','未采集']);
    assert.match(await page.locator('#overview-quality-best').innerText(),/没有可展示/);
    assert.deepEqual(errors,[]);assert.deepEqual(network,[]);
    console.log('PASS: merged overview, Case/success counts, independent best metric sources; '+
      'all 12 manual performance Cases, model best scores, summary-row jumps; '+
      'PPL lower-is-better, unknown directions, comparable datasets, no aggregate score; '+
      'default dark, left header/aligned nav, tabs/history/reload, mobile, offline, raw data preserved');
  } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
