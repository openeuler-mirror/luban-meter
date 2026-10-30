// Offline resource views consume saved observations; no exporter is contacted.
function initResources() {
  const picker = $('resource-case'), gpuPicker = $('resource-gpu');
  const entries = [], samples = new Map();
  const colors = Array.from({length:8},(_,i)=>'var(--resource-color-'+i+')');
  const metrics = [
    {key:'utilization',title:'GPU利用率',unit:'%',avg:'utilization_avg'},
    {key:'memory_used',title:'已用显存',unit:'MiB',avg:'memory_used_avg_mb'},
    {key:'power',title:'GPU功耗',unit:'W',avg:'power_avg_watts'},
    {key:'temperature',title:'GPU温度',unit:'°C',avg:'temperature_avg_celsius'}
  ];
  const valid = (value, key) => numeric(value) &&
    (key === 'utilization' ? value >= 0 && value <= 100 :
      key === 'temperature' ? value >= -273.15 : value >= 0);
  const number = value => numeric(value) ?
    value.toLocaleString('zh-CN',{maximumFractionDigits:2}).replace(/^-0$/,'0') : '不可用';
  const time = value => numeric(value) ? new Date(value*1000).toLocaleString(
    'zh-CN',{hour12:false}) : '时间未保存';
  const clock = value => new Date(value*1000).toLocaleTimeString(
    'zh-CN',{hour12:false});
  const keyFor = (ti,ci) => ti+':'+ci;
  for (const sample of D.observations || []) {
    const key = keyFor(sample.task_index,sample.case_index);
    if (!samples.has(key)) samples.set(key,[]);
    samples.get(key).push(sample);
  }
  tasks.forEach((task,ti) => {
    const group = el('optgroup');
    group.label = [taskDisplayName(task.name),scenarioDisplayName(task.scenario)]
      .filter(Boolean).join(' · ');
    (task.cases || []).forEach((c,ci) => {
      const key = keyFor(ti,ci), points = samples.get(key) || [];
      entries.push({task,c,ti,ci,key,points});
      option(group,key,c.id+' · '+(names[c.status] || c.status)+
        (points.length ? '' : ' · 无采样序列'));
    });
    if (group.children.length) picker.append(group);
  });

  function groupSeries(points, label) {
    const grouped = new Map();
    for (const p of points) {
      if (!numeric(p.time)) continue;
      const labels = Object.entries(p.labels || {}).sort(([a],[b])=>a.localeCompare(b));
      const key = JSON.stringify([p.metric,labels]);
      if (!grouped.has(key)) grouped.set(key,{name:label(p),points:[]});
      grouped.get(key).points.push({time:p.time,value:p.value});
    }
    return [...grouped.values()].map((s,i)=>({...s,color:colors[i%colors.length],
      points:s.points.sort((a,b)=>a.time-b.time)}));
  }
  function domain(entry) {
    const values = entry.points.map(p=>p.time).filter(numeric);
    const start = entry.c.window?.started_at, end = entry.c.window?.ended_at;
    if (numeric(start)) values.push(start);
    if (numeric(end)) values.push(end);
    return values.length ? values.reduce(([lo,hi],v)=>
      [Math.min(lo,v),Math.max(hi,v)],[values[0],values[0]]) : null;
  }
  function panel(parent,title,unit,id) {
    const box = el('div',undefined,'resource-panel');
    if (id) box.id = id;
    box.append(el('h4',title),el('span',unit,'resource-unit'));
    parent.append(box);
    return box;
  }
  function chart(parent,title,unit,series,entry,key='',id='',source='device') {
    const box = panel(parent,title,unit,id), range = domain(entry);
    const usable = series.map(s=>({...s,points:s.points.map(p=>({...p,
      value:(key ? valid(p.value,key) : numeric(p.value)) ? p.value : null}))}));
    const values = usable.flatMap(s=>s.points.map(p=>p.value)).filter(numeric);
    if (!values.length || !range) {
      box.append(el('div','不可用 · 未保存有效采样','resource-unavailable'));
      return;
    }
    const width=640,height=250,left=70,right=20,top=26,bottom=48;
    const lo=Math.min(0,...values), hi=key==='utilization'?100:Math.max(1,...values);
    const x=t=>left+(range[1]===range[0] ? .5 :
      (t-range[0])/(range[1]-range[0]))*(width-left-right);
    const y=v=>height-bottom-(v-lo)/(hi-lo)*(height-top-bottom);
    const svg = svgElement('svg',{viewBox:`0 0 ${width} ${height}`,
      role:'img','aria-label':title+'（'+unit+'）',
      'data-start':range[0],'data-end':range[1]});
    const start=entry.c.window?.started_at, end=entry.c.window?.ended_at;
    if (numeric(start) && numeric(end) && end>=start) {
      svg.append(svgElement('rect',{x:x(start),y:top,width:x(end)-x(start),
        height:height-top-bottom,fill:'var(--accent)',opacity:'.045'}));
      for (const t of [start,end]) svg.append(svgElement('line',{
        x1:x(t),x2:x(t),y1:top,y2:height-bottom,stroke:'var(--accent)',
        'stroke-dasharray':'3 4',opacity:'.45'}));
    }
    function text(label,attrs) {
      const node=svgElement('text',{fill:'currentColor','font-size':12,...attrs});
      node.textContent=label;svg.append(node);
    }
    text(unit,{x:left,y:15});
    for (let i=0;i<=4;i++) {
      const value=lo+(hi-lo)*i/4,py=y(value);
      svg.append(svgElement('line',{x1:left,x2:width-right,y1:py,y2:py,
        stroke:'var(--line)'}));
      text(number(value),{x:left-10,y:py+4,'text-anchor':'end'});
    }
    const ticks=range[0]===range[1] ? [range[0]] :
      [range[0],(range[0]+range[1])/2,range[1]];
    for (const t of ticks) {
      text(clock(t),{x:x(t),y:height-25,'text-anchor':'middle'});
      if (numeric(start)) text((t>=start?'+':'')+number(t-start)+' s',
        {x:x(t),y:height-8,'text-anchor':'middle','font-size':10});
    }
    const legend=el('div',undefined,'resource-legend');
    for (const s of usable) {
      const steps=s.points.slice(1).map((p,i)=>p.time-s.points[i].time)
        .filter(n=>n>0).sort((a,b)=>a-b);
      const hardware=(source==='host' ? entry.c.host_observation :
        entry.c.device_observation) || {};
      const cadence=hardware.interval_seconds;
      // The export keeps at most 500 snapshots: do not treat downsampling
      // itself as a collection outage.
      const stride=Math.max(1,Math.ceil((hardware.sample_count || 0)/500));
      const gap=(key && numeric(cadence) && cadence>0 ? cadence*stride :
        steps[Math.floor((steps.length-1)/2)])*2.5;
      let segment=[];
      function flush() {
        if (segment.length>1) svg.append(svgElement('polyline',{
          points:segment.map(p=>`${x(p.time)},${y(p.value)}`).join(' '),
          fill:'none',stroke:s.color,'stroke-width':2,'pointer-events':'none',
          'vector-effect':'non-scaling-stroke'}));
        segment=[];
      }
      s.points.forEach((p,i)=>{
        if (!numeric(p.value)) {flush();return;}
        if (i && p.time-s.points[i-1].time>gap) flush();
        segment.push(p);
        const dot=svgElement('circle',{cx:x(p.time),cy:y(p.value),r:2.4,
          fill:s.color});
        const tooltip=svgElement('title',{});
        tooltip.textContent=s.name+' · '+time(p.time)+' · '+number(p.value)+' '+unit;
        dot.append(tooltip);svg.append(dot);
      });
      flush();
      const item=el('span',s.name);
      item.style.setProperty('--series-color',s.color);legend.append(item);
    }
    box.append(svg,legend);
  }

  function renderContext(entry) {
    const {c,task,ti,ci}=entry, box=$('resource-context');
    box.replaceChildren();
    const heading=el('div',undefined,'resource-context-heading');
    heading.append(el('strong',taskDisplayName(task.name)+' / '+c.id),
      link('查看Case结果','summary-'+ti+'-'+ci));
    box.append(heading);
    const grid=el('div',undefined,'resource-meta');
    const device=c.device_observation || {}, service=c.observation || {};
    const duration=c.window?.ended_at-c.window?.started_at;
    for (const [title,value] of [
      ['运行ID',R.id || '未保存'],['模型',c.parameters?.model || '未保存'],
      ['Case测试窗口',time(c.window?.started_at)+' → '+time(c.window?.ended_at)],
      ['测试时长',numeric(duration) && duration>=0 ? number(duration)+' s' : '不可用'],
      ['硬件采样 / 错误',number(device.sample_count)+' / '+number(device.error_count)],
      ['主机采样 / 错误',number(c.host_observation?.sample_count)+' / '+number(c.host_observation?.error_count)],
      ['服务采样成功 / 失败',number(service.valid_samples)+' / '+number(service.failed_samples)]
    ]) {
      const item=el('div');item.append(el('span',title),el('strong',value));grid.append(item);
    }
    box.append(grid);
    const runStart=R.started_at,runEnd=R.finished_at;
    if (numeric(runStart) && numeric(runEnd) && runEnd>runStart &&
        numeric(c.window?.started_at) && numeric(c.window?.ended_at)) {
      const track=el('div',undefined,'resource-run-track');
      const mark=el('span');
      const percent=t=>Math.max(0,Math.min(100,(t-runStart)/(runEnd-runStart)*100));
      const left=percent(c.window.started_at),right=percent(c.window.ended_at);
      mark.style.left=left+'%';mark.style.width=Math.max(0,right-left)+'%';
      mark.title=c.id+' · '+time(c.window.started_at)+' → '+time(c.window.ended_at);
      track.append(mark);
      box.append(el('p','当前Case在整次运行中的位置','resource-caption'),track,
        el('div',time(runStart)+' → '+time(runEnd),'resource-caption'));
    }
  }
  function devices(entry) {
    if (entry.c.status === 'discarded') return [];
    const result=new Map();
    for (const device of entry.c.device_observation?.devices || [])
      result.set(String(device.index),device);
    for (const p of entry.points) if (p.metric.startsWith('device/') &&
      p.labels?.device != null && !result.has(String(p.labels.device)))
      result.set(String(p.labels.device),{index:p.labels.device});
    return [...result.values()].sort((a,b)=>String(a.index).localeCompare(
      String(b.index),undefined,{numeric:true}));
  }
  function renderGpu(entry) {
    const box=$('resource-gpu-content'), all=devices(entry);
    const selected=all.filter(d=>!gpuPicker.value || String(d.index)===gpuPicker.value);
    box.replaceChildren();
    if (!selected.length) {
      box.append(el('div','不可用 · 此Case未保存GPU采样或汇总','resource-unavailable'));
      return;
    }
    const table=el('table',undefined,'resource-device-table'),thead=el('thead'),row=el('tr');
    for (const title of ['设备','平均利用率（%）','平均已用显存（MiB）',
      '显存总量（MiB）','平均功耗（W）','平均温度（°C）']) row.append(el('th',title));
    thead.append(row);table.append(thead);
    const body=el('tbody');
    for (const device of selected) {
      const tr=el('tr'),name=el('td');
      name.append(el('strong','GPU '+device.index),el('small',device.name || '型号未保存'));
      name.style.setProperty('--series-color',colors[all.indexOf(device)%colors.length]);
      tr.append(name);
      for (const [field,key] of [['utilization_avg','utilization'],
        ['memory_used_avg_mb','memory_used'],['memory_total_mb','memory_used'],
        ['power_avg_watts','power'],['temperature_avg_celsius','temperature']])
        tr.append(el('td',valid(device[field],key)?number(device[field]):'不可用','num'));
      body.append(tr);
    }
    table.append(body);
    const wrap=el('div',undefined,'table-wrap');wrap.append(table);box.append(wrap);
    const grid=el('div',undefined,'resource-chart-grid');
    for (const metric of metrics) {
      const series=selected.map(device=>({name:'GPU '+device.index,
        color:colors[all.indexOf(device)%colors.length],points:entry.points.filter(p=>
          p.metric==='device/'+metric.key && String(p.labels?.device)===String(device.index))
          .map(p=>({time:p.time,value:p.value})).filter(p=>numeric(p.time)).sort((a,b)=>a.time-b.time)}));
      chart(grid,metric.title,metric.unit,series,entry,metric.key,'resource-'+metric.key);
    }
    box.append(grid);
  }
  function renderHost(entry) {
    if (entry.c.host_observation && entry.c.status!=='discarded') {
      renderHostMetrics(entry);return;
    }
    const box=$('resource-host-content'),cpu=entry.c.status==='discarded' ? {} :
      entry.c.device_observation?.cpu || {};
    box.replaceChildren();
    for (const [key,title,unit,field] of [
      ['utilization','CPU利用率','%','utilization_avg'],
      ['memory_used','主机已用内存','MiB','memory_used_avg_mb']
    ]) {
      const points=entry.points.filter(p=>p.metric==='cpu/'+key);
      if (points.length) chart(box,title,unit,groupSeries(points,()=> '主机'),
        entry,key,'resource-host-'+key);
      else {
        const p=panel(box,title,unit,'resource-host-'+key);
        p.append(el('strong',valid(cpu[field],key)?number(cpu[field])+' '+unit:'不可用',
          'resource-host-value'),el('p',valid(cpu[field],key)?
          '仅保存汇总均值，未保存时间序列':'此Case未保存主机采样','resource-caption'));
      }
    }
    for (const title of ['磁盘','网络']) {
      const p=panel(box,title,'');
      p.append(el('strong','不可用','resource-host-value'),
        el('p','此Case未保存'+title+'指标','resource-caption'));
    }
  }
  function renderHostMetrics(entry) {
    const box=$('resource-host-content'),host=entry.c.host_observation;
    box.replaceChildren();
    const controls=el('div',undefined,'resource-host-controls');
    const meta=el('p','主机：'+(host.node?.nodename || '未保存')+' · '+
      (host.source || '数据源未保存')+' · '+(host.status || '状态未保存'),
      'resource-caption');
    controls.append(meta);
    const selectors=el('div',undefined,'analysis-toolbar');
    const pickers={};
    for(const [group,title] of [['filesystems','文件系统'],['disks','块设备'],
      ['networks','网络接口（含lo）']]) {
      const label=el('label',title),select=el('select');
      select.id='resource-host-'+group;select.setAttribute('aria-label',title);
      const identities=new Map();
      for(const p of entry.points.filter(p=>p.metric.startsWith('host/'+group+'/'))) {
        const key=JSON.stringify(p.labels);identities.set(key,p.labels);
      }
      for(const [key,labels] of [...identities].sort(([a],[b])=>a.localeCompare(b)))
        option(select,key,[labels.device,labels.mountpoint,labels.fstype].filter(Boolean).join(' · '));
      if(!identities.size){option(select,'','不可用');select.disabled=true;}
      // Prefer real storage; all exported mounts/interfaces remain selectable.
      if(group==='filesystems') {
        const data=[...identities].find(([,l])=>l.mountpoint==='/data') ||
          [...identities].find(([,l])=>l.mountpoint==='/');
        if(data)select.value=data[0];
      }
      if(group==='networks') {
        const data=[...identities].find(([,l])=>l.device==='lo');
        if(data && entry.c.parameters?.base_url?.includes('127.0.0.1'))select.value=data[0];
      }
      label.append(select);selectors.append(label);pickers[group]=select;
    }
    controls.append(selectors);box.append(controls);
    const plots=el('div',undefined,'resource-chart-grid resource-host-plots');box.append(plots);
    function draw() {
      plots.replaceChildren();
      function series(group,field,name,scale=1) {
        let points=entry.points.filter(p=>p.metric==='host/'+group+'/'+field);
        if(pickers[group] && pickers[group].value) {
          const labels=JSON.parse(pickers[group].value);
          points=points.filter(p=>Object.entries(labels).every(([k,v])=>p.labels[k]===v));
        }
        return {name,color:colors[0],points:points.map(p=>({time:p.time,
          value:numeric(p.value)?p.value*scale:null})).sort((a,b)=>a.time-b.time)};
      }
      function plot(title,unit,group,fields,id,key='',scale=1) {
        const rows=fields.map(([field,name],i)=>({...series(group,field,name,scale),
          color:colors[i%colors.length]}));
        chart(plots,title,unit,rows,entry,key,id,'host');
      }
      plot('CPU利用率（忙碌率含I/O等待）','%','cpu',[
        ['utilization','忙碌率'],['user','user'],['system','system'],['iowait','iowait']
      ],'resource-host-utilization','utilization');
      plot('主机内存','GiB','memory',[
        ['used_bytes','已用（总量−可用）'],['available_bytes','可用']
      ],'resource-host-memory','memory_used',1/1024**3);
      plot('文件系统空间使用率','%','filesystems',[
        ['utilization','已用（总量−空闲）/总量']
      ],'resource-host-filesystem-usage','utilization');
      plot('磁盘读写速度','MiB/s','disks',[
        ['read_bytes_per_second','读取'],['write_bytes_per_second','写入']
      ],'resource-host-disk-throughput','memory_used',1/1024**2);
      plot('磁盘IOPS','IOPS','disks',[
        ['reads_per_second','读IOPS'],['writes_per_second','写IOPS']
      ],'resource-host-disk-iops','memory_used');
      plot('磁盘I/O忙碌率','%','disks',[
        ['utilization','I/O忙碌率']
      ],'resource-host-disk-busy','utilization');
      plot('网络收发速度','MiB/s','networks',[
        ['receive_bytes_per_second','接收'],['transmit_bytes_per_second','发送']
      ],'resource-host-network-throughput','memory_used',1/1024**2);
      plot('网络错误与丢包速率','events/s','networks',[
        ['receive_errors_per_second','接收错误'],['transmit_errors_per_second','发送错误'],
        ['receive_dropped_per_second','接收丢包'],['transmit_dropped_per_second','发送丢包']
      ],'resource-host-network-errors','memory_used');
    }
    for(const picker of Object.values(pickers))picker.onchange=draw;
    draw();
  }
  function renderService(entry) {
    const box=$('resource-service-content');box.replaceChildren();
    const points=entry.points.filter(p=>!p.metric.startsWith('device/') &&
      !p.metric.startsWith('cpu/') && !p.metric.startsWith('host/'));
    const grid=el('div',undefined,'resource-chart-grid');
    for (const [name,title] of [['vllm:num_requests_running','服务端运行请求'],
      ['vllm:num_requests_waiting','服务端排队请求']]) {
      const rows=points.filter(p=>p.metric===name);
      if (rows.length) chart(grid,title,'请求',groupSeries(rows,p=>
        JSON.stringify(p.labels || {})),entry,'',name.endsWith('running')?
        'resource-running':'resource-waiting');
    }
    box.append(grid);
    const detail=el('details'),picker=el('select');
    picker.id='resource-series';picker.setAttribute('aria-label','此Case的服务观测序列');
    detail.append(el('summary','其他服务观测 · 保存的原始值'));
    const grouped=groupSeries(points,p=>p.metric+' · '+JSON.stringify(p.labels || {}));
    grouped.sort((a,b)=>a.name.localeCompare(b.name));
    grouped.forEach((s,i)=>option(picker,String(i),s.name));
    const output=el('div');detail.append(picker,output);
    function update() {
      output.replaceChildren();
      const s=grouped[Number(picker.value)];
      if (s) chart(output,'服务观测原始值','原始值（单位未保存）',[s],entry);
      else output.append(el('div','不可用 · 此Case未保存服务采样','resource-unavailable'));
    }
    picker.onchange=update;
    // Build this secondary chart only when requested, not every raw series.
    detail.addEventListener('toggle',()=>{if(detail.open) update();});
    box.append(detail);
  }
  function render() {
    const entry=entries.find(e=>e.key===picker.value);
    if (!entry) return;
    renderContext(entry);
    gpuPicker.replaceChildren();option(gpuPicker,'','全部GPU');
    for (const device of devices(entry)) option(gpuPicker,String(device.index),
      'GPU '+device.index+' · '+(device.name || '型号未保存'));
    gpuPicker.disabled=devices(entry).length===0;
    renderGpu(entry);renderHost(entry);renderService(entry);
  }
  if (!entries.length) {
    picker.disabled=true;
    $('resource-context').append(el('div','没有可展示的Case','resource-unavailable'));
  } else {
    // Start with a sampled Case; all unsampled Cases remain selectable.
    picker.value=(entries.find(e=>e.points.length) || entries[0]).key;
    render();
  }
  picker.onchange=render;
  gpuPicker.onchange=()=>{
    const entry=entries.find(e=>e.key===picker.value);if(entry) renderGpu(entry);
  };
}
