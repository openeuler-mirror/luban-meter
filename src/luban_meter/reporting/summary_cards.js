// Reusable summary cards; only actual proportions have SVG progress rings.
function summaryCard(config) {
  const node=(tag,text,className)=>{
    const result=document.createElement(tag);
    if(text!==undefined)result.textContent=text;
    if(className)result.className=className;
    return result;
  };
  const svgNode=(tag,attributes)=>{
    const result=document.createElementNS('http://www.w3.org/2000/svg',tag);
    for(const [key,value] of Object.entries(attributes))result.setAttribute(key,value);
    return result;
  };
  const percentage=config.kind==='percentage';
  const available=typeof config.value==='number' && Number.isFinite(config.value) &&
    (!percentage || (config.value>=0 && config.value<=1));
  const value=available?(percentage?config.value*100:config.value):null;
  let formatted=value==null?'Unavailable':value!==0 &&
    (Math.abs(value)<0.001 || Math.abs(value)>=1e7)?
    value.toExponential(2):value.toLocaleString('en-US',
      {maximumFractionDigits:percentage?2:3});
  if(available && percentage && config.value<1 && formatted==='100')formatted='<100';
  const card=node('a',undefined,'summary-card');
  card.href=config.href;
  card.dataset.metric=config.id;
  card.dataset.tone=['cyan','green','purple','yellow'].includes(config.tone)?config.tone:'cyan';
  card.dataset.available=String(available);
  if(available&&percentage) {
    const ring=node('div',undefined,'summary-ring');
    ring.dataset.mode='progress';
    ring.setAttribute('role','progressbar');
    ring.setAttribute('aria-label',config.title);
    ring.setAttribute('aria-valuemin','0');ring.setAttribute('aria-valuemax','100');
    ring.setAttribute('aria-valuenow',String(value));
    const svg=svgNode('svg',{viewBox:'0 0 80 80','aria-hidden':'true'});
    const circle={cx:40,cy:40,r:32,fill:'none','stroke-width':5,pathLength:100};
    svg.append(svgNode('circle',{...circle,stroke:'var(--ring-track)'}));
    if(config.value>0)svg.append(svgNode('circle',{
      ...circle,stroke:'var(--card-accent)','stroke-linecap':'round',
      'stroke-dasharray':(config.value*100)+' 100',
      transform:'rotate(-90 40 40)',class:'summary-ring-fill'}));
    const glyph=svgNode('text',{x:40,y:46,'text-anchor':'middle',
      fill:'var(--card-accent)','font-size':19,'font-family':'system-ui,sans-serif'});
    glyph.textContent='%';
    svg.append(glyph);ring.append(svg);card.append(ring);
  }
  const body=node('div',undefined,'summary-card-body');
  const main=node('div',undefined,'summary-card-main');
  const number=node('span',formatted,'summary-card-value');
  if(available)number.title=percentage?'原始比例：'+String(config.value):
    '原始值：'+String(config.value)+' '+config.unit;
  main.append(number,node('span',config.unit,'summary-card-unit'));
  body.append(node('div',config.title,'summary-card-title'),main,
    node('div',config.description,'summary-card-description'),
    node('div',config.source,'summary-card-source'));
  card.append(body);
  return card;
}
function renderSummaryCards(parent,configs) {
  parent.replaceChildren(...configs.map(summaryCard));
}
