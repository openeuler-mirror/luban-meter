// Embedded in the offline HTML; never fetch raw files in the browser.
const requestViews = new Map();
const failedStates = new Set(['failed', 'service_failed', 'sandbox_failed',
  'errored', 'cancelled', 'incomplete', 'not_started']);
function requestFailed(row) {
  return failedStates.has(row.status) ||
    row.sample_states.some(state => failedStates.has(state));
}
function lazyJson(parent, title, value) {
  const details = el('details');
  details.append(el('summary', title));
  details.addEventListener('toggle', () => {
    if (details.open && details.children.length === 1)
      details.append(el('pre', value == null ? '未采集' :
        typeof value === 'string' ? value : JSON.stringify(value, null, 2)));
  });
  parent.append(details);
}
function requestPanel(parent, summary, id, data) {
  const section = el('details');
  section.id = id + '-requests';
  section.tabIndex = -1;
  section.append(el('summary', '请求与样本详情'));
  parent.append(section);
  const navigation = el('p');
  summary.append(navigation);
  if (data.status !== 'available') {
    section.append(el('p', '详情不可用：' + data.reason));
    navigation.append(link('请求详情不可用（查看原因）', section.id));
    return;
  }
  section.append(el('p', '已保存的正式请求记录 ' + data.requests.length + ' 条，样本 ' +
    data.samples.length + ' 条。包含已保存输入输出；分页只减少页面渲染，' +
    '不会截断内嵌记录。批次记录数不等于逻辑请求数；请求与样本状态分别展示。'));
  section.append(el('p', '仅展示原始 Case 文件中保存的记录，不包含预热事件；' +
    '未写入原始文件的请求不会补造。', 'muted'));
  section.append(el('p', '耗时只采用响应中保存的 e2el_ms / latency_ms / ' +
    'duration_ms，不用调度占位时间替代。缺少正文、Token 或时间点时显示未采集。',
    'muted'));
  const toolbar = el('div', undefined, 'toolbar');
  const search = el('input');
  search.placeholder = '搜索样本编号或请求编号';
  search.setAttribute('aria-label', '搜索样本编号或请求编号');
  const status = el('select');
  status.setAttribute('aria-label', '请求状态筛选');
  option(status, 'all', '全部请求与样本');
  option(status, 'failed', '异常请求或异常样本');
  option(status, 'success', '成功请求且无样本异常');
  const sort = el('select');
  sort.setAttribute('aria-label', '请求排序');
  option(sort, 'original', '原始记录顺序');
  option(sort, 'latency_ms', '完整耗时从高到低');
  option(sort, 'ttft_ms', 'TTFT 从高到低');
  toolbar.append(search, status, sort);
  const table = el('table'), thead = el('thead'), heading = el('tr');
  for (const name of ['样本 / 请求', '请求状态 / 样本状态', '输入 / 输出 Token',
    'TTFT（ms）', '完整耗时（ms）', '详情']) heading.append(el('th', name));
  thead.append(heading);
  const tbody = el('tbody');
  table.append(thead, tbody);
  const wrap = el('div', undefined, 'table-wrap');
  wrap.append(table);
  const pager = el('div', undefined, 'toolbar');
  const previous = el('button', '上一页'), next = el('button', '下一页');
  const label = el('span');
  label.setAttribute('aria-live', 'polite');
  pager.append(previous, label, next);
  section.append(toolbar, wrap, pager, link('返回摘要', 'summary-section'));
  let page = 0, filtered = [], activeIndex = null;
  const pageSize = 50;
  function expand(index, button, tr) {
    if (button.getAttribute('aria-expanded') === 'true') {
      tr.nextElementSibling.remove();
      button.setAttribute('aria-expanded', 'false');
      return;
    }
    const row = data.rows[index];
    const detailRow = el('tr'), cell = el('td');
    cell.colSpan = 6;
    const request = row.request_index == null ? null :
      data.requests[row.request_index];
    if (request) {
      const response = request.response || {};
      const text = response.text ?? response.output_text;
      lazyJson(cell, '请求输出正文（未采集时不会补造）', text);
      lazyJson(cell, '请求错误', request.error ?? response.error);
      lazyJson(cell, '调度时间线（原始偏移，单位秒）', Object.fromEntries(
        Object.entries(request).filter(([key]) =>
          key.endsWith('_offset_s') || key.endsWith('_delay_s'))));
      lazyJson(cell, '响应计时（原始字段，单位见字段后缀）', Object.fromEntries(
        Object.entries(response).filter(([key]) =>
          key.endsWith('_ms') || key.endsWith('_monotonic_s'))));
      lazyJson(cell, '原始请求记录与计时字段', request);
    } else cell.append(el('p', '该样本没有关联的已保存请求记录。'));
    for (const si of row.sample_indices) {
      const sample = data.samples[si];
      if (sample.facts?.prompt != null)
        lazyJson(cell, '样本保存的提示词（不代表每次请求体）', sample.facts.prompt);
      if (sample.facts?.raw_output != null)
        lazyJson(cell, '样本保存的输出', sample.facts.raw_output);
      lazyJson(cell, '样本 ' + sample.sample_id + ' 来源（不等同于实际请求体）',
        sample.source);
      lazyJson(cell, '样本 ' + sample.sample_id + ' 原始事实与输出', sample.facts);
      if (sample.error) lazyJson(cell, '样本错误', sample.error);
    }
    if (row.sample_indices.length > 1) cell.append(el('p',
      '多个样本使用相同 sample_id；全部列出，未任意选择其一。'));
    if (!row.sample_indices.length) cell.append(el('p', '未找到关联样本。'));
    detailRow.append(cell);
    tr.after(detailRow);
    button.setAttribute('aria-expanded', 'true');
  }
  function render() {
    const q = search.value.trim().toLowerCase();
    filtered = data.rows.map((row, index) => ({row, index})).filter(({row}) =>
      (status.value === 'all' || (status.value === 'failed' ?
        requestFailed(row) : row.status === 'success' && !requestFailed(row))) &&
      [row.sample_id, row.request_id].join(' ').toLowerCase().includes(q));
    if (sort.value !== 'original') filtered.sort((a, b) => {
      const x = a.row[sort.value], y = b.row[sort.value];
      if (x == null) return y == null ? a.index - b.index : 1;
      if (y == null) return -1;
      return y - x || a.index - b.index;
    });
    if (activeIndex != null) {
      page = Math.floor(filtered.findIndex(r => r.index === activeIndex) /
        pageSize);
    }
    const pages = Math.max(1, Math.ceil(filtered.length / pageSize));
    page = Math.max(0, Math.min(page, pages - 1));
    tbody.replaceChildren();
    for (const {row, index} of filtered.slice(page * pageSize,
      (page + 1) * pageSize)) {
      const tr = el('tr');
      tr.id = id + '-request-' + index;
      tr.tabIndex = -1;
      const cells = [recorded(row.sample_id) + ' / ' +
        (row.request_index == null ? '无请求记录' : recorded(row.request_id)),
        recorded(row.status) + ' / ' + (row.sample_states.join(', ') || '未采集'),
        recorded(row.input_tokens) + ' / ' + recorded(row.output_tokens),
        recorded(row.ttft_ms), recorded(row.latency_ms)];
      for (const value of cells) tr.append(el('td', value));
      const td = el('td'), button = el('button', '展开');
      button.setAttribute('aria-expanded', 'false');
      button.setAttribute('aria-label', '展开请求记录 ' + index);
      button.onclick = () => expand(index, button, tr);
      td.append(button, document.createTextNode(' '),
        link('定位', tr.id));
      tr.append(td);
      tbody.append(tr);
      if (activeIndex === index) expand(index, button, tr);
    }
    label.textContent = '第 ' + (page + 1) + ' / ' + pages + ' 页 · ' +
      filtered.length + ' 条记录（每页 ' + pageSize + ' 条）';
    previous.disabled = page === 0;
    next.disabled = page >= pages - 1;
    if (!filtered.length) {
      const tr = el('tr'), td = el('td', '没有匹配的记录');
      td.colSpan = 6;
      tr.append(td);
      tbody.append(tr);
    }
    activeIndex = null;
  }
  for (const input of [search, status, sort]) input.oninput = () => {
    page = 0;
    render();
  };
  previous.onclick = () => {page--; render();};
  next.onclick = () => {page++; render();};
  requestViews.set(id, action => {
    search.value = '';
    status.value = action === 'failed' ? 'failed' : 'all';
    sort.value = 'original';
    page = 0;
    if (typeof action === 'number') activeIndex = action;
    render();
  });
  navigation.append(link('查看请求列表', section.id), document.createTextNode(' · '),
    link('异常请求/样本', section.id + '-failed'));
  const slowest = data.rows.reduce((best, row, index) =>
    row.latency_ms != null && (best == null ||
      row.latency_ms > data.rows[best].latency_ms) ? index : best, null);
  if (slowest != null) navigation.append(document.createTextNode(' · '),
    link('最慢请求：' + fmt(data.rows[slowest].latency_ms) + ' ms',
      id + '-request-' + slowest));
  // Render a bounded page only when this section is opened.
  section.addEventListener('toggle', () => {
    if (section.open && !tbody.children.length) render();
  });
}
function ensureRequestTarget(target) {
  const match = /^(case-\d+-\d+)-(requests-failed|request-(\d+))$/.exec(target);
  if (!match || !requestViews.has(match[1])) return target;
  requestViews.get(match[1])(match[3] == null ? 'failed' : Number(match[3]));
  return match[3] == null ? match[1] + '-requests' : target;
}
