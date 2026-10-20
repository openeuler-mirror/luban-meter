// Optional browser regression: requires Playwright and an installed Edge.
// python tests/test_report_requests.py results/html-v2-preview
// node tests/report_requests.browser.cjs results/html-v2-preview/report-full.html
const assert = require('node:assert/strict');
const fs = require('node:fs');
const {chromium} = require('playwright');

(async () => {
  const browser = await chromium.launch({headless: true, channel: 'msedge'});
  try {
    const page = await browser.newPage({viewport: {width: 1280, height: 900}});
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('dialog', () => {throw Error('Untrusted content executed');});
    await page.route('**/*', route => route.abort());
    await page.setContent(fs.readFileSync(process.argv[2], 'utf8'));
    const panel = page.locator('#case-0-0-requests');
    assert.equal(await panel.locator('tbody tr').count(), 0);
    await page.getByRole('link', {name: '查看请求列表', exact: true}).click();
    await panel.locator('tbody tr').nth(49).waitFor();
    assert.equal(await panel.locator('tbody tr').count(), 50);
    await panel.getByRole('combobox', {name: '请求排序'}).selectOption('ttft_ms');
    assert.equal(await panel.locator('tbody tr').first().getAttribute('id'),
      'case-0-0-request-119');
    await panel.getByRole('combobox', {name: '请求排序'}).selectOption('original');
    await panel.getByRole('button', {name: '下一页'}).click();
    assert.match(await panel.innerText(), /第 2 \/ 3 页/);
    await page.getByRole('link', {name: '最慢请求：219 ms', exact: true}).click();
    assert.equal(await page.evaluate(() => document.activeElement.id),
      'case-0-0-request-119');
    assert.match(await panel.innerText(), /第 3 \/ 3 页/);
    await panel.getByText('请求输出正文（未采集时不会补造）', {exact: true}).click();
    await panel.locator('pre').first().waitFor({state: 'visible'});
    assert.match(await panel.innerText(), /<script>alert\(1\)<\/script>/);
    await page.getByRole('link', {name: '失败：12', exact: true}).click();
    assert.match(await panel.innerText(), /13 条记录/);
    const search = panel.getByRole('textbox');
    await search.fill('orphan');
    assert.match(await panel.innerText(), /无请求记录/);
    await panel.getByRole('button', {name: '展开请求记录 120'}).click();
    assert.match(await panel.innerText(), /没有关联的已保存请求/);
    await search.fill('does-not-exist');
    assert.match(await panel.innerText(), /没有匹配的记录/);
    await page.evaluate(() => {location.hash = 'case-0-0-request-119';});
    await page.waitForFunction(() =>
      document.activeElement.id === 'case-0-0-request-119');
    assert.equal(await search.inputValue(), '');
    await page.getByRole('link', {name: '请求详情不可用（查看原因）'}).click();
    assert.match(await page.locator('#case-0-1-requests').innerText(),
      /原始文件缺失/);
    assert.deepEqual(errors, []);
    if (process.argv[3]) {
      await page.evaluate(() => {location.hash = 'overview';});
      await page.screenshot({path: process.argv[3], fullPage: true});
    }
    console.log('PASS: pagination, filters, deep links, lazy text, escaping, ' +
      'missing raw files, orphan samples, no network or JS errors');
  } finally {
    await browser.close();
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
