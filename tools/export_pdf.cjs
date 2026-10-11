/** Print the offline textbook with installed Chrome/Chromium and Node.js >=22.
 * No npm packages are required. The HTML embeds Chinese/KaTeX fonts and figures.
 * Update the HTML with tools/build_book.py when changing textbook content.
 *
 * node tools/export_pdf.cjs --chrome '/path/to/chrome'
 */
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawn, spawnSync } = require('node:child_process');
const { pathToFileURL } = require('node:url');
const { createHash } = require('node:crypto');

const root = path.resolve(__dirname, '..');
const name = 'GPU_Interview_12_Exercises_ZH';
const options = {
  html: path.join(root, `${name}.html`),
  output: path.join(root, 'output', 'pdf', `${name}.pdf`),
};
for (let i = 2; i < process.argv.length; i += 2) {
  const key = process.argv[i].replace(/^--/, '');
  if (!['html', 'output', 'chrome', 'report'].includes(key) || !process.argv[i + 1]) {
    throw new Error('Usage: node tools/export_pdf.cjs [--html PATH] [--output PATH] [--chrome PATH] [--report PATH]');
  }
  options[key] = path.resolve(process.argv[i + 1]);
}

const printCSS = `
@page { size: A4; }
html { scroll-behavior: auto; }
body { background: white; font-size: 9pt; line-height: 1.7; }
.layout { display: block; max-width: none; }
nav, .copy { display: none !important; }
main { padding: 0; }
article { max-width: none; }
.tagline { font-size: 8pt; margin-bottom: 14pt; }
h1 { font-size: 23pt; margin: 0 0 16pt; }
h2 { font-size: 18pt; margin: 18pt 0 16pt; padding-top: 12pt;
     break-before: auto; break-after: avoid; }
h2.pdf-chapter { break-before: page; margin-top: 0; }
h3 { font-size: 13pt; margin: 19pt 0 8pt; break-after: avoid; }
h4 { font-size: 10.5pt; margin: 14pt 0 7pt; break-after: avoid; }
p { margin: 7pt 0; orphans: 3; widows: 3; }
p.pdf-question { break-inside: avoid; break-after: avoid; }
li { margin: 4pt 0; }
table { display: table; width: 100%; table-layout: fixed; font-size: 7.8pt;
        line-height: 1.55; margin: 12pt 0; overflow: visible; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
td, th { min-width: 0; padding: 5pt 6pt; overflow-wrap: anywhere; }
code { font-family: Menlo, Consolas, 'Liberation Mono', 'Noto Sans SC', monospace;
       font-size: 0.86em; padding: 1pt 2pt; }
pre { font-family: Menlo, Consolas, 'Liberation Mono', 'Noto Sans SC', monospace;
      font-size: 7.1pt; line-height: 1.55; white-space: pre-wrap;
      overflow-wrap: anywhere; word-break: normal; overflow: visible;
      padding: 9pt 10pt; margin: 0; orphans: 3; widows: 3; }
pre code { white-space: inherit; overflow-wrap: inherit; word-break: inherit;
           font-size: inherit; }
.codehilite { border-radius: 3pt; margin: 10pt 0; break-inside: auto;
              box-decoration-break: clone; }
blockquote { margin: 10pt 0; padding: 3pt 10pt; }
img { max-width: 100%; max-height: 170mm; break-inside: avoid; }
.katex { font-size: 1.03em; }
.katex-display { margin: 9pt 0; padding: 3pt 0; overflow: visible;
                 break-inside: avoid; }
a { color: #006e61; }
.print-toc { break-before: page; }
.print-toc ol { padding-left: 18pt; margin: 0; }
.print-toc li { margin: 6pt 0; font-size: 10pt; break-inside: avoid; }
.print-toc a { text-decoration: none; }
.footer { font-size: 7pt; margin-top: 24pt; }
`;

function findChrome() {
  if (options.chrome) return options.chrome;
  const mac = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
  if (fs.existsSync(mac)) return mac;
  for (const command of ['google-chrome', 'chromium', 'chromium-browser']) {
    const result = spawnSync('which', [command], { encoding: 'utf8' });
    if (result.status === 0) return result.stdout.trim();
  }
  throw new Error('Install Chrome/Chromium or pass its executable with --chrome.');
}

async function main() {
  if (typeof WebSocket === 'undefined') throw new Error('Node.js >=22 is required.');
  const chrome = findChrome();
  const source = path.resolve(options.html);
  if (!fs.existsSync(source)) throw new Error(`HTML source not found: ${source}`);
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'gpu-book-pdf-'));
  const browser = spawn(chrome, [
    '--headless=new', '--disable-gpu', '--no-first-run', '--no-default-browser-check',
    '--allow-file-access-from-files', '--remote-debugging-port=0', `--user-data-dir=${profile}`,
    'about:blank',
  ], { stdio: 'ignore' });
  let socket;
  try {
    const portFile = path.join(profile, 'DevToolsActivePort');
    for (let attempt = 0; !fs.existsSync(portFile); attempt++) {
      if (attempt >= 200 || browser.exitCode !== null) throw new Error('Chrome did not start.');
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    const [port, endpoint] = fs.readFileSync(portFile, 'utf8').trim().split('\n');
    socket = new WebSocket(`ws://127.0.0.1:${port}${endpoint}`);
    await new Promise((resolve, reject) => {
      socket.addEventListener('open', resolve, { once: true });
      socket.addEventListener('error', reject, { once: true });
    });
    let sequence = 0;
    const pending = new Map();
    const events = new Map();
    socket.addEventListener('message', event => {
      const response = JSON.parse(event.data);
      if (response.id && pending.has(response.id)) {
        const { resolve, reject, timer } = pending.get(response.id);
        pending.delete(response.id);
        clearTimeout(timer);
        response.error ? reject(new Error(JSON.stringify(response.error))) : resolve(response.result);
      }
      if (response.method && events.has(response.method)) {
        events.get(response.method)(response.params);
        events.delete(response.method);
      }
    });
    function call(method, params = {}, sessionId) {
      return new Promise((resolve, reject) => {
        const id = ++sequence;
        const timer = setTimeout(() => { pending.delete(id); reject(new Error(`Timeout: ${method}`)); }, 180000);
        pending.set(id, { resolve, reject, timer });
        socket.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }));
      });
    }
    const { targetId } = await call('Target.createTarget', { url: 'about:blank' });
    const { sessionId } = await call('Target.attachToTarget', { targetId, flatten: true });
    const send = (method, params) => call(method, params, sessionId);
    const evaluate = async expression => {
      const result = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
      if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
      return result.result.value;
    };
    await send('Page.enable');
    await send('Runtime.enable');
    await send('Emulation.setEmulatedMedia', { media: 'print' });
    await send('Emulation.setDeviceMetricsOverride', { width: 681, height: 1000, deviceScaleFactor: 1, mobile: false });
    const loaded = new Promise(resolve => events.set('Page.loadEventFired', resolve));
    await send('Page.navigate', { url: pathToFileURL(source).href });
    await loaded;
    await evaluate(`(() => { const style = document.createElement('style'); style.textContent = ${JSON.stringify(printCSS)}; document.head.append(style); })()`);
    await evaluate(`(() => {
      const article = document.querySelector('article');
      article.querySelectorAll('p').forEach(paragraph => {
        if (/^问[：:]/.test(paragraph.textContent.trim())) paragraph.classList.add('pdf-question');
      });
      const headings = [...article.querySelectorAll('h2')];
      headings.forEach(target => {
        const text = target.textContent;
        if (/^第 \\d+ 题｜/.test(text) || ['怎样使用这套材料', '验证记录与适用范围',
          '各后端共享的工程约定', 'Triton 实现、逐题解释与面试追问',
          '参考资料与来源边界', '完整源码附录'].includes(text)) target.classList.add('pdf-chapter');
      });
      const toc = document.createElement('section');
      toc.className = 'print-toc';
      const title = document.createElement('h2');
      title.id = 'pdf-contents'; title.textContent = '目录'; toc.append(title);
      const list = document.createElement('ol');
      headings.forEach((target, index) => {
        if (!target.id) target.id = 'pdf-section-' + index;
        const item = document.createElement('li');
        const link = document.createElement('a');
        link.href = '#' + target.id; link.textContent = target.textContent;
        item.append(link); list.append(item);
      });
      toc.append(list); headings[0].before(toc);
    })()`);
    await evaluate('document.fonts.ready.then(() => true)');
    await evaluate('Promise.all([...document.images].map(image => image.decode())).then(() => true)');
    await evaluate(`(() => {
      document.querySelectorAll('.katex-display').forEach(block => {
        const math = block.querySelector('.katex');
        if (!math) return;
        const width = block.getBoundingClientRect().width;
        if (math.scrollWidth > width) {
          const size = parseFloat(getComputedStyle(math).fontSize);
          math.style.fontSize = size * width / math.scrollWidth * 0.97 + 'px';
        }
      });
    })()`);
    const layout = await evaluate(`(() => ({
      exercises: [...document.querySelectorAll('h2')].filter(h => /^第 \\d+ 题/.test(h.textContent)).length,
      code_blocks: document.querySelectorAll('.codehilite').length,
      formulas: document.querySelectorAll('.katex').length,
      fonts: document.fonts.status,
      broken_images: [...document.images].filter(i => !i.complete || !i.naturalWidth).length,
      missing_internal_links: [...document.querySelectorAll('article a[href^="#"]')]
        .map(a => a.getAttribute('href').slice(1)).filter(id => id && !document.getElementById(decodeURIComponent(id))),
      overflow: [...document.querySelectorAll('pre, table, .katex-display')]
        .filter(e => e.scrollWidth > e.clientWidth + 2)
        .map(e => ({tag: e.tagName, text: e.innerText.slice(0, 100), width: e.clientWidth, scroll_width: e.scrollWidth}))
    }))()`);
    if (layout.exercises !== 12 || layout.broken_images || layout.missing_internal_links.length || layout.overflow.length) {
      throw new Error(`PDF layout checks failed: ${JSON.stringify(layout)}`);
    }
    console.log(`Checked ${layout.exercises} exercises, ${layout.code_blocks} code blocks, ${layout.formulas} formulas.`);
    const pdf = await send('Page.printToPDF', {
      printBackground: true, paperWidth: 210 / 25.4, paperHeight: 297 / 25.4,
      marginTop: 17 / 25.4, marginRight: 15 / 25.4, marginBottom: 18 / 25.4, marginLeft: 15 / 25.4,
      displayHeaderFooter: true,
      headerTemplate: '<div style="font:8px sans-serif;color:#687785;width:100%;padding:0 15mm;">GPU PROGRAMMING | CUDA / TRITON / PYTORCH</div>',
      footerTemplate: '<div style="font:9px sans-serif;color:#687785;text-align:center;width:100%;"><span class="pageNumber"></span> / <span class="totalPages"></span></div>',
      generateDocumentOutline: true, generateTaggedPDF: true,
    });
    fs.mkdirSync(path.dirname(options.output), { recursive: true });
    fs.writeFileSync(options.output, Buffer.from(pdf.data, 'base64'));
    const report = {
      source_html: path.basename(source), source_html_sha256: createHash('sha256').update(fs.readFileSync(source)).digest('hex'),
      pdf: path.basename(options.output), pdf_bytes: fs.statSync(options.output).size, layout,
    };
    if (options.report) {
      fs.mkdirSync(path.dirname(options.report), { recursive: true });
      fs.writeFileSync(options.report, JSON.stringify(report, null, 2) + '\n');
    }
    console.log(JSON.stringify(report, null, 2));
    await call('Browser.close');
  } finally {
    if (socket) socket.close();
    if (browser.exitCode === null) {
      browser.kill('SIGTERM');
      await new Promise(resolve => browser.once('exit', resolve));
    }
    fs.rmSync(profile, { recursive: true, force: true });
  }
}

main().catch(error => { console.error(error.message); process.exitCode = 1; });
