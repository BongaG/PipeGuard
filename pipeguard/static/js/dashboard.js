const topo = window.TOPOLOGY;
const svgNS = 'http://www.w3.org/2000/svg';
const svg = document.getElementById('schematic');
const pos = {};
pos[topo.reservoir.id] = {x: topo.reservoir.x, y: topo.reservoir.y};
topo.junctions.forEach(j => { pos[j.id] = {x: j.x, y: j.y}; });
const palette = {J1: '#1f6f8b', J2: '#2e7d5b', J3: '#b83227', J4: '#c98a0b', J5: '#6a4c93', J6: '#3c6e71', J7: '#8c564b', J8: '#5b6b78'};
let linkUp = true;

function el(tag, attrs, parent) {
  const e = document.createElementNS(svgNS, tag);
  Object.entries(attrs || {}).forEach(([k, v]) => e.setAttribute(k, v));
  (parent || svg).appendChild(e);
  return e;
}

const pipeEls = {};
const pipeLabelEls = {};
const valveEls = {};
const nodeEls = {};

function drawStatic() {
  const gPipes = el('g');
  topo.pipes.forEach(p => {
    const a = pos[p.start], b = pos[p.end];
    const w = 3 + p.diameter * 28;
    pipeEls[p.id] = el('line', {x1: a.x, y1: a.y, x2: b.x, y2: b.y, 'stroke-width': w, class: 'pipe-line flowing'}, gPipes);
    const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
    const dx = b.x - a.x, dy = b.y - a.y, len = Math.hypot(dx, dy);
    const nx = -dy / len, ny = dx / len;
    const lbl = el('text', {x: mx + nx * 20, y: my + ny * 20 + 4, class: 'pipe-label', 'text-anchor': 'middle'}, gPipes);
    lbl.textContent = p.id;
    pipeLabelEls[p.id] = lbl;
    const vx = a.x + dx * 0.36, vy = a.y + dy * 0.36;
    const ang = Math.atan2(dy, dx) * 180 / Math.PI;
    const v = el('path', {d: 'M -8 -6 L 8 6 L 8 -6 L -8 6 Z', class: 'valve', transform: `translate(${vx},${vy}) rotate(${ang})`}, gPipes);
    valveEls[p.id] = v;
  });
  const r = pos[topo.reservoir.id];
  el('rect', {x: r.x - 26, y: r.y - 30, width: 52, height: 60, rx: 4, class: 'reservoir'});
  const rt = el('text', {x: r.x, y: r.y + 5, class: 'node-id'}); rt.textContent = 'R1';
  const rl = el('text', {x: r.x, y: r.y + 48, class: 'node-edge'}); rl.textContent = 'Reservoir';
  topo.junctions.forEach(j => {
    const g = el('g', {transform: `translate(${j.x},${j.y})`, class: 'node'});
    const ring = el('circle', {r: 30, class: 'node-ring ok'}, g);
    const id = el('text', {y: -8, class: 'node-id'}, g); id.textContent = j.id;
    const kpa = el('text', {y: 11, class: 'node-kpa'}, g); kpa.textContent = '-';
    const res = el('text', {y: 46, class: 'node-res'}, g);
    const edge = el('text', {y: 60, class: 'node-edge'}, g);
    nodeEls[j.id] = {ring, kpa, res, edge};
  });
}

function levelFor(received, demand) {
  const ratio = demand > 0 ? received / demand : 1;
  if (ratio >= 0.95) return 'ok';
  if (ratio >= 0.6) return 'warn';
  return 'danger';
}

function isolatedNodes(valves) {
  const closedPipes = new Set(Object.entries(valves).filter(([, s]) => s === 'closed').map(([p]) => p));
  const into = {};
  topo.pipes.forEach(p => { into[p.end] = p; });
  const out = new Set();
  topo.junctions.forEach(j => {
    let n = j.id;
    while (into[n]) {
      if (closedPipes.has(into[n].id)) { out.add(j.id); break; }
      n = into[n].start;
    }
  });
  return out;
}

function updateSchematic(s) {
  const iso = isolatedNodes(s.valves);
  const suspect = s.alert && s.alert.pipe;
  topo.pipes.forEach(p => {
    let cls = 'pipe-line';
    if (s.valves[p.id] === 'closed' || iso.has(p.start)) cls += ' isolated';
    else if (p.id === suspect) cls += ' suspect';
    else cls += ' flowing';
    pipeEls[p.id].setAttribute('class', cls);
    valveEls[p.id].setAttribute('class', `valve ${s.valves[p.id]}`);
    pipeLabelEls[p.id].textContent = `${p.id}  ${(s.pipe_flow[p.id] || 0).toFixed(1)} L/s`;
  });
  s.nodes.forEach(n => {
    const e = nodeEls[n.id];
    e.ring.setAttribute('class', `node-ring ${iso.has(n.id) ? 'warn' : levelFor(n.received_lps, n.demand_lps)}`);
    e.kpa.textContent = iso.has(n.id) ? 'off' : n.received_lps.toFixed(1);
    e.res.textContent = iso.has(n.id) ? 'isolated' : `of ${n.demand_lps.toFixed(1)} L/s, ${Math.round(n.pressure)} kPa`;
    e.edge.textContent = n.source === 'device' ? 'live device' : '';
  });
}

const labels = {normal: 'Normal', demand_spike: 'Demand change', leak: 'Leak', burst: 'Burst'};
function updateSide(s) {
  document.getElementById('zoneLine').textContent = `t = ${s.t} s, demand at ${(s.demand_multiplier * 100).toFixed(0)}% of design, ${s.model_loaded ? 'model loaded' : 'model not trained yet, run scripts/train.py'}`;
  document.getElementById('inlet').textContent = `${s.inlet_lps.toFixed(2)} L/s`;
  document.getElementById('metered').textContent = `${s.metered_lps.toFixed(2)} L/s`;
  document.getElementById('imbalance').textContent = `${s.imbalance_lps.toFixed(2)} L/s`;
  const bars = document.getElementById('probaBars');
  if (!s.proba) { bars.innerHTML = `<p class="calm">${s.link_up ? 'Collecting the first 70 seconds of data before the model starts.' : 'Model paused while the cloud link is down.'}</p>`; } else
  bars.innerHTML = Object.entries(s.proba).map(([k, v]) => `<div class="bar-row ${k}"><span>${labels[k]}</span><div class="bar-track"><div class="bar-fill" style="width:${(v * 100).toFixed(1)}%"></div></div><span>${(v * 100).toFixed(0)}%</span></div>`).join('');
  const panel = document.getElementById('alertPanel');
  const body = document.getElementById('alertBody');
  if (s.alert) {
    const a = s.alert;
    panel.className = 'panel active' + (a.detected_by === 'edge fail-safe' ? ' edge' : '');
    const rank = (a.ranking || []).map(r => `<li><span>${r.pipe}</span><div class="bar"><i style="width:${(r.p * 100).toFixed(0)}%"></i></div><span>${(r.p * 100).toFixed(0)}%</span></li>`).join('');
    body.className = '';
    const where = a.kind === 'burst' && a.junction ? `at ${a.junction} (fed by ${a.pipe})` : `on ${a.pipe || 'unknown pipe'}`;
    body.innerHTML = `<div class="alert-kind">${labels[a.kind]} ${where}</div>
      <div class="alert-meta">Incident #${a.id}, found by ${a.detected_by}${a.latency_s != null ? `, ${a.latency_s.toFixed(0)} s after it started` : ''}.</div>
      ${rank ? `<ol class="rank">${rank}</ol>` : ''}
      <div class="btns">${a.pipe && a.detected_by !== 'edge fail-safe' && s.valves[a.pipe] === 'open' ? `<button class="danger" data-close="${a.pipe}">Close valve on ${a.pipe}</button>` : ''}<button class="ghost" data-resolve="${a.id}">Mark resolved</button></div>`;
  } else {
    panel.className = 'panel';
    body.className = 'calm';
    body.textContent = s.link_up ? 'No leak or burst detected. The model is comparing every sensor against the digital twin once a second.' : 'Cloud link is down. Edge nodes are watching for large pressure drops and will close valves on their own.';
  }
  const status = document.getElementById('linkStatus');
  linkUp = s.link_up;
  status.textContent = s.link_up ? `Cloud link up on ${s.link_mode}` : 'Cloud link down, edge fail-safe active';
  status.className = 'status' + (s.link_up ? '' : ' down');
  document.getElementById('linkBtn').textContent = s.link_up ? 'Cut cloud link' : 'Restore cloud link';
  const tbody = document.querySelector('#valveTable tbody');
  tbody.innerHTML = topo.pipes.map(p => {
    const st = s.valves[p.id];
    const act = st === 'open' ? 'close' : 'open';
    return `<tr><td>${p.id}</td><td>${(s.pipe_flow[p.id] || 0).toFixed(2)} L/s</td><td class="state-${st}">${st}</td><td><button class="ghost small" data-valve="${p.id}" data-action="${act}" ${st.endsWith('ing') ? 'disabled' : ''}>${act === 'close' ? 'Close' : 'Open'}</button></td></tr>`;
  }).join('');
  document.getElementById('log').innerHTML = s.log.map(l => `<li class="${l.level}"><time>t ${l.t}</time><span>${l.text}</span></li>`).join('');
}

let pChart, fChart;
function initCharts() {
  const base = {animation: false, responsive: true, maintainAspectRatio: false, interaction: {mode: 'nearest', intersect: false},
    plugins: {legend: {labels: {boxWidth: 12, font: {family: 'Barlow'}}}},
    scales: {x: {ticks: {maxTicksLimit: 8}, title: {display: true, text: 'Time (s)'}}}};
  pChart = new Chart(document.getElementById('pressureChart'), {type: 'line', data: {labels: [], datasets: []},
    options: {...base, elements: {point: {radius: 0}}, scales: {...base.scales, y: {title: {display: true, text: 'Water received (L/s)'}}}}});
  fChart = new Chart(document.getElementById('flowChart'), {type: 'line', data: {labels: [], datasets: [
    {label: 'Inlet flow', data: [], borderColor: '#1f6f8b', borderWidth: 2},
    {label: 'Delivered to junctions', data: [], borderColor: '#c98a0b', borderWidth: 2, borderDash: [6, 4]}]},
    options: {...base, elements: {point: {radius: 0}}, scales: {...base.scales, y: {title: {display: true, text: 'Flow (L/s)'}}}}});
}

function updateCharts(s) {
  const sel = document.getElementById('chartNode').value;
  const metric = document.getElementById('chartMetric').value;
  const ids = sel === 'all' ? Object.keys(s.series[metric]) : [sel];
  const ds = [];
  ids.forEach(id => {
    ds.push({label: id, data: s.series[metric][id], borderColor: palette[id], borderWidth: 1.8});
    if (metric === 'pressure' && sel !== 'all') ds.push({label: `${id} twin`, data: s.series.twin[id], borderColor: '#5b6b78', borderDash: [6, 4], borderWidth: 1.5});
  });
  pChart.options.scales.y.title.text = metric === 'pressure' ? 'Pressure (kPa)' : 'Water received (L/s)';
  pChart.data.labels = s.series.t;
  pChart.data.datasets = ds;
  pChart.update();
  fChart.data.labels = s.series.t;
  fChart.data.datasets[0].data = s.series.inlet;
  fChart.data.datasets[1].data = s.series.metered;
  fChart.update();
}

async function post(url, body) {
  const r = await fetch(url, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body || {})});
  return r.json();
}

document.addEventListener('click', async ev => {
  const b = ev.target.closest('button');
  if (!b) return;
  if (b.dataset.scn) {
    const kind = b.dataset.scn;
    if (kind === 'demand_spike') await post('/api/scenario', {kind, node: document.getElementById('nodeSel').value, factor: 2});
    else if (kind === 'burst') await post('/api/scenario', {kind, node: document.getElementById('nodeSel').value, percent: parseFloat(document.getElementById('burstSel').value)});
    else await post('/api/scenario', {kind, pipe: document.getElementById('pipeSel').value});
  } else if (b.dataset.valve) {
    await post('/api/valve', {pipe: b.dataset.valve, action: b.dataset.action});
  } else if (b.dataset.close) {
    await post('/api/valve', {pipe: b.dataset.close, action: 'close'});
  } else if (b.dataset.resolve) {
    await post(`/api/alerts/${b.dataset.resolve}/resolve`);
  } else if (b.id === 'repairBtn') {
    await post('/api/repair');
  } else if (b.id === 'linkBtn') {
    await post('/api/link', {up: !linkUp});
  }
  refresh();
});

async function refresh() {
  try {
    const s = await (await fetch('/api/state?seconds=180')).json();
    if (!s.ready) return;
    updateSchematic(s);
    updateSide(s);
    updateCharts(s);
  } catch (e) {
    document.getElementById('linkStatus').textContent = 'Dashboard lost contact with the server';
  }
}

drawStatic();
initCharts();
refresh();
setInterval(refresh, 1000);
