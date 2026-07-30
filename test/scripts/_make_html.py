"""Generate test/validation_ihm.html from the canonical TESTS list.

Reads ``_make_matrix.py::TESTS`` (single source of truth — shared with
the XLSX generator) and produces a self-contained HTML checklist:

- Sections collapsibles par catégorie (8 sections, striping coloré).
- 1 clic pour valider (OK / NOK / NA en boutons toggle).
- "Tout OK section" et "Tout OK global" pour cocher en masse.
- Micro-description : Pré-requis → Action → Attendu, visible
  d'emblée.
- Compteur par section + barre de progression locale.
- Persistance localStorage, export JSON + Markdown, impression OK.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation_ihm.html"


def _load_tests() -> list[tuple]:
    """Import TESTS from the matrix generator (shared source of truth)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_make_matrix", Path(__file__).with_name("_make_matrix.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TESTS


# Visual accent colour per category (left-border + chip tint).
CATEGORY_COLOR = {
    "IHM": "#1f4e78",
    "Paramètres": "#7c3aed",
    "Entrées": "#ea580c",
    "Sorties": "#16a34a",
    "Cas limites": "#ca8a04",
    "Performance": "#0891b2",
    "Robustesse": "#dc2626",
    "Régression": "#475569",
}

CATEGORY_ICON = {
    "IHM": "🖥️",
    "Paramètres": "⚙️",
    "Entrées": "📥",
    "Sorties": "📤",
    "Cas limites": "⚠️",
    "Performance": "⏱️",
    "Robustesse": "🛡️",
    "Régression": "🔁",
}


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Validation IHM — ShutterstockAnalyzer v2.1.0</title>
<style>
  :root {
    --bg: #f7f8fa;
    --card: #ffffff;
    --border: #d4d8df;
    --border-strong: #b3bac4;
    --text: #1a1d24;
    --muted: #6b7280;
    --subtle: #9ca3af;
    --primary: #1f4e78;
    --ok: #16a34a;
    --ok-bg: #f0fdf4;
    --nok: #dc2626;
    --nok-bg: #fef2f2;
    --na: #94a3b8;
    --na-bg: #f1f5f9;
    --shadow: 0 1px 3px rgba(0,0,0,0.05);
    --shadow-md: 0 4px 12px rgba(0,0,0,0.08);
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    padding: 24px 24px 80px;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
  }
  h1 { margin: 0 0 4px; font-size: 24px; color: var(--primary); }
  h2 { margin: 0; font-size: 16px; font-weight: 600; }
  .meta { color: var(--muted); font-size: 13px; margin-bottom: 16px; }

  /* --- Meta & summary ----------------------------------------- */
  .info-grid {
    display: grid; grid-template-columns: auto 1fr auto 1fr; gap: 6px 16px;
    background: var(--card); padding: 16px; border-radius: 6px;
    box-shadow: var(--shadow); margin-bottom: 24px; align-items: center;
  }
  .info-grid label { font-weight: 600; color: var(--muted); font-size: 13px; }
  .info-grid input {
    border: 1px solid var(--border); padding: 6px 10px;
    border-radius: 4px; font-size: 13px; max-width: 320px;
    background: var(--bg);
  }
  .summary-bar {
    background: var(--card); border-radius: 6px; padding: 14px 18px;
    margin-bottom: 16px; box-shadow: var(--shadow);
    display: flex; gap: 24px; flex-wrap: wrap; align-items: center;
  }
  .summary-bar .item { font-size: 13px; color: var(--muted); }
  .summary-bar .item strong { font-size: 20px; margin-right: 6px; color: var(--text); }
  .summary-bar .ok strong { color: var(--ok); }
  .summary-bar .nok strong { color: var(--nok); }
  .global-progress {
    flex: 1; min-width: 220px; height: 8px; background: var(--na-bg);
    border-radius: 4px; overflow: hidden; position: relative;
  }
  .global-progress > div { height: 100%; background: linear-gradient(90deg, var(--ok), #22c55e); transition: width 0.3s ease; }

  /* --- Toolbar ------------------------------------------------ */
  .toolbar {
    position: sticky; top: 0; z-index: 50;
    background: rgba(247,248,250,0.95); backdrop-filter: blur(8px);
    padding: 12px 0; margin: 0 -24px 16px; padding-left: 24px; padding-right: 24px;
    display: flex; gap: 8px; flex-wrap: wrap; align-items: center;
    border-bottom: 1px solid var(--border);
  }
  .toolbar button {
    padding: 8px 14px; border-radius: 4px; border: 1px solid var(--border);
    background: var(--card); color: var(--text);
    cursor: pointer; font-size: 13px; font-weight: 500;
    transition: all 0.15s ease;
  }
  .toolbar button:hover { filter: brightness(0.97); transform: translateY(-1px); }
  .toolbar button.primary {
    background: var(--primary); color: white; border-color: var(--primary);
  }
  .toolbar button.success {
    background: var(--ok); color: white; border-color: var(--ok);
  }
  .toolbar button.danger {
    background: white; color: var(--nok); border-color: var(--nok);
  }
  .toolbar input[type="search"] {
    padding: 8px 12px; border: 1px solid var(--border);
    border-radius: 4px; font-size: 13px; min-width: 240px;
    background: var(--card);
  }
  .toolbar select {
    padding: 8px 12px; border: 1px solid var(--border);
    border-radius: 4px; font-size: 13px; background: var(--card);
  }

  /* --- Sections ----------------------------------------------- */
  .section {
    background: var(--card); border-radius: 8px; margin-bottom: 18px;
    box-shadow: var(--shadow); overflow: hidden;
    border-left: 4px solid var(--accent);
  }
  .section-header {
    display: flex; align-items: center; gap: 16px;
    padding: 14px 18px; cursor: pointer;
    background: linear-gradient(135deg, var(--accent-soft), transparent 80%);
    user-select: none;
  }
  .section-header:hover { filter: brightness(0.98); }
  .section-header .icon { font-size: 22px; }
  .section-header .name { font-weight: 600; color: var(--accent); font-size: 16px; flex: 1; }
  .section-header .count {
    font-size: 13px; color: var(--muted);
    background: var(--card); padding: 3px 10px; border-radius: 99px;
    border: 1px solid var(--border);
  }
  .section-header .count.complete { color: var(--ok); border-color: var(--ok); }
  .section-progress {
    width: 140px; height: 6px; background: var(--na-bg);
    border-radius: 3px; overflow: hidden;
  }
  .section-progress > div {
    height: 100%; background: var(--accent); transition: width 0.3s ease;
  }
  .section-actions { display: flex; gap: 6px; }
  .section-actions button {
    padding: 4px 10px; font-size: 12px; border-radius: 4px;
    border: 1px solid var(--border); background: var(--card);
    cursor: pointer; font-weight: 500;
  }
  .section-actions button.all-ok { background: var(--ok-bg); color: var(--ok); border-color: var(--ok); }
  .section-actions button.reset { color: var(--muted); }
  .section-actions button:hover { filter: brightness(0.95); }
  .chevron { transition: transform 0.2s ease; color: var(--subtle); }
  details[open] .chevron { transform: rotate(90deg); }

  /* --- Test cards --------------------------------------------- */
  .section-body { padding: 0 18px 18px; }
  .test-card {
    padding: 14px 16px; margin-top: 8px;
    border-radius: 6px; border: 1px solid var(--border);
    background: var(--card);
    display: grid;
    grid-template-columns: 90px 1fr 250px;
    gap: 16px;
    align-items: start;
    transition: all 0.15s ease;
  }
  .test-card.status-OK { background: var(--ok-bg); border-color: var(--ok); }
  .test-card.status-NOK { background: var(--nok-bg); border-color: var(--nok); }
  .test-card.status-NA { background: var(--card); }

  /* Striping inside section (alternance subtile). */
  .test-card.alt { background-color: rgba(0,0,0,0.012); }
  .test-card.status-OK.alt { background: var(--ok-bg); }
  .test-card.status-NOK.alt { background: var(--nok-bg); }

  .test-id-block { display: flex; flex-direction: column; gap: 4px; }
  .test-id {
    font-family: 'Cascadia Code', 'Consolas', monospace;
    font-weight: 700; font-size: 14px; color: var(--text);
  }
  .severity {
    display: inline-block; padding: 1px 6px; border-radius: 3px;
    font-size: 10px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.4px; width: fit-content;
  }
  .severity.critique { background: #fef2f2; color: var(--nok); }
  .severity.majeure { background: #fff7ed; color: #ea580c; }
  .severity.mineure { background: #f1f5f9; color: var(--muted); }
  .req {
    font-size: 10px; color: var(--subtle);
    font-family: 'Cascadia Code', 'Consolas', monospace;
  }

  .test-body { display: flex; flex-direction: column; gap: 6px; }
  .test-feat { font-weight: 600; font-size: 14px; color: var(--text); }
  .micro {
    background: rgba(0,0,0,0.025); border-radius: 4px;
    padding: 8px 10px; font-size: 12px; color: var(--text);
    border-left: 3px solid var(--accent); margin-top: 2px;
  }
  .micro dt {
    display: inline-block; font-weight: 700; color: var(--accent);
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.4px;
    width: 76px; vertical-align: top;
  }
  .micro dd { display: inline; margin: 0; }
  .micro div { margin-bottom: 3px; }
  .micro div:last-child { margin-bottom: 0; }

  .status-block { display: flex; flex-direction: column; gap: 6px; }
  .status-buttons { display: flex; gap: 4px; }
  .status-buttons button {
    flex: 1; padding: 8px 0; border-radius: 4px;
    border: 1px solid var(--border); background: var(--card);
    cursor: pointer; font-size: 12px; font-weight: 600;
    color: var(--muted); transition: all 0.1s ease;
  }
  .status-buttons button:hover { filter: brightness(0.95); transform: scale(0.98); }
  .status-buttons button.btn-OK.active { background: var(--ok); color: white; border-color: var(--ok); }
  .status-buttons button.btn-NOK.active { background: var(--nok); color: white; border-color: var(--nok); }
  .status-buttons button.btn-NA.active { background: var(--na); color: white; border-color: var(--na); }
  .status-buttons button.btn-OK:hover:not(.active) { color: var(--ok); border-color: var(--ok); }
  .status-buttons button.btn-NOK:hover:not(.active) { color: var(--nok); border-color: var(--nok); }
  .status-buttons button.btn-NA:hover:not(.active) { color: var(--na); border-color: var(--na); }
  textarea.comment {
    width: 100%; padding: 6px 8px; border: 1px solid var(--border);
    border-radius: 4px; font-size: 12px; font-family: inherit;
    resize: vertical; min-height: 36px; max-height: 100px;
    background: var(--card);
  }

  /* --- Export panel ------------------------------------------ */
  details.export-section {
    background: var(--card); padding: 14px 18px; border-radius: 6px;
    margin: 16px 0; box-shadow: var(--shadow);
  }
  summary { font-weight: 600; cursor: pointer; }
  pre.export {
    background: #1a1d24; color: #e5e7eb; padding: 16px;
    border-radius: 4px; overflow-x: auto; font-size: 12px;
    font-family: 'Cascadia Code', 'Consolas', monospace;
    max-height: 400px; overflow-y: auto;
    margin-top: 8px;
  }

  /* --- Toast ------------------------------------------------- */
  .toast {
    position: fixed; bottom: 24px; right: 24px; z-index: 100;
    background: var(--text); color: white; padding: 10px 18px;
    border-radius: 6px; box-shadow: var(--shadow-md);
    font-size: 13px; opacity: 0;
    transform: translateY(20px); transition: all 0.25s ease;
    pointer-events: none;
  }
  .toast.show { opacity: 1; transform: translateY(0); }

  /* --- Print ------------------------------------------------- */
  @media print {
    body { background: white; padding: 0; }
    .toolbar, .toast { display: none; }
    .section { box-shadow: none; break-inside: avoid; }
    .test-card { break-inside: avoid; }
    details { open: true; }
    details > summary { list-style: none; }
    details > summary::-webkit-details-marker { display: none; }
  }

  /* --- Responsive -------------------------------------------- */
  @media (max-width: 900px) {
    .test-card { grid-template-columns: 1fr; }
    .test-id-block { flex-direction: row; gap: 12px; align-items: center; }
    .info-grid { grid-template-columns: auto 1fr; }
  }
</style>
</head>
<body>

<h1>Validation IHM — ShutterstockAnalyzer</h1>
<p class="meta">
  Outil : <strong>ShutterstockAnalyzer</strong> ·
  Version : <strong id="meta-version">v2.1.0</strong> ·
  Persistance navigateur (<code>localStorage</code>) ·
  __NB_TESTS__ tests répartis en __NB_SECTIONS__ sections.
</p>

<div class="info-grid">
  <label for="testeur">Testeur</label>
  <input type="text" id="testeur" placeholder="Nom du testeur">
  <label for="env">Environnement</label>
  <input type="text" id="env" placeholder="Windows 11 — i7 — sans Ollama">
  <label for="date">Date</label>
  <input type="date" id="date">
  <label for="build">Build EXE</label>
  <input type="text" id="build" placeholder="dist/ShutterstockAnalyzer.exe — 24,7 Mo">
</div>

<div class="summary-bar">
  <div class="item ok"><strong id="count-ok">0</strong>OK</div>
  <div class="item nok"><strong id="count-nok">0</strong>NOK</div>
  <div class="item"><strong id="count-na">0</strong>NA</div>
  <div class="item">/ <strong id="count-total">0</strong> tests</div>
  <div class="global-progress" title="Taux d'avancement (OK+NOK / Total)">
    <div id="global-progress-bar" style="width: 0%"></div>
  </div>
  <div class="item">Taux OK : <strong id="count-rate">0%</strong></div>
</div>

<div class="toolbar">
  <input type="search" id="search" placeholder="🔍 Filtrer ID, fonctionnalité, description…">
  <select id="filter-status">
    <option value="">Tous statuts</option>
    <option value="OK">OK uniquement</option>
    <option value="NOK">NOK uniquement</option>
    <option value="NA">NA uniquement</option>
  </select>
  <select id="filter-severity">
    <option value="">Toutes sévérités</option>
    <option value="critique">Critique</option>
    <option value="majeure">Majeure</option>
    <option value="mineure">Mineure</option>
  </select>
  <button id="btn-expand-all">⊞ Tout déplier</button>
  <button id="btn-collapse-all">⊟ Tout replier</button>
  <button id="btn-all-ok-global" class="success">✓ Tout valider OK</button>
  <button id="btn-reset-global" class="danger">↻ Réinitialiser</button>
  <button id="btn-export-json" class="primary">📥 Exporter JSON</button>
  <button id="btn-export-md">📝 Générer Markdown</button>
  <button onclick="window.print()">🖨 Imprimer</button>
</div>

<details class="export-section" id="export-section">
  <summary>Export (JSON / Markdown)</summary>
  <pre class="export" id="export-output">(Clique sur un bouton Exporter pour générer)</pre>
</details>

<div id="sections-container"></div>

<div class="toast" id="toast"></div>

<script>
// ============================================================================
// Test list — generated from test/scripts/_make_matrix.py::TESTS
// ============================================================================
const TESTS = __TESTS_JSON__;
const CATEGORY_COLORS = __CATEGORY_COLORS__;
const CATEGORY_ICONS = __CATEGORY_ICONS__;
const STORAGE_KEY = 'ssanalyzer_qa_ihm_v2';

// ----------------------------------------------------------------------------
// Persistence
// ----------------------------------------------------------------------------
function loadState() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}'); }
  catch (e) { return {}; }
}
function saveState(state) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}

// ----------------------------------------------------------------------------
// Group + filter
// ----------------------------------------------------------------------------
function groupByCategory(tests) {
  const out = {};
  tests.forEach(t => {
    out[t.categorie] = out[t.categorie] || [];
    out[t.categorie].push(t);
  });
  return out;
}

function passesFilter(t) {
  const s = document.getElementById('search').value.toLowerCase();
  const st = document.getElementById('filter-status').value;
  const sev = document.getElementById('filter-severity').value;
  const state = loadState();
  const status = (state[t.id] || {}).status || 'NA';
  if (s && ![t.id, t.fonctionnalite, t.description, t.attendu]
      .some(x => (x || '').toLowerCase().includes(s))) return false;
  if (st && status !== st) return false;
  if (sev && t.severite !== sev) return false;
  return true;
}

// ----------------------------------------------------------------------------
// Light helpers
// ----------------------------------------------------------------------------
function hexToRgba(hex, alpha) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function escapeHtml(s) {
  return (s || '').replace(/[&<>"']/g, c => ({
    '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
  })[c]);
}

// ----------------------------------------------------------------------------
// Render
// ----------------------------------------------------------------------------
function render() {
  const container = document.getElementById('sections-container');
  container.innerHTML = '';
  const state = loadState();
  const grouped = groupByCategory(TESTS);

  Object.entries(grouped).forEach(([cat, tests]) => {
    const filtered = tests.filter(passesFilter);
    if (filtered.length === 0) return;

    const color = CATEGORY_COLORS[cat] || '#475569';
    const icon = CATEGORY_ICONS[cat] || '📋';

    const okCount = tests.filter(t => (state[t.id] || {}).status === 'OK').length;
    const nokCount = tests.filter(t => (state[t.id] || {}).status === 'NOK').length;
    const naCount = tests.length - okCount - nokCount;
    const done = okCount + nokCount;
    const pct = tests.length > 0 ? Math.round(done / tests.length * 100) : 0;
    const complete = naCount === 0 && tests.length > 0;

    const section = document.createElement('details');
    section.open = true;
    section.className = 'section';
    section.style.setProperty('--accent', color);
    section.style.setProperty('--accent-soft', hexToRgba(color, 0.10));
    section.dataset.category = cat;

    section.innerHTML = `
      <summary class="section-header">
        <span class="chevron">▶</span>
        <span class="icon">${icon}</span>
        <span class="name">${escapeHtml(cat)}</span>
        <span class="count ${complete ? 'complete' : ''}">
          ${okCount}/${tests.length} OK${nokCount ? ' · ' + nokCount + ' NOK' : ''}
        </span>
        <div class="section-progress"><div style="width: ${pct}%"></div></div>
        <div class="section-actions">
          <button class="all-ok" data-action="all-ok" data-cat="${escapeHtml(cat)}">✓ Tout OK</button>
          <button class="reset" data-action="reset" data-cat="${escapeHtml(cat)}">↻ Reset</button>
        </div>
      </summary>
      <div class="section-body">
        ${filtered.map((t, i) => renderTestCard(t, state, i)).join('')}
      </div>
    `;
    container.appendChild(section);
  });

  updateSummary();
}

function renderTestCard(t, state, idx) {
  const s = state[t.id] || { status: 'NA', comment: '' };
  const altClass = idx % 2 === 1 ? ' alt' : '';
  return `
    <div class="test-card status-${s.status}${altClass}" data-id="${t.id}">
      <div class="test-id-block">
        <span class="test-id">${t.id}</span>
        <span class="severity ${t.severite}">${t.severite}</span>
        <span class="req">${escapeHtml(t.exigence)}</span>
      </div>
      <div class="test-body">
        <div class="test-feat">${escapeHtml(t.fonctionnalite)}</div>
        <div class="micro">
          <div><dt>Pré-requis</dt><dd>${escapeHtml(t.prerequis || '—')}</dd></div>
          <div><dt>Donnée</dt><dd>${escapeHtml(t.donnee || '—')}</dd></div>
          <div><dt>Action</dt><dd>${escapeHtml(t.description)}</dd></div>
          <div><dt>Attendu</dt><dd>${escapeHtml(t.attendu)}</dd></div>
        </div>
      </div>
      <div class="status-block">
        <div class="status-buttons">
          <button class="btn-OK ${s.status === 'OK' ? 'active' : ''}" data-id="${t.id}" data-status="OK">OK</button>
          <button class="btn-NOK ${s.status === 'NOK' ? 'active' : ''}" data-id="${t.id}" data-status="NOK">NOK</button>
          <button class="btn-NA ${s.status === 'NA' ? 'active' : ''}" data-id="${t.id}" data-status="NA">NA</button>
        </div>
        <textarea class="comment" data-id="${t.id}" placeholder="Commentaire / observation…">${escapeHtml(s.comment || '')}</textarea>
      </div>
    </div>
  `;
}

function updateSummary() {
  const state = loadState();
  let ok = 0, nok = 0, na = 0;
  TESTS.forEach(t => {
    const s = (state[t.id] || {}).status || 'NA';
    if (s === 'OK') ok++;
    else if (s === 'NOK') nok++;
    else na++;
  });
  document.getElementById('count-ok').textContent = ok;
  document.getElementById('count-nok').textContent = nok;
  document.getElementById('count-na').textContent = na;
  document.getElementById('count-total').textContent = TESTS.length;
  const done = ok + nok;
  const pct = TESTS.length > 0 ? Math.round(done / TESTS.length * 100) : 0;
  document.getElementById('global-progress-bar').style.width = pct + '%';
  document.getElementById('count-rate').textContent = done > 0
    ? Math.round(ok / done * 100) + '%' : '—';
}

// ----------------------------------------------------------------------------
// Actions
// ----------------------------------------------------------------------------
function setStatus(id, status) {
  const state = loadState();
  state[id] = state[id] || {};
  state[id].status = status;
  saveState(state);
}
function setComment(id, comment) {
  const state = loadState();
  state[id] = state[id] || {};
  state[id].comment = comment;
  saveState(state);
}

function allOkSection(cat) {
  const state = loadState();
  TESTS.filter(t => t.categorie === cat).forEach(t => {
    state[t.id] = state[t.id] || {};
    state[t.id].status = 'OK';
  });
  saveState(state);
  render();
  showToast(`Section « ${cat} » : tous les tests validés OK`);
}

function resetSection(cat) {
  if (!confirm(`Remettre tous les tests de la section « ${cat} » à NA ?`)) return;
  const state = loadState();
  TESTS.filter(t => t.categorie === cat).forEach(t => {
    state[t.id] = state[t.id] || {};
    state[t.id].status = 'NA';
  });
  saveState(state);
  render();
}

function allOkGlobal() {
  if (!confirm(`Valider OK l'intégralité des ${TESTS.length} tests ?`)) return;
  const state = loadState();
  TESTS.forEach(t => {
    state[t.id] = state[t.id] || {};
    state[t.id].status = 'OK';
  });
  saveState(state);
  render();
  showToast('Tous les tests validés OK');
}

function resetGlobal() {
  if (!confirm('Réinitialiser tous les statuts et commentaires ?')) return;
  localStorage.removeItem(STORAGE_KEY);
  render();
}

function showToast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2200);
}

// ----------------------------------------------------------------------------
// Export
// ----------------------------------------------------------------------------
function buildPayload() {
  const state = loadState();
  return {
    tool: 'ShutterstockAnalyzer',
    version: document.getElementById('meta-version').textContent,
    testeur: document.getElementById('testeur').value,
    environnement: document.getElementById('env').value,
    date: document.getElementById('date').value,
    build: document.getElementById('build').value,
    generated_at: new Date().toISOString(),
    tests: TESTS.map(t => {
      const row = state[t.id] || { status: 'NA', comment: '' };
      return { ...t, statut: row.status || 'NA', commentaire: row.comment || '' };
    })
  };
}

function exportJson() {
  const data = buildPayload();
  document.getElementById('export-output').textContent = JSON.stringify(data, null, 2);
  document.getElementById('export-section').open = true;
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `validation_ihm_${data.testeur || 'anonymous'}_${new Date().toISOString().slice(0,10)}.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function exportMarkdown() {
  const data = buildPayload();
  const lines = [];
  lines.push(`# Validation IHM — ${data.tool} ${data.version}`);
  lines.push(``);
  lines.push(`- **Testeur** : ${data.testeur || '_(non renseigné)_'}`);
  lines.push(`- **Environnement** : ${data.environnement || '_(non renseigné)_'}`);
  lines.push(`- **Date** : ${data.date || data.generated_at.slice(0,10)}`);
  lines.push(`- **Build** : ${data.build || '_(non renseigné)_'}`);
  lines.push(``);
  const ok = data.tests.filter(t => t.statut === 'OK').length;
  const nok = data.tests.filter(t => t.statut === 'NOK').length;
  const na = data.tests.filter(t => t.statut === 'NA').length;
  const done = ok + nok;
  lines.push(`## Synthèse`);
  lines.push(``);
  lines.push(`| Total | OK | NOK | NA | Taux OK |`);
  lines.push(`|---|---|---|---|---|`);
  lines.push(`| ${data.tests.length} | ${ok} | ${nok} | ${na} | ${done > 0 ? Math.round(ok/done*100)+'%' : '—'} |`);
  lines.push(``);

  // Group by category
  const byCat = {};
  data.tests.forEach(t => { (byCat[t.categorie] = byCat[t.categorie] || []).push(t); });
  Object.entries(byCat).forEach(([cat, tests]) => {
    const co = tests.filter(t => t.statut === 'OK').length;
    lines.push(`## ${cat} (${co}/${tests.length} OK)`);
    lines.push(``);
    lines.push(`| ID | Fonctionnalité | Sévérité | Statut | Commentaire |`);
    lines.push(`|---|---|---|---|---|`);
    tests.forEach(t => {
      lines.push(`| ${t.id} | ${t.fonctionnalite} | ${t.severite} | ${t.statut} | ${(t.commentaire || '').replace(/\\|/g, '\\\\|').replace(/\\n/g, ' ')} |`);
    });
    lines.push(``);
  });

  if (nok > 0) {
    lines.push(`## Anomalies détectées (NOK)`);
    data.tests.filter(t => t.statut === 'NOK').forEach(t => {
      lines.push(``);
      lines.push(`### ${t.id} — ${t.fonctionnalite} (${t.severite})`);
      lines.push(`- **Pré-requis** : ${t.prerequis}`);
      lines.push(`- **Action** : ${t.description}`);
      lines.push(`- **Attendu** : ${t.attendu}`);
      lines.push(`- **Commentaire** : ${t.commentaire || '_(à compléter)_'}`);
    });
  }

  document.getElementById('export-output').textContent = lines.join('\\n');
  document.getElementById('export-section').open = true;
}

// ----------------------------------------------------------------------------
// Event delegation
// ----------------------------------------------------------------------------
function init() {
  // Restore meta
  const state = loadState();
  if (state._meta) {
    document.getElementById('testeur').value = state._meta.testeur || '';
    document.getElementById('env').value = state._meta.env || '';
    document.getElementById('date').value = state._meta.date || '';
    document.getElementById('build').value = state._meta.build || '';
  }
  if (!document.getElementById('date').value) {
    document.getElementById('date').value = new Date().toISOString().slice(0,10);
  }
  ['testeur','env','date','build'].forEach(id => {
    document.getElementById(id).addEventListener('input', () => {
      const s = loadState();
      s._meta = s._meta || {};
      s._meta[id] = document.getElementById(id).value;
      saveState(s);
    });
  });

  // Delegated clicks for status / section actions
  document.getElementById('sections-container').addEventListener('click', (e) => {
    const status = e.target.dataset.status;
    if (status && e.target.dataset.id) {
      setStatus(e.target.dataset.id, status);
      render();
      return;
    }
    const action = e.target.dataset.action;
    if (action === 'all-ok') {
      e.preventDefault(); e.stopPropagation();
      allOkSection(e.target.dataset.cat);
    } else if (action === 'reset') {
      e.preventDefault(); e.stopPropagation();
      resetSection(e.target.dataset.cat);
    }
  });

  // Comment inputs
  document.getElementById('sections-container').addEventListener('input', (e) => {
    if (e.target.matches('textarea.comment')) {
      setComment(e.target.dataset.id, e.target.value);
      updateSummary();
    }
  });

  // Toolbar
  document.getElementById('search').addEventListener('input', render);
  document.getElementById('filter-status').addEventListener('change', render);
  document.getElementById('filter-severity').addEventListener('change', render);
  document.getElementById('btn-expand-all').addEventListener('click', () => {
    document.querySelectorAll('.section').forEach(s => s.open = true);
  });
  document.getElementById('btn-collapse-all').addEventListener('click', () => {
    document.querySelectorAll('.section').forEach(s => s.open = false);
  });
  document.getElementById('btn-all-ok-global').addEventListener('click', allOkGlobal);
  document.getElementById('btn-reset-global').addEventListener('click', resetGlobal);
  document.getElementById('btn-export-json').addEventListener('click', exportJson);
  document.getElementById('btn-export-md').addEventListener('click', exportMarkdown);

  render();
}

init();
</script>
</body>
</html>
"""


def main() -> int:
    tests = _load_tests()
    payload = []
    for t in tests:
        tid, cat, req, feat, desc, prereq, inp, expected, severity = t
        payload.append({
            "id": tid,
            "categorie": cat,
            "exigence": req,
            "fonctionnalite": feat,
            "description": desc,
            "prerequis": prereq,
            "donnee": inp,
            "attendu": expected,
            "severite": severity,
        })

    nb_sections = len({t[1] for t in tests})

    out = (HTML_TEMPLATE
           .replace("__TESTS_JSON__", json.dumps(payload, ensure_ascii=False))
           .replace("__CATEGORY_COLORS__", json.dumps(CATEGORY_COLOR, ensure_ascii=False))
           .replace("__CATEGORY_ICONS__", json.dumps(CATEGORY_ICON, ensure_ascii=False))
           .replace("__NB_TESTS__", str(len(tests)))
           .replace("__NB_SECTIONS__", str(nb_sections)))

    OUT.write_text(out, encoding="utf-8")
    size_kb = OUT.stat().st_size / 1024
    print(f"OK -> {OUT}  ({len(tests)} tests · {nb_sections} sections · {size_kb:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
