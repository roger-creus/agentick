"""Shared utilities for LLM/VLM debugging tools.

Provides ANSI-to-HTML conversion, image encoding, message serialisation,
and the parameterised HTML template used by both llm_debugging and vlm_debugging.
"""

from __future__ import annotations

import base64
import re
from io import BytesIO
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# ANSI → HTML converter
# ---------------------------------------------------------------------------
_ANSI_RE = re.compile(r"\x1b\[([0-9;]*)m")

_FG_COLORS = {
    "30": "#1e1e1e", "31": "#e74c3c", "32": "#2ecc71", "33": "#f39c12",
    "34": "#3498db", "35": "#9b59b6", "36": "#1abc9c", "37": "#ecf0f1",
    "90": "#7f8c8d", "91": "#ff6b6b", "92": "#55efc4", "93": "#ffeaa7",
    "94": "#74b9ff", "95": "#a29bfe", "96": "#81ecec", "97": "#ffffff",
}

_BG_COLORS = {
    "40": "#1e1e1e", "41": "#e74c3c", "42": "#2ecc71", "43": "#f39c12",
    "44": "#3498db", "45": "#9b59b6", "46": "#1abc9c", "47": "#ecf0f1",
    "100": "#7f8c8d", "101": "#ff6b6b", "102": "#55efc4", "103": "#ffeaa7",
    "104": "#74b9ff", "105": "#a29bfe", "106": "#81ecec", "107": "#ffffff",
}


def _html_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def ansi_to_html(text: str) -> str:
    """Convert ANSI escape codes to HTML spans with inline styles."""
    result: list[str] = []
    last_end = 0
    open_spans = 0

    for m in _ANSI_RE.finditer(text):
        result.append(_html_escape(text[last_end:m.start()]))
        last_end = m.end()

        codes = [c for c in m.group(1).split(";") if c]
        if not codes or codes == ["0"]:
            result.append("</span>" * open_spans)
            open_spans = 0
            continue

        styles: list[str] = []
        for code in codes:
            if code in _FG_COLORS:
                styles.append(f"color:{_FG_COLORS[code]}")
            elif code in _BG_COLORS:
                styles.append(f"background:{_BG_COLORS[code]}")
            elif code == "1":
                styles.append("font-weight:bold")
            elif code == "2":
                styles.append("opacity:0.7")
            elif code == "4":
                styles.append("text-decoration:underline")

        if styles:
            result.append(f'<span style="{";".join(styles)}">')
            open_spans += 1

    result.append(_html_escape(text[last_end:]))
    result.append("</span>" * open_spans)
    return "".join(result)


# ---------------------------------------------------------------------------
# numpy → base64 PNG
# ---------------------------------------------------------------------------
def np_to_b64(img: np.ndarray) -> str:
    """Convert a numpy RGB array to a base64-encoded PNG string."""
    from PIL import Image

    pil = Image.fromarray(img.astype(np.uint8))
    buf = BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Message serialisation
# ---------------------------------------------------------------------------
def serialise_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Flatten messages into {role, content} strings for JSON serialisation."""
    out = []
    for msg in messages:
        c = msg["content"]
        if isinstance(c, list):
            parts = [b.get("text", "[image]") for b in c if isinstance(b, dict)]
            c = "\n".join(parts)
        out.append({"role": msg["role"], "content": str(c)})
    return out


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------
def capture_renders(
    env: Any,
    primary_obs: Any,
    modalities: list[str],
) -> dict[str, Any]:
    """Capture renders in all requested modalities.

    Returns a dict keyed by modality name. Values are either str (ascii/language)
    or numpy arrays (rgb_array/rgb_array_flat).
    """
    renders: dict[str, Any] = {}
    primary_mode = getattr(env, "render_mode", None)

    for mode in modalities:
        if mode == primary_mode:
            renders[mode] = primary_obs
        else:
            try:
                renders[mode] = env.render_in_mode(mode)
            except Exception:
                pass  # mode not available for this env
    return renders


def renders_to_html_data(renders: dict[str, Any]) -> dict[str, str]:
    """Convert raw renders to HTML-ready data (base64 images, HTML-escaped text)."""
    data: dict[str, str] = {}
    for mode, value in renders.items():
        if mode in ("rgb_array", "rgb_array_flat"):
            if isinstance(value, np.ndarray):
                data[mode] = np_to_b64(value)
        elif mode == "ascii":
            text = str(value)
            data[f"{mode}_html"] = ansi_to_html(text)
            data[f"{mode}_clean"] = re.sub(r"\x1b\[[0-9;]*m", "", text)
        elif mode in ("language", "language_structured"):
            data[mode] = str(value)
    return data


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------
def generate_html(
    episodes: list[dict[str, Any]],
    output: str,
    *,
    title: str = "Agentick LLM Debugger",
    extra_sections_js: str = "",
) -> None:
    """Write a self-contained HTML viewer.

    Args:
        episodes: Episode data (list of episode dicts with steps).
        output: Output file path.
        title: Page title.
        extra_sections_js: Additional JS function body appended inside
            ``renderExtraSections(st, ep)`` — return HTML string for extra panels.
    """
    import json
    from pathlib import Path

    data_json = json.dumps(episodes, ensure_ascii=False)
    html = _HTML_TEMPLATE.replace("/*__DATA__*/", f"const DATA = {data_json};")
    html = html.replace("__TITLE__", title)
    html = html.replace("/*__EXTRA_SECTIONS__*/", extra_sections_js)
    Path(output).write_text(html, encoding="utf-8")
    size_mb = Path(output).stat().st_size / 1024 / 1024
    print(f"\nViewer saved to: {output}  ({size_mb:.1f} MB)")


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
:root {
  --bg: #0d1117;
  --surface: #161b22;
  --surface2: #1c2333;
  --border: #30363d;
  --text: #e6edf3;
  --text-dim: #8b949e;
  --accent: #58a6ff;
  --green: #3fb950;
  --red: #f85149;
  --yellow: #d29922;
  --purple: #bc8cff;
}
* { margin:0; padding:0; box-sizing:border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
  background: var(--bg);
  color: var(--text);
  min-height: 100vh;
}

/* Header */
.header {
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  padding: 12px 20px;
  display: flex;
  align-items: center;
  gap: 16px;
  flex-wrap: wrap;
}
.header h1 {
  font-size: 16px;
  font-weight: 600;
  color: var(--accent);
}
.header .meta {
  font-size: 13px;
  color: var(--text-dim);
}
.header .meta b { color: var(--text); font-weight: 500; }

/* Episode bar */
.episode-bar {
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  padding: 8px 20px;
  display: flex;
  gap: 8px;
  align-items: center;
  flex-wrap: wrap;
}
.ep-btn {
  border: 1px solid var(--border);
  background: var(--surface2);
  color: var(--text);
  padding: 5px 14px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 13px;
  transition: all 0.15s;
}
.ep-btn:hover { border-color: var(--accent); }
.ep-btn.active { background: var(--accent); color: #000; border-color: var(--accent); font-weight: 600; }
.ep-btn.success { border-left: 3px solid var(--green); }
.ep-btn.fail { border-left: 3px solid var(--red); }

/* Step controls */
.step-bar {
  background: var(--surface2);
  border-bottom: 1px solid var(--border);
  padding: 10px 20px;
  display: flex;
  align-items: center;
  gap: 16px;
  flex-wrap: wrap;
}
.step-bar label { font-size: 13px; color: var(--text-dim); }
.step-bar input[type=range] { flex: 1; min-width: 200px; accent-color: var(--accent); }
.step-info {
  font-size: 13px;
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
}
.step-info span { white-space: nowrap; }
.badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 12px;
  font-weight: 600;
}
.badge-action { background: #1f3a5f; color: var(--accent); }
.badge-reward { background: #1a3a1a; color: var(--green); }
.badge-tokens { background: #2d2040; color: var(--purple); }
.badge-time { background: #3a2a10; color: var(--yellow); }

/* Main layout */
.main {
  display: grid;
  grid-template-columns: minmax(300px, 480px) 1fr;
  height: calc(100vh - 130px);
  overflow: hidden;
}

/* Left panel: renders */
.renders {
  border-right: 1px solid var(--border);
  overflow-y: auto;
  padding: 12px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.render-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
  max-height: 70vh;
}
.render-card h3 {
  font-size: 12px;
  padding: 6px 10px;
  background: var(--surface2);
  border-bottom: 1px solid var(--border);
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
.ascii-content {
  padding: 8px 10px;
  font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', 'Menlo', monospace;
  font-size: 14px;
  line-height: 1.35;
  white-space: pre;
  overflow-y: scroll;
  overflow-x: auto;
  max-height: 60vh;
  background: #000;
  color: #ccc;
}
.lang-content {
  padding: 10px 12px;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  font-size: 13px;
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-word;
  background: var(--surface);
  color: var(--text);
  overflow-y: scroll;
  max-height: 60vh;
}
.iso-content { text-align: center; background: #000; overflow-y: scroll; max-height: 60vh; }
.iso-content img { max-width: 100%; height: auto; image-rendering: auto; }

/* Right panel: LLM interaction */
.llm-panel {
  overflow-y: auto;
  padding: 12px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

/* Collapsible sections */
.section {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
  max-height: 70vh;
}
.section-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  background: var(--surface2);
  cursor: pointer;
  user-select: none;
  font-size: 13px;
  font-weight: 600;
  border-bottom: 1px solid var(--border);
}
.section-header:hover { background: #22293a; }
.section-header .arrow { transition: transform 0.2s; font-size: 10px; }
.section-header .arrow.open { transform: rotate(90deg); }
.section-header .count {
  margin-left: auto;
  font-size: 11px;
  color: var(--text-dim);
  font-weight: 400;
}
.section-body { display: none; }
.section-body.open { display: block; max-height: 60vh; overflow-y: scroll; }
.section-body pre {
  padding: 10px 12px;
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  font-size: 12px;
  line-height: 1.5;
  white-space: pre-wrap;
  word-break: break-word;
  color: var(--text);
}

/* Chat-style messages */
.msg {
  padding: 8px 12px;
  border-bottom: 1px solid var(--border);
}
.msg:last-child { border-bottom: none; }
.msg-role {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 4px;
}
.msg-role.system { color: var(--yellow); }
.msg-role.user { color: var(--accent); }
.msg-role.assistant { color: var(--green); }
.msg-content {
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  font-size: 12px;
  line-height: 1.5;
  white-space: pre-wrap;
  word-break: break-word;
  color: var(--text);
  max-height: 60vh;
  overflow-y: scroll;
}
.msg.current-step {
  background: rgba(88, 166, 255, 0.06);
  border-left: 3px solid var(--accent);
}

/* Response card */
.response-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
  max-height: 70vh;
}
.response-label {
  font-size: 12px;
  font-weight: 600;
  padding: 6px 10px;
  background: var(--surface2);
  border-bottom: 1px solid var(--border);
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
.response-body {
  padding: 10px 12px;
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  font-size: 12px;
  line-height: 1.5;
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 60vh;
  overflow-y: scroll;
}
.reasoning-text { color: var(--purple); }
.action-line { color: var(--green); font-weight: bold; }

/* VLM description card */
.vlm-desc-card {
  background: var(--surface);
  border: 1px solid #3a2a10;
  border-radius: 8px;
  overflow: hidden;
  max-height: 70vh;
}
.vlm-desc-label {
  font-size: 12px;
  font-weight: 600;
  padding: 6px 10px;
  background: #2a2010;
  border-bottom: 1px solid #3a2a10;
  color: var(--yellow);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
.vlm-desc-body {
  padding: 10px 12px;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  font-size: 13px;
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-word;
  color: var(--text);
  max-height: 50vh;
  overflow-y: scroll;
}
.vlm-desc-tokens {
  padding: 4px 10px 8px;
  font-size: 11px;
  color: var(--text-dim);
}

/* Playback controls */
.play-controls {
  display: flex;
  gap: 6px;
  align-items: center;
}
.play-btn {
  border: 1px solid var(--border);
  background: var(--surface2);
  color: var(--text);
  width: 32px;
  height: 28px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  display: flex;
  align-items: center;
  justify-content: center;
}
.play-btn:hover { border-color: var(--accent); }
.play-btn.active { background: var(--accent); color: #000; }

/* Scrollbar styling */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #484f58; }

@media (max-width: 900px) {
  .main { grid-template-columns: 1fr; height: auto; }
  .renders { border-right: none; border-bottom: 1px solid var(--border); }
}
</style>
</head>
<body>

<div class="header">
  <h1 id="hTitle">__TITLE__</h1>
  <span class="meta" id="hMeta"></span>
</div>

<div class="episode-bar" id="epBar"></div>

<div class="step-bar">
  <div class="play-controls">
    <button class="play-btn" onclick="prevStep()" title="Previous">&#9664;</button>
    <button class="play-btn" id="playBtn" onclick="togglePlay()" title="Play/Pause">&#9654;</button>
    <button class="play-btn" onclick="nextStep()" title="Next">&#9654;</button>
  </div>
  <label>Step</label>
  <input type="range" id="slider" min="0" max="0" value="0" oninput="setStep(+this.value)">
  <div class="step-info" id="stepInfo"></div>
</div>

<div class="main">
  <div class="renders" id="rendersPanel"></div>
  <div class="llm-panel" id="llmPanel"></div>
</div>

<script>
/*__DATA__*/

let curEp = 0;
let curStep = 0;
let playTimer = null;

function init() {
  const ep = DATA[0];
  document.getElementById('hTitle').textContent = `__TITLE__ \u2014 ${ep.task}`;
  document.getElementById('hMeta').innerHTML =
    `Model: <b>${ep.model}</b> &nbsp;|&nbsp; Harness: <b>${ep.harness}</b> &nbsp;|&nbsp; Difficulty: <b>${ep.difficulty}</b>`;

  const bar = document.getElementById('epBar');
  DATA.forEach((ep, i) => {
    const btn = document.createElement('button');
    btn.className = `ep-btn ${ep.success ? 'success' : 'fail'}`;
    btn.textContent = `Ep ${i+1} (seed ${ep.seed}) \u2014 ${ep.success ? '\u2713' : '\u2717'} ${ep.n_steps} steps  r=${ep.total_reward}`;
    btn.onclick = () => selectEpisode(i);
    bar.appendChild(btn);
  });

  selectEpisode(0);
}

function selectEpisode(idx) {
  curEp = idx;
  curStep = 0;
  stopPlay();

  document.querySelectorAll('.ep-btn').forEach((b, i) => {
    b.classList.toggle('active', i === idx);
  });

  const ep = DATA[idx];
  const slider = document.getElementById('slider');
  slider.max = ep.steps.length - 1;
  slider.value = 0;

  renderStep();
}

function setStep(s) {
  curStep = s;
  document.getElementById('slider').value = s;
  renderStep();
}

function prevStep() { if (curStep > 0) setStep(curStep - 1); }
function nextStep() {
  const ep = DATA[curEp];
  if (curStep < ep.steps.length - 1) setStep(curStep + 1);
  else stopPlay();
}

function togglePlay() {
  if (playTimer) { stopPlay(); return; }
  playTimer = setInterval(() => nextStep(), 600);
  document.getElementById('playBtn').classList.add('active');
}

function stopPlay() {
  if (playTimer) { clearInterval(playTimer); playTimer = null; }
  document.getElementById('playBtn').classList.remove('active');
}

function renderStep() {
  const ep = DATA[curEp];
  const st = ep.steps[curStep];

  // Step info badges
  const info = document.getElementById('stepInfo');
  info.innerHTML = `
    <span><b>Step ${st.step + 1}/${ep.n_steps}</b></span>
    <span class="badge badge-action">${st.action_name} (${st.parsed_action})</span>
    <span class="badge badge-reward">r=${st.reward} cum=${st.cumulative_reward}</span>
    <span class="badge badge-tokens">${st.input_tokens}+${st.output_tokens} tok</span>
    <span class="badge badge-time">${st.latency}s</span>
    ${st.done ? (st.terminated ? '<span class="badge" style="background:#1a3a1a;color:var(--green)">TERMINATED</span>' : '<span class="badge" style="background:#3a1a1a;color:var(--red)">TRUNCATED</span>') : ''}
  `;

  // Build render cards dynamically
  const rendersPanel = document.getElementById('rendersPanel');
  let rhtml = '';
  const r = st.renders || {};

  if (r.ascii_html) {
    rhtml += `<div class="render-card"><h3>ASCII Render</h3><div class="ascii-content">${r.ascii_html}</div></div>`;
  }
  if (r.language) {
    rhtml += `<div class="render-card"><h3>Language Render</h3><div class="lang-content">${escHtml(r.language)}</div></div>`;
  }
  if (r.language_structured) {
    rhtml += `<div class="render-card"><h3>Language (Structured)</h3><div class="lang-content">${escHtml(r.language_structured)}</div></div>`;
  }
  if (r.rgb_array) {
    rhtml += `<div class="render-card"><h3>Isometric Render</h3><div class="iso-content"><img src="data:image/png;base64,${r.rgb_array}" alt="isometric"></div></div>`;
  }
  if (r.rgb_array_flat) {
    rhtml += `<div class="render-card"><h3>Flat 2D Render</h3><div class="iso-content"><img src="data:image/png;base64,${r.rgb_array_flat}" alt="flat 2d"></div></div>`;
  }

  // Fallback for legacy data format (no renders dict)
  if (!rhtml && st.ascii_html) {
    rhtml += `<div class="render-card"><h3>ASCII Render</h3><div class="ascii-content">${st.ascii_html}</div></div>`;
  }
  if (!rhtml && st.iso_b64) {
    rhtml += `<div class="render-card"><h3>Isometric Render</h3><div class="iso-content"><img src="data:image/png;base64,${st.iso_b64}" alt="isometric"></div></div>`;
  }

  rendersPanel.innerHTML = rhtml;

  // LLM panel
  renderLLMPanel(st, ep);
}

function renderLLMPanel(st, ep) {
  const panel = document.getElementById('llmPanel');
  let html = '';

  // 1. System prompt (collapsible, closed by default)
  const sysMsg = st.messages.find(m => m.role === 'system');
  if (sysMsg) {
    html += `
    <div class="section">
      <div class="section-header" onclick="toggleSection(this)">
        <span class="arrow">\u25B6</span> System Prompt
      </div>
      <div class="section-body">
        <pre>${escHtml(sysMsg.content)}</pre>
      </div>
    </div>`;
  }

  // 2. Conversation messages
  const nonSys = st.messages.filter(m => m.role !== 'system');
  const histLabel = nonSys.length > 1
    ? `Conversation (${nonSys.length} messages \u2014 ${Math.floor((nonSys.length - 1) / 2)} history turns)`
    : `Current Observation`;

  const isOpen = nonSys.length <= 4;
  html += `
  <div class="section">
    <div class="section-header" onclick="toggleSection(this)">
      <span class="arrow ${isOpen ? 'open' : ''}">\u25B6</span> ${histLabel}
      <span class="count">${nonSys.length} msg${nonSys.length !== 1 ? 's' : ''}</span>
    </div>
    <div class="section-body ${isOpen ? 'open' : ''}">`;

  nonSys.forEach((msg, i) => {
    const isCurrent = (i === nonSys.length - 1 && msg.role === 'user');
    html += `
      <div class="msg ${isCurrent ? 'current-step' : ''}">
        <div class="msg-role ${msg.role}">${msg.role}${isCurrent ? ' (current observation)' : ''}</div>
        <div class="msg-content">${escHtml(msg.content)}</div>
      </div>`;
  });

  html += `</div></div>`;

  // 3. Extra sections (VLM description, etc.)
  html += renderExtraSections(st, ep);

  // 4. LLM Response
  const respHtml = formatResponse(st.response, st.reasoning);
  html += `
  <div class="response-card">
    <div class="response-label">LLM Response</div>
    <div class="response-body">${respHtml}</div>
  </div>`;

  panel.innerHTML = html;
}

function renderExtraSections(st, ep) {
  /*__EXTRA_SECTIONS__*/
  return '';
}

function formatResponse(response, reasoning) {
  if (!response) return '<span style="color:var(--red)">No response</span>';

  const lines = escHtml(response).split('\n');
  const formatted = lines.map(line => {
    if (/ACTION:\s*\d+/i.test(line)) {
      return `<span class="action-line">${line}</span>`;
    }
    return line;
  }).join('\n');

  if (reasoning) {
    const parts = escHtml(response).split(/ACTION:/i);
    if (parts.length >= 2) {
      return `<span class="reasoning-text">${parts[0].trim()}</span>\n<span class="action-line">ACTION:${parts.slice(1).join('ACTION:')}</span>`;
    }
  }

  return formatted;
}

function toggleSection(header) {
  const arrow = header.querySelector('.arrow');
  const body = header.nextElementSibling;
  arrow.classList.toggle('open');
  body.classList.toggle('open');
}

function escHtml(s) {
  if (!s) return '';
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

init();
</script>
</body>
</html>
"""
