"""Web-based human play interface for Agentick tasks.

Provides a fast, compact multi-modal play UI:
- Isometric pixel view (main)
- ASCII grid view
- Language description view
- State dict view
- Keyboard + button controls
"""

from __future__ import annotations

import base64
import io
import os
import re
import uuid
from typing import Any

import numpy as np

try:
    from flask import Flask, jsonify, request
except ImportError:
    Flask = None  # type: ignore[assignment,misc]


# ── Active play sessions ────────────────────────────────────────────────────

_sessions: dict[str, dict[str, Any]] = {}


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from a string."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _render_multimodal(env: Any, iso_renderer: Any = None) -> dict[str, str]:
    """Render current env state in all modalities. Reuses iso_renderer for speed."""
    unwrapped = env.unwrapped

    # ASCII
    try:
        ascii_raw = unwrapped.render_in_mode("ascii")
        ascii_text = _strip_ansi(ascii_raw) if isinstance(ascii_raw, str) else str(ascii_raw)
    except Exception:
        ascii_text = "(unavailable)"

    # Language
    try:
        lang_text = unwrapped.render_in_mode("language")
        if not isinstance(lang_text, str):
            lang_text = str(lang_text)
    except Exception:
        lang_text = "(unavailable)"

    # Pixel — reuse cached renderer, encode as JPEG for speed
    pixel_b64 = ""
    try:
        if iso_renderer is not None:
            info = unwrapped._get_info()
            rgb = iso_renderer.render(
                unwrapped.grid, unwrapped.entities, unwrapped.agent, info
            )
        else:
            rgb = unwrapped.render_in_mode("rgb_array")
        if isinstance(rgb, np.ndarray):
            from PIL import Image

            img = Image.fromarray(rgb.astype(np.uint8))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            pixel_b64 = "data:image/jpeg;base64," + base64.b64encode(
                buf.getvalue()
            ).decode()
    except Exception:
        pass

    # State dict (compact)
    state_text = ""
    try:
        sd = unwrapped.render_in_mode("state_dict")
        if isinstance(sd, dict):
            lines = []
            for k, v in sd.items():
                if isinstance(v, np.ndarray):
                    lines.append(f"{k}: ndarray{v.shape}")
                elif isinstance(v, dict):
                    lines.append(f"{k}: {{{', '.join(f'{sk}: {sv}' for sk, sv in v.items())}}}")
                elif isinstance(v, list) and len(v) > 5:
                    lines.append(f"{k}: [{len(v)} items]")
                else:
                    lines.append(f"{k}: {v}")
            state_text = "\n".join(lines)
    except Exception:
        state_text = "(unavailable)"

    return {"ascii": ascii_text, "language": lang_text, "pixel": pixel_b64, "state": state_text}


# ── Action mapping (matches ActionType enum in core/types.py) ───────────────

_ACTION_MAP = {
    "noop": 0,
    "move_up": 1,
    "move_down": 2,
    "move_left": 3,
    "move_right": 4,
    "interact": 5,
    "rotate_left": 6,
    "rotate_right": 7,
    "move_forward": 8,
}


def _action_name_to_int(name: str) -> int:
    return _ACTION_MAP.get(name.lower().strip(), 0)


# ── HTML ────────────────────────────────────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Agentick — Human Play</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0d1117;--s1:#161b22;--s2:#1c2129;--s3:#21262d;
  --border:#30363d;--text:#e6edf3;--dim:#8b949e;--muted:#484f58;
  --accent:#58a6ff;--green:#3fb950;--red:#f85149;--yellow:#d29922;
  --mono:'JetBrains Mono','SF Mono','Consolas',monospace;
  --sans:-apple-system,'Segoe UI',Helvetica,Arial,sans-serif;
  --r:6px;
}
html,body{height:100%;overflow:hidden}
body{font-family:var(--sans);background:var(--bg);color:var(--text);font-size:13px}

/* Layout */
.app{display:flex;flex-direction:column;height:100vh}
.topbar{display:flex;align-items:center;gap:12px;padding:6px 16px;background:var(--s1);border-bottom:1px solid var(--border);min-height:42px;flex-shrink:0}
.topbar .logo{font-weight:700;font-size:15px;color:var(--accent);letter-spacing:-.5px;white-space:nowrap}
.topbar select,.topbar button{font-size:12px;background:var(--s2);color:var(--text);border:1px solid var(--border);border-radius:var(--r);padding:4px 8px;cursor:pointer;outline:none}
.topbar select:hover,.topbar button:hover{border-color:var(--accent)}
.topbar button{padding:4px 12px}
.topbar .sep{width:1px;height:20px;background:var(--border)}
.topbar .stats{font-family:var(--mono);font-size:11px;color:var(--dim);display:flex;gap:12px;margin-left:auto}
.topbar .stats span{white-space:nowrap}
.topbar .stats .val{color:var(--text)}

/* Main area */
.main{display:flex;flex:1;min-height:0}

/* Left: pixel view */
.pixel-panel{flex:0 0 auto;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:12px;background:var(--bg);position:relative}
.pixel-panel img{max-width:100%;max-height:100%;object-fit:contain;border-radius:var(--r);image-rendering:auto}
.pixel-panel .overlay{position:absolute;top:0;left:0;right:0;bottom:0;display:flex;align-items:center;justify-content:center;background:rgba(0,0,0,.7);border-radius:var(--r);pointer-events:none;opacity:0;transition:opacity .2s}
.pixel-panel .overlay.show{opacity:1}
.pixel-panel .overlay span{font-size:28px;font-weight:700;letter-spacing:1px}
.pixel-panel .overlay .win{color:var(--green)}
.pixel-panel .overlay .lose{color:var(--red)}

/* Right panels */
.side{flex:1;display:flex;flex-direction:column;min-width:0;border-left:1px solid var(--border)}

/* Tabs */
.tabs{display:flex;border-bottom:1px solid var(--border);flex-shrink:0}
.tabs button{flex:1;padding:6px 0;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.5px;background:none;border:none;color:var(--muted);cursor:pointer;border-bottom:2px solid transparent;transition:all .15s}
.tabs button.active{color:var(--accent);border-bottom-color:var(--accent)}
.tabs button:hover{color:var(--dim)}

/* Tab content */
.tab-content{flex:1;overflow:auto;padding:10px;font-family:var(--mono);font-size:11.5px;line-height:1.5;white-space:pre-wrap;word-break:break-word;color:var(--dim);background:var(--s1)}
.tab-content.lang{font-family:var(--sans);font-size:12.5px;line-height:1.6;color:var(--text)}

/* Controls bar */
.controls{display:flex;align-items:center;justify-content:center;gap:6px;padding:8px 16px;background:var(--s1);border-top:1px solid var(--border);flex-shrink:0}
.controls .dpad{display:grid;grid-template-columns:repeat(3,1fr);gap:3px}
.controls .dpad button{width:38px;height:32px;font-size:16px;background:var(--s2);color:var(--text);border:1px solid var(--border);border-radius:var(--r);cursor:pointer;display:flex;align-items:center;justify-content:center;transition:all .1s}
.controls .dpad button:hover{background:var(--s3);border-color:var(--accent)}
.controls .dpad button:active,.controls .dpad button.flash{background:var(--accent);color:#fff;border-color:var(--accent)}
.controls .dpad button.empty{visibility:hidden}
.controls .act-btn{height:32px;padding:0 14px;font-size:11px;font-weight:600;background:var(--s2);color:var(--text);border:1px solid var(--border);border-radius:var(--r);cursor:pointer;font-family:var(--sans);text-transform:uppercase;letter-spacing:.5px;transition:all .1s}
.controls .act-btn:hover{background:var(--s3);border-color:var(--accent)}
.controls .act-btn:active,.controls .act-btn.flash{background:var(--accent);color:#fff}
.controls .act-btn .key{font-size:9px;color:var(--muted);margin-left:4px;font-weight:400}

/* Responsive */
@media(max-width:700px){
  .main{flex-direction:column}
  .pixel-panel{max-height:45vh}
  .side{border-left:none;border-top:1px solid var(--border)}
}

/* Start screen */
.start-screen{display:flex;align-items:center;justify-content:center;flex:1;flex-direction:column;gap:16px;color:var(--dim)}
.start-screen p{font-size:14px}
</style>
</head>
<body>
<div class="app">
  <!-- Top bar -->
  <div class="topbar">
    <span class="logo">agentick</span>
    <div class="sep"></div>
    <select id="task-select"><option value="">Select task...</option></select>
    <select id="diff-select">
      <option value="easy">Easy</option>
      <option value="medium">Medium</option>
      <option value="hard">Hard</option>
      <option value="expert">Expert</option>
    </select>
    <button id="btn-start" title="Start / Restart">Play</button>
    <button id="btn-reset" title="Reset same task" style="display:none">Reset</button>
    <div class="stats" id="stats" style="display:none">
      <span>Step <span class="val" id="st-step">0</span></span>
      <span>Reward <span class="val" id="st-reward">0.00</span></span>
    </div>
  </div>

  <!-- Main -->
  <div class="main">
    <!-- Pixel view -->
    <div class="pixel-panel" id="pixel-panel">
      <div class="start-screen" id="start-screen">
        <p>Select a task and press <b>Play</b> to begin</p>
      </div>
      <img id="pixel-img" style="display:none" alt="Game view">
      <div class="overlay" id="overlay">
        <span id="overlay-text"></span>
      </div>
    </div>

    <!-- Side panels -->
    <div class="side">
      <div class="tabs">
        <button class="active" data-tab="ascii">ASCII</button>
        <button data-tab="lang">Language</button>
        <button data-tab="state">State</button>
      </div>
      <div class="tab-content" id="tab-ascii"></div>
      <div class="tab-content lang" id="tab-lang" style="display:none"></div>
      <div class="tab-content" id="tab-state" style="display:none"></div>
    </div>
  </div>

  <!-- Controls -->
  <div class="controls">
    <div class="dpad">
      <button class="empty"></button>
      <button data-act="move_up" id="k-up" title="W / ArrowUp">&#9650;</button>
      <button class="empty"></button>
      <button data-act="move_left" id="k-left" title="A / ArrowLeft">&#9664;</button>
      <button data-act="noop" id="k-noop" title="Space">&#183;</button>
      <button data-act="move_right" id="k-right" title="D / ArrowRight">&#9654;</button>
      <button class="empty"></button>
      <button data-act="move_down" id="k-down" title="S / ArrowDown">&#9660;</button>
      <button class="empty"></button>
    </div>
    <button class="act-btn" data-act="interact" id="k-interact" title="E">Interact <span class="key">E</span></button>
    <button class="act-btn" data-act="rotate_left" id="k-rotl" title="Q">Rot L <span class="key">Q</span></button>
    <button class="act-btn" data-act="rotate_right" id="k-rotr" title="R">Rot R <span class="key">R</span></button>
  </div>
</div>

<script>
const $ = s => document.querySelector(s);
const taskSel = $('#task-select'), diffSel = $('#diff-select');
const btnStart = $('#btn-start'), btnReset = $('#btn-reset');
const pixelImg = $('#pixel-img'), startScreen = $('#start-screen');
const overlay = $('#overlay'), overlayText = $('#overlay-text');
const stats = $('#stats'), stStep = $('#st-step'), stReward = $('#st-reward');
const tabAscii = $('#tab-ascii'), tabLang = $('#tab-lang'), tabState = $('#tab-state');
let sessionId = null, busy = false, done = false;

// Tabs
document.querySelectorAll('.tabs button').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tabs button').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const t = btn.dataset.tab;
    tabAscii.style.display = t === 'ascii' ? '' : 'none';
    tabLang.style.display = t === 'lang' ? '' : 'none';
    tabState.style.display = t === 'state' ? '' : 'none';
  });
});

// Load tasks
fetch('/api/tasks').then(r=>r.json()).then(tasks => {
  // Group by category
  const cats = {};
  tasks.forEach(t => {
    const c = t.category || 'other';
    if (!cats[c]) cats[c] = [];
    cats[c].push(t);
  });
  for (const [cat, list] of Object.entries(cats).sort()) {
    const og = document.createElement('optgroup');
    og.label = cat.replace(/_/g,' ');
    list.forEach(t => {
      const o = document.createElement('option');
      o.value = t.name;
      o.textContent = t.name.replace('-v0','');
      o.title = t.summary || '';
      og.appendChild(o);
    });
    taskSel.appendChild(og);
  }
});

function updateUI(data) {
  if (data.pixel) {
    pixelImg.src = data.pixel;
    pixelImg.style.display = '';
    startScreen.style.display = 'none';
  }
  tabAscii.textContent = data.ascii || '';
  tabLang.textContent = data.language || '';
  tabState.textContent = data.state || '';
  stStep.textContent = data.steps ?? 0;
  stReward.textContent = (data.total_reward ?? 0).toFixed(2);
}

async function startTask() {
  const task = taskSel.value;
  if (!task) return;
  busy = true; done = false;
  overlay.classList.remove('show');
  const r = await fetch('/api/start', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({task_id: task, difficulty: diffSel.value})
  });
  const d = await r.json();
  if (d.success) {
    sessionId = d.session_id;
    updateUI(d);
    stats.style.display = '';
    btnReset.style.display = '';
  }
  busy = false;
}

async function resetTask() {
  if (!sessionId) return;
  busy = true; done = false;
  overlay.classList.remove('show');
  const r = await fetch('/api/reset', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({session_id: sessionId})
  });
  const d = await r.json();
  if (d.success) updateUI(d);
  busy = false;
}

async function step(action) {
  if (!sessionId || busy || done) return;
  busy = true;
  const r = await fetch('/api/step', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({session_id: sessionId, action})
  });
  const d = await r.json();
  updateUI(d);
  if (d.terminated || d.truncated) {
    done = true;
    overlay.classList.add('show');
    if (d.success) {
      overlayText.className = 'win';
      overlayText.textContent = 'SUCCESS';
    } else {
      overlayText.className = 'lose';
      overlayText.textContent = d.truncated ? 'TIME UP' : 'FAILED';
    }
  }
  busy = false;
}

btnStart.addEventListener('click', startTask);
btnReset.addEventListener('click', resetTask);

// D-pad + action buttons
document.querySelectorAll('[data-act]').forEach(btn => {
  btn.addEventListener('click', () => step(btn.dataset.act));
});

// Keyboard
const keyMap = {
  ArrowUp:'move_up', ArrowDown:'move_down', ArrowLeft:'move_left', ArrowRight:'move_right',
  w:'move_up', s:'move_down', a:'move_left', d:'move_right',
  W:'move_up', S:'move_down', A:'move_left', D:'move_right',
  e:'interact', E:'interact',
  q:'rotate_left', Q:'rotate_left',
  r:'rotate_right', R:'rotate_right',
  ' ':'noop',
};
const keyBtnMap = {
  ArrowUp:'k-up', ArrowDown:'k-down', ArrowLeft:'k-left', ArrowRight:'k-right',
  w:'k-up', s:'k-down', a:'k-left', d:'k-right',
  W:'k-up', S:'k-down', A:'k-left', D:'k-right',
  e:'k-interact', E:'k-interact',
  q:'k-rotl', Q:'k-rotl',
  r:'k-rotr', R:'k-rotr',
  ' ':'k-noop',
};

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'SELECT') return;
  const act = keyMap[e.key];
  if (act) {
    e.preventDefault();
    step(act);
    const bid = keyBtnMap[e.key];
    if (bid) {
      const b = document.getElementById(bid);
      if (b) { b.classList.add('flash'); setTimeout(() => b.classList.remove('flash'), 120); }
    }
  }
  // Enter = start/reset
  if (e.key === 'Enter') {
    e.preventDefault();
    if (done || !sessionId) startTask(); else resetTask();
  }
});
</script>
</body>
</html>"""


class PlayWebApp:
    """Minimal web app for human play of Agentick tasks."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8080):
        if Flask is None:
            raise ImportError("Flask is required: pip install flask")
        self.app = Flask(__name__)
        self.app.secret_key = os.urandom(24)
        self.host = host
        self.port = port
        self._setup_routes()

    def _setup_routes(self):
        @self.app.route("/")
        def index():
            return _HTML

        @self.app.route("/api/tasks")
        def api_tasks():
            from agentick.tasks.descriptions import get_all_task_descriptions

            descs = get_all_task_descriptions()
            return jsonify([
                {"name": n, "category": d.category, "summary": d.summary}
                for n, d in sorted(descs.items())
            ])

        @self.app.route("/api/start", methods=["POST"])
        def api_start():
            import agentick
            from agentick.rendering.iso_renderer import IsometricRenderer

            data = request.get_json(force=True)
            task_id = data.get("task_id", "GoToGoal-v0")
            difficulty = data.get("difficulty", "easy")
            seed = data.get("seed")

            try:
                env = agentick.make(task_id, difficulty=difficulty, render_mode="rgb_array")
                reset_kw: dict[str, Any] = {}
                if seed is not None:
                    reset_kw["seed"] = int(seed)
                env.reset(**reset_kw)

                # Create a cached iso renderer for this session
                iso = IsometricRenderer(output_size=(512, 512))

                session_id = str(uuid.uuid4())[:8]
                _sessions[session_id] = {
                    "env": env,
                    "iso": iso,
                    "task_id": task_id,
                    "difficulty": difficulty,
                    "steps": 0,
                    "total_reward": 0.0,
                    "done": False,
                }

                renders = _render_multimodal(env, iso)
                return jsonify({
                    "success": True,
                    "session_id": session_id,
                    **renders,
                    "steps": 0,
                    "total_reward": 0.0,
                })
            except Exception as e:
                return jsonify({"success": False, "error": str(e)}), 400

        @self.app.route("/api/step", methods=["POST"])
        def api_step():
            data = request.get_json(force=True)
            action_name = data.get("action", "noop")
            session_id = data.get("session_id")

            session = _sessions.get(session_id) if session_id else None
            if session is None and _sessions:
                session = next(reversed(_sessions.values()))

            if session is None:
                return jsonify({"error": "No active session"}), 400

            if session["done"]:
                renders = _render_multimodal(session["env"], session.get("iso"))
                return jsonify({
                    **renders,
                    "reward": 0,
                    "terminated": True,
                    "truncated": False,
                    "success": False,
                    "steps": session["steps"],
                    "total_reward": session["total_reward"],
                })

            action_int = _action_name_to_int(action_name)
            env = session["env"]
            _obs, reward, terminated, truncated, info = env.step(action_int)
            session["steps"] += 1
            session["total_reward"] += reward
            session["done"] = terminated or truncated

            renders = _render_multimodal(env, session.get("iso"))
            success = info.get("success", False) if terminated else False

            return jsonify({
                **renders,
                "reward": float(reward),
                "terminated": terminated,
                "truncated": truncated,
                "success": success,
                "steps": session["steps"],
                "total_reward": float(session["total_reward"]),
            })

        @self.app.route("/api/reset", methods=["POST"])
        def api_reset():
            data = request.get_json(force=True) if request.data else {}
            session_id = data.get("session_id") if data else None

            session = _sessions.get(session_id) if session_id else None
            if session is None and _sessions:
                session = next(reversed(_sessions.values()))
            if session is None:
                return jsonify({"error": "No active session"}), 400

            env = session["env"]
            env.reset()
            session["steps"] = 0
            session["total_reward"] = 0.0
            session["done"] = False

            renders = _render_multimodal(env, session.get("iso"))
            return jsonify({"success": True, **renders, "steps": 0, "total_reward": 0.0})

        @self.app.route("/api/quit", methods=["POST"])
        def api_quit():
            data = request.get_json(force=True) if request.data else {}
            session_id = data.get("session_id") if data else None

            if session_id and session_id in _sessions:
                s = _sessions.pop(session_id)
                try:
                    s["env"].close()
                except Exception:
                    pass
            elif _sessions:
                _, s = _sessions.popitem()
                try:
                    s["env"].close()
                except Exception:
                    pass
            return jsonify({"success": True})

    def run(self, debug: bool = False):
        self.app.run(host=self.host, port=self.port, debug=debug)


# ── Public API ──────────────────────────────────────────────────────────────


def create_app(**kwargs) -> Flask:
    """Create and return a Flask app instance."""
    webapp = PlayWebApp(**kwargs)
    return webapp.app


def run_webapp(
    host: str = "127.0.0.1",
    port: int = 8080,
    debug: bool = False,
):
    """Run the Agentick human play web application.

    Example:
        >>> run_webapp(port=8080)
        # Visit http://127.0.0.1:8080 in your browser
    """
    webapp = PlayWebApp(host=host, port=port)
    print(f"\n  Agentick Play → http://{host}:{port}\n")
    webapp.run(debug=debug)


# Backward compat alias
ShowcaseWebApp = PlayWebApp


if __name__ == "__main__":
    run_webapp()
