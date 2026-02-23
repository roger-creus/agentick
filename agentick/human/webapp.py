"""Web-based showcase interface for Agentick tasks using Flask.

Provides:
- Task carousel with video previews
- Human play mode with multi-modal observations (ASCII + language + pixels)
- API endpoints for task browsing and interactive play
"""

from __future__ import annotations

import base64
import io
import os
import re
import uuid
from pathlib import Path
from typing import Any

import numpy as np

try:
    from flask import Flask, jsonify, request, send_from_directory
except ImportError:
    Flask = None  # type: ignore[assignment,misc]


# ── Active play sessions ──────────────────────────────────────────────────────

_sessions: dict[str, dict[str, Any]] = {}


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from a string."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _render_multimodal(env: Any) -> dict[str, str]:
    """Render the current env state in all three modalities.

    Returns dict with keys: ascii, language, pixel (base64 PNG).
    """
    from PIL import Image

    unwrapped = env.unwrapped

    # ASCII (strip ANSI codes for clean HTML display)
    try:
        ascii_raw = unwrapped.render_in_mode("ascii")
        ascii_text = _strip_ansi(ascii_raw) if isinstance(ascii_raw, str) else str(ascii_raw)
    except Exception:
        ascii_text = "(ASCII render unavailable)"

    # Language
    try:
        lang_text = unwrapped.render_in_mode("language")
        if not isinstance(lang_text, str):
            lang_text = str(lang_text)
    except Exception:
        lang_text = "(Language render unavailable)"

    # Pixel (rgb_array -> base64 PNG)
    pixel_b64 = ""
    try:
        rgb = unwrapped.render_in_mode("rgb_array")
        if isinstance(rgb, np.ndarray):
            img = Image.fromarray(rgb.astype(np.uint8))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            pixel_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        pass

    return {"ascii": ascii_text, "language": lang_text, "pixel": pixel_b64}


class ShowcaseWebApp:
    """Web application for the Agentick task showcase with human play."""

    def __init__(self, host: str = "0.0.0.0", port: int = 5000):
        if Flask is None:
            raise ImportError(
                "Flask is required for the web interface. Install it with: pip install flask"
            )
        self.app = Flask(__name__)
        self.app.secret_key = os.urandom(24)
        self.host = host
        self.port = port
        self._setup_routes()

    def _setup_routes(self):
        """Setup Flask routes."""

        @self.app.route("/")
        def index():
            """Serve the showcase HTML page."""
            showcase_dir = _project_root() / "showcase"
            return send_from_directory(str(showcase_dir), "index.html")

        @self.app.route("/api/task_descriptions")
        def api_task_descriptions():
            """Return JSON array of all task descriptions."""
            from agentick.tasks.descriptions import get_all_task_descriptions

            descs = get_all_task_descriptions()
            videos_dir = _project_root() / "videos"
            gallery_dir = _project_root() / "gallery"
            result = []
            for name, desc in sorted(descs.items()):
                entry = desc.to_dict()
                entry["video"] = _find_video(name, videos_dir)
                entry["gallery_gif"] = _find_gallery_gif(name, gallery_dir)
                result.append(entry)
            return jsonify(result)

        @self.app.route("/videos/<path:filename>")
        def serve_video(filename):
            """Serve video files."""
            videos_dir = _project_root() / "videos"
            return send_from_directory(str(videos_dir), filename)

        @self.app.route("/gallery/<path:filename>")
        def serve_gallery(filename):
            """Serve gallery GIF files (iso or 2D)."""
            # Prefer isometric GIFs from showcase/videos/iso/
            iso_dir = _project_root() / "showcase" / "videos" / "iso"
            iso_path = iso_dir / filename
            if iso_path.is_file():
                return send_from_directory(str(iso_dir), filename)
            # Fall back to 2D gallery
            gallery_dir = _project_root() / "gallery"
            return send_from_directory(str(gallery_dir), filename)

        # ── Play-mode API endpoints ───────────────────────────────────────

        @self.app.route("/api/start_task", methods=["POST"])
        def api_start_task():
            """Start a new play session for a task.

            Request JSON: {task_id: str, difficulty: str, seed?: int}
            Response JSON: {success, session_id, ascii, language, pixel, valid_actions}
            """
            import agentick

            data = request.get_json(force=True)
            task_id = data.get("task_id", "GoToGoal-v0")
            difficulty = data.get("difficulty", "easy")
            seed = data.get("seed")

            try:
                env = agentick.make(
                    task_id,
                    difficulty=difficulty,
                    render_mode="rgb_array",
                )
                reset_kwargs: dict[str, Any] = {}
                if seed is not None:
                    reset_kwargs["seed"] = int(seed)
                env.reset(**reset_kwargs)

                session_id = str(uuid.uuid4())[:8]
                _sessions[session_id] = {
                    "env": env,
                    "task_id": task_id,
                    "difficulty": difficulty,
                    "steps": 0,
                    "total_reward": 0.0,
                    "done": False,
                }

                renders = _render_multimodal(env)
                return jsonify(
                    {
                        "success": True,
                        "session_id": session_id,
                        "render": renders["pixel"],
                        "ascii": renders["ascii"],
                        "language": renders["language"],
                        "pixel": renders["pixel"],
                    }
                )
            except Exception as e:
                return jsonify({"success": False, "error": str(e)}), 400

        @self.app.route("/api/step", methods=["POST"])
        def api_step():
            """Take an action in the active play session.

            Request JSON: {session_id?: str, action: str}
            Response JSON: {render, ascii, language, pixel, reward, terminated,
                           truncated, success, steps, total_reward}
            """
            data = request.get_json(force=True)
            action_name = data.get("action", "noop")
            session_id = data.get("session_id")

            # Find session (use provided ID or most recent)
            session = None
            if session_id and session_id in _sessions:
                session = _sessions[session_id]
            elif _sessions:
                session = list(_sessions.values())[-1]

            if session is None:
                return jsonify({"error": "No active session"}), 400

            if session["done"]:
                renders = _render_multimodal(session["env"])
                return jsonify(
                    {
                        "render": renders["pixel"],
                        "ascii": renders["ascii"],
                        "language": renders["language"],
                        "pixel": renders["pixel"],
                        "reward": 0,
                        "terminated": True,
                        "truncated": False,
                        "success": False,
                        "steps": session["steps"],
                        "total_reward": session["total_reward"],
                    }
                )

            # Map action name to integer
            action_int = _action_name_to_int(action_name)

            env = session["env"]
            obs, reward, terminated, truncated, info = env.step(action_int)
            session["steps"] += 1
            session["total_reward"] += reward
            session["done"] = terminated or truncated

            renders = _render_multimodal(env)
            success = info.get("success", False) if terminated else False

            return jsonify(
                {
                    "render": renders["pixel"],
                    "ascii": renders["ascii"],
                    "language": renders["language"],
                    "pixel": renders["pixel"],
                    "reward": float(reward),
                    "terminated": terminated,
                    "truncated": truncated,
                    "success": success,
                    "steps": session["steps"],
                    "total_reward": float(session["total_reward"]),
                }
            )

        @self.app.route("/api/reset", methods=["POST"])
        def api_reset():
            """Reset the current play session.

            Response JSON: {success, render, ascii, language, pixel}
            """
            data = request.get_json(force=True) if request.data else {}
            session_id = data.get("session_id") if data else None

            session = None
            if session_id and session_id in _sessions:
                session = _sessions[session_id]
            elif _sessions:
                session = list(_sessions.values())[-1]

            if session is None:
                return jsonify({"error": "No active session"}), 400

            env = session["env"]
            env.reset()
            session["steps"] = 0
            session["total_reward"] = 0.0
            session["done"] = False

            renders = _render_multimodal(env)
            return jsonify(
                {
                    "success": True,
                    "render": renders["pixel"],
                    "ascii": renders["ascii"],
                    "language": renders["language"],
                    "pixel": renders["pixel"],
                }
            )

        @self.app.route("/api/quit", methods=["POST"])
        def api_quit():
            """Close the play session and free resources."""
            data = request.get_json(force=True) if request.data else {}
            session_id = data.get("session_id") if data else None

            if session_id and session_id in _sessions:
                session = _sessions.pop(session_id)
                try:
                    session["env"].close()
                except Exception:
                    pass
            elif _sessions:
                sid, session = _sessions.popitem()
                try:
                    session["env"].close()
                except Exception:
                    pass

            return jsonify({"success": True})

    def run(self, debug: bool = False):
        """Run the web application."""
        self.app.run(host=self.host, port=self.port, debug=debug)


# ── Helpers ───────────────────────────────────────────────────────────────────

_ACTION_MAP = {
    "noop": 0,
    "move_up": 1,
    "move_down": 2,
    "move_left": 3,
    "move_right": 4,
    "pickup": 5,
    "drop": 6,
    "use": 7,
    "interact": 8,
    "rotate_left": 9,
    "rotate_right": 10,
    "move_forward": 11,
}


def _action_name_to_int(name: str) -> int:
    """Convert action name to ActionType integer."""
    return _ACTION_MAP.get(name.lower().strip(), 0)


def _project_root() -> Path:
    """Return the Agentick project root (two levels up from this file)."""
    return Path(__file__).resolve().parent.parent.parent


def _find_gallery_gif(
    task_name: str,
    gallery_dir: Path,
) -> dict[str, str] | None:
    """Return dict of difficulty -> gallery-relative GIF path, or None.

    Prefers isometric GIFs from ``showcase/videos/iso/`` (served via the
    same ``/gallery/`` route).  Falls back to 2D GIFs in the ``gallery/``
    directory.
    """
    result: dict[str, str] = {}

    # 1. Prefer isometric oracle GIFs (showcase/videos/iso/{task}_{diff}.gif)
    iso_dir = _project_root() / "showcase" / "videos" / "iso"
    if iso_dir.is_dir():
        for diff in ("easy", "medium", "hard", "expert"):
            gif = iso_dir / f"{task_name}_{diff}.gif"
            if gif.is_file():
                # Served via /gallery/ route which checks iso_dir first
                result[diff] = f"{task_name}_{diff}.gif"
    if result:
        return result

    # 2. Fall back to 2D gallery (gallery/{diff}/{task}.gif)
    if gallery_dir.is_dir():
        for diff in ("easy", "medium", "hard", "expert"):
            gif = gallery_dir / diff / f"{task_name}.gif"
            if gif.is_file():
                result[diff] = f"{diff}/{task_name}.gif"
        if result:
            return result
        # Flat fallback
        gif = gallery_dir / f"{task_name}.gif"
        if gif.is_file():
            return {"easy": gif.name}

    return None


def _find_video(task_name: str, videos_dir: Path) -> str | None:
    """Return the video filename for a task (easy dense preferred)."""
    if not videos_dir.is_dir():
        return None
    base = task_name.replace("-v0", "").replace("-v1", "")
    for f in sorted(videos_dir.iterdir()):
        if f.suffix == ".mp4" and base in f.name and "easy_dense" in f.name:
            return f.name
    for f in sorted(videos_dir.iterdir()):
        if f.suffix == ".mp4" and base in f.name:
            return f.name
    return None


def create_app(**kwargs) -> Flask:
    """Create and return a Flask app instance."""
    webapp = ShowcaseWebApp(**kwargs)
    return webapp.app


def run_webapp(
    host: str = "0.0.0.0",
    port: int = 5000,
    debug: bool = False,
):
    """
    Run the Agentick showcase web application.

    Example:
        >>> run_webapp(port=8080)
        # Visit http://localhost:8080 in your browser
    """
    webapp = ShowcaseWebApp(host=host, port=port)
    print(f"Starting Agentick Showcase on http://{host}:{port}")
    webapp.run(debug=debug)


if __name__ == "__main__":
    run_webapp()
