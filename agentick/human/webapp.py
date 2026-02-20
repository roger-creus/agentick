"""Web-based showcase interface for Agentick tasks using Flask."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from flask import Flask, jsonify, send_from_directory
except ImportError:
    Flask = None  # type: ignore[assignment,misc]


class ShowcaseWebApp:
    """Web application for the Agentick task showcase."""

    def __init__(self, host: str = "0.0.0.0", port: int = 5000):
        if Flask is None:
            raise ImportError(
                "Flask is required for the web interface. "
                "Install it with: pip install flask"
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
            result = []
            for name, desc in sorted(descs.items()):
                entry = desc.to_dict()
                entry["video"] = _find_video(name, videos_dir)
                result.append(entry)
            return jsonify(result)

        @self.app.route("/videos/<path:filename>")
        def serve_video(filename):
            """Serve video files."""
            videos_dir = _project_root() / "videos"
            return send_from_directory(str(videos_dir), filename)

    def run(self, debug: bool = False):
        """Run the web application."""
        self.app.run(host=self.host, port=self.port, debug=debug)


def _project_root() -> Path:
    """Return the Agentick project root (two levels up from this file)."""
    return Path(__file__).resolve().parent.parent.parent


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
