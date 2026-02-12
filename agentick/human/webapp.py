"""Web-based human evaluation interface using Flask.

Provides a simple web interface for humans to interact with Agentick tasks
and record their performance for baseline evaluation.
"""

from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, jsonify, render_template_string, request, session

from agentick import make
from agentick.human.recorder import HumanDataRecorder
from agentick.tasks.registry import list_tasks


@dataclass
class EvaluationSession:
    """Tracks a human evaluation session."""

    task_id: str
    difficulty: str
    env: Any
    recorder: HumanDataRecorder
    session_start: datetime
    episode_actions: list[int]
    episode_action_names: list[str]
    episode_rewards: list[float]
    episode_steps: int


class HumanEvaluationWebApp:
    """Web application for human evaluation of Agentick tasks."""

    def __init__(
        self, output_dir: str = "human_eval_data", host: str = "0.0.0.0", port: int = 5000
    ):
        """
        Initialize web application.

        Args:
            output_dir: Directory to save evaluation data
            host: Host to bind server to
            port: Port to run server on
        """
        self.app = Flask(__name__)
        self.app.secret_key = os.urandom(24)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.host = host
        self.port = port

        # Active sessions (keyed by session ID)
        self.sessions: dict[str, EvaluationSession] = {}

        self._setup_routes()

    @staticmethod
    def _make_json_serializable(obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: HumanEvaluationWebApp._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [HumanEvaluationWebApp._make_json_serializable(item) for item in obj]
        else:
            return obj

    def _setup_routes(self):
        """Setup Flask routes."""

        @self.app.route("/")
        def index():
            """Main page with task selection."""
            tasks = list_tasks()
            return render_template_string(INDEX_TEMPLATE, tasks=tasks)

        @self.app.route("/api/start_task", methods=["POST"])
        def start_task():
            """Start a new evaluation task."""
            data = request.json
            task_id = data.get("task_id")
            difficulty = data.get("difficulty", "easy")

            if not task_id:
                return jsonify({"error": "task_id required"}), 400

            # Create environment
            env = make(task_id, difficulty=difficulty, render_mode="rgb_array")

            # Create recorder
            session_id = session.get("session_id", os.urandom(16).hex())
            session["session_id"] = session_id

            recorder = HumanDataRecorder(save_dir=str(self.output_dir))

            # Reset environment
            obs, info = env.reset()

            # Store session
            self.sessions[session_id] = EvaluationSession(
                task_id=task_id,
                difficulty=difficulty,
                env=env,
                recorder=recorder,
                session_start=datetime.now(),
                episode_actions=[],
                episode_action_names=[],
                episode_rewards=[],
                episode_steps=0,
            )

            # Get initial render
            render_data = self._get_render_data(env)

            return jsonify(
                {
                    "success": True,
                    "session_id": session_id,
                    "task_id": task_id,
                    "difficulty": difficulty,
                    "render": render_data,
                    "info": self._make_json_serializable(info),
                    "valid_actions": info.get("valid_actions", []),
                }
            )

        @self.app.route("/api/step", methods=["POST"])
        def step():
            """Execute an action step."""
            data = request.json
            action_name = data.get("action")

            session_id = session.get("session_id")
            if not session_id or session_id not in self.sessions:
                return jsonify({"error": "No active session"}), 400

            eval_session = self.sessions[session_id]
            env = eval_session.env
            recorder = eval_session.recorder

            # Map action name to action index
            action_map = {
                "noop": 0,
                "move_up": 1,
                "move_down": 2,
                "move_left": 3,
                "move_right": 4,
                "pickup": 5,
                "drop": 6,
                "rotate_left": 7,
                "rotate_right": 8,
                "move_forward": 9,
            }

            action_idx = action_map.get(action_name, 0)

            # Execute step
            obs, reward, terminated, truncated, info = env.step(action_idx)

            # Record action and reward
            eval_session.episode_actions.append(action_idx)
            eval_session.episode_action_names.append(action_name)
            eval_session.episode_rewards.append(reward)
            eval_session.episode_steps += 1

            # Get render
            render_data = self._get_render_data(env)

            # Handle episode end
            if terminated or truncated:
                # Calculate episode duration
                duration = (datetime.now() - eval_session.session_start).total_seconds()

                # Record episode with Human DataRecorder
                episode_stats = {
                    "success": terminated,
                    "total_reward": sum(eval_session.episode_rewards),
                    "step_count": eval_session.episode_steps,
                    "duration": duration,
                    "actions": eval_session.episode_action_names,
                }

                recorder.record_episode(
                    task_name=eval_session.task_id,
                    difficulty=eval_session.difficulty,
                    episode_stats=episode_stats,
                )

                # Save session
                recorder.save_session()

                total_reward = sum(eval_session.episode_rewards)

                # Cleanup session
                env.close()
                del self.sessions[session_id]

                return jsonify(
                    {
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "success": bool(terminated),
                        "reward": float(reward),
                        "total_reward": float(total_reward),
                        "render": render_data,
                        "info": self._make_json_serializable(info),
                        "message": "Task completed!"
                        if terminated
                        else "Episode truncated (max steps reached)",
                    }
                )

            return jsonify(
                {
                    "terminated": False,
                    "truncated": False,
                    "reward": float(reward),
                    "render": render_data,
                    "info": self._make_json_serializable(info),
                    "valid_actions": info.get("valid_actions", []),
                }
            )

        @self.app.route("/api/reset", methods=["POST"])
        def reset():
            """Reset the current task."""
            session_id = session.get("session_id")
            if not session_id or session_id not in self.sessions:
                return jsonify({"error": "No active session"}), 400

            eval_session = self.sessions[session_id]
            env = eval_session.env
            recorder = eval_session.recorder

            # Record previous episode if it had any steps
            if eval_session.episode_steps > 0:
                duration = (datetime.now() - eval_session.session_start).total_seconds()

                episode_stats = {
                    "success": False,
                    "total_reward": sum(eval_session.episode_rewards),
                    "step_count": eval_session.episode_steps,
                    "duration": duration,
                    "actions": eval_session.episode_action_names,
                    "reset": True,  # Mark as reset episode
                }

                recorder.record_episode(
                    task_name=eval_session.task_id,
                    difficulty=eval_session.difficulty,
                    episode_stats=episode_stats,
                )

            # Reset environment and episode tracking
            obs, info = env.reset()
            eval_session.episode_actions = []
            eval_session.episode_action_names = []
            eval_session.episode_rewards = []
            eval_session.episode_steps = 0
            eval_session.session_start = datetime.now()

            # Get render
            render_data = self._get_render_data(env)

            return jsonify(
                {
                    "success": True,
                    "render": render_data,
                    "info": self._make_json_serializable(info),
                    "valid_actions": info.get("valid_actions", []),
                }
            )

        @self.app.route("/api/quit", methods=["POST"])
        def quit_task():
            """Quit the current task."""
            session_id = session.get("session_id")
            if session_id and session_id in self.sessions:
                eval_session = self.sessions[session_id]
                eval_session.env.close()
                del self.sessions[session_id]

            return jsonify({"success": True})

    def _get_render_data(self, env) -> str:
        """Get base64-encoded render data."""
        render = env.render()
        if isinstance(render, np.ndarray):
            # Convert RGB array to base64-encoded PNG
            from PIL import Image

            img = Image.fromarray(render.astype("uint8"), "RGB")
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            buffer.seek(0)
            img_str = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/png;base64,{img_str}"
        else:
            # Text render
            return str(render)

    def run(self, debug: bool = False):
        """
        Run the web application.

        Args:
            debug: Enable Flask debug mode
        """
        self.app.run(host=self.host, port=self.port, debug=debug)


# HTML template for the main page
INDEX_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Agentick Human Evaluation</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .container {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .task-selector {
            margin: 20px 0;
        }
        select, button {
            padding: 10px;
            margin: 5px;
            font-size: 16px;
        }
        button {
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        button:hover {
            background-color: #45a049;
        }
        button:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }
        .render-area {
            margin: 20px 0;
            text-align: center;
        }
        .render-area img {
            max-width: 100%;
            border: 2px solid #ddd;
            border-radius: 4px;
        }
        .controls {
            margin: 20px 0;
            text-align: center;
        }
        .action-btn {
            margin: 5px;
            padding: 15px 30px;
            background-color: #008CBA;
        }
        .action-btn:hover {
            background-color: #007399;
        }
        .info-panel {
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 4px;
            margin: 10px 0;
        }
        .status {
            font-weight: bold;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎮 Agentick Human Evaluation</h1>

        <div class="task-selector" id="taskSelector">
            <h2>Select a Task</h2>
            <select id="taskSelect">
                <option value="">-- Select a task --</option>
                {% for task in tasks %}
                <option value="{{ task }}">{{ task }}</option>
                {% endfor %}
            </select>
            <select id="difficultySelect">
                <option value="easy">Easy</option>
                <option value="medium">Medium</option>
                <option value="hard">Hard</option>
                <option value="expert">Expert</option>
            </select>
            <button onclick="startTask()">Start Task</button>
        </div>

        <div id="gameArea" style="display:none;">
            <div class="info-panel">
                <div class="status" id="status">Ready</div>
                <div id="taskInfo"></div>
            </div>

            <div class="render-area">
                <img id="renderImage" src="" alt="Task rendering">
            </div>

            <div class="controls">
                <h3>Controls</h3>
                <div>
                    <button class="action-btn" onclick="sendAction('move_up')">↑ Up (W)</button>
                </div>
                <div>
                    <button class="action-btn" onclick="sendAction('move_left')">← Left (A)</button>
                    <button class="action-btn" onclick="sendAction('noop')">⊗ No-op (Space)</button>
                    <button class="action-btn" onclick="sendAction('move_right')">→ Right (D)</button>
                </div>
                <div>
                    <button class="action-btn" onclick="sendAction('move_down')">↓ Down (S)</button>
                </div>
                <div style="margin-top: 10px;">
                    <button class="action-btn" onclick="sendAction('pickup')">📦 Pickup (E)</button>
                    <button class="action-btn" onclick="sendAction('drop')">📤 Drop (Q)</button>
                    <button class="action-btn" onclick="sendAction('rotate_left')">↺ Rotate Left (Z)</button>
                    <button class="action-btn" onclick="sendAction('rotate_right')">↻ Rotate Right (X)</button>
                </div>
                <div style="margin-top: 20px;">
                    <button onclick="resetTask()">🔄 Reset</button>
                    <button onclick="quitTask()">❌ Quit</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentSession = null;

        // Keyboard controls
        document.addEventListener('keydown', function(event) {
            if (!currentSession) return;

            const keyMap = {
                'w': 'move_up',
                'a': 'move_left',
                's': 'move_down',
                'd': 'move_right',
                'e': 'pickup',
                'q': 'drop',
                'z': 'rotate_left',
                'x': 'rotate_right',
                ' ': 'noop',
                'ArrowUp': 'move_up',
                'ArrowLeft': 'move_left',
                'ArrowDown': 'move_down',
                'ArrowRight': 'move_right'
            };

            const action = keyMap[event.key.toLowerCase()];
            if (action) {
                event.preventDefault();
                sendAction(action);
            }
        });

        async function startTask() {
            const taskId = document.getElementById('taskSelect').value;
            const difficulty = document.getElementById('difficultySelect').value;

            if (!taskId) {
                alert('Please select a task');
                return;
            }

            const response = await fetch('/api/start_task', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({task_id: taskId, difficulty: difficulty})
            });

            const data = await response.json();

            if (data.success) {
                currentSession = data.session_id;
                document.getElementById('taskSelector').style.display = 'none';
                document.getElementById('gameArea').style.display = 'block';
                document.getElementById('renderImage').src = data.render;
                updateStatus(`Playing: ${taskId} (${difficulty})`);
            }
        }

        async function sendAction(action) {
            if (!currentSession) return;

            const response = await fetch('/api/step', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({action: action})
            });

            const data = await response.json();

            document.getElementById('renderImage').src = data.render;

            if (data.terminated || data.truncated) {
                updateStatus(data.message || 'Episode ended');
                currentSession = null;
                setTimeout(() => {
                    document.getElementById('gameArea').style.display = 'none';
                    document.getElementById('taskSelector').style.display = 'block';
                }, 2000);
            }
        }

        async function resetTask() {
            if (!currentSession) return;

            const response = await fetch('/api/reset', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'}
            });

            const data = await response.json();
            document.getElementById('renderImage').src = data.render;
            updateStatus('Task reset');
        }

        async function quitTask() {
            if (!currentSession) return;

            await fetch('/api/quit', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'}
            });

            currentSession = null;
            document.getElementById('gameArea').style.display = 'none';
            document.getElementById('taskSelector').style.display = 'block';
        }

        function updateStatus(message) {
            document.getElementById('status').textContent = message;
        }
    </script>
</body>
</html>
"""


def create_app(output_dir: str = "human_eval_data", **kwargs) -> Flask:
    """
    Create and return a Flask app instance.

    Args:
        output_dir: Directory to save evaluation data
        **kwargs: Additional arguments for HumanEvaluationWebApp

    Returns:
        Flask application instance
    """
    webapp = HumanEvaluationWebApp(output_dir=output_dir, **kwargs)
    return webapp.app


def run_webapp(
    output_dir: str = "human_eval_data",
    host: str = "0.0.0.0",
    port: int = 5000,
    debug: bool = False,
):
    """
    Run the human evaluation web application.

    Args:
        output_dir: Directory to save evaluation data
        host: Host to bind server to
        port: Port to run server on
        debug: Enable Flask debug mode

    Example:
        >>> run_webapp(port=8080)
        # Visit http://localhost:8080 in your browser
    """
    webapp = HumanEvaluationWebApp(output_dir=output_dir, host=host, port=port)
    print(f"Starting Agentick Human Evaluation Web App on http://{host}:{port}")
    print(f"Evaluation data will be saved to: {output_dir}")
    webapp.run(debug=debug)


if __name__ == "__main__":
    run_webapp()
