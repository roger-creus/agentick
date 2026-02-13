"""Interactive HTML dashboards with Plotly."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import plotly.graph_objects as go


def generate_dashboard(
    results_dict: dict[str, Any],
    output_dir: str | Path,
    include_videos: bool = True,
    include_trajectories: bool = True,
) -> str:
    """
    Generate interactive HTML dashboard.

    Args:
        results_dict: Results dictionary with experiment results
        output_dir: Output directory
        include_videos: Include embedded videos
        include_trajectories: Include trajectory replay viewer

    Returns:
        Path to generated HTML file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dashboard_path = output_dir / "dashboard.html"

    # Generate Plotly figures
    figures_html = _generate_figures(results_dict)

    # Generate dropdowns/controls
    controls_html = _generate_controls(results_dict)

    # Generate summary stats
    summary_html = _generate_summary_stats(results_dict)

    # Generate video embeds if requested
    videos_html = ""
    if include_videos:
        videos_html = _generate_video_embeds(output_dir)

    # Generate trajectory viewer if requested
    trajectory_html = ""
    if include_trajectories:
        trajectory_html = _generate_trajectory_viewer(output_dir)

    # Assemble full HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Agentick Results Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #333;
                border-bottom: 3px solid #4CAF50;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #555;
                margin-top: 30px;
            }}
            .controls {{
                margin: 20px 0;
                padding: 15px;
                background-color: #f9f9f9;
                border-radius: 5px;
            }}
            select, button {{
                padding: 8px 12px;
                margin: 5px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 14px;
            }}
            button {{
                background-color: #4CAF50;
                color: white;
                cursor: pointer;
            }}
            button:hover {{
                background-color: #45a049;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }}
            .stat-card {{
                padding: 15px;
                background-color: #f0f7ff;
                border-radius: 5px;
                border-left: 4px solid #2196F3;
            }}
            .stat-label {{
                font-size: 12px;
                color: #666;
                text-transform: uppercase;
            }}
            .stat-value {{
                font-size: 24px;
                font-weight: bold;
                color: #333;
            }}
            .figure-container {{
                margin: 30px 0;
            }}
            .video-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            video {{
                width: 100%;
                border-radius: 5px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 Agentick Results Dashboard</h1>

            {summary_html}

            <h2>🎛️ Controls</h2>
            <div class="controls">
                {controls_html}
            </div>

            <h2>📊 Interactive Plots</h2>
            {figures_html}

            {videos_html}

            {trajectory_html}
        </div>

        <script>
            // Interactive filtering logic
            function filterByTask(task) {{
                // Update plots based on selected task
                console.log('Filtering by task:', task);
            }}

            function filterByAgent(agent) {{
                // Update plots based on selected agent
                console.log('Filtering by agent:', agent);
            }}
        </script>
    </body>
    </html>
    """

    with open(dashboard_path, "w") as f:
        f.write(html)

    return str(dashboard_path)


def _generate_figures(results_dict: dict[str, Any]) -> str:
    """Generate Plotly figures HTML."""
    figures_html = ""

    # Extract data
    if "metrics" in results_dict:
        # Bar chart of success rates by task
        if "per_task" in results_dict:
            tasks = list(results_dict["per_task"].keys())
            success_rates = [results_dict["per_task"][t].get("success_rate", 0) for t in tasks]

            fig = go.Figure(
                data=[
                    go.Bar(
                        x=tasks,
                        y=success_rates,
                        marker_color="#4CAF50",
                        text=[f"{sr:.1%}" for sr in success_rates],
                        textposition="outside",
                    )
                ]
            )
            fig.update_layout(
                title="Success Rate by Task",
                xaxis_title="Task",
                yaxis_title="Success Rate",
                yaxis_tickformat=".0%",
                height=400,
            )
            figures_html += '<div class="figure-container" id="fig1"></div>\n'
            figures_html += f"<script>Plotly.newPlot('fig1', {fig.to_json()});</script>\n"

        # Line plot placeholder for learning curves
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=[0, 10, 20, 30], y=[0.1, 0.3, 0.5, 0.7], mode="lines+markers"))
        fig2.update_layout(title="Learning Curve", xaxis_title="Episodes", yaxis_title="Return")
        figures_html += '<div class="figure-container" id="fig2"></div>\n'
        figures_html += f"<script>Plotly.newPlot('fig2', {fig2.to_json()});</script>\n"

    return figures_html


def _generate_controls(results_dict: dict[str, Any]) -> str:
    """Generate interactive controls HTML."""
    controls = []

    # Task selector
    if "per_task" in results_dict:
        tasks = list(results_dict["per_task"].keys())
        task_options = "".join(f'<option value="{t}">{t}</option>' for t in tasks)
        controls.append(
            f"""
            <label for="task-select">Select Task:</label>
            <select id="task-select" onchange="filterByTask(this.value)">
                <option value="all">All Tasks</option>
                {task_options}
            </select>
        """
        )

    # Agent selector (placeholder)
    controls.append(
        """
        <label for="agent-select">Select Agent:</label>
        <select id="agent-select" onchange="filterByAgent(this.value)">
            <option value="all">All Agents</option>
        </select>
    """
    )

    # Refresh button
    controls.append('<button onclick="location.reload()">🔄 Refresh</button>')

    return "\n".join(controls)


def _generate_summary_stats(results_dict: dict[str, Any]) -> str:
    """Generate summary statistics cards."""
    if "metrics" not in results_dict:
        return ""

    metrics = results_dict["metrics"]

    stats = [
        {
            "label": "Success Rate",
            "value": f"{metrics.get('success_rate', 0):.1%}",
        },
        {
            "label": "Mean Return",
            "value": f"{metrics.get('mean_return', 0):.2f}",
        },
        {
            "label": "Mean Episode Length",
            "value": f"{metrics.get('mean_episode_length', 0):.1f}",
        },
        {
            "label": "Total Episodes",
            "value": f"{metrics.get('total_episodes', 0)}",
        },
    ]

    cards = []
    for stat in stats:
        cards.append(
            f"""
            <div class="stat-card">
                <div class="stat-label">{stat["label"]}</div>
                <div class="stat-value">{stat["value"]}</div>
            </div>
        """
        )

    return f'<div class="stats-grid">{"".join(cards)}</div>'


def _generate_video_embeds(output_dir: Path) -> str:
    """Generate embedded videos section."""
    video_dir = output_dir / "videos"
    if not video_dir.exists():
        return ""

    videos = list(video_dir.glob("*.mp4"))
    if not videos:
        return ""

    video_html = '<h2>🎬 Episode Videos</h2>\n<div class="video-grid">\n'

    for video_path in videos[:6]:  # Limit to 6 videos
        # Make path relative
        rel_path = video_path.relative_to(output_dir)
        video_html += f"""
        <div>
            <h4>{video_path.stem}</h4>
            <video controls>
                <source src="{rel_path}" type="video/mp4">
                Your browser does not support video playback.
            </video>
        </div>
        """

    video_html += "</div>\n"
    return video_html


def _generate_trajectory_viewer(output_dir: Path) -> str:
    """Generate trajectory replay viewer."""
    return """
    <h2>🔄 Episode Replay</h2>
    <div id="trajectory-viewer">
        <p>Select an episode to replay:</p>
        <select id="episode-select">
            <option value="">-- Select Episode --</option>
        </select>
        <div id="replay-container" style="margin-top: 20px;"></div>
    </div>
    <script>
        // Trajectory replay logic would go here
        // Load trajectory, step through frames, etc.
    </script>
    """
