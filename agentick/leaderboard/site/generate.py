"""Static leaderboard site generator."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from agentick.leaderboard.database import load_entries
from agentick.leaderboard.scoring import TASK_CAPABILITY_MAP


class SiteGenerator:
    """Generate static leaderboard website."""

    def __init__(
        self,
        data_dir: str | Path = "leaderboard_data",
        output_dir: str | Path = "leaderboard_site",
    ):
        """Initialize site generator.

        Args:
            data_dir: Leaderboard data directory (contains entries.json)
            output_dir: Output directory for generated site
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.entries_path = self.data_dir / "entries.json"

        # Setup Jinja2
        template_dir = Path(__file__).parent / "templates"
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(["html", "xml"]),
        )

    def generate(self) -> None:
        """Generate complete static site."""
        print("Generating leaderboard site...")

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Copy assets
        self._copy_assets()

        # Generate pages
        self._generate_index()
        self._generate_about()
        self._generate_submit()

        print(f"Site generated in {self.output_dir}")

    def _copy_assets(self) -> None:
        """Copy static assets."""
        assets_src = Path(__file__).parent / "assets"
        assets_dst = self.output_dir / "assets"

        if assets_src.exists():
            shutil.copytree(assets_src, assets_dst, dirs_exist_ok=True)

    @staticmethod
    def _display_field(agent_type: str, raw_value: str, field: str) -> str:
        """Return display value for modality/harness columns.

        Non-applicable combinations show '-' instead of the raw value.
        """
        is_baseline = agent_type in ("other",)
        if field == "modality":
            if is_baseline:
                return "\u2013"
            return raw_value or "\u2013"
        if field == "harness":
            if is_baseline or agent_type == "rl":
                return "\u2013"
            return raw_value or "\u2013"
        return raw_value

    def _generate_index(self) -> None:
        """Generate main leaderboard page."""
        entries = load_entries(self.entries_path)

        # Build rankings from entries (sorted by agentick_score descending)
        rankings = []
        for entry in entries:
            scores = entry.get("scores", {})
            ci = scores.get("agentick_score_ci", [0.0, 0.0])
            agent_type = entry.get("agent_type", "")
            overall = scores.get("agentick_score", 0.0)
            has_overall = overall > 0 or len(scores.get("per_category", {})) == 6
            rankings.append({
                "rank": 0,
                "agent_name": entry.get("agent_name", ""),
                "author": entry.get("author", ""),
                "agent_type": agent_type,
                "observation_mode": entry.get("observation_mode", ""),
                "modality": self._display_field(
                    agent_type, entry.get("observation_mode", ""), "modality",
                ),
                "harness": self._display_field(
                    agent_type, entry.get("harness", ""), "harness",
                ),
                "model": entry.get("model", ""),
                "score": overall,
                "has_overall": has_overall,
                "score_ci_lower": ci[0] if len(ci) >= 2 else 0.0,
                "score_ci_upper": ci[1] if len(ci) >= 2 else 0.0,
                "per_category": scores.get("per_category", {}),
                "per_task": scores.get("per_task", {}),
                "open_weights": entry.get("open_weights", False),
                "date": entry.get("date", ""),
            })

        # Entries with full results first (sorted by score), then partial entries
        rankings.sort(key=lambda x: (x["has_overall"], x["score"]), reverse=True)
        for i, r in enumerate(rankings):
            r["rank"] = i + 1

        # Capability list for category tabs
        categories = sorted(set(TASK_CAPABILITY_MAP.values()))

        # Task list sorted alphabetically, grouped by category
        task_names = sorted(TASK_CAPABILITY_MAP.keys())
        task_categories = dict(TASK_CAPABILITY_MAP)

        # Load task descriptions from showcase JSON (truncate for compact display)
        task_descriptions = {}
        desc_path = Path("docs/showcase/task_descriptions.json")
        if desc_path.exists():
            with open(desc_path) as f:
                for td in json.load(f):
                    summary = td.get("goal", td.get("summary", ""))
                    # Keep it short — first sentence only, max 120 chars
                    if ". " in summary:
                        summary = summary[: summary.index(". ") + 1]
                    if len(summary) > 120:
                        summary = summary[:117] + "..."
                    task_descriptions[td["name"]] = summary

        # Build chart data as JSON for Chart.js
        chart_data = {
            "agent_names": [r["agent_name"] for r in rankings],
            "overall_scores": [round(r["score"], 3) for r in rankings],
            "categories": categories,
            "per_category": {
                cat: [round(r["per_category"].get(cat, 0), 3) for r in rankings]
                for cat in categories
            },
        }

        template = self.env.get_template("index.html")
        html = template.render(
            title="Agentick Leaderboard",
            rankings=rankings,
            categories=categories,
            task_names=task_names,
            task_categories=task_categories,
            task_descriptions=task_descriptions,
            chart_data_json=json.dumps(chart_data),
        )

        (self.output_dir / "index.html").write_text(html)

    def _generate_about(self) -> None:
        """Generate about page."""
        template = self.env.get_template("about.html")
        html = template.render(title="About Agentick")
        (self.output_dir / "about.html").write_text(html)

    def _generate_submit(self) -> None:
        """Generate submission instructions page."""
        template = self.env.get_template("submit.html")
        html = template.render(title="Submit Your Agent")
        (self.output_dir / "submit.html").write_text(html)


def main() -> None:
    """CLI entrypoint for site generation."""
    generator = SiteGenerator()
    generator.generate()


if __name__ == "__main__":
    main()
