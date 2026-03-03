"""Human evaluation module.

Provides a web interface for humans to play Agentick tasks and record session data.
"""

from agentick.human.recorder import HumanDataRecorder
from agentick.human.webapp import ShowcaseWebApp, run_webapp

__all__ = [
    "HumanDataRecorder",
    "ShowcaseWebApp",
    "run_webapp",
]
