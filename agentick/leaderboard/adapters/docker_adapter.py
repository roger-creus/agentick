"""Docker container adapter for agent servers."""

from __future__ import annotations

import time
from typing import Any

import requests


class DockerAgent:
    """
    Adapter for dockerized agent servers.

    The agent runs in a Docker container and exposes an HTTP API.
    """

    def __init__(
        self,
        image: str,
        port: int = 8080,
        endpoint: str = "/predict",
        timeout: float = 30.0,
        health_check_endpoint: str = "/health",
        **container_kwargs,
    ):
        """
        Initialize Docker agent adapter.

        Args:
            image: Docker image name
            port: Port to expose
            endpoint: Prediction endpoint
            timeout: Request timeout
            health_check_endpoint: Health check endpoint
            **container_kwargs: Additional docker run arguments
        """
        self.image = image
        self.port = port
        self.endpoint = endpoint
        self.timeout = timeout
        self.health_check_endpoint = health_check_endpoint
        self.container_kwargs = container_kwargs

        self.container_id: str | None = None
        self.base_url = f"http://localhost:{port}"

        # Start container
        self._start_container()

    def _start_container(self):
        """Start Docker container."""
        try:
            import docker
        except ImportError:
            raise ImportError(
                "docker package not installed. Install with: uv sync --extra leaderboard"
            )

        client = docker.from_env()

        # Run container
        self.container = client.containers.run(
            self.image,
            detach=True,
            ports={f"{self.port}/tcp": self.port},
            **self.container_kwargs,
        )
        self.container_id = self.container.id

        # Wait for container to be healthy
        self._wait_for_health()

    def _wait_for_health(self, max_wait: int = 30):
        """Wait for container to be ready."""
        health_url = f"{self.base_url}{self.health_check_endpoint}"
        start_time = time.time()

        while time.time() - start_time < max_wait:
            try:
                response = requests.get(health_url, timeout=1.0)
                if response.status_code == 200:
                    return
            except requests.exceptions.RequestException:
                pass

            time.sleep(1)

        raise RuntimeError(f"Container failed to become healthy after {max_wait}s")

    def reset(self) -> None:
        """Reset agent state (via API call)."""
        reset_url = f"{self.base_url}/reset"
        try:
            requests.post(reset_url, timeout=self.timeout)
        except requests.exceptions.RequestException:
            pass  # Optional endpoint

    @property
    def name(self) -> str:
        """Get agent name."""
        return f"docker_{self.image.replace('/', '_').replace(':', '_')}"

    def act(self, observation: Any, info: dict[str, Any]) -> int:
        """
        Select action by calling container API.

        Args:
            observation: Environment observation
            info: Info dict

        Returns:
            Action index
        """
        predict_url = f"{self.base_url}{self.endpoint}"

        # Convert observation to JSON-serializable format
        if hasattr(observation, "tolist"):
            obs_data = observation.tolist()
        else:
            obs_data = observation

        payload = {
            "observation": obs_data,
            "info": info,
        }

        response = requests.post(
            predict_url,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()

        data = response.json()

        # Extract action
        if "action" in data:
            return int(data["action"])
        else:
            raise ValueError(f"Invalid response format: {data}")

    def stop(self):
        """Stop and remove container."""
        if self.container_id:
            try:
                import docker

                client = docker.from_env()
                container = client.containers.get(self.container_id)
                container.stop()
                container.remove()
            except Exception:
                pass

    def __del__(self):
        """Cleanup on deletion."""
        self.stop()
