"""Local weights adapter for loading PyTorch models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class LocalWeightsAgent:
    """
    Adapter for loading agents from local PyTorch weight files.

    Supports CNN policies (pixel obs), MLP policies (state obs), and language models.
    """

    def __init__(
        self,
        weights_path: str,
        model_class: str,  # "agents.MyAgent" or path to class
        observation_mode: str = "rgb_array",
        device: str = "cpu",
        **model_kwargs,
    ):
        """
        Initialize local weights agent.

        Args:
            weights_path: Path to .pt or .safetensors file
            model_class: Importable path to model class
            observation_mode: Observation format
            device: Device to run on
            **model_kwargs: Additional model kwargs
        """
        self.weights_path = Path(weights_path)
        self.model_class_path = model_class
        self.observation_mode = observation_mode
        self.device = device
        self.model_kwargs = model_kwargs

        # Load model
        self._load_model()

    def _load_model(self):
        """Load model from weights file."""
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch not installed. Install with: uv sync --extra llm")

        if not self.weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {self.weights_path}")

        # Import model class
        module_path, class_name = self.model_class_path.rsplit(".", 1)
        module = __import__(module_path, fromlist=[class_name])
        model_class = getattr(module, class_name)

        # Instantiate model
        self.model = model_class(**self.model_kwargs)

        # Load weights
        state_dict = torch.load(self.weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

    def reset(self) -> None:
        """Reset state."""
        pass

    @property
    def name(self) -> str:
        """Get agent name."""
        return f"local_{self.weights_path.stem}"

    def act(self, observation: Any, info: dict[str, Any]) -> int:
        """
        Select action using local model.

        Args:
            observation: Environment observation
            info: Info dict

        Returns:
            Action index
        """
        import torch

        # Convert observation to tensor
        if isinstance(observation, np.ndarray):
            obs_tensor = torch.from_numpy(observation).float()
        else:
            obs_tensor = torch.tensor(observation).float()

        # Add batch dimension if needed
        if obs_tensor.ndim == len(observation.shape):
            obs_tensor = obs_tensor.unsqueeze(0)

        obs_tensor = obs_tensor.to(self.device)

        # Forward pass
        with torch.no_grad():
            output = self.model(obs_tensor)

        # Get action
        if output.ndim == 2:
            # Policy output: (batch, actions)
            action = output.argmax(dim=1).item()
        else:
            # Scalar output
            action = int(output.item())

        # Validate action
        valid_actions = info.get("valid_actions", [])
        if valid_actions and action not in valid_actions:
            # Fallback to random valid action
            rng = np.random.default_rng()
            action = int(rng.choice(valid_actions))

        return action
