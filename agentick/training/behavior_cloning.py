"""Behavior Cloning — train a CNN from pixel observations to actions.

Pure PyTorch implementation using the Nature CNN architecture
(Mnih et al., 2015). Trains on oracle demonstration data collected
via :class:`agentick.data.DataCollector` with ``rgb_array`` modality.

Example::

    from agentick.training.behavior_cloning import BehaviorCloningTrainer

    trainer = BehaviorCloningTrainer(
        dataset_path="trajectories/oracle_pixels/",
        output_dir="models/bc/",
        num_epochs=50,
    )
    trainer.train()
    agent = trainer.as_agent()
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Nature CNN (Mnih et al., 2015)
# ---------------------------------------------------------------------------


class NatureCNN(nn.Module):
    """Nature CNN: 3-layer convolution + 2-layer MLP head.

    Architecture::

        Conv2d(1, 32, 8, stride=4) -> ReLU
        Conv2d(32, 64, 4, stride=2) -> ReLU
        Conv2d(64, 64, 3, stride=1) -> ReLU
        Flatten -> Linear(3136, 512) -> ReLU -> Linear(512, n_actions)
    """

    def __init__(self, n_actions: int, in_channels: int = 1) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        # For 84x84 input: 32@20x20 -> 64@9x9 -> 64@7x7 = 3136
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(B, C, 84, 84)`` with values in [0, 1].

        Returns:
            Action logits of shape ``(B, n_actions)``.
        """
        features = self.conv(x)
        features = features.reshape(features.size(0), -1)
        return self.fc(features)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class BCPixelDataset(Dataset):
    """Dataset of (observation, action) pairs from pixel trajectories."""

    def __init__(
        self,
        data_path: str | Path | None = None,
        image_size: tuple[int, int] = (84, 84),
        grayscale: bool = True,
    ) -> None:
        self.image_size = image_size
        self.grayscale = grayscale
        self.observations: list[np.ndarray] = []
        self.actions: list[int] = []

        if data_path is not None:
            data_path = Path(data_path)
            traj_file = data_path / "trajectories.jsonl"
            if traj_file.exists():
                self._load_from_jsonl(traj_file)
            else:
                raise FileNotFoundError(
                    f"No trajectories.jsonl found in {data_path}. "
                    "Save a CollectedDataset with save_pixels=True."
                )

    @classmethod
    def from_collected_dataset(
        cls,
        dataset: Any,
        image_size: tuple[int, int] = (84, 84),
        grayscale: bool = True,
    ) -> BCPixelDataset:
        """Create from an in-memory CollectedDataset.

        This avoids serializing/deserializing pixel arrays through JSON.
        """
        ds = cls(data_path=None, image_size=image_size, grayscale=grayscale)
        for traj in dataset.trajectories:
            for step in traj.steps:
                pixels = step.observations.get("rgb_array")
                if pixels is None:
                    pixels = step.observations.get("rgb_array_2d")
                if pixels is not None:
                    arr = (
                        np.array(pixels, dtype=np.uint8)
                        if not isinstance(pixels, np.ndarray)
                        else pixels
                    )
                    ds.observations.append(arr)
                    ds.actions.append(step.action)
        if not ds.observations:
            raise ValueError(
                "No pixel observations found. Collect with record_modalities=['rgb_array']"
            )
        return ds

    def _load_from_jsonl(self, path: Path) -> None:
        with open(path) as f:
            for line in f:
                traj = json.loads(line)
                for step in traj["steps"]:
                    obs = step["observations"]
                    # Prefer rgb_array, fall back to state_dict
                    pixels = obs.get("rgb_array") or obs.get("rgb_array_2d")
                    if pixels is not None:
                        arr = np.array(pixels, dtype=np.uint8)
                        self.observations.append(arr)
                        self.actions.append(step["action"])

        if not self.observations:
            raise ValueError(
                "No pixel observations found. Ensure data was collected "
                "with record_modalities=['rgb_array']"
            )

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """Resize to image_size and convert to grayscale float [0, 1]."""
        # img may be (H, W, 3) uint8
        if img.ndim == 2:
            gray = img.astype(np.float32) / 255.0
        elif img.ndim == 3 and img.shape[2] >= 3:
            # RGB to grayscale: 0.299R + 0.587G + 0.114B
            if self.grayscale:
                gray = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(
                    np.float32
                ) / 255.0
            else:
                gray = img.astype(np.float32) / 255.0
                gray = gray.transpose(2, 0, 1)  # HWC -> CHW
        else:
            gray = img.astype(np.float32) / 255.0

        # Resize using simple nearest-neighbor
        if self.grayscale and gray.ndim == 2:
            gray = self._resize_2d(gray, self.image_size)
            gray = gray[np.newaxis, :, :]  # Add channel dim
        elif not self.grayscale and gray.ndim == 3:
            # CHW format
            resized = np.zeros(
                (gray.shape[0], *self.image_size),
                dtype=np.float32,
            )
            for c in range(gray.shape[0]):
                resized[c] = self._resize_2d(gray[c], self.image_size)
            gray = resized
        else:
            gray = self._resize_2d(gray, self.image_size)
            gray = gray[np.newaxis, :, :]

        return gray

    @staticmethod
    def _resize_2d(
        arr: np.ndarray,
        target: tuple[int, int],
    ) -> np.ndarray:
        """Simple bilinear resize for 2D array."""
        th, tw = target
        sh, sw = arr.shape
        if sh == th and sw == tw:
            return arr
        # Use numpy-based resize via indexing
        row_idx = (np.arange(th) * sh / th).astype(int).clip(0, sh - 1)
        col_idx = (np.arange(tw) * sw / tw).astype(int).clip(0, sw - 1)
        return arr[np.ix_(row_idx, col_idx)]

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = self._preprocess(self.observations[idx])
        return torch.from_numpy(img), self.actions[idx]


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class BehaviorCloningTrainer:
    """Train a Nature CNN via behavior cloning on pixel demonstrations.

    Args:
        dataset_path: Path to saved ``CollectedDataset`` directory
            (must contain rgb_array observations).
        output_dir: Directory for checkpoints and final model.
        model: Model architecture (``"nature_cnn"``).
        n_actions: Number of discrete actions (auto-detected if None).
        image_size: Input image dimensions.
        grayscale: Convert to grayscale.
        num_epochs: Training epochs.
        batch_size: Mini-batch size.
        learning_rate: Optimizer learning rate.
        weight_decay: L2 regularization.
        device: Compute device.
        log_interval: Print loss every N batches.
        save_interval: Save checkpoint every N epochs.
        wandb_project: Optional W&B project name.
    """

    def __init__(
        self,
        dataset_path: str | Path | None = None,
        collected_dataset: Any | None = None,
        output_dir: str | Path = "models/bc",
        model: str = "nature_cnn",
        n_actions: int | None = None,
        image_size: tuple[int, int] = (84, 84),
        grayscale: bool = True,
        num_epochs: int = 50,
        batch_size: int = 64,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        device: str = "auto",
        log_interval: int = 10,
        save_interval: int = 10,
        wandb_project: str | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_type = model
        self.image_size = image_size
        self.grayscale = grayscale
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.wandb_project = wandb_project

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load dataset
        self._dataset: BCPixelDataset | None = None
        if collected_dataset is not None:
            self._dataset = BCPixelDataset.from_collected_dataset(
                collected_dataset,
                image_size=image_size,
                grayscale=grayscale,
            )
        elif dataset_path is not None:
            self._dataset = BCPixelDataset(
                dataset_path,
                image_size=image_size,
                grayscale=grayscale,
            )

        # Determine n_actions
        if n_actions is not None:
            self.n_actions = n_actions
        elif self._dataset is not None:
            self.n_actions = max(self._dataset.actions) + 1
        else:
            self.n_actions = 5  # default for agentick

        # Build model
        in_channels = 1 if grayscale else 3
        if model == "nature_cnn":
            self._model = NatureCNN(self.n_actions, in_channels=in_channels)
        else:
            raise ValueError(f"Unknown model: {model}")
        self._model = self._model.to(self.device)

        self._trained = False
        self._train_losses: list[float] = []

    def train(self) -> dict[str, Any]:
        """Run behavior cloning training loop.

        Returns:
            Dict with training metrics.
        """
        if self._dataset is None:
            raise RuntimeError("No dataset loaded")

        dataloader = DataLoader(
            self._dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )

        optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Optional wandb
        wandb_run = None
        if self.wandb_project:
            try:
                import wandb

                wandb_run = wandb.init(project=self.wandb_project)
            except ImportError:
                pass

        self._model.train()
        epoch_losses = []

        for epoch in range(self.num_epochs):
            running_loss = 0.0
            n_batches = 0

            for batch_idx, (images, actions) in enumerate(dataloader):
                images = images.to(self.device)
                if isinstance(actions, torch.Tensor):
                    actions = actions.to(dtype=torch.long, device=self.device)
                else:
                    actions = torch.tensor(actions, dtype=torch.long, device=self.device)

                logits = self._model(images)
                loss = F.cross_entropy(logits, actions)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                n_batches += 1

                if (batch_idx + 1) % self.log_interval == 0:
                    avg = running_loss / n_batches
                    logger.info(
                        f"Epoch {epoch + 1}/{self.num_epochs} batch {batch_idx + 1} loss={avg:.4f}"
                    )

            avg_loss = running_loss / max(n_batches, 1)
            epoch_losses.append(avg_loss)
            self._train_losses.append(avg_loss)

            if wandb_run:
                import wandb

                wandb.log({"epoch": epoch + 1, "loss": avg_loss})

            # Checkpoint
            if (epoch + 1) % self.save_interval == 0:
                ckpt = self.output_dir / f"checkpoint_epoch{epoch + 1}.pt"
                torch.save(self._model.state_dict(), ckpt)

            print(f"Epoch {epoch + 1}/{self.num_epochs}: loss={avg_loss:.4f}")

        # Save final model
        final_path = self.output_dir / "model_final.pt"
        torch.save(self._model.state_dict(), final_path)

        # Save metadata
        meta = {
            "model_type": self.model_type,
            "n_actions": self.n_actions,
            "image_size": list(self.image_size),
            "grayscale": self.grayscale,
            "num_epochs": self.num_epochs,
            "final_loss": epoch_losses[-1] if epoch_losses else None,
            "dataset_size": len(self._dataset),
        }
        with open(self.output_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        self._trained = True

        if wandb_run:
            wandb_run.finish()

        return {
            "final_loss": epoch_losses[-1] if epoch_losses else None,
            "epoch_losses": epoch_losses,
            "dataset_size": len(self._dataset),
        }

    def evaluate(
        self,
        tasks: list[str] | None = None,
        num_episodes: int = 20,
        difficulty: str = "easy",
    ) -> dict[str, float]:
        """Evaluate the trained model on tasks.

        Args:
            tasks: Task IDs to evaluate on.
            num_episodes: Episodes per task.
            difficulty: Difficulty level.

        Returns:
            Dict mapping task_id to success rate.
        """
        import agentick

        agent = self.as_agent()
        results = {}

        for task_id in tasks or ["GoToGoal-v0"]:
            env = agentick.make(
                task_id,
                difficulty=difficulty,
                render_mode="rgb_array",
            )
            successes = 0
            for seed in range(num_episodes):
                obs, info = env.reset(seed=seed)
                agent.reset(obs, info)
                done = False
                while not done:
                    action = agent.act(obs, info)
                    obs, reward, done, trunc, info = env.step(action)
                    done = done or trunc
                if info.get("success", False):
                    successes += 1
            results[task_id] = successes / num_episodes
            env.close()

        return results

    def as_agent(self, device: str | None = None) -> BCAgent:
        """Wrap trained model as an agentick agent."""
        return BCAgent(
            model_path=self.output_dir,
            device=device or str(self.device),
        )


class BCAgent:
    """Agent wrapper for a trained BC model.

    Implements ``act(obs, info) -> int`` for agentick environments.
    Expects rgb_array observations.
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cpu",
    ) -> None:
        model_path = Path(model_path)
        with open(model_path / "meta.json") as f:
            meta = json.load(f)

        self.n_actions = meta["n_actions"]
        self.image_size = tuple(meta["image_size"])
        self.grayscale = meta["grayscale"]

        in_channels = 1 if self.grayscale else 3
        self.model = NatureCNN(self.n_actions, in_channels=in_channels)
        weights = torch.load(
            model_path / "model_final.pt",
            map_location=device,
            weights_only=True,
        )
        self.model.load_state_dict(weights)
        self.model.eval()
        self.device = torch.device(device)
        self.model.to(self.device)

    def reset(self, obs: Any, info: dict[str, Any]) -> None:
        """Reset (no-op)."""
        pass

    def act(self, obs: Any, info: dict[str, Any]) -> int:
        """Select action from pixel observation."""
        img = self._preprocess(obs)
        with torch.no_grad():
            logits = self.model(img.unsqueeze(0).to(self.device))
        return int(logits.argmax(dim=1).item())

    def _preprocess(self, obs: Any) -> torch.Tensor:
        """Convert observation to tensor."""
        if isinstance(obs, np.ndarray):
            img = obs
        elif isinstance(obs, dict):
            img = obs.get("rgb_array", obs.get("image", np.zeros((84, 84, 3))))
            img = np.array(img, dtype=np.uint8)
        else:
            img = np.array(obs, dtype=np.uint8)

        # To float [0, 1]
        img = img.astype(np.float32) / 255.0

        # Grayscale
        if self.grayscale and img.ndim == 3 and img.shape[2] >= 3:
            img = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

        # Resize
        if img.ndim == 2:
            img = BCPixelDataset._resize_2d(img, self.image_size)
            img = img[np.newaxis, :, :]  # (1, H, W)
        elif img.ndim == 3:
            # CHW
            img = img.transpose(2, 0, 1)
            resized = np.zeros(
                (img.shape[0], *self.image_size),
                dtype=np.float32,
            )
            for c in range(img.shape[0]):
                resized[c] = BCPixelDataset._resize_2d(img[c], self.image_size)
            img = resized

        return torch.from_numpy(img)
