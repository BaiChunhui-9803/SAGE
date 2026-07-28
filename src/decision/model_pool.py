"""
Model Pool Manager for Adaptive Decision System
Manages multiple DT models with different context windows
"""

import torch
import torch.nn as nn
import logging
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ModelType(Enum):
    DECISION_TRANSFORMER = "decision_transformer"
    Q_NETWORK = "q_network"
    STATE_PREDICTOR = "state_predictor"


@dataclass
class ModelInfo:
    """Information about a model in the pool"""

    key: str
    model_type: ModelType
    model_path: Optional[Path]
    context_window: int
    min_history: int
    max_history: int
    prediction_steps: List[int]
    fallback: Optional[str]
    strategy: str
    description: str
    model: Optional[nn.Module] = None
    loaded: bool = False


# Default model pool configuration
DEFAULT_MODEL_POOL_CONFIG = {
    "q_only": {
        "model_type": "q_network",
        "model_path": None,
        "context_window": 0,
        "min_history": 0,
        "max_history": 4,
        "prediction_steps": [1],
        "fallback": None,
        "strategy": "q_only",
        "description": "Q-value only for very short history (0-4 steps)",
    },
    "DT_ctx5": {
        "model_type": "decision_transformer",
        "model_path": "cache/model/dt_ctx5.pth",
        "context_window": 5,
        "min_history": 5,
        "max_history": 9,
        "prediction_steps": [1, 3],
        "fallback": "q_only",
        "strategy": "hybrid",
        "description": "DT with 5-step context for early phase",
    },
    "DT_ctx10": {
        "model_type": "decision_transformer",
        "model_path": "cache/model/dt_ctx10.pth",
        "context_window": 10,
        "min_history": 10,
        "max_history": 19,
        "prediction_steps": [1, 3, 5],
        "fallback": "DT_ctx5",
        "strategy": "hybrid",
        "description": "DT with 10-step context for mid phase",
    },
    "DT_ctx20": {
        "model_type": "decision_transformer",
        "model_path": "cache/model/best_model.pth",
        "context_window": 20,
        "min_history": 20,
        "max_history": 1000,
        "prediction_steps": [1, 3, 5],
        "fallback": "DT_ctx10",
        "strategy": "hybrid",
        "description": "DT with 20-step context for mature phase",
    },
}


class ModelPoolManager:
    """
    Manages a pool of models for adaptive decision making
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        model_dir: Optional[Path] = None,
        device: torch.device = torch.device("cpu"),
        preload_all: bool = False,
    ):
        """
        Initialize model pool manager

        Args:
            config: Model pool configuration dict
            model_dir: Base directory for model files
            device: Device to load models on
            preload_all: Whether to preload all models
        """
        self.config = config or DEFAULT_MODEL_POOL_CONFIG
        self.model_dir = model_dir or Path.cwd()
        self.device = device

        self.models: Dict[str, ModelInfo] = {}
        self._shared_models: Dict[str, nn.Module] = {}

        # Initialize model info
        for key, cfg in self.config.items():
            self.models[key] = ModelInfo(
                key=key,
                model_type=ModelType(cfg["model_type"]),
                model_path=Path(cfg["model_path"]) if cfg.get("model_path") else None,
                context_window=cfg["context_window"],
                min_history=cfg["min_history"],
                max_history=cfg["max_history"],
                prediction_steps=cfg["prediction_steps"],
                fallback=cfg.get("fallback"),
                strategy=cfg["strategy"],
                description=cfg["description"],
            )

        if preload_all:
            self.preload_all()

    def load_model(self, model_key: str) -> Optional[nn.Module]:
        """
        Load a specific model by key

        Args:
            model_key: Key of the model to load

        Returns:
            Loaded model or None if not applicable
        """
        if model_key not in self.models:
            logger.warning(f"Unknown model key: {model_key}")
            return None

        info = self.models[model_key]

        if info.loaded and info.model is not None:
            return info.model

        if info.model_type == ModelType.Q_NETWORK:
            return self._get_shared_model("q_network")

        if info.model_type == ModelType.STATE_PREDICTOR:
            return self._get_shared_model("state_predictor")

        if info.model_type == ModelType.DECISION_TRANSFORMER:
            model = self._load_dt_model(info)
            if model is not None:
                info.model = model
                info.loaded = True
            return model

        return None

    def _load_dt_model(self, info: ModelInfo) -> Optional[nn.Module]:
        """Load a Decision Transformer model"""
        from src.models.DecisionTransformer import DecisionTransformer

        model_path = self.model_dir / info.model_path if info.model_path else None

        if model_path is None or not model_path.exists():
            logger.warning(f"Model file not found: {model_path}")
            return None

        try:
            checkpoint = torch.load(model_path, map_location="cpu")

            state_dim = checkpoint.get("state_dim", 940)
            action_dim = checkpoint.get("action_dim", 11)

            state_dict = checkpoint["model_state_dict"]
            max_len = state_dict["embed_timestep.weight"].shape[0]

            model = DecisionTransformer(
                state_dim=state_dim,
                act_vocab_size=action_dim,
                n_layer=4,
                n_head=4,
                n_embd=128,
                max_len=max_len,
            )

            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()

            logger.info(
                f"Loaded model {info.key} from {model_path} (max_len={max_len})"
            )
            return model

        except Exception as e:
            logger.error(f"Failed to load model {info.key}: {e}")
            return None

    def _get_shared_model(self, model_type: str) -> Optional[nn.Module]:
        """Get or load a shared model (Q-network or State Predictor)"""
        if model_type in self._shared_models:
            return self._shared_models[model_type]

        if model_type == "q_network":
            from src.models.QNetwork import QNetwork

            model_path = self.model_dir / "cache/model/q_network.pth"
            if model_path.exists():
                checkpoint = torch.load(model_path, map_location="cpu")
                model = QNetwork(
                    state_dim=checkpoint.get("state_dim", 940),
                    action_dim=checkpoint.get("action_dim", 11),
                )
                model.load_state_dict(checkpoint["model_state_dict"])
                model.to(self.device)
                model.eval()
                self._shared_models["q_network"] = model
                return model

        elif model_type == "state_predictor":
            from src.models.StateTransitionPredictor import StateTransitionPredictor

            model_path = self.model_dir / "cache/model/state_predictor.pth"
            if model_path.exists():
                checkpoint = torch.load(model_path, map_location="cpu")
                model = StateTransitionPredictor(
                    state_dim=checkpoint.get("state_dim", 940),
                    action_dim=checkpoint.get("action_dim", 11),
                )
                model.load_state_dict(checkpoint["model_state_dict"])
                model.to(self.device)
                model.eval()
                self._shared_models["state_predictor"] = model
                return model

        return None

    def preload_all(self):
        """Preload all models in the pool"""
        for key in self.models:
            self.load_model(key)

    def select_model_for_history(self, available_history: int) -> str:
        """
        Select the best model for given history length

        Args:
            available_history: Number of available historical steps

        Returns:
            Key of the selected model
        """
        for key, info in self.models.items():
            if info.min_history <= available_history <= info.max_history:
                return key

        # Fallback to q_only if no model matches
        return "q_only"

    def get_model(self, model_key: str) -> Optional[nn.Module]:
        """Get a loaded model by key"""
        if model_key not in self.models:
            return None

        info = self.models[model_key]
        if not info.loaded:
            return self.load_model(model_key)

        return info.model

    def get_model_info(self, model_key: str) -> Optional[ModelInfo]:
        """Get model info by key"""
        return self.models.get(model_key)

    def get_fallback_chain(self, model_key: str) -> List[str]:
        """
        Get the fallback chain for a model

        Returns:
            List of model keys to try in order
        """
        chain = [model_key]
        info = self.models.get(model_key)

        while info and info.fallback:
            chain.append(info.fallback)
            info = self.models.get(info.fallback)

        return chain

    def get_all_available_models(self) -> List[str]:
        """Get list of all available (loaded) model keys"""
        return [
            k
            for k, v in self.models.items()
            if v.loaded or v.model_type == ModelType.Q_NETWORK
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the model pool"""
        stats = {
            "total_models": len(self.models),
            "loaded_models": sum(1 for v in self.models.values() if v.loaded),
            "models_by_type": {},
        }

        for key, info in self.models.items():
            type_name = info.model_type.value
            if type_name not in stats["models_by_type"]:
                stats["models_by_type"][type_name] = []
            stats["models_by_type"][type_name].append(
                {
                    "key": key,
                    "context_window": info.context_window,
                    "history_range": (info.min_history, info.max_history),
                    "loaded": info.loaded,
                }
            )

        return stats
