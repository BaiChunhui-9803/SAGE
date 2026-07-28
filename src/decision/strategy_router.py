"""
Strategy Router for Adaptive Decision System
Selects appropriate model and strategy based on available history
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class DecisionStrategy(Enum):
    Q_ONLY = "q_only"
    DT_ONLY = "dt_only"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"


@dataclass
class RoutingDecision:
    """Result of strategy routing"""

    model_key: str
    strategy: DecisionStrategy
    prediction_steps: int
    context_window: int
    confidence_threshold: float
    fallback_model: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class StrategyRouter:
    """
    Routes to appropriate decision strategy based on context
    """

    def __init__(
        self,
        model_pool,
        confidence_thresholds: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize strategy router

        Args:
            model_pool: ModelPoolManager instance
            confidence_thresholds: Custom confidence thresholds
        """
        self.model_pool = model_pool
        self.confidence_thresholds = confidence_thresholds or {
            "multi_step": 0.8,
            "single_step": 0.5,
            "fallback": 0.3,
        }

    def route(
        self,
        available_history: int,
        current_confidence: Optional[float] = None,
        context: Optional[Dict] = None,
    ) -> RoutingDecision:
        """
        Route to appropriate model and strategy

        Args:
            available_history: Number of historical steps available
            current_confidence: Current model confidence (if known)
            context: Additional context information

        Returns:
            RoutingDecision with selected model, strategy, and parameters
        """
        model_key = self.model_pool.select_model_for_history(available_history)
        model_info = self.model_pool.get_model_info(model_key)

        if model_key == "q_only":
            return RoutingDecision(
                model_key="q_only",
                strategy=DecisionStrategy.Q_ONLY,
                prediction_steps=1,
                context_window=0,
                confidence_threshold=0.0,
                fallback_model=None,
                metadata={
                    "reason": "insufficient_history",
                    "available_history": available_history,
                },
            )

        prediction_steps = self._determine_prediction_steps(
            model_info.prediction_steps, current_confidence, available_history
        )

        strategy = self._determine_strategy(
            model_info.strategy, current_confidence, available_history
        )

        confidence_threshold = self._get_confidence_threshold(prediction_steps)

        return RoutingDecision(
            model_key=model_key,
            strategy=strategy,
            prediction_steps=prediction_steps,
            context_window=model_info.context_window,
            confidence_threshold=confidence_threshold,
            fallback_model=model_info.fallback,
            metadata={
                "available_history": available_history,
                "model_context_window": model_info.context_window,
                "model_description": model_info.description,
            },
        )

    def _determine_prediction_steps(
        self,
        available_steps: List[int],
        current_confidence: Optional[float],
        available_history: int,
    ) -> int:
        """Determine number of steps to predict"""
        if current_confidence is None:
            return 1

        if current_confidence >= self.confidence_thresholds["multi_step"]:
            if available_history >= 15 and max(available_steps) > 1:
                return max(available_steps)

        if current_confidence >= self.confidence_thresholds["single_step"]:
            return 1

        return 1

    def _determine_strategy(
        self,
        default_strategy: str,
        current_confidence: Optional[float],
        available_history: int,
    ) -> DecisionStrategy:
        """Determine decision strategy"""
        if available_history < 5:
            return DecisionStrategy.Q_ONLY

        if current_confidence is not None:
            if current_confidence < self.confidence_thresholds["fallback"]:
                return DecisionStrategy.Q_ONLY

        if default_strategy == "hybrid":
            return DecisionStrategy.HYBRID
        elif default_strategy == "dt_only":
            return DecisionStrategy.DT_ONLY
        else:
            return DecisionStrategy.HYBRID

    def _get_confidence_threshold(self, prediction_steps: int) -> float:
        """Get confidence threshold for prediction steps"""
        if prediction_steps > 1:
            return self.confidence_thresholds["multi_step"]
        return self.confidence_thresholds["single_step"]


class ConfidenceEstimator:
    """
    Estimates decision confidence from multiple signals
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.weights = weights or {
            "dt_prob": 0.4,
            "q_consistency": 0.2,
            "history_score": 0.2,
            "validation_score": 0.2,
        }

    def estimate(
        self,
        dt_probs: List[float],
        q_values: Dict[int, float],
        selected_action: int,
        available_history: int,
        max_history: int = 20,
        historical_validation_loss: Optional[float] = None,
    ) -> float:
        """
        Estimate overall decision confidence

        Args:
            dt_probs: DT probability distribution
            q_values: Q-values for actions
            selected_action: Selected action ID
            available_history: Available history length
            max_history: Maximum history for full confidence
            historical_validation_loss: Historical validation loss for the model

        Returns:
            Confidence score between 0 and 1
        """
        dt_confidence = max(dt_probs) if dt_probs else 0.0

        best_q_action = max(q_values, key=q_values.get) if q_values else selected_action
        q_consistency = 1.0 if selected_action == best_q_action else 0.5

        history_score = min(available_history / max_history, 1.0)

        validation_score = 1.0
        if historical_validation_loss is not None:
            validation_score = max(0.0, 1.0 - historical_validation_loss)

        confidence = (
            self.weights["dt_prob"] * dt_confidence
            + self.weights["q_consistency"] * q_consistency
            + self.weights["history_score"] * history_score
            + self.weights["validation_score"] * validation_score
        )

        return min(1.0, max(0.0, confidence))
