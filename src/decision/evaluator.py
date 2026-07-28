"""
Decision Evaluator
Evaluates the quality of action decisions
"""

import torch
import numpy as np
from typing import List, Dict, Any
from collections import defaultdict


class DecisionEvaluator:
    def __init__(
        self,
        dt_model,
        q_network,
        state_predictor,
        action_vocab: Dict[str, int],
        id_to_action: Dict[int, str],
        device,
    ):
        self.dt_model = dt_model
        self.q_network = q_network
        self.state_predictor = state_predictor
        self.action_vocab = action_vocab
        self.id_to_action = id_to_action
        self.device = device
        self.num_actions = len(action_vocab)

    def evaluate_sample(
        self,
        states: List[int],
        actions: List[int],
        rtgs: List[float],
        cutoff: int,
    ) -> Dict[str, Any]:
        """
        Evaluate a single sample's decision quality

        Args:
            states: full state sequence
            actions: full action sequence
            rtgs: full RTG sequence
            cutoff: index to split history and future

        Returns:
            evaluation dict with detailed metrics
        """
        # Extract history and future
        history_states = states[:cutoff]
        history_actions = actions[:cutoff]
        history_rtgs = rtgs[:cutoff]

        true_future_actions = actions[cutoff:]
        true_future_states = states[cutoff:] if cutoff < len(states) else []

        current_state = history_states[-1]
        true_action = true_future_actions[0] if true_future_actions else None

        # Prepare DT input
        seq_len = len(history_states)
        states_tensor = torch.tensor([history_states], dtype=torch.long).to(self.device)
        actions_tensor = torch.tensor([history_actions], dtype=torch.long).to(
            self.device
        )
        rtgs_tensor = torch.tensor([history_rtgs], dtype=torch.float).to(self.device)
        timesteps_tensor = torch.arange(seq_len).unsqueeze(0).to(self.device)

        # Get DT predictions
        self.dt_model.eval()
        with torch.no_grad():
            logits = self.dt_model(
                states_tensor, actions_tensor, rtgs_tensor, timesteps_tensor
            )
            probs = torch.softmax(logits[0, -1, :], dim=-1)
            dt_pred_action = torch.argmax(probs).item()
            dt_pred_prob = probs[dt_pred_action].item()

        # Get top-3 actions
        topk_probs, topk_indices = torch.topk(probs, 3)
        dt_top3_actions = topk_indices.cpu().numpy().tolist()
        dt_top3_probs = topk_probs.cpu().numpy().tolist()

        # Evaluate Q-values for all actions
        self.q_network.eval()
        q_values = {}
        with torch.no_grad():
            state_tensor = torch.tensor([current_state], dtype=torch.long).to(
                self.device
            )
            for a in range(self.num_actions):
                action_tensor = torch.tensor([a], dtype=torch.long).to(self.device)
                q = self.q_network(state_tensor, action_tensor).item()
                q_values[a] = q

        # Best action by Q-value
        q_pred_action = max(q_values, key=q_values.get)

        # Hybrid score (DT prob + Q value)
        dt_weight = 0.3
        q_weight = 0.7

        q_array = np.array([q_values.get(a, 0) for a in dt_top3_actions])
        q_min, q_max = q_array.min(), q_array.max()
        if q_max > q_min:
            q_normalized = (q_array - q_min) / (q_max - q_min)
        else:
            q_normalized = np.ones_like(q_array)

        hybrid_scores = dt_weight * np.array(dt_top3_probs) + q_weight * q_normalized
        hybrid_pred_action = dt_top3_actions[np.argmax(hybrid_scores)]

        # Predict next state
        self.state_predictor.eval()
        with torch.no_grad():
            state_tensor = torch.tensor([current_state], dtype=torch.long).to(
                self.device
            )
            action_tensor = torch.tensor([hybrid_pred_action], dtype=torch.long).to(
                self.device
            )
            next_logits = self.state_predictor(state_tensor, action_tensor)
            next_probs = torch.softmax(next_logits[0], dim=-1)
            pred_next_state = torch.argmax(next_probs).item()
            pred_next_state_prob = next_probs[pred_next_state].item()

        true_next_state = true_future_states[0] if true_future_states else None

        # Compare with ground truth
        result = {
            # Basic info
            "episode_length": len(states),
            "cutoff": cutoff,
            "current_state": current_state,
            # Ground truth
            "true_action": true_action,
            "true_action_name": self.id_to_action.get(true_action, "unknown"),
            "true_next_state": true_next_state,
            # DT predictions
            "dt_pred_action": dt_pred_action,
            "dt_pred_action_name": self.id_to_action.get(dt_pred_action, "unknown"),
            "dt_pred_prob": dt_pred_prob,
            "dt_top3_actions": dt_top3_actions,
            "dt_top3_probs": dt_top3_probs,
            "dt_correct": dt_pred_action == true_action
            if true_action is not None
            else None,
            "dt_in_top3": true_action in dt_top3_actions
            if true_action is not None
            else None,
            # Q-values
            "q_values": q_values,
            "q_pred_action": q_pred_action,
            "q_pred_action_name": self.id_to_action.get(q_pred_action, "unknown"),
            "q_correct": q_pred_action == true_action
            if true_action is not None
            else None,
            "q_value_for_true_action": q_values.get(true_action, None)
            if true_action is not None
            else None,
            "q_value_for_pred_action": q_values.get(q_pred_action, None),
            # Hybrid prediction
            "hybrid_pred_action": hybrid_pred_action,
            "hybrid_pred_action_name": self.id_to_action.get(
                hybrid_pred_action, "unknown"
            ),
            "hybrid_correct": hybrid_pred_action == true_action
            if true_action is not None
            else None,
            "hybrid_scores": {a: s for a, s in zip(dt_top3_actions, hybrid_scores)},
            # State prediction
            "pred_next_state": pred_next_state,
            "pred_next_state_prob": pred_next_state_prob,
            "state_correct": pred_next_state == true_next_state
            if true_next_state is not None
            else None,
            # Quality metrics
            "better_than_true": q_values.get(hybrid_pred_action, 0)
            >= q_values.get(true_action, 0)
            if true_action is not None
            else None,
        }

        return result

    def evaluate_batch(
        self,
        samples: List[Dict],
        cutoff_ratio: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of samples

        Args:
            samples: list of sample dicts with 'states', 'actions', 'rtgs'
            cutoff_ratio: ratio to split history and future

        Returns:
            aggregated metrics
        """
        results = []

        for sample in samples:
            states = sample["states"]
            actions = sample["actions"]
            rtgs = sample["rtgs"]

            if len(states) < 5:
                continue

            cutoff = int(len(states) * cutoff_ratio)

            result = self.evaluate_sample(states, actions, rtgs, cutoff)
            result["sample_idx"] = sample.get("idx", None)
            results.append(result)

        # Aggregate metrics
        metrics = self._aggregate_metrics(results)

        return {
            "individual_results": results,
            "aggregated_metrics": metrics,
        }

    def _aggregate_metrics(self, results: List[Dict]) -> Dict[str, Any]:
        """Aggregate individual results into summary metrics"""
        n = len(results)
        if n == 0:
            return {}

        metrics = {
            "total_samples": n,
            # DT metrics
            "dt_accuracy": sum(1 for r in results if r["dt_correct"]) / n,
            "dt_top3_accuracy": sum(1 for r in results if r["dt_in_top3"]) / n,
            "avg_dt_prob": np.mean([r["dt_pred_prob"] for r in results]),
            # Q metrics
            "q_accuracy": sum(1 for r in results if r["q_correct"]) / n,
            "avg_q_value_for_pred": np.mean(
                [r["q_value_for_pred_action"] for r in results]
            ),
            "avg_q_value_for_true": np.mean(
                [
                    r["q_value_for_true_action"]
                    for r in results
                    if r["q_value_for_true_action"] is not None
                ]
            ),
            # Hybrid metrics
            "hybrid_accuracy": sum(1 for r in results if r["hybrid_correct"]) / n,
            # State prediction metrics
            "state_accuracy": sum(1 for r in results if r["state_correct"]) / n,
            "avg_state_prob": np.mean([r["pred_next_state_prob"] for r in results]),
            # Quality metrics
            "better_than_true_ratio": sum(1 for r in results if r["better_than_true"])
            / n,
        }

        return metrics
