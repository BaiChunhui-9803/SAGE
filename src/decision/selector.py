"""
Optimal Action Selector
Combines DT predictions with Q-values for optimal action selection
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional


class OptimalActionSelector:
    def __init__(
        self,
        dt_model,
        q_network,
        state_predictor,
        action_vocab: Dict[str, int],
        device,
        top_k: int = 3,
        dt_weight: float = 0.3,
        q_weight: float = 0.7,
    ):
        self.dt_model = dt_model
        self.q_network = q_network
        self.state_predictor = state_predictor
        self.action_vocab = action_vocab
        self.id_to_action = {v: k for k, v in action_vocab.items()}
        self.device = device
        self.top_k = top_k
        self.dt_weight = dt_weight
        self.q_weight = q_weight

        self.num_actions = len(action_vocab)

    def get_dt_topk_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rtgs: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> Tuple[List[int], np.ndarray]:
        """
        Get top-k actions from DT model predictions

        Returns:
            topk_actions: list of action indices
            topk_probs: array of probabilities
        """
        self.dt_model.eval()
        with torch.no_grad():
            logits = self.dt_model(states, actions, rtgs, timesteps)
            probs = F.softmax(logits[0, -1, :], dim=-1)

            topk_probs, topk_indices = torch.topk(probs, self.top_k)

            topk_actions = topk_indices.cpu().numpy().tolist()
            topk_probs = topk_probs.cpu().numpy()

        return topk_actions, topk_probs

    def evaluate_action_q_values(
        self, state: int, actions: List[int]
    ) -> Dict[int, float]:
        """
        Evaluate Q-values for each candidate action

        Args:
            state: current state index
            actions: list of candidate action indices

        Returns:
            q_values: dict mapping action -> Q value
        """
        self.q_network.eval()
        q_values = {}

        with torch.no_grad():
            state_tensor = torch.tensor([state], dtype=torch.long).to(self.device)

            for action in actions:
                action_tensor = torch.tensor([action], dtype=torch.long).to(self.device)
                q_value = self.q_network(state_tensor, action_tensor).item()
                q_values[action] = q_value

        return q_values

    def predict_next_state(
        self, state: int, action: int, temperature: float = 1.0
    ) -> int:
        """
        Predict the next state after taking an action

        Args:
            state: current state index
            action: action to take
            temperature: sampling temperature

        Returns:
            next_state: predicted next state index
        """
        self.state_predictor.eval()

        with torch.no_grad():
            state_tensor = torch.tensor([state], dtype=torch.long).to(self.device)
            action_tensor = torch.tensor([action], dtype=torch.long).to(self.device)

            logits = self.state_predictor(state_tensor, action_tensor)
            probs = F.softmax(logits / temperature, dim=-1)

            next_state = torch.multinomial(probs, num_samples=1).squeeze(-1).item()

        return next_state

    def select_action(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rtgs: torch.Tensor,
        timesteps: torch.Tensor,
        mode: str = "hybrid",
    ) -> Tuple[int, Dict]:
        """
        Select the optimal action using specified mode

        Args:
            states, actions, rtgs, timesteps: input tensors for DT model
            mode:
                - 'optimal': pure Q-value maximization
                - 'dt_top3': DT's top-3 actions, pick max Q
                - 'hybrid': DT probability + Q value weighted (recommended)

        Returns:
            selected_action: int - the selected action index
            info: dict - detailed information about selection
        """
        current_state = states[0, -1].item()

        # Get DT's top-k predictions
        topk_actions, topk_probs = self.get_dt_topk_actions(
            states, actions, rtgs, timesteps
        )

        if mode == "dt_only":
            selected_action = topk_actions[0]
            return selected_action, {
                "mode": mode,
                "dt_probs": {a: p for a, p in zip(topk_actions, topk_probs)},
                "q_values": None,
                "combined_scores": None,
            }

        # Evaluate Q-values for top-k actions
        q_values = self.evaluate_action_q_values(current_state, topk_actions)

        if mode == "dt_top3":
            # Select action with max Q among DT's top-3
            best_action = max(topk_actions, key=lambda a: q_values[a])
            selected_action = best_action
            combined_scores = {a: q_values[a] for a in topk_actions}

        elif mode == "optimal":
            # Evaluate ALL actions and pick max Q
            all_actions = list(range(self.num_actions))
            all_q_values = self.evaluate_action_q_values(current_state, all_actions)
            selected_action = max(all_actions, key=lambda a: all_q_values[a])
            q_values = all_q_values
            combined_scores = all_q_values

        elif mode == "hybrid":
            # Weighted combination of DT probs and Q values
            q_array = np.array([q_values.get(a, 0) for a in topk_actions])

            # Normalize Q values to [0, 1]
            q_min, q_max = q_array.min(), q_array.max()
            if q_max > q_min:
                q_normalized = (q_array - q_min) / (q_max - q_min)
            else:
                q_normalized = np.ones_like(q_array)

            # Combine scores
            combined_scores = self.dt_weight * topk_probs + self.q_weight * q_normalized

            # Select best action
            best_idx = np.argmax(combined_scores)
            selected_action = topk_actions[best_idx]

            combined_scores = {a: s for a, s in zip(topk_actions, combined_scores)}

        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Get next state prediction for selected action
        pred_next_state = self.predict_next_state(current_state, selected_action)

        return selected_action, {
            "mode": mode,
            "dt_probs": {a: p for a, p in zip(topk_actions, topk_probs)},
            "q_values": q_values,
            "combined_scores": combined_scores,
            "pred_next_state": pred_next_state,
        }
