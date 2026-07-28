"""
Q-Network for Action Value Estimation
Q(s, a) -> expected cumulative reward
"""

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, state_dim=940, action_dim=11, hidden_dim=128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.state_embed = nn.Embedding(state_dim, hidden_dim)
        self.action_embed = nn.Embedding(action_dim, hidden_dim)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, state, action):
        """
        Args:
            state: (batch_size,) or (batch_size, seq_len) - state indices
            action: (batch_size,) or (batch_size, seq_len) - action indices
        Returns:
            q_value: (batch_size,) or (batch_size, seq_len) - Q values
        """
        if state.dim() == 2:
            s_emb = self.state_embed(state)
            a_emb = self.action_embed(action)
        else:
            s_emb = self.state_embed(state)
            a_emb = self.action_embed(action)

        combined = torch.cat([s_emb, a_emb], dim=-1)
        q_value = self.fc(combined).squeeze(-1)

        return q_value


class QNetworkTrainer:
    def __init__(self, model, lr=1e-4, weight_decay=0.01):
        self.model = model
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.criterion = nn.MSELoss()

    def train_step(self, states, actions, q_targets):
        """
        Args:
            states: (batch_size,)
            actions: (batch_size,)
            q_targets: (batch_size,) - ground truth Q values
        """
        self.model.train()
        self.optimizer.zero_grad()

        q_preds = self.model(states, actions)
        loss = self.criterion(q_preds, q_targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def evaluate(self, states, actions, q_targets):
        """Evaluate without gradient"""
        self.model.eval()
        with torch.no_grad():
            q_preds = self.model(states, actions)
            loss = self.criterion(q_preds, q_targets)
        return loss.item(), q_preds
