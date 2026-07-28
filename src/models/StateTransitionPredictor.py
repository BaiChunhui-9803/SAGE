"""
State Transition Predictor
Predicts the next state given current state and action: (s_t, a_t) -> s_{t+1}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StateTransitionPredictor(nn.Module):
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
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, state, action):
        """
        Args:
            state: (batch_size,) - current state index
            action: (batch_size,) - action index
        Returns:
            logits: (batch_size, state_dim) - logits for next state prediction
        """
        s_emb = self.state_embed(state)
        a_emb = self.action_embed(action)

        combined = torch.cat([s_emb, a_emb], dim=-1)
        logits = self.fc(combined)

        return logits

    def predict(self, state, action, temperature=1.0):
        """Predict next state with sampling"""
        logits = self.forward(state, action)
        probs = F.softmax(logits / temperature, dim=-1)

        if self.training:
            next_state = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            next_state = torch.argmax(probs, dim=-1)

        return next_state

    def predict_top_k(self, state, action, k=3, temperature=1.0):
        """Predict top-k most likely next states"""
        logits = self.forward(state, action)
        probs = F.softmax(logits / temperature, dim=-1)

        top_k_probs, top_k_indices = torch.topk(probs, k, dim=-1)

        return top_k_indices, top_k_probs


class StateTransitionTrainer:
    def __init__(self, model, lr=1e-4, weight_decay=0.01):
        self.model = model
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()

    def train_step(self, state, action, next_state):
        """
        Args:
            state: (batch_size,)
            action: (batch_size,)
            next_state: (batch_size,) - target state index
        """
        self.model.train()
        self.optimizer.zero_grad()

        logits = self.model(state, action)
        loss = self.criterion(logits, next_state)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def evaluate(self, state, action, next_state):
        """Evaluate accuracy and loss"""
        self.model.eval()
        with torch.no_grad():
            logits = self.model(state, action)
            loss = self.criterion(logits, next_state)

            preds = torch.argmax(logits, dim=-1)
            accuracy = (preds == next_state).float().mean()

            top3_preds = torch.topk(logits, 3, dim=-1).indices
            top3_correct = (
                torch.any(top3_preds == next_state.unsqueeze(1), dim=1).float().mean()
            )

        return loss.item(), accuracy.item(), top3_correct.item()
