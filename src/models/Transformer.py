import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TrajectoryTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=8, num_layers=4, max_len=512):
        super().__init__()
        self.d_model = d_model
        # vocab_size 已经包含了 PAD_ID
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            batch_first=True, norm_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, x, padding_mask=None):
        batch_size, seq_len = x.shape

        # 1. 因果掩码 (Causal Mask)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1)

        # 2. Embedding + Position
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = x + self.pos_embedding[:, :seq_len, :]

        # 3. Transformer 处理
        # padding_mask 对应的是 [batch, seq_len] 的布尔矩阵
        output = self.transformer(x, mask=mask.bool(), src_key_padding_mask=padding_mask)

        # 4. 预测 logits
        logits = self.output_head(output)
        return logits