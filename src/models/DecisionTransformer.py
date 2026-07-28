import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_vocab_size, n_layer=4, n_head=4, n_embd=128, max_len=100):
        super().__init__()
        self.n_embd = n_embd

        # 1. 各个模态的 Embedding
        # RTG 是连续值，用线性层；State 如果是索引则用 Embedding
        self.embed_rtg = nn.Linear(1, n_embd)
        self.embed_state = nn.Embedding(state_dim + 1, n_embd)  # +1 为了处理 PAD
        self.embed_action = nn.Embedding(act_vocab_size + 1, n_embd)

        # 时间步 Embedding
        self.embed_timestep = nn.Embedding(max_len, n_embd)

        # 2. Transformer 主体 (GPT 架构)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=n_embd,
                nhead=n_head,
                dim_feedforward=n_embd * 4,
                batch_first=True,
                norm_first=True
            ),
            num_layers=n_layer
        )

        # 3. 预测头：预测 Action
        self.predict_action = nn.Sequential(
            nn.Linear(n_embd, act_vocab_size)
        )

    def forward(self, states, actions, rtgs, timesteps, padding_mask=None):
        batch_size, seq_len = states.shape

        # 计算各种 Embedding
        time_emb = self.embed_timestep(timesteps)
        # rtgs 形状补全为 (B, K, 1)
        rtg_emb = self.embed_rtg(rtgs.unsqueeze(-1)) + time_emb
        state_emb = self.embed_state(states) + time_emb
        action_emb = self.embed_action(actions) + time_emb

        # 交织序列: [R1, S1, A1, R2, S2, A2, ...]
        # 形状: (B, 3*K, D)
        stacked_inputs = torch.stack([rtg_emb, state_emb, action_emb], dim=2).reshape(batch_size, 3 * seq_len,
                                                                                      self.n_embd)

        # 如果有 Mask，需要扩展 3 倍长度
        if padding_mask is not None:
            # 简单处理：这里可以使用 causal mask 和 padding mask 的组合
            pass

            # 经过 Transformer
        outputs = self.transformer(stacked_inputs)

        # 我们要用 (Rt, St) 来预测 At，所以取每个三元组中 state 输出的位置
        # 在 3*K 长度中，state 处于索引 1, 4, 7... (即 3*i + 1)
        state_preds = outputs[:, 1::3, :]
        action_logits = self.predict_action(state_preds)

        return action_logits


# class DecisionTransformer(nn.Module):
#     def __init__(self, state_dim, act_vocab_size, n_embd=128, n_head=8, n_layer=4, max_len=512):
#         super().__init__()
#         self.d_model = n_embd
#
#         # 1. 输入投影层
#         # state_dim 应包含 PAD，所以大小为 num_states + 1
#         self.embed_rtg = nn.Linear(1, n_embd)
#         self.embed_state = nn.Embedding(state_dim + 1, n_embd)
#         self.embed_action = nn.Embedding(act_vocab_size + 1, n_embd)
#
#         # 位置编码 (针对时间步)
#         self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, n_embd))
#
#         # 2. Transformer 主体
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=n_embd, nhead=n_head, dim_feedforward=n_embd * 4,
#             batch_first=True, norm_first=True, dropout=0.1
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
#
#         # 3. 输出头：预测动作 logits
#         self.predict_action = nn.Linear(n_embd, act_vocab_size)
#
#     def forward(self, states, actions, rtgs, timesteps, padding_mask=None):
#         batch_size, seq_len = states.shape
#
#         # 因果掩码 (针对 3 * seq_len 长度)
#         mask = torch.triu(torch.ones(3 * seq_len, 3 * seq_len, device=states.device) * float('-inf'), diagonal=1)
#
#         # 4. 计算 Embeddings 并注入时间步信息 (Positional Encoding)
#         # 这里使用 timesteps 索引从 pos_embedding 中取值
#         time_emb = self.pos_embedding[:, :seq_len, :]
#
#         r_emb = self.embed_rtg(rtgs.unsqueeze(-1)) + time_emb
#         s_emb = self.embed_state(states) + time_emb
#         a_emb = self.embed_action(actions) + time_emb
#
#         # 5. 交织排列 [R1, S1, A1, R2, S2, A2, ...]
#         # 结果形状: (batch, 3 * seq_len, n_embd)
#         stacked_inputs = torch.stack([r_emb, s_emb, a_emb], dim=2).reshape(batch_size, 3 * seq_len, self.d_model)
#
#         # 6. 处理 Padding Mask (DT 的特殊处理)
#         # 原始 mask 是 [B, K]，需要拉伸到 [B, 3*K]
#         if padding_mask is not None:
#             dt_padding_mask = torch.stack([padding_mask, padding_mask, padding_mask], dim=2).reshape(batch_size,
#                                                                                                      3 * seq_len)
#         else:
#             dt_padding_mask = None
#
#         # 7. Transformer 计算
#         outputs = self.transformer(stacked_inputs, mask=mask.bool(), src_key_padding_mask=dt_padding_mask)
#
#         # 8. 预测动作：根据 (Rt, St) 预测 At，即取索引为 1, 4, 7... 的输出
#         action_logits = self.predict_action(outputs[:, 1::3, :])
#
#         return action_logits

# class DecisionTransformer(nn.Module):
#     def __init__(self, state_dim, act_vocab_size, n_embd=128, n_head=8, n_layer=4, max_len=2048):
#         super().__init__()
#         self.d_model = n_embd
#
#         # 1. 输入投影层 (注意：state_dim + 1 用于处理可能的 PAD)
#         self.embed_rtg = nn.Linear(1, n_embd)
#         self.embed_state = nn.Embedding(state_dim + 2, n_embd)
#         self.embed_action = nn.Embedding(act_vocab_size + 1, n_embd)
#         self.embed_ln = nn.LayerNorm(n_embd)
#
#         # 位置编码：DT 通常使用学习到的时间步编码
#         self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, n_embd))
#
#         # 2. Transformer 主体
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=n_embd, nhead=n_head, dim_feedforward=n_embd * 4,
#             batch_first=True, norm_first=True, dropout=0.1
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
#
#         # 3. 输出头
#         self.predict_action = nn.Linear(n_embd, act_vocab_size)
#
#     def forward(self, states, actions, rtgs, timesteps, padding_mask=None):
#         batch_size, seq_len = states.shape
#
#         # 时间步位置信息
#         time_emb = self.pos_embedding[:, :seq_len, :]
#
#         # 计算各模态 Embedding 并加入时间信息
#         r_emb = self.embed_rtg(rtgs.unsqueeze(-1)) + time_emb
#         s_emb = self.embed_state(states) + time_emb
#         a_emb = self.embed_action(actions) + time_emb
#
#         # 交织序列 [R1, S1, A1, R2, S2, A2, ...]
#         stacked_inputs = torch.stack([r_emb, s_emb, a_emb], dim=2).reshape(batch_size, 3 * seq_len, self.d_model)
#         stacked_inputs = self.embed_ln(stacked_inputs)
#
#         # 构造因果掩码 (Causal Mask)
#         mask = torch.triu(torch.ones(3 * seq_len, 3 * seq_len, device=states.device), diagonal=1).bool()
#
#         # 处理 Padding Mask
#         if padding_mask is not None:
#             dt_padding_mask = padding_mask.unsqueeze(-1).repeat(1, 1, 3).reshape(batch_size, 3 * seq_len)
#         else:
#             dt_padding_mask = None
#
#         outputs = self.transformer(stacked_inputs, mask=mask, src_key_padding_mask=dt_padding_mask)
#
#         # 预测动作：根据 [Rt, St] 预测 At，提取 St 对应的输出 (索引 1, 4, 7...)
#         action_logits = self.predict_action(outputs[:, 1::3, :])
#         return action_logits