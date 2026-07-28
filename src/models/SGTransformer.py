import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GuidedMultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # 可学习的缩放因子，让模型决定对距离矩阵的依赖程度
        self.bias_scale = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x, attn_mask=None, key_padding_mask=None, sim_bias=None):
        """
        sim_bias: [batch, seq_len, seq_len]
        """
        if sim_bias is not None:
            # 将 sim_bias 扩展并与 causal mask 合并
            # MHA 的 attn_mask 形状要求是 [num_heads*batch, seq_len, seq_len]
            # 或者 [seq_len, seq_len] (如果全 batch 通用)

            # 这里的逻辑是：原有的因果掩码 + 缩放后的距离偏置
            combined_mask = attn_mask + (sim_bias * self.bias_scale)
        else:
            combined_mask = attn_mask

        # 注意：PyTorch 的 MultiheadAttention 在传入 attn_mask 时，
        # 内部实际上是做加法。我们直接利用这一点。
        return self.mha(x, x, x, attn_mask=combined_mask.bool(), key_padding_mask=key_padding_mask)


class SimilarityGuidedTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=8, num_layers=4, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))

        # 手动构建层列表，因为我们要传入自定义参数
        self.layers = nn.ModuleList([
            # 这里简化处理，实际中可以封装成自定义的 EncoderLayer 类
            # 重点在于每一层都要能处理 sim_bias
            GuidedMultiHeadAttention(d_model, nhead) for _ in range(num_layers)
        ])

        # 为了演示简洁，这里定义层归一化和前馈网络
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.ReLU(),
                nn.Linear(d_model * 4, d_model)
            ) for _ in range(num_layers)
        ])

        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, x, padding_mask=None, sim_bias=None):
        """
        sim_bias: 从外部传入的已经计算好的距离偏置矩阵 [batch, seq_len, seq_len]
        """
        batch_size, seq_len = x.shape

        # 1. 因果掩码
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1)

        # 2. Embedding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = x + self.pos_embedding[:, :seq_len, :]

        # 3. 逐层处理 (Guided Attention)
        for attn_layer, norm, ffn in zip(self.layers, self.norms, self.ffns):
            # 残差连接 + Attention
            # 注意：PyTorch MHA 返回 (output, weights)，取第一个
            attn_out, _ = attn_layer(x, attn_mask=causal_mask.bool(),
                                     key_padding_mask=padding_mask,
                                     sim_bias=sim_bias)
            x = norm(x + attn_out)

            # 残差连接 + FFN
            ffn_out = ffn(x)
            x = norm(x + ffn_out)

        logits = self.output_head(x)
        return logits