from sklearn.manifold import MDS
import torch
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import logging


def init_embedding_with_dm(model, state_dm, embedding_dim):
    """利用 MDS 初始化，并进行归一化防止梯度爆炸"""
    logging.info("Initializing Embedding with MDS...")
    # 归一化距离矩阵到 [0, 1]
    max_d = np.max(state_dm) if np.max(state_dm) > 0 else 1
    norm_dm = state_dm / max_d

    mds = MDS(n_components=embedding_dim, dissimilarity='precomputed', random_state=42, n_init=1, normalized_stress='auto')
    weights = mds.fit_transform(norm_dm)

    # 标准化权重范围
    weights = (weights - weights.mean()) / (weights.std() + 1e-5) * 0.02

    with torch.no_grad():
        # 注意：PAD_ID 对应的权重保持为 0
        num_states = weights.shape[0]
        model.embedding.weight[:num_states].copy_(torch.from_numpy(weights))
    logging.info(f"MDS Embedding ready. Range: {weights.min():.4f} to {weights.max():.4f}")

class RLTrajectoryDataset(Dataset):
    def __init__(self, state_log):
        # 过滤过短的序列
        self.samples = [torch.tensor(seq, dtype=torch.long) for seq in state_log if len(seq) > 2]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn_builder(pad_id):
    def _collate(batch):
        return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=pad_id)
    return _collate



class DTDataset(Dataset):
    def __init__(self, data, context_window=20):
        self.states = data['states']
        self.actions = data['actions']
        self.rtgs = data['rtgs']
        self.K = context_window

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        s = self.states[idx]
        a = self.actions[idx]
        r = self.rtgs[idx]

        # 截取最后的 K 步 (或随机截取)
        tlen = len(s)
        s = torch.tensor(s[-self.K:], dtype=torch.long)
        a = torch.tensor(a[-self.K:], dtype=torch.long)
        r = torch.tensor(r[-self.K:], dtype=torch.float32)

        # 构造时间步索引 (比如 0, 1, 2...)
        timesteps = torch.arange(max(0, tlen - self.K), tlen, dtype=torch.long)

        # Padding (如果序列长度不足 K)
        p_len = self.K - len(s)
        if p_len > 0:
            s = torch.cat([torch.zeros(p_len, dtype=torch.long), s])
            a = torch.cat([torch.zeros(p_len, dtype=torch.long), a])
            r = torch.cat([torch.zeros(p_len), r])
            timesteps = torch.cat([torch.zeros(p_len, dtype=torch.long), timesteps])

        return s, a, r, timesteps


def init_dt_state_embedding(model, state_dm, embedding_dim):
    """专门针对 DT 的 embed_state 层进行 MDS 初始化"""
    logging.info("Using MDS to initialize DT State Embedding...")

    # 1. 归一化距离矩阵
    max_d = np.max(state_dm) if np.max(state_dm) > 0 else 1
    norm_dm = state_dm / max_d

    # 2. MDS 计算
    mds = MDS(n_components=embedding_dim, dissimilarity='precomputed',
              random_state=42, n_init=1, normalized_stress='auto')
    weights = mds.fit_transform(norm_dm)

    # 3. 标准化
    weights = (weights - weights.mean()) / (weights.std() + 1e-5) * 0.02

    # 4. 拷贝到模型
    with torch.no_grad():
        num_states = weights.shape[0]
        # 确保只初始化实际的状态 ID，保留最后的 PAD_ID 默认为随机或 0
        model.embed_state.weight[:num_states].copy_(torch.from_numpy(weights))

    logging.info(f"DT State Embedding initialized. Shape: {weights.shape}")