import matplotlib.pyplot as plt
import os

import numpy as np

# 设置为你想要使用的逻辑核心数，例如 4 或 8
os.environ["LOKY_MAX_CPU_COUNT"] = "8"

from sklearn.manifold import TSNE


def plot_embedding(model, save_path, state_names=None):
    # 1. 提取权重 (假设 num_states 是你实际感兴趣的状态数量)
    weights = model.embedding.weight.detach().cpu().numpy()

    # 2. 使用 t-SNE 降维到 2D
    tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)
    embeddings_2d = tsne.fit_transform(weights)

    colors = np.linspace(0, 1, len(weights))

    plt.figure(figsize=(12, 9))
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=colors,  # 传入数值映射到颜色
        cmap='viridis',  # 指定使用 viridis 配色
        s=50,  # 点的大小
        alpha=0.8,  # 透明度
        edgecolors='none'
    )

    # 4. 添加色彩条
    cbar = plt.colorbar(scatter)
    cbar.set_label('State Index / Property Value', rotation=270, labelpad=15)

    plt.title("Embedding Space Visualization (MDS Initialized)", fontsize=15)
    plt.xlabel("t-SNE component 1")
    plt.ylabel("t-SNE component 2")
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.savefig(save_path, dpi=300)
