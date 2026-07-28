from src.config.base_config import *
from src.train.init import *
from src.data.load_data import *
from src.models.Transformer import TrajectoryTransformer
from src.models.SGTransformer import SimilarityGuidedTransformer
from src.models.DecisionTransformer import DecisionTransformer
from src.utils.model_utils import *
from src.plot.plot_prediction import *
from src.plot.plot_model import *

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


def train_dt(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for states, actions, rtgs, timesteps in train_loader:
        states, actions, rtgs, timesteps = [
            x.to(device) for x in [states, actions, rtgs, timesteps]
        ]

        # 预测动作
        action_logits = model(states, actions, rtgs, timesteps)

        # Loss 只计算有效的动作预测 (不含最后一个动作或 padding)
        # 注意：这里 labels 是真实 actions
        loss = criterion(
            action_logits.view(-1, action_logits.size(-1)), actions.view(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def batch_dt_func(d_model=128, epochs=50, batch_size=64, lr=1e-4, context_window=20):
    # --- 0. 环境配置与路径 ---
    set_seed(42)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CACHE_DIR = "cache/model"
    os.makedirs(CACHE_DIR, exist_ok=True)

    MODEL_NAME = f"decision_transformer_d{d_model}_e{epochs}{suffix}.pth"
    MODEL_PATH = os.path.join(CACHE_DIR, MODEL_NAME)

    # --- 1. 数据准备与 RTG 分布提取 ---
    # 计算训练集中每条轨迹的总奖励 (Return)
    # 假设 r_log 是 List[List[float]]，即每个元素是一个轨迹的奖励序列
    traj_returns = [sum(traj) for traj in r_log]
    rtg_stats = {
        "max": float(np.max(traj_returns)),
        "min": float(np.min(traj_returns)),
        "mean": float(np.mean(traj_returns)),
        "p90": float(np.percentile(traj_returns, 90)),
        "std": float(np.std(traj_returns)),
    }
    logging.info(
        f"数据分布: Max Return={rtg_stats['max']:.4f}, Mean={rtg_stats['mean']:.4f}"
    )

    dt_data, action_vocab = preprocess_decision_transformer_data(
        state_log, action_log, r_log
    )
    num_real_states = len(state_distance_matrix)
    ACT_VOCAB_SIZE = len(action_vocab)

    # 划分数据集
    indices = np.arange(len(dt_data["states"]))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_data = {k: [v[i] for i in train_idx] for k, v in dt_data.items()}

    train_dataset = DTDataset(train_data, context_window=context_window)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # --- 2. 模型初始化 ---
    model = DecisionTransformer(
        state_dim=num_real_states,
        act_vocab_size=ACT_VOCAB_SIZE,
        n_embd=d_model,
        max_len=2048,
    ).to(DEVICE)

    # --- 3. 应用 MDS 初始化 ---
    if params.get("mdl_init_embedding_freeze") or params.get(
        "mdl_init_embedding_train"
    ):
        init_dt_state_embedding(model, state_distance_matrix, d_model)
        if params.get("mdl_init_embedding_freeze"):
            model.embed_state.weight.requires_grad = False
            logging.info("MDS State Embedding fixed.")

    # --- 4. 训练流程 ---
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    criterion = nn.CrossEntropyLoss()

    loss_history = []  # 用于存储每一轮的 loss
    logging.info(f"开始训练. 目标路径: {MODEL_PATH}")

    for epoch in range(epochs):
        avg_loss = train_dt(model, train_loader, optimizer, criterion, DEVICE)
        loss_history.append(avg_loss)  # 记录 Loss

        if (epoch + 1) % 5 == 0:
            logging.info(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f}")

    # --- 5. 导出 Loss 到文本文件 (新增逻辑) ---
    # 定义导出路径（和模型放在同一目录下，但后缀不同）
    # LOSS_LOG_PATH = 'output/data/decision/train_loss_with_cause_mask.txt'
    # try:
    #     with open(LOSS_LOG_PATH, 'w', encoding='utf-8') as f:
    #         f.write("Epoch,Average_Loss\n")
    #         for i, loss_val in enumerate(loss_history):
    #             f.write(f"{i + 1},{loss_val:.6f}\n")
    #     logging.info(f"训练 Loss 已导出至: {LOSS_LOG_PATH}")
    # except Exception as e:
    #     logging.error(f"Loss 导出失败: {e}")

    # --- 6. 增强的保存逻辑 ---
    # 包含了推理所需的全部上下文：权重、词典、模型参数、奖励分布
    save_content = {
        "model_state_dict": model.state_dict(),
        "action_vocab": action_vocab,
        "id_to_action": {v: k for k, v in action_vocab.items()},
        "rtg_stats": rtg_stats,  # 关键：存入奖励分布，供推理时设置 target_return
        "config": {
            "state_dim": num_real_states,
            "act_vocab_size": ACT_VOCAB_SIZE,
            "n_embd": d_model,
            "context_window": context_window,
            "max_len": 2048,
        },
    }

    torch.save(save_content, MODEL_PATH)
    logging.info(f"训练完成！RTG 统计与权重已持久化。")

    return model, action_vocab


def load_dt_model(path, device):
    checkpoint = torch.load(path, map_location=device)
    cfg = checkpoint["config"]

    max_len = cfg.get("max_len", 2048)

    # 1. 重建结构
    model = DecisionTransformer(
        state_dim=cfg["state_dim"],
        act_vocab_size=cfg["act_vocab_size"],
        n_embd=cfg["n_embd"],
        max_len=max_len,
    ).to(device)

    # 2. 加载权重
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint["action_vocab"]


def evaluate_action_accuracy_detailed(model_path, dt_data, device):
    """
    1. 还原动作字符串
    2. 流式呈现全序列预测结果
    3. 关联 log_fitness 指标
    """
    # --- 1. 加载模型及元数据 ---
    checkpoint = torch.load(model_path, map_location=device)
    cfg = checkpoint["config"]
    action_vocab = checkpoint["action_vocab"]
    # 建立 ID 到字符串的映射
    id_to_action = {v: k for k, v in action_vocab.items()}
    ACTION_PAD_ID = cfg["act_vocab_size"]

    # --- 2. 严格对齐测试集划分 (包括 Fitness) ---
    indices = np.arange(len(dt_data["states"]))
    _, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

    # 提取测试数据及对应的 fitness 值
    test_data = {k: [v[i] for i in test_idx] for k, v in dt_data.items()}
    test_fitness = [log_fitness[i] for i in test_idx]

    test_dataset = DTDataset(test_data, context_window=cfg["context_window"])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # --- 3. 重建模型 ---
    model = DecisionTransformer(
        state_dim=cfg["state_dim"],
        act_vocab_size=cfg["act_vocab_size"],
        n_embd=cfg["n_embd"],
        max_len=cfg.get("max_len", 2048),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    stats_results = []  # 用于存储统计数据

    total_correct = 0
    total_valid_steps = 0

    print("\n" + "=" * 120)
    print(
        f"{'测试索引':<8} | {'Fitness':<10} | {'预测流 (Prediction Stream)':<45} | {'准确率'}"
    )
    print("-" * 120)

    with torch.no_grad():
        for i, (states, actions, rtgs, timesteps) in enumerate(test_loader):
            states, actions, rtgs, timesteps = [
                x.to(device) for x in [states, actions, rtgs, timesteps]
            ]

            # 前向传播
            action_logits = model(states, actions, rtgs, timesteps)
            preds = torch.argmax(action_logits, dim=-1)[0]  # 取出 batch 中的第一条
            target_actions = actions[0]

            # 过滤 Padding 后的对比
            valid_idx = (target_actions != ACTION_PAD_ID).nonzero(as_tuple=True)[0]

            p_stream = []
            a_stream = []
            num_correct = 0

            for idx in valid_idx:
                p_id = preds[idx].item()
                a_id = target_actions[idx].item()

                # 转换回字符串动作，如果 ID 没在字典里（例如预测了 PAD），标记为 [P]
                p_str = id_to_action.get(p_id, "[P]")
                a_str = id_to_action.get(a_id, "[P]")

                # 为了可视化美观，如果是预测正确显示动作，错误则显示 "预测/实际"
                if p_id == a_id:
                    p_stream.append(f"{p_str}")
                    num_correct += 1
                else:
                    p_stream.append(
                        f"\033[91m{p_str}({a_str})\033[00m"
                    )  # 红色高亮错误项

            # 统计
            num_valid = len(valid_idx)
            acc = (num_correct / num_valid) * 100 if num_valid > 0 else 0
            total_correct += num_correct
            total_valid_steps += num_valid

            stats_results.append(
                {
                    "test_index": int(test_idx[i]),
                    "fitness": float(test_fitness[i]),
                    "accuracy": float(acc),
                }
            )

            # 输出这一轨迹的流式结果
            stream_str = " -> ".join(p_stream[:8]) + (
                " ..." if len(p_stream) > 8 else ""
            )
            print(
                f"{test_idx[i]:<12} | {test_fitness[i]:<10.4f} | {stream_str:<45} | {acc:>6.2f}%"
            )

    final_acc = (total_correct / total_valid_steps) * 100
    print("-" * 120)
    print(
        f"评估总结: 总有效步数 {total_valid_steps} | 总体平均准确率: {final_acc:.2f}%"
    )
    return final_acc, stats_results
