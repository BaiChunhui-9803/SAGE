from src.config.base_config import *
from src.train.init import *
from src.data.load_data import *
from src.models.Transformer import TrajectoryTransformer
from src.models.SGTransformer import SimilarityGuidedTransformer
from src.utils.model_utils import (
    RLTrajectoryDataset,
    init_embedding_with_dm,
    collate_fn_builder,
)
from src.plot.plot_prediction import *
from src.plot.plot_model import *

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


def train_one_epoch(model, loader, optimizer, criterion, device, pad_id):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        if batch.size(1) < 2:
            continue
        inputs, targets = batch[:, :-1], batch[:, 1:]
        mask = inputs == pad_id
        optimizer.zero_grad()
        logits = model(inputs, padding_mask=mask)
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def predict_k_steps(model, initial_seq, k, state_dm, device, pad_id):
    model.eval()  # 确保处于评估模式
    current_tokens = list(initial_seq)
    for _ in range(k):
        x = torch.tensor([current_tokens], dtype=torch.long).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits[0, -1, :], dim=-1)
            if params["mdl_spatial_prior"]:
                probs[pad_id] = 0
                curr_s = current_tokens[-1]
                if curr_s < len(state_dm):
                    dists = torch.from_numpy(state_dm[curr_s]).to(device)
                    # 空间先验引导
                    spatial_prior = torch.exp(-(dists**2) / (2 * 1.5**2))
                    probs[: len(dists)] *= spatial_prior
            next_token = torch.argmax(probs).item()
            current_tokens.append(next_token)
    return current_tokens[len(initial_seq) :]


def evaluate_test_performance(model, test_logs, k, state_dm, device, pad_id):
    model.eval()
    all_step_errors = []
    n_in = N
    for traj in test_logs:
        if len(traj) < n_in + k:
            continue
        input_data = traj[:n_in]
        actual_next = traj[n_in : n_in + k]
        preds = predict_k_steps(model, input_data, k, state_dm, device, pad_id)
        traj_errors = [state_dm[p][a] for p, a in zip(preds, actual_next)]
        all_step_errors.append(traj_errors)
    if not all_step_errors:
        return None
    return np.mean(all_step_errors, axis=0)


def print_final_comparison(
    model,
    test_indices,
    test_fitness_list,
    test_logs,
    state_dm,
    device,
    pad_id,
    k_steps=K,
    n_input=N,
):
    print("\n" + "=" * 110)
    header = "{:<8} | {:<6} | {:<65} | {}".format(
        "索引", "类型", f"序列内容 (前{N}步输入 -> 后{K}步预测/实际)", "平均误差"
    )
    print(header)
    print("-" * 110)
    # 固定选取测试集的前几个，方便对比
    # sample_indices = [0, 1, 3, 4, 7, 9, 10, 11]
    sample_indices = range(int(len(test_logs)))
    results = {}
    excel_data = []
    for idx in sample_indices:
        traj = test_logs[idx]
        if len(traj) < n_input + k_steps:
            continue
        input_data = traj[:n_input]
        actual_next = traj[n_input : n_input + k_steps]
        preds = predict_k_steps(model, input_data, k_steps, state_dm, device, pad_id)
        step_errors = [state_dm[p][a] for p, a in zip(preds, actual_next)]
        avg_err = np.mean(step_errors)
        input_str = str(list(input_data))
        results[idx] = {
            "input": input_data,
            "actual": actual_next,
            "preds": preds,
            "actual_fitness": test_fitness_list[idx],
            "step_errors": step_errors,
            "avg_err": avg_err,
        }

        row_info = {
            "Index": idx,
            "Original_Index": test_indices[idx],  # 之前提到的原 state_log 下标
            "Input_Path": input_str,
            "Actual_Next": str(actual_next),
            "Predicted_Next": str(preds),
            "Avg_Error": round(avg_err, 3),
            "Step_Details": " -> ".join([f"{e:.2f}" for e in step_errors]),
            "Fitness": test_fitness_list[idx],  # 关联的 fitness
        }
        excel_data.append(row_info)
        print(f"Test_{idx:<4} | 实际 | {input_str} -> {actual_next} | {avg_err:.3f}")
        print(f"{'':<9} | 预测 | {' ' * len(input_str)} -> {preds} | 距离流:")
        print(
            f"{'':<9} | 详情 | {' ' * len(input_str)} {' -> '.join([f'{e:.2f}' for e in step_errors])} |"
        )
        print("-" * 110)

    df = pd.DataFrame(excel_data)
    save_filename = f"output/data/{model_name}/prediction_results{suffix}.xlsx"
    df.to_excel(save_filename, index=False)
    return results


def batch_predict_func(
    d_model, epochs, batch_size, lr, f_plot_pred_FL=False, f_plot_embedding=False
):
    # --- 0. 固定随机性 ---
    set_seed(42)

    # --- 1. 参数与环境配置 ---
    NUM_REAL_STATES = len(state_distance_matrix)
    PAD_ID = NUM_REAL_STATES
    VOCAB_SIZE = NUM_REAL_STATES + 1

    D_MODEL = d_model
    EPOCHS = epochs
    BATCH_SIZE = batch_size
    LR = lr
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    METRICS_FILE = f"output/data/{model_name}/test_metrics_step_wise{suffix}.txt"
    CACHE_DIR = "cache/model"
    if params["mdl_attn_sim_bias"]:
        MODEL_NAME = f"sg_transformer_v{VOCAB_SIZE}_d{D_MODEL}_e{EPOCHS}{suffix}.pth"
    else:
        MODEL_NAME = f"traj_transformer_v{VOCAB_SIZE}_d{D_MODEL}_e{EPOCHS}{suffix}.pth"
    MODEL_PATH = os.path.join(CACHE_DIR, MODEL_NAME)

    os.makedirs(CACHE_DIR, exist_ok=True)

    # --- 2. 数据准备 (因为固定了随机种子，每次划分的 test_logs 将完全一致) ---
    all_indices = np.arange(len(state_log))
    train_indices, test_indices = train_test_split(
        all_indices, test_size=0.2, random_state=42, shuffle=True
    )
    train_logs = [state_log[i] for i in train_indices]
    test_logs = [state_log[i] for i in test_indices]
    test_fitness_list = [log_fitness[i] for i in test_indices]

    # train_logs, test_logs = train_test_split(state_log, test_size=0.2, random_state=42, shuffle=True)
    train_dataset = RLTrajectoryDataset(train_logs)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn_builder(PAD_ID),
    )

    # --- 3. 模型初始化 ---
    if params["mdl_attn_sim_bias"]:
        model = SimilarityGuidedTransformer(VOCAB_SIZE, d_model=D_MODEL).to(DEVICE)
    else:
        model = TrajectoryTransformer(VOCAB_SIZE, d_model=D_MODEL).to(DEVICE)

    if params["mdl_init_embedding_freeze"] or params["mdl_init_embedding_train"]:
        init_embedding_with_dm(model, state_distance_matrix, D_MODEL)
        if params["mdl_init_embedding_freeze"]:
            model.embedding.weight.requires_grad = False

    if os.path.exists(MODEL_PATH):
        logging.info(f"检测到匹配的模型文件: {MODEL_PATH}，正在加载权重...")
        # 修正：使用 load_state_dict 而不是直接赋值
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        logging.info("模型权重加载成功。")
    else:
        logging.info(f"未找到预训练模型，开始进入训练流程...")
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR,
            weight_decay=0.01,
        )
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

        steps_list = [f"Step_{i}_Err" for i in range(1, K + 1)]
        header = (
            ",".join(["Epoch", "Train_Loss"] + steps_list + ["Total_Mean_Err"]) + "\n"
        )

        with open(METRICS_FILE, "w") as f:
            f.write(header)

        for epoch in range(EPOCHS):
            avg_train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, DEVICE, PAD_ID
            )
            test_step_errors = evaluate_test_performance(
                model, test_logs, K, state_distance_matrix, DEVICE, PAD_ID
            )

            if f_plot_embedding:
                model_folder_name = MODEL_NAME.replace(".pth", "")
                embedding_dir = f"output/embedding/{model_folder_name}"
                embedding_filename = f"train_epoch_{epoch}.pdf"

                save_path = os.path.join(embedding_dir, embedding_filename)

                if not os.path.exists(embedding_dir):
                    os.makedirs(embedding_dir, exist_ok=True)
                    logging.info(f"创建 Embedding 存储目录: {embedding_dir}")

                    # 4. 执行绘图逻辑
                plot_embedding(model, save_path)

            if test_step_errors is not None:
                overall_mean = np.mean(test_step_errors)
                metrics_row = (
                    f"{epoch + 1},{avg_train_loss:.4f},"
                    + ",".join([f"{e:.4f}" for e in test_step_errors])
                    + f",{overall_mean:.4f}\n"
                )
                with open(METRICS_FILE, "a") as f:
                    f.write(metrics_row)
                logging.info(
                    f"Epoch {epoch + 1}/{EPOCHS} - Loss: {avg_train_loss:.4f} - Test Mean Err: {overall_mean:.4f}"
                )

        torch.save(model.state_dict(), MODEL_PATH)
        logging.info(f"模型已持久化至: {MODEL_PATH}")

    # --- 4. 最终展示 (在固定的测试集上进行验证) ---
    pred_results = print_final_comparison(
        model,
        test_indices,
        test_fitness_list,
        test_logs,
        state_distance_matrix,
        DEVICE,
        PAD_ID,
        k_steps=K,
        n_input=N,
    )
    if f_plot_pred_FL:
        batch_plot_pred_FL(
            pred_results,
            state_distance_matrix,
            state_value,
            save_dir=f"{FIG_PATH}/{model_name}/pred_FL{suffix}",
        )


def load_and_predict(
    input_seq, k, d_model=128, epochs=20, batch_size=32, lr=1e-4, f_plot_embedding=False
):
    """
    加载保存的模型权重，并对输入的指定序列进行后续 K 步预测。

    参数:
    - input_seq: list[int], 输入的状态索引序列 (长度建议为 N)
    - model_path: str, .pth 模型文件路径
    - d_model: int, 模型的隐藏层维度
    - state_dm: np.ndarray, 状态距离矩阵 (用于空间先验)
    - device: torch.device
    - pad_id: int, 填充 ID (通常为 VOCAB_SIZE - 1)
    """
    # 1. 重新构建模型结构
    num_real_states = len(state_distance_matrix)
    vocab_size = num_real_states + 1
    PAD_ID = num_real_states
    D_MODEL = d_model
    EPOCHS = epochs
    BATCH_SIZE = batch_size
    LR = lr
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    METRICS_FILE = f"output/data/{model_name}/test_metrics_step_wise{suffix}.txt"
    CACHE_DIR = "cache/model"
    if params["mdl_attn_sim_bias"]:
        MODEL_NAME = f"sg_transformer_v{vocab_size}_d{D_MODEL}_e{EPOCHS}{suffix}.pth"
    else:
        MODEL_NAME = f"traj_transformer_v{vocab_size}_d{D_MODEL}_e{EPOCHS}{suffix}.pth"
    MODEL_PATH = os.path.join(CACHE_DIR, MODEL_NAME)

    model = TrajectoryTransformer(vocab_size, d_model=d_model).to(DEVICE)

    if f_plot_embedding:
        plot_embedding(model)

    # 2. 加载权重
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"未找到模型文件: {MODEL_PATH}")

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 3. 执行预测 (调用你现有的 predict_k_steps 函数)
    # 假设预测 K 步，K 可以从全局变量获取或作为参数传入
    predicted_steps = predict_k_steps(
        model, input_seq, k, state_distance_matrix, DEVICE, PAD_ID
    )

    return predicted_steps
