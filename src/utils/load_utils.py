import csv

import numpy as np
from src.config.base_config import *
from src.data.global_variable import *


def load_distance_matrix(file_path):
    """
    从文件加载距离矩阵
    :param file_path: 文件路径
    :return: 距离矩阵
    """
    return np.load(file_path)


def save_distance_matrix(matrix, file_path):
    """
    保存距离矩阵到文件
    :param matrix: 距离矩阵
    :param file_path: 文件路径
    """
    np.save(file_path, matrix)


def generate_suffix(params_dict):
    # 定义参数名到缩写的映射表
    # 您可以在这里添加更多参数，如：'learning_rate': 'lr'
    state_seq_ABBR_MAP = {
        "mdl_spatial_prior": "sp",
        "mdl_init_embedding_freeze": "embedF",
        "mdl_init_embedding_train": "embedT",
        "N": N,
        "K": K,
        # 'use_batch_norm': 'bn',
        # 'dropout_rate': 'dr'
    }

    dt_ABBR_MAP = {
        "mdl_init_embedding_freeze": "embedF",
        "mdl_init_embedding_train": "embedT",
    }

    if model_name == "decisionTransformer":
        ABBR_MAP = dt_ABBR_MAP
    else:
        ABBR_MAP = state_seq_ABBR_MAP

    suffix_parts = []

    for key, value in params_dict.items():
        if key in ABBR_MAP:
            abbr = ABBR_MAP[key]
            # 根据值类型处理：如果是布尔值且为 True，只加缩写；否则加值
            if isinstance(value, bool):
                if value:
                    suffix_parts.append(abbr)
            else:
                suffix_parts.append(f"{key}{value}")

    # 用下划线连接，并在最前面补一个下划线
    return "_" + "_".join(suffix_parts) if suffix_parts else ""


def create_action_dictionary(action_path):
    """
    动态创建动作字典，通过读取action_path目录下的CSV文件获取真实的动作名称，
    key是从a开始的字母，value是对应的动作名
    """
    import os
    import pandas as pd

    try:
        # 获取action_path目录下的CSV文件列表
        if os.path.exists(action_path):
            csv_files = [f for f in os.listdir(action_path) if f.endswith(".csv")]
            if csv_files:
                # 读取第一个CSV文件
                first_csv = csv_files[0]
                csv_path = os.path.join(action_path, first_csv)

                # 读取CSV文件
                df = pd.read_csv(csv_path)

                # 获取所有动作列（排除Unnamed列）
                action_columns = [col for col in df.columns if col != "Unnamed: 0"]

                # 创建从a,b,c,...到动作名的映射
                action_dict = {}
                for i, action_name in enumerate(action_columns):
                    key = chr(ord("a") + i)
                    action_dict[key] = action_name

                print(f"Loaded {len(action_dict)} actions from CSV: {first_csv}")
                return action_dict

        # 如果无法读取CSV文件，使用默认的字典作为后备
        print(
            "Warning: Could not load actions from CSV, using default action dictionary"
        )
        default_action_dict = {
            "a": "action_ATK_nearest",
            "b": "action_ATK_clu_nearest",
            "c": "action_ATK_nearest_weakest",
            "d": "action_ATK_clu_nearest_weakest",
            "e": "action_ATK_threatening",
            "f": "action_DEF_clu_nearest",
            "g": "action_MIX_gather",
            "h": "action_MIX_lure",
            "i": "action_MIX_sacrifice_lure",
            "j": "do_randomly",
            "k": "do_nothing",
        }
        return default_action_dict

    except Exception as e:
        print(f"Error loading action dictionary: {e}")
        print("Using default action dictionary")
        default_action_dict = {
            "a": "action_ATK_nearest",
            "b": "action_ATK_clu_nearest",
            "c": "action_ATK_nearest_weakest",
            "d": "action_ATK_clu_nearest_weakest",
            "e": "action_ATK_threatening",
            "f": "action_DEF_clu_nearest",
            "g": "action_MIX_gather",
            "h": "action_MIX_lure",
            "i": "action_MIX_sacrifice_lure",
            "j": "do_randomly",
            "k": "do_nothing",
        }
        return default_action_dict


def preprocess_decision_transformer_data(state_log, action_log, r_log):
    """
    输入:
        state_log, action_log, r_log: 均为 list[list]
    返回:
        processed_data: 包含对齐后的 s, a, rtg 的字典
        action_vocab: 动作字符串到 ID 的映射表
    """
    # 1. 构建动作词典 (Action Vocabulary)
    # 提取所有轨迹中出现过的唯一动作
    all_actions = sorted(list(set([a for sublist in action_log for a in sublist])))
    action_to_id = {act: i for i, act in enumerate(all_actions)}

    processed_states = []
    processed_actions = []
    processed_rtgs = []

    for s_raw, a_raw, r_raw in zip(state_log, action_log, r_log):
        # --- 对齐逻辑 ---
        # s: 去掉最后一位
        # a: 去掉最后一位 (无效动作)
        # r: 去掉第一位 (无效占位)
        s_aligned = s_raw[:-1]
        a_aligned = a_raw[:-1]
        r_aligned = r_raw[1:]

        # 确保三者长度一致
        assert len(s_aligned) == len(a_aligned) == len(r_aligned)

        # --- 动作转 ID ---
        a_ids = [action_to_id[act] for act in a_aligned]

        # --- 计算 RTG ---
        rtg_aligned = []
        current_val = 0
        for r in reversed(r_aligned):
            current_val += r
            rtg_aligned.append(current_val)
        rtg_aligned.reverse()  # 翻转回来

        # 存入结果
        processed_states.append(s_aligned)
        processed_actions.append(a_ids)
        processed_rtgs.append(rtg_aligned)

    return {
        "states": processed_states,
        "actions": processed_actions,
        "rtgs": processed_rtgs,
    }, action_to_id


def get_sampling_masks(log_fitness):
    data = np.array(log_fitness)

    # 1. 全局最优掩码
    max_val = np.max(data)
    best_mask = data == max_val

    # 2. 全局前 5% 掩码
    top_5_threshold = np.percentile(data, 95)
    top_5_mask = data >= top_5_threshold

    # 3. 全局最差掩码
    min_val = np.min(data)
    worst_mask = data == min_val

    # 4. 全局后 5% 掩码
    bottom_5_threshold = np.percentile(data, 5)
    bottom_5_mask = data <= bottom_5_threshold

    # 5. 全局中位数附近 (抽样中间 5% 的数据作为代表)
    median_low = np.percentile(data, 47.5)
    median_high = np.percentile(data, 52.5)
    median_mask = (data >= median_low) & (data <= median_high)

    return {
        "best": best_mask,
        "top_5pct": top_5_mask,
        "worst": worst_mask,
        "bottom_5pct": bottom_5_mask,
        "median": median_mask,
    }
