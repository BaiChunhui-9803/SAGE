import logging
import os
import time
from sklearn.manifold import MDS
import numpy as np
from scipy.interpolate import griddata

logger = logging.getLogger(__name__)

from src.config.base_config import *


def get_fitness_landscape(log_distance_matrix, log_object, n=None, filename=None):
    """
    使用 MDS 降维并插值生成适应度地形数据，带日志控制
    """
    # 如果没有指定文件名，使用默认命名规则
    if filename is None:
        filename = f"cache/npy/fitness_landscape_data_{map_id}_{data_id}.npy"

    # 1. 样本切片处理
    if n is not None:
        logger.info(f"Subsampling enabled: Using first {n} samples for landscape.")
        log_distance_matrix = log_distance_matrix[:n, :n]
        log_object = log_object[:n]

    # 2. 检查缓存文件
    if os.path.exists(filename):
        logger.info(f"Cache hit: Loading fitness landscape data from {filename}")
        try:
            data = np.load(filename, allow_pickle=True)
            grid_x, grid_y, grid_z = data
            return grid_x, grid_y, grid_z
        except Exception as e:
            logger.error(f"Failed to load cache file {filename}: {e}. Recomputing...")

    # 3. 执行计算逻辑
    logger.warning(
        "No valid cache found. Starting MDS dim-reduction and Interpolation..."
    )
    start_time = time.time()

    try:
        # 使用 MDS 降维到 2D
        # dissimilarity='precomputed' 表示输入的是距离矩阵而非原始坐标
        mds = MDS(
            n_components=2,
            dissimilarity="precomputed",
            random_state=42,
            normalized_stress="auto",
        )
        mds_coords = mds.fit_transform(log_distance_matrix)
        logger.debug("MDS dimension reduction completed.")

        # 准备插值数据
        x = mds_coords[:, 0]
        y = mds_coords[:, 1]
        z = log_object

        # 定义插值网格 (1000j 代表 1000 个采样点)
        grid_x, grid_y = np.mgrid[x.min() : x.max() : 1000j, y.min() : y.max() : 1000j]

        # 使用插值方法估计网格上的适应度值 (linear 线性插值)
        grid_z = griddata((x, y), z, (grid_x, grid_y), method="linear")

        calc_time = time.time() - start_time
        logger.info(f"Computation finished. Time elapsed: {calc_time:.2f}s")

        # 4. 保存结果
        # 确保目录存在
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        data = np.array([grid_x, grid_y, grid_z])
        np.save(filename, data)
        logger.info(f"Fitness landscape data saved to {filename}")

        return grid_x, grid_y, grid_z

    except Exception as e:
        logger.error(f"Landscape computation failed: {e}")
        raise


def generate_fitness_landscape(game_results, log_distance_matrix, n=None):
    """
    基于游戏结果和距离矩阵生成适应度景观数据。

    :param game_results: 原始游戏结果列表
    :param log_distance_matrix: 预先计算好的 DTW 距离矩阵
    :param n: 景观平滑参数或采样点数（默认为 None）
    :return: (log_object, grid_x, grid_y, grid_z)
    """
    logger.info("Starting fitness landscape generation...")

    try:
        # 1. 提取 log_object (假设 log[2] 是分值，log[3] 是某种修正值或权重)
        # 这里使用列表推导式高效生成
        log_object = [log[2] + log[3] for log in game_results]
        logger.debug(f"Extracted log_object with {len(log_object)} entries.")

        # 2. 调用景观生成算法 (通常涉及降维如 MDS/TSNE 和插值)
        start_time = time.time()
        grid_x, grid_y, grid_z = get_fitness_landscape(
            log_distance_matrix, log_object, n=n
        )

        elapsed = time.time() - start_time
        logger.info(f"Fitness landscape calculated successfully in {elapsed:.2f}s.")

        return log_object, grid_x, grid_y, grid_z

    except Exception as e:
        logger.error(f"Error during fitness landscape generation: {e}")
        raise


def generate_state_landscape(state_value_dict, state_distance_matrix, n=None):
    """
    基于状态评价值字典和距离矩阵生成状态景观数据。

    :param state_value_dict: 状态 ID 到 Value 的映射（或直接为列表）
    :param state_distance_matrix: 状态距离矩阵
    :param n: 采样点数
    """
    logger.info("启动状态空间景观生成流程...")

    try:
        # 确保输入为数组格式
        if isinstance(state_value_dict, dict):
            # 按 ID 排序确保与矩阵索引对应
            sorted_keys = sorted(state_value_dict.keys())
            state_values = [state_value_dict[k] for k in sorted_keys]
        else:
            state_values = state_value_dict

        grid_x, grid_y, grid_z = get_state_landscape(
            state_distance_matrix, state_values, n=n
        )

        return state_values, grid_x, grid_y, grid_z

    except Exception as e:
        logger.error(f"State Landscape 流程异常: {e}")
        raise


def get_state_landscape(state_distance_matrix, state_values, n=None, filename=None):
    """
    使用 MDS 将状态距离矩阵降维，并结合状态价值生成状态空间地形。

    参数:
    :param state_distance_matrix: 状态间的物理/逻辑距离矩阵 (V x V)
    :param state_values: 每个状态对应的评价值/奖励值 (V,)
    :param n: 采样点数，若状态数过多可进行截断
    :param filename: 缓存文件名
    """
    # 默认缓存路径
    if filename is None:
        filename = "cache/npy/state_landscape_data.npy"

    # 1. 数据采样处理 (防止 OOM 或计算时间过长)
    if n is not None and n < len(state_values):
        logger.info(f"状态采样已启用: 仅使用前 {n} 个状态生成地形")
        state_distance_matrix = state_distance_matrix[:n, :n]
        state_values = state_values[:n]

    # 2. 缓存检查
    if os.path.exists(filename):
        logger.info(f"命中缓存: 从 {filename} 加载状态地形数据")
        return np.load(filename, allow_pickle=True)

    # 3. 计算逻辑
    logger.warning("未发现缓存，开始执行 MDS 降维与表面插值...")
    start_time = time.time()

    try:
        # MDS 降维：将状态间的距离关系映射到二维坐标 (X, Y)
        # 这一步是为了在平面上找到每个状态的“位置”
        mds = MDS(
            n_components=2,
            dissimilarity="precomputed",
            random_state=42,
            n_init=1,
            normalized_stress="auto",
        )
        coords = mds.fit_transform(state_distance_matrix)

        x_obs = coords[:, 0]
        y_obs = coords[:, 1]
        z_obs = np.array(state_values)

        # 定义网格范围与密度 (500x500 的分辨率)
        grid_x, grid_y = np.mgrid[
            x_obs.min() : x_obs.max() : 500j, y_obs.min() : y_obs.max() : 500j
        ]

        # 执行插值：将离散的状态点连接成连续的地形表面
        # 使用 'cubic' (三次样条) 相比 'linear' 能产生更平滑的学术图表效果
        grid_z = griddata((x_obs, y_obs), z_obs, (grid_x, grid_y), method="cubic")

        # 填充缺失值 (插值边界外可能存在 NaN)
        grid_z = np.nan_to_num(grid_z, nan=np.nanmin(grid_z))

        # 4. 存储与返回
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        landscape_data = np.array([grid_x, grid_y, grid_z])
        np.save(filename, landscape_data)

        elapsed = time.time() - start_time
        logger.info(f"状态地形生成完成，耗时: {elapsed:.2f}s")

        return grid_x, grid_y, grid_z

    except Exception as e:
        logger.error(f"状态地形计算失败: {e}")
        raise
