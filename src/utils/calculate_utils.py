import math
import os
import time
import logging

from sklearn.manifold import MDS

from src.utils.load_utils import *
from src.config.base_config import *

# 获取当前模块的 logger
logger = logging.getLogger(__name__)


def dtw_distance(seq1, seq2, distance_matrix):
    """
    计算两个序列之间的DTW距离
    :param seq1: 第一个序列
    :param seq2: 第二个序列
    :param distance_matrix: 距离矩阵
    :return: DTW距离
    """
    m = len(seq1)
    n = len(seq2)
    dtw_matrix = np.zeros((m + 1, n + 1))
    dtw_matrix[0, 0] = 0
    for i in range(1, m + 1):
        dtw_matrix[i, 0] = np.inf
    for j in range(1, n + 1):
        dtw_matrix[0, j] = np.inf

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = distance_matrix[seq1[i - 1], seq2[j - 1]]
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1]
            )

    return dtw_matrix[m, n]


def calculate_distance_matrix(reverse_dict, custom_distance, secondary_bk_trees):
    """
    计算距离矩阵
    """
    num_clusters = len(reverse_dict)
    # 初始化距离矩阵
    distance_matrix = np.zeros((num_clusters, num_clusters))
    clusters = list(reverse_dict.values())

    last_output_time = time.time()
    progress_threshold = 0.01  # 每 1% 更新一次进度

    logger.info(f"Starting distance matrix calculation for {num_clusters} clusters...")

    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            state1 = clusters[i]["cluster"]
            state2 = clusters[j]["cluster"]

            # 获取两个状态的节点
            # 建议：这里如果频繁查询，可以考虑先将 node 提取出来缓存
            node1 = (
                secondary_bk_trees[state1[0]].find_node_by_cluster_id(state1[1]).state
            )
            node2 = (
                secondary_bk_trees[state2[0]].find_node_by_cluster_id(state2[1]).state
            )

            # 计算多维距离并转为欧几里得距离
            dist = custom_distance.multi_distance(node1, node2)
            euclidean_distance = math.sqrt(dist[0] ** 2 + dist[1] ** 2)

            distance_matrix[i, j] = euclidean_distance
            distance_matrix[j, i] = euclidean_distance

        # 进度控制
        progress = (i + 1) / num_clusters
        if progress >= progress_threshold or i == num_clusters - 1:
            current_time = time.time()
            time_elapsed = current_time - last_output_time

            # 使用 info 记录进度
            logger.info(
                f"Progress: {progress * 100:.1f}% | Processed: {i + 1}/{num_clusters} | Step Time: {time_elapsed:.2f}s"
            )

            last_output_time = current_time
            progress_threshold += 0.01

    logger.info("Distance matrix calculation completed.")
    return distance_matrix


def calculate_distance_matrix(reverse_dict, custom_distance, secondary_bk_trees):
    """
    Robust distance-matrix calculation for collected augmented datasets.
    Missing secondary BKTree nodes are assigned a large fallback distance so the
    optional matrix build does not abort after the ETG has already been built.
    """
    num_clusters = len(reverse_dict)
    distance_matrix = np.zeros((num_clusters, num_clusters))
    clusters = list(reverse_dict.values())
    fallback_distance = 1e6

    resolved_nodes = []
    missing_clusters = []
    for cluster in clusters:
        state = cluster["cluster"]
        tree = secondary_bk_trees.get(state[0])
        node = tree.find_node_by_cluster_id(state[1]) if tree is not None else None
        if node is None:
            resolved_nodes.append(None)
            missing_clusters.append(state)
        else:
            resolved_nodes.append(node.state)

    if missing_clusters:
        logger.warning(
            "Missing %s state clusters in secondary BKTrees; using fallback distance for affected pairs. Sample: %s",
            len(missing_clusters),
            missing_clusters[:10],
        )

    last_output_time = time.time()
    progress_threshold = 0.01

    logger.info(f"Starting distance matrix calculation for {num_clusters} clusters...")

    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            node1 = resolved_nodes[i]
            node2 = resolved_nodes[j]

            if node1 is None or node2 is None:
                distance_matrix[i, j] = fallback_distance
                distance_matrix[j, i] = fallback_distance
                continue

            dist = custom_distance.multi_distance(node1, node2)
            euclidean_distance = math.sqrt(dist[0] ** 2 + dist[1] ** 2)

            distance_matrix[i, j] = euclidean_distance
            distance_matrix[j, i] = euclidean_distance

        progress = (i + 1) / num_clusters
        if progress >= progress_threshold or i == num_clusters - 1:
            current_time = time.time()
            time_elapsed = current_time - last_output_time
            logger.info(
                f"Progress: {progress * 100:.1f}% | Processed: {i + 1}/{num_clusters} | Step Time: {time_elapsed:.2f}s"
            )
            last_output_time = current_time
            progress_threshold += 0.01

    logger.info("Distance matrix calculation completed.")
    return distance_matrix


def calculate_and_save_distance_matrix(
    reverse_dict, custom_distance, secondary_bk_trees, distance_matrix_folder
):
    """
    计算距离矩阵并保存到文件，带日志控制
    """
    # 确保文件夹存在
    if not os.path.exists(distance_matrix_folder):
        try:
            os.makedirs(distance_matrix_folder)
            logger.info(f"Created new directory: {distance_matrix_folder}")
        except Exception as e:
            logger.error(f"Failed to create directory {distance_matrix_folder}: {e}")
            raise

    state_distance_matrix_path = os.path.join(
        distance_matrix_folder, "state_distance_matrix.npy"
    )

    # 检查缓存是否存在
    if os.path.exists(state_distance_matrix_path):
        logger.info(f"Cache hit. Loading distance matrix: {state_distance_matrix_path}")
        return load_distance_matrix(state_distance_matrix_path)
    else:
        logger.warning(
            "No cache hit. Starting fresh calculation (this may take a while)..."
        )

        start_time = time.time()
        distance_matrix = calculate_distance_matrix(
            reverse_dict, custom_distance, secondary_bk_trees
        )
        total_time = time.time() - start_time

        try:
            save_distance_matrix(distance_matrix, state_distance_matrix_path)
            logger.info(
                f"Matrix saved successfully to {state_distance_matrix_path} | Total time: {total_time:.2f}s"
            )
        except Exception as e:
            logger.error(f"Failed to save matrix: {e}")

        return distance_matrix


def calculate_dtw_distance_matrix(state_log, distance_matrix):
    """
    计算所有序列之间的DTW距离矩阵
    """
    num_sequences = len(state_log)
    dtw_distance_matrix = np.zeros((num_sequences, num_sequences))

    last_output_time = time.time()
    progress_threshold = 0.01

    logger.info(f"Starting DTW distance matrix calculation for {num_sequences} logs...")

    for i in range(num_sequences):
        for j in range(i + 1, num_sequences):
            seq1 = state_log[i]
            seq2 = state_log[j]
            dtw_dist = dtw_distance(seq1, seq2, distance_matrix)
            dtw_distance_matrix[i, j] = dtw_dist
            dtw_distance_matrix[j, i] = dtw_dist

        # 进度记录
        progress = (i + 1) / num_sequences
        if progress >= progress_threshold or i == num_sequences - 1:
            current_time = time.time()
            time_elapsed = current_time - last_output_time
            logger.info(
                f"DTW Progress: {progress * 100:.1f}% | Processed: {i + 1}/{num_sequences} | Step Time: {time_elapsed:.2f}s"
            )
            last_output_time = current_time
            progress_threshold += 0.01

    logger.info("DTW distance matrix calculation completed.")
    return dtw_distance_matrix


def calculate_and_save_dtw_distance_matrix(
    state_log, distance_matrix, dtw_distance_matrix_folder
):
    """
    计算DTW距离矩阵并保存到文件，带日志控制
    """
    if not os.path.exists(dtw_distance_matrix_folder):
        try:
            os.makedirs(dtw_distance_matrix_folder)
            logger.info(f"Created directory: {dtw_distance_matrix_folder}")
        except Exception as e:
            logger.error(f"Failed to create DTW directory: {e}")
            raise

    log_distance_matrix_path = os.path.join(
        dtw_distance_matrix_folder, "log_distance_matrix.npy"
    )

    if os.path.exists(log_distance_matrix_path):
        logger.info(
            f"Cache hit. Loading DTW distance matrix from {log_distance_matrix_path}"
        )
        return load_distance_matrix(log_distance_matrix_path)
    else:
        logger.warning(
            "No DTW cache hit. Starting calculation (this may be very slow)..."
        )

        start_time = time.time()
        dtw_matrix = calculate_dtw_distance_matrix(state_log, distance_matrix)
        total_time = time.time() - start_time

        try:
            save_distance_matrix(dtw_matrix, log_distance_matrix_path)
            logger.info(
                f"DTW matrix saved to {log_distance_matrix_path} | Total time: {total_time:.2f}s"
            )
        except Exception as e:
            logger.error(f"Failed to save DTW matrix: {e}")

        return dtw_matrix


def get_or_create_mds_coords(
    distance_matrix: np.ndarray, random_state: int = 42
) -> np.ndarray:
    """
    如果坐标文件存在则直接读取；否则用 MDS 计算并保存。
    """
    coords_path = f"cache/npy/state_landscape_data_{map_id}_{data_id}.npy"
    if os.path.exists(coords_path):
        print(f"[MDS] load coords from {coords_path}")
        return np.load(coords_path)
    print("[MDS] fit & save coords ...")
    reducer = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=random_state,
        n_init=4,
        normalized_stress="auto",
    )
    coords = reducer.fit_transform(distance_matrix)
    np.save(coords_path, coords)
    print(f"[MDS] coords saved to {coords_path}")
    return coords
