import logging
import os
import time
from sklearn.manifold import MDS
import numpy as np
from scipy.interpolate import griddata

logger = logging.getLogger(__name__)

from src.config.base_config import *


def get_fitness_landscape(log_distance_matrix, log_object, n=None, filename=None):
    """Generate a fitness landscape using MDS and interpolation, with progress logging."""
    # Use the default cache name when no filename is supplied.
    if filename is None:
        filename = f"cache/npy/fitness_landscape_data_{map_id}_{data_id}.npy"

    # 1. Select the requested sample slice.
    if n is not None:
        logger.info(f"Subsampling enabled: Using first {n} samples for landscape.")
        log_distance_matrix = log_distance_matrix[:n, :n]
        log_object = log_object[:n]

    # 2. Check the cached result.
    if os.path.exists(filename):
        logger.info(f"Cache hit: Loading fitness landscape data from {filename}")
        try:
            data = np.load(filename, allow_pickle=True)
            grid_x, grid_y, grid_z = data
            return grid_x, grid_y, grid_z
        except Exception as e:
            logger.error(f"Failed to load cache file {filename}: {e}. Recomputing...")

    # 3. Compute the landscape.
    logger.warning(
        "No valid cache found. Starting MDS dim-reduction and Interpolation..."
    )
    start_time = time.time()

    try:
        # Embed the distances in two dimensions using MDS.
        # dissimilarity='precomputed' expects a distance matrix, not coordinates.
        mds = MDS(
            n_components=2,
            dissimilarity="precomputed",
            random_state=42,
            normalized_stress="auto",
        )
        mds_coords = mds.fit_transform(log_distance_matrix)
        logger.debug("MDS dimension reduction completed.")

        # Prepare interpolation values.
        x = mds_coords[:, 0]
        y = mds_coords[:, 1]
        z = log_object

        # Build the interpolation grid; 1000j requests 1,000 samples.
        grid_x, grid_y = np.mgrid[x.min() : x.max() : 1000j, y.min() : y.max() : 1000j]

        # Linearly interpolate the fitness values on the grid.
        grid_z = griddata((x, y), z, (grid_x, grid_y), method="linear")

        calc_time = time.time() - start_time
        logger.info(f"Computation finished. Time elapsed: {calc_time:.2f}s")

        # 4. Save the result.
        # Create the output directory if necessary.
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        data = np.array([grid_x, grid_y, grid_z])
        np.save(filename, data)
        logger.info(f"Fitness landscape data saved to {filename}")

        return grid_x, grid_y, grid_z

    except Exception as e:
        logger.error(f"Landscape computation failed: {e}")
        raise


def generate_fitness_landscape(game_results, log_distance_matrix, n=None):
    """Generate a landscape from game results and a precomputed DTW distance matrix. Optionally limit the sample to n. Return log values and the interpolated grid (x, y, z)."""
    logger.info("Starting fitness landscape generation...")

    try:
        # 1. Compute landscape values using the recorded score and correction fields.
        # Collect one value per log entry.
        log_object = [log[2] + log[3] for log in game_results]
        logger.debug(f"Extracted log_object with {len(log_object)} entries.")

        # 2. Generate the MDS embedding and interpolated landscape.
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
    """Generate a state landscape from a state-ID-to-value mapping (or a value list) and a state-distance matrix, optionally limiting the sample to n."""
    logger.info("启动状态空间景观生成流程...")

    try:
        # Convert the input to an array.
        if isinstance(state_value_dict, dict):
            # Sort by state ID to match the distance-matrix indexing.
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
    """Embed the state-distance matrix with MDS and interpolate the corresponding state values. Optionally truncate to n states and cache under filename."""
    # Default cache path.
    if filename is None:
        filename = "cache/npy/state_landscape_data.npy"

    # 1. Limit the sample to bound memory and computation.
    if n is not None and n < len(state_values):
        logger.info(f"状态采样已启用: 仅使用前 {n} 个状态生成地形")
        state_distance_matrix = state_distance_matrix[:n, :n]
        state_values = state_values[:n]

    # 2. Check the cache.
    if os.path.exists(filename):
        logger.info(f"命中缓存: 从 {filename} 加载状态地形数据")
        return np.load(filename, allow_pickle=True)

    # 3. Compute the embedding.
    logger.warning("未发现缓存，开始执行 MDS 降维与表面插值...")
    start_time = time.time()

    try:
        # Use MDS to map state distances into two-dimensional coordinates.
        # Each embedded point represents one state.
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

        # Define a 500-by-500 interpolation grid.
        grid_x, grid_y = np.mgrid[
            x_obs.min() : x_obs.max() : 500j, y_obs.min() : y_obs.max() : 500j
        ]

        # Interpolate discrete state values into a continuous surface.
        # Cubic interpolation produces a smoother surface than linear interpolation.
        grid_z = griddata((x_obs, y_obs), z_obs, (grid_x, grid_y), method="cubic")

        # Fill missing values outside the interpolation boundary.
        grid_z = np.nan_to_num(grid_z, nan=np.nanmin(grid_z))

        # 4. Store and return the result.
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        landscape_data = np.array([grid_x, grid_y, grid_z])
        np.save(filename, landscape_data)

        elapsed = time.time() - start_time
        logger.info(f"状态地形生成完成，耗时: {elapsed:.2f}s")

        return grid_x, grid_y, grid_z

    except Exception as e:
        logger.error(f"状态地形计算失败: {e}")
        raise
