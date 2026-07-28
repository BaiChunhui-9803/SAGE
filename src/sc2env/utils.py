import numpy as np
from sklearn.cluster import KMeans
import math
import os
import csv
import shutil

from .config import get_map_config

_MAP_CONFIG, _MAP, _ENV_CONFIG, _ALG_CONFIG, _PATH_CONFIG = get_map_config("sce-1")


class GameContext:
    def __init__(self):
        self.episode_count = 0
        self.node_dict = {}
        self.episode_node_list = []


def init_game(ctx, path_config):
    ctx.episode_count = 0
    ctx.node_dict = {}
    ctx.episode_node_list = []

    os.makedirs(path_config["_DATA_TRANSIT_PATH"], exist_ok=True)
    shutil.rmtree(path_config["_DATA_TRANSIT_PATH"])
    os.makedirs(path_config["_DATA_TRANSIT_PATH"])
    os.makedirs(os.path.dirname(path_config["_UNITS_ATTRIBUTE_PATH"]), exist_ok=True)
    os.makedirs(path_config["_GAME_GRAPH_PATH"], exist_ok=True)
    os.makedirs(path_config["_GAME_BKTREE_PATH"], exist_ok=True)
    os.makedirs(path_config["_GAME_SUB_QTABLE_PATH"], exist_ok=True)
    os.makedirs(path_config["_GAME_SUB_EPISODE_PATH"], exist_ok=True)
    os.makedirs(path_config["_GAME_SHORT_TERM_RESULT_PATH"], exist_ok=True)

    for path_key in [
        "_GAME_CLUS_PATH",
        "_GAME_ACTION_LOG_PATH",
        "_GAME_ACTION_PATH",
        "_GAME_STATE_NODE_PATH",
        "_GAME_NODE_LOG_PATH",
        "_GAME_RESULT_PATH",
        "_GAME_RESULT_TEST_PATH",
    ]:
        with open(path_config[path_key], "w") as f:
            f.write("")


def distance(pos1, pos2):
    return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)


def circle_fitting(my_units_lst: list, ex_radius):
    center_x_sum = 0
    center_y_sum = 0
    num_points = len(my_units_lst)
    for point in my_units_lst:
        center_x_sum += point[1]
        center_y_sum += point[2]
    center_point = (center_x_sum / num_points, center_y_sum / num_points)
    dist_max = 0.0
    for point in my_units_lst:
        dist = distance((point[1], point[2]), center_point)
        if dist >= dist_max:
            dist_max = dist
    radius = dist_max + ex_radius
    return center_point, radius


def calculate_std_deviation(numbers):
    n = len(numbers)
    mean = sum(numbers) / n
    squared_diff_sum = sum((x - mean) ** 2 for x in numbers)
    variance = squared_diff_sum / n
    std_deviation = math.sqrt(variance)
    return std_deviation


def calculate_variance_sum(my_units_lst: list):
    x = [point[1] for point in my_units_lst]
    y = [point[2] for point in my_units_lst]
    x_var = np.var(x)
    y_var = np.var(y)
    variance_sum = x_var + y_var
    return variance_sum


def calculate_coefficient_of_variation(numbers):
    if len(numbers) > 1:
        mean = sum(numbers) / len(numbers)
        if mean == 0.0:
            return 0.0
        std_deviation = calculate_std_deviation(numbers)
        coefficient_of_variation = std_deviation / mean
        return coefficient_of_variation
    else:
        return 0.0


def calculate_clu_uniformity(my_units_lst: list):
    center_point, radius = circle_fitting(my_units_lst, 0.0)
    distances = []
    uniformity = 0.0
    if len(my_units_lst):
        if len(my_units_lst) == 1:
            return uniformity
        else:
            for point in my_units_lst:
                x = point[1]
                y = point[2]
                distance = math.sqrt(
                    (x - center_point[0]) ** 2 + (y - center_point[1]) ** 2
                )
                distances.append(distance)
            uniformity = 1.0 - calculate_coefficient_of_variation(distances)
    else:
        uniformity = 0.0
    return round(uniformity, 2)


def calculate_clu_crowding(my_units_lst: list, min_radius):
    center_point, radius = circle_fitting(my_units_lst, 0.0)
    if radius == 0.0:
        return 1.0
    else:
        total_points = len(my_units_lst)
        total_distance = 0
        if len(my_units_lst):
            if len(my_units_lst) == 1:
                return 1.0
            else:
                for i in range(total_points):
                    min_distance = float("inf")
                    for j in range(total_points):
                        if i != j:
                            d = distance(
                                (my_units_lst[i][1], my_units_lst[i][2]),
                                (my_units_lst[j][1], my_units_lst[j][2]),
                            )
                            if d < min_distance:
                                min_distance = d
                    total_distance += min_distance
                max_distance = (
                    2 * total_points * radius * math.sin(math.pi / total_points)
                )
                crowding = 1.0 - total_distance / max_distance
        else:
            crowding = 0.0
        return round(crowding, 2)


def kmeans(my_units_lst, k):
    x = []
    for point in my_units_lst:
        x.append(point[1:])
    km_result = KMeans(n_clusters=k, n_init="auto")
    km_result.fit(x)
    labels = km_result.predict(x)
    clustered_points = []
    for i in range(len(my_units_lst)):
        clustered_points.append(my_units_lst[i] + (labels[i],))
    return clustered_points


def merge_units_csv(file1, file2):
    f_in = open(file1, "r")
    reader = csv.reader(f_in)
    header = next(reader)
    f_in.close()
    f_in = open(file2, "r")
    reader = csv.reader(f_in)
    next(reader)
    f_out = open(_PATH_CONFIG["_UNITS_ATTRIBUTE_PATH"], "w", newline="")
    writer = csv.writer(f_out)
    writer.writerow(header)
    for row in reader:
        writer.writerow(row)
    f_out.close()


def save_short_term_result(short_term_result, folder, episode, step):
    if not os.path.exists(folder):
        os.makedirs(folder)
    filepath = os.path.join(folder, f"{episode - 1}.csv")
    if not os.path.exists(filepath):
        with open(filepath, "w", newline="") as file:
            writer = csv.writer(file)
    with open(filepath, "a") as file:
        file.write("step{}\n\t".format(step))
        file.write("{}\n".format(short_term_result.__str__()))


def save_clusters_health_to_csv(cluster_health_dict, folder, episode, step):
    if not os.path.exists(folder):
        os.makedirs(folder)
    filepath = os.path.join(folder, f"{episode - 1}.csv")
    if not os.path.exists(filepath):
        with open(filepath, "w", newline="") as file:
            writer = csv.writer(file)
    with open(filepath, "a") as file:
        file.write("step{}\n\t".format(step))
        for cluster_id, cluster_item in cluster_health_dict.items():
            file.write("cluster_{}: \n\t\t".format(cluster_id))
            for unit in cluster_item:
                file.write(
                    "{},{},{},{},{};".format(
                        unit[0], unit[1], unit[2], unit[3], unit[4]
                    )
                )
            if cluster_id == -1:
                file.write("\n")
            else:
                file.write("\n\t")


def append_qtable_csv(episode_count, qtable, actions):
    if episode_count == 2:
        with open(_PATH_CONFIG["_EPISODE_QTABLE_PATH"], mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(actions)
        # with open(_PATH_CONFIG["_GAME_RESULT_PATH"], "w") as f:
        #     f.write("")
    with open(_PATH_CONFIG["_EPISODE_QTABLE_PATH"], mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(np.sum(qtable, axis=0))


def save_dataframes_to_csv(dataframes_dict, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

    for key, df in dataframes_dict.items():
        filepath = os.path.join(folder, f"{key}.csv")
        if os.path.exists(filepath):
            df.q_table.to_csv(filepath, mode="w", header=True, index=True, sep=",")
        else:
            df.q_table.to_csv(filepath, index=True)


def save_node(state, reward, ctx):
    if state in ctx.node_dict:
        ctx.episode_node_list.append(ctx.node_dict[state][0])
        ctx.node_dict[state][1] = (ctx.node_dict[state][1] + reward) / 2
    else:
        nid = len(ctx.node_dict)
        ctx.node_dict[state] = [str(nid), reward]
        ctx.episode_node_list.append(str(nid))


def save_node_log(ctx, dict_path, log_path):
    with open(dict_path, "w", newline="") as file:
        for key, value in ctx.node_dict.items():
            line = f"{key}\t{value[0]}\t{value[1]}\n"
            file.write(line)
    with open(log_path, "a", newline="") as file:
        for item in ctx.episode_node_list:
            file.write(item + " ")
        file.write("\n")
    ctx.episode_node_list.clear()
