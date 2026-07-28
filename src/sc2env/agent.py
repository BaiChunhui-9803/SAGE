from pysc2.agents import base_agent
from pysc2.lib import actions, features
import pandas as pd

# from scipy.spatial import distance
import random
from collections import deque, defaultdict
import json
import matplotlib
import matplotlib.pyplot as plt

from src.sc2env.config import get_map_config

_MAP_CONFIG, _MAP, _ENV_CONFIG, _ALG_CONFIG, _PATH_CONFIG = get_map_config("sce-1")
from src.sc2env.utils import *

from src.structure.custom_distance_sc2 import CustomDistance
from src.structure.BKTree_sc2 import ClusterNode, BKTree

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
pd.set_option("display.max_columns", None)


class QLearningTable:
    def __init__(
        self,
        actions,
        learning_rate=_ALG_CONFIG["learning_rate"],
        reward_decay=_ALG_CONFIG["reward_decay"],
    ):
        self.actions = actions
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation, e_greedy=0.9):
        self.check_state_exist(observation)
        if np.random.uniform() < e_greedy:
            state_action = self.q_table.loc[observation, :]
            action = np.random.choice(
                state_action[state_action == np.max(state_action)].index
            )
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != "terminal":
            q_target = r + self.reward_decay * self.q_table.loc[s_, :].max()
        else:
            q_target = r
        self.q_table.loc[s, a] += self.learning_rate * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = pd.concat(
                [
                    self.q_table,
                    pd.Series(
                        [0] * len(self.actions), index=self.q_table.columns, name=state
                    )
                    .to_frame()
                    .T,
                ]
            )


class Agent(base_agent.BaseAgent):
    clusters = (
        "k_means_000",
        "k_means_025",
        "k_means_050",
        "k_means_075",
        "k_means_100",
    )

    actions = (
        "action_ATK_nearest",
        "action_ATK_clu_nearest",
        "action_ATK_nearest_weakest",
        "action_ATK_clu_nearest_weakest",
        "action_ATK_threatening",
        # "action_DEF_nearest",
        "action_DEF_clu_nearest",
        "action_MIX_gather",
        "action_MIX_lure",
        "action_MIX_sacrifice_lure",
        "do_randomly",
        "do_nothing",
    )

    def get_units_list(self, file):
        unit_my_list = []
        unit_enemy_list = []
        f_in = open(file, "r")
        reader = csv.reader(f_in)
        header = next(reader)
        for row in reader:
            if row[0] == str(_MAP["unit_type_id"]):
                unit_my_list.append((row[0], row[12], row[13]))
                continue
            if row[0] == str(_MAP["unit_type_id"]):
                unit_enemy_list.append((row[0], row[12], row[13]))
                continue
        return unit_my_list, unit_enemy_list

    def get_my_units_by_type(self, obs, unit_type):
        return [
            unit
            for unit in obs.observation.raw_units
            if unit.unit_type == unit_type
            and unit.alliance == features.PlayerRelative.SELF
        ]

    def get_enemy_units_by_type(self, obs, unit_type):
        return [
            unit
            for unit in obs.observation.raw_units
            if unit.unit_type == unit_type
            and unit.alliance == features.PlayerRelative.ENEMY
        ]

    def get_distances(self, obs, units_list, xy):
        units_xy = [(unit.x, unit.y) for unit in units_list]
        return np.linalg.norm(np.array(units_xy) - np.array(xy), axis=1)

    def get_nearest_enemy(self, mp, enemy_units):
        enemy_units_list = sorted(
            [(item["tag"], item["x"], item["y"]) for item in enemy_units],
            key=lambda x: x[0],
        )
        min_dis = 99.0
        min_tag = -1
        for unit in enemy_units_list:
            dis = distance((unit[1], unit[2]), mp)
            if dis < min_dis:
                min_dis = dis
                min_tag = unit[0]
        return min_tag

    def get_center_position(self, obs, alliance, unit_type):
        position = (0, 0)
        if alliance == "Self":
            my_units = [
                unit
                for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.alliance == features.PlayerRelative.SELF
            ]
            if len(my_units) == 0:
                return position
            for unit in my_units:
                position = tuple(map(lambda x, y: x + y, position, (unit.x, unit.y)))
            return (position[0] / len(my_units), position[1] / len(my_units))
        elif alliance == "Enemy":
            position = (0, 0)
            enemy_units = [
                unit
                for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.alliance == features.PlayerRelative.ENEMY
            ]
            if len(enemy_units) == 0:
                return position
            for unit in enemy_units:
                position = tuple(map(lambda x, y: x + y, position, (unit.x, unit.y)))
            return (position[0] / len(enemy_units), position[1] / len(enemy_units))
        return None

    def get_center_position_point(self, points):
        position = (0, 0)
        my_units = [(unit[1], unit[2]) for unit in points]
        if len(my_units) == 0:
            return position
        for unit in my_units:
            position = tuple(map(lambda x, y: x + y, position, (unit[0], unit[1])))
        return (position[0] / len(my_units), position[1] / len(my_units))

    def choose_nearest_weakest_enemy(self, mp, enemy_list):
        sorted_enemy_lst = sorted(
            [
                (
                    item["tag"],
                    item["x"],
                    item["y"],
                    item["health"],
                    distance(mp, (item["x"], item["y"])),
                )
                for item in enemy_list
            ],
            key=lambda x: x[3],
        )
        closest_enemy = min(sorted_enemy_lst, key=lambda x: x[4])
        return closest_enemy[0]

    def choose_threatening_enemy(self, mp, enemy_list):
        sorted_enemy_lst = sorted(
            [
                (
                    item["tag"],
                    item["x"],
                    item["y"],
                    item["health"],
                    distance(mp, (item["x"], item["y"])),
                )
                for item in enemy_list
            ],
            key=lambda x: x[3],
        )
        max_health = sorted_enemy_lst[-1][3]
        max_health_enemies = [
            enemy for enemy in sorted_enemy_lst if enemy[3] == max_health
        ]
        closest_enemy = min(max_health_enemies, key=lambda x: x[4])
        return closest_enemy[0]

    def step(self, obs, env):
        super(Agent, self).step(obs, env)

    def do_nothing(self, obs):
        return actions.RAW_FUNCTIONS.no_op()


class ShortTermReward:
    def __init__(self):
        self.r_kill = 0
        self.r_fall = 0
        self.r_inferior = 0
        self.r_dominant = 0
        self.r_self_health_loss_ratio = 0
        self.r_enemy_health_loss_ratio = 0
        self.r_fire_coverage = 0
        self.r_covered_in_fire = 0

    def __str__(self):
        return "{} {} {} {} {} {} {} {}".format(
            self.r_kill,
            self.r_fall,
            self.r_inferior,
            self.r_dominant,
            self.r_self_health_loss_ratio,
            self.r_enemy_health_loss_ratio,
            self.r_fire_coverage,
            self.r_covered_in_fire,
        )


class SmartAgent(Agent):
    def __init__(self):
        super(SmartAgent, self).__init__()
        self.ctx = None
        self._termination_signaled = False
        self._initial_units_my = []
        self._initial_units_enemy = []
        self._initial_spawned = False
        self.previous_clu_state = None
        self.previous_clu_action = None
        self.previous_combat_state = {}
        self.previous_combat_action = {}
        self.cluster_result = None
        self._move_back = True
        self._dis_move_back = [True, True, True, True, True, True, True, True]
        self.action_lst = []
        self.score_cumulative_attack_last = 0
        self.score_cumulative_defense_last = 0
        self.score_cumulative_attack_now = 0
        self.score_cumulative_defense_now = 0
        self.score_attack_max = 0
        self.score_defense_max = 0
        self._backup_target_grid = (0, 0)
        self._backup_target_map = (0.0, 0.0)
        self.action_queue = deque()
        self.end_game_frames = _ENV_CONFIG["_MAX_STEP"] * _ENV_CONFIG["_STEP_MUL"]
        self.end_game_state = "Dogfall"
        self.end_game_flag = False
        self.clusters_qtable = QLearningTable(self.clusters)
        self.sub_clusters_qtable_list = {}
        self.sub_clusters_qtable_tag = None
        self.previous_sub_tag = None
        self.previous_obs = {}
        self.current_obs = {}
        self.previous_reward = ShortTermReward()
        self.new_game()
        # self.primary_clusters = np.zeros(total_states, dtype=int)
        self.custom_distance_manager = CustomDistance(threshold=0.5)
        self.primary_bktree = BKTree(
            self.custom_distance_manager.multi_distance, distance_index=0
        )
        self.secondary_bktree = defaultdict(
            lambda: BKTree(
                self.custom_distance_manager.multi_distance, distance_index=1
            )
        )

    def map_to_grid(self, x, y):
        i, j = int(x), int(_ENV_CONFIG["_MAP_RESOLUTION"] - y)
        return i, j

    def grid_to_map(self, i, j):
        x, y = (
            random.uniform(i, i + 1),
            random.uniform(
                int(_ENV_CONFIG["_MAP_RESOLUTION"] - j) - 1,
                int(_ENV_CONFIG["_MAP_RESOLUTION"] - j),
            ),
        )
        return x, y

    def ripple(self, influence_map, ripple_level, ripple_center, alliance):
        max_x = _ENV_CONFIG["_MAP_RESOLUTION"] - 1
        max_y = _ENV_CONFIG["_MAP_RESOLUTION"] - 1
        if alliance == "Self":
            if ripple_level == 0:
                influence_map[int(ripple_center[1])][int(ripple_center[2])] += (
                    _ALG_CONFIG["_MY_UNIT_INFLUENCE"][ripple_level]
                )
            else:
                for dx in range(
                    -abs(ripple_level) if int(ripple_center[1]) > 0 else 0,
                    abs(ripple_level) + 1 if int(ripple_center[1]) < max_x else 0,
                ):
                    for dy in range(
                        -abs(ripple_level) if int(ripple_center[2]) > 0 else 0,
                        abs(ripple_level) + 1 if int(ripple_center[2]) < max_y else 0,
                    ):
                        if dx != 0 or dy != 0:
                            influence_map[int(ripple_center[1]) + dx][
                                int(ripple_center[2]) + dy
                            ] += _ALG_CONFIG["_MY_UNIT_INFLUENCE"][ripple_level]

        elif alliance == "Enemy":
            if ripple_level == 0:
                influence_map[int(ripple_center[1])][int(ripple_center[2])] += (
                    _ALG_CONFIG["_ENEMY_UNIT_INFLUENCE"][ripple_level]
                )
            else:
                for dx in range(
                    -abs(ripple_level) if int(ripple_center[1]) > 0 else 0,
                    abs(ripple_level) + 1 if int(ripple_center[1]) < max_x else 0,
                ):
                    for dy in range(
                        -abs(ripple_level) if int(ripple_center[2]) > 0 else 0,
                        abs(ripple_level) + 1 if int(ripple_center[2]) < max_y else 0,
                    ):
                        if dx != 0 or dy != 0:
                            influence_map[int(ripple_center[1]) + dx][
                                int(ripple_center[2]) + dy
                            ] += _ALG_CONFIG["_ENEMY_UNIT_INFLUENCE"][ripple_level]
        elif alliance == "Target":
            influence_map[int(ripple_center[0])][int(ripple_center[1])] += 100

    def get_influence_map(self, unit_my_list, unit_enemy_list):
        side = int((pow(_ENV_CONFIG["_MAP_RESOLUTION"], 1)))
        influence_map = np.zeros((side, side))
        for my_unit in unit_my_list:
            for index in range(len(_ALG_CONFIG["_MY_UNIT_INFLUENCE"])):
                self.ripple(influence_map, index, my_unit, "Self")
        for enemy_unit in unit_enemy_list:
            for index in range(len(_ALG_CONFIG["_ENEMY_UNIT_INFLUENCE"])):
                self.ripple(influence_map, index, enemy_unit, "Enemy")
        return influence_map

    def analyze_influence_map(self, influence_map):
        positive_list, negative_list = [], []
        for i in range(len(influence_map)):
            for j in range(len(influence_map[0])):
                if influence_map[i][j] > 0:
                    positive_list.append((i, j, influence_map[i][j]))
                elif influence_map[i][j] < 0:
                    negative_list.append((i, j, influence_map[i][j]))
        sort_negative_list = sorted(negative_list, key=lambda x: x[2], reverse=True)
        if len(positive_list) == 0:
            return (
                _ENV_CONFIG["_MAP_RESOLUTION"] / 2,
                _ENV_CONFIG["_MAP_RESOLUTION"] / 2,
            )
        if len(sort_negative_list) == 0:
            return (
                _ENV_CONFIG["_MAP_RESOLUTION"] / 2,
                _ENV_CONFIG["_MAP_RESOLUTION"] / 2,
            )
        percentile = 80
        p = np.percentile([t[2] for t in sort_negative_list], percentile)
        filtered_negative_list = [x for x in sort_negative_list if x[2] >= p]
        sum_x = 0
        sum_y = 0
        for coord in positive_list:
            sum_x += coord[0]
            sum_y += coord[1]
        avg_x = sum_x / len(positive_list)
        avg_y = sum_y / len(positive_list)
        positive_ctr = (avg_x, avg_y)
        min_distance = float("inf")
        min_index = -1
        for i, coord in enumerate(filtered_negative_list):
            d = distance(coord, positive_ctr)
            if d < min_distance:
                min_distance = d
                min_index = i
        return filtered_negative_list[min_index]

    def get_map_boundary(self, influence_map, width):
        rows, cols = len(influence_map), len(influence_map[0])
        non_zero = [
            (i, j) for i in range(rows) for j in range(cols) if influence_map[i][j] != 0
        ]
        if not non_zero:
            return 40, 44, 40, 44
        min_row, min_col = non_zero[0]
        max_row, max_col = non_zero[0]
        for i, j in non_zero:
            if i < min_row:
                min_row = i
            if i > max_row:
                max_row = i
            if j < min_col:
                min_col = j
            if j > max_col:
                max_col = j
        top = max(min_row - width, 0)
        bottom = min(max_row + 1 + width, _ENV_CONFIG["_MAP_RESOLUTION"])
        left = max(min_col - width, 0)
        right = min(max_col + 1 + width, _ENV_CONFIG["_MAP_RESOLUTION"])
        return top, bottom, left, right

    def check_sub_table_exist(self, sub_table_tag):
        if sub_table_tag in self.sub_clusters_qtable_list:
            return True
        else:
            return False

    def update_sub_clusters_qtable_list(self, clu_lists):
        sub_table_tag = (clu_lists[0], clu_lists[1])
        if not self.check_sub_table_exist(sub_table_tag):
            self.sub_clusters_qtable_list.update(
                {sub_table_tag: QLearningTable(self.actions)}
            )
            self.previous_combat_state.update({sub_table_tag: None})
            self.previous_combat_action.update({sub_table_tag: None})

    def get_local_enemy(self, my_local_units, enemy_units):
        local_enemy_unit_list = []
        for enemy_unit in enemy_units:
            sum_distance = 0
            count = 0
            for my_unit in my_local_units:
                sum_distance += distance(
                    (enemy_unit[1], enemy_unit[2]), (my_unit[1], my_unit[2])
                )
                if (
                    distance((enemy_unit[1], enemy_unit[2]), (my_unit[1], my_unit[2]))
                    < _ENV_CONFIG["_UNIT_RADIUS"]
                ):
                    count += 1
            local_enemy_unit_list.append((enemy_unit, count, sum_distance))
        local_enemy_unit_list.sort(key=lambda x: (x[1], -x[2]), reverse=True)
        return local_enemy_unit_list

    def get_local_weak_enemy(self, my_local_units, enemy_units):
        local_enemy_unit_list = []
        for enemy_unit in enemy_units:
            count = 0
            for my_unit in my_local_units:
                if (
                    distance((enemy_unit[1], enemy_unit[2]), (my_unit[1], my_unit[2]))
                    < _ENV_CONFIG["_UNIT_RADIUS"]
                ):
                    count += 1
            local_enemy_unit_list.append((enemy_unit, count, enemy_unit[3]))
        local_enemy_unit_list.sort(key=lambda x: (x[1], -x[2]), reverse=True)
        return local_enemy_unit_list

    def k_means_000(self, obs):
        my_units = self.get_my_units_by_type(obs, _MAP["unit_type"])
        my_units_lst = sorted(
            [(item["tag"], item["x"], item["y"]) for item in my_units],
            key=lambda x: x[0],
        )
        clu_number = len(my_units_lst)
        clu_lists = [clu_number, 0.0, []]
        if clu_number > 0:
            clu_variance = calculate_variance_sum(my_units_lst)
            clu_lists[1] = round(clu_variance * 2) / 2
            for i in range(clu_number):
                clu_lists[2].append(
                    (
                        i,
                        (my_units_lst[i][1], my_units_lst[i][2]),
                        0.0,
                        1.0,
                        [my_units_lst[i]],
                    )
                )
        self.update_sub_clusters_qtable_list(clu_lists)
        return clu_lists

    def k_means_025(self, obs):
        my_units = self.get_my_units_by_type(obs, _MAP["unit_type"])
        my_units_lst = sorted(
            [(item["tag"], item["x"], item["y"]) for item in my_units],
            key=lambda x: x[0],
        )
        clu_number = 1
        clu_lists = [clu_number, 0.0, []]
        if len(my_units_lst) * 0.25 > 1:
            clu_number = int(len(my_units_lst) * 0.25)
            clu_lists[0] = clu_number
        if len(my_units_lst) > 0:
            clusters = kmeans(my_units_lst, clu_number)
            clu_center_list = []
            unique_labels = set([cluster[-1] for cluster in clusters])
            for label in unique_labels:
                cluster_points = [
                    cluster[:-1] for cluster in clusters if cluster[-1] == label
                ]
                clu_uniformity = calculate_clu_uniformity(cluster_points)
                clu_crowding = calculate_clu_crowding(
                    cluster_points, _ENV_CONFIG["_UNIT_RADIUS"]
                )
                clu_0_center, clu_0_radius = circle_fitting(cluster_points, 0.0)
                clu_center_list.append((0, clu_0_center[0], clu_0_center[1]))
                clu_lists[2].append(
                    (label, clu_0_center, clu_uniformity, clu_crowding, cluster_points)
                )
            clu_variance = calculate_variance_sum(clu_center_list)
            clu_lists[1] = round(clu_variance * 2) / 2
        self.update_sub_clusters_qtable_list(clu_lists)
        return clu_lists

    def k_means_050(self, obs):
        my_units = self.get_my_units_by_type(obs, _MAP["unit_type"])
        my_units_lst = sorted(
            [(item["tag"], item["x"], item["y"]) for item in my_units],
            key=lambda x: x[0],
        )
        clu_number = 1
        clu_lists = [clu_number, 0.0, []]
        if len(my_units_lst) * 0.5 > 1:
            clu_number = int(len(my_units_lst) * 0.5)
            clu_lists[0] = clu_number
        if len(my_units_lst) > 0:
            clusters = kmeans(my_units_lst, clu_number)
            clu_center_list = []
            unique_labels = set([cluster[-1] for cluster in clusters])
            for label in unique_labels:
                cluster_points = [
                    cluster[:-1] for cluster in clusters if cluster[-1] == label
                ]
                clu_uniformity = calculate_clu_uniformity(cluster_points)
                clu_crowding = calculate_clu_crowding(
                    cluster_points, _ENV_CONFIG["_UNIT_RADIUS"]
                )
                clu_0_center, clu_0_radius = circle_fitting(cluster_points, 0.0)
                clu_center_list.append((0, clu_0_center[0], clu_0_center[1]))
                clu_lists[2].append(
                    (label, clu_0_center, clu_uniformity, clu_crowding, cluster_points)
                )
            clu_variance = calculate_variance_sum(clu_center_list)
            clu_lists[1] = round(clu_variance * 2) / 2
        self.update_sub_clusters_qtable_list(clu_lists)
        # print(clu_lists[2])
        return clu_lists

    def k_means_075(self, obs):
        my_units = self.get_my_units_by_type(obs, _MAP["unit_type"])
        my_units_lst = sorted(
            [(item["tag"], item["x"], item["y"]) for item in my_units],
            key=lambda x: x[0],
        )
        clu_number = 1
        clu_lists = [clu_number, 0.0, []]
        if len(my_units_lst) * 0.75 > 1:
            clu_number = int(len(my_units_lst) * 0.75)
            clu_lists[0] = clu_number
        if len(my_units_lst) > 0:
            clusters = kmeans(my_units_lst, clu_number)
            clu_center_list = []
            unique_labels = set([cluster[-1] for cluster in clusters])
            for label in unique_labels:
                cluster_points = [
                    cluster[:-1] for cluster in clusters if cluster[-1] == label
                ]
                clu_uniformity = calculate_clu_uniformity(cluster_points)
                clu_crowding = calculate_clu_crowding(
                    cluster_points, _ENV_CONFIG["_UNIT_RADIUS"]
                )
                clu_0_center, clu_0_radius = circle_fitting(cluster_points, 0.0)
                clu_center_list.append((0, clu_0_center[0], clu_0_center[1]))
                clu_lists[2].append(
                    (label, clu_0_center, clu_uniformity, clu_crowding, cluster_points)
                )
            clu_variance = calculate_variance_sum(clu_center_list)
            clu_lists[1] = round(clu_variance * 2) / 2
        self.update_sub_clusters_qtable_list(clu_lists)
        return clu_lists

    def k_means_100(self, obs):
        my_units = self.get_my_units_by_type(obs, _MAP["unit_type"])
        my_units_lst = sorted(
            [(item["tag"], item["x"], item["y"]) for item in my_units],
            key=lambda x: x[0],
        )
        clu_number = 1
        clu_variance = 0.0
        clu_lists = [clu_number, clu_variance, []]
        if len(my_units_lst) > 0:
            clu_uniformity = calculate_clu_uniformity(my_units_lst)
            clu_crowding = calculate_clu_crowding(
                my_units_lst, _ENV_CONFIG["_UNIT_RADIUS"]
            )
            clu_0_center, clu_0_radius = circle_fitting(my_units_lst, 0.0)
            clu_lists[2].append(
                (0, clu_0_center, clu_uniformity, clu_crowding, my_units_lst)
            )
        self.update_sub_clusters_qtable_list(clu_lists)
        return clu_lists

    def action_greedy(self, obs):
        my_units = self.get_my_units_by_type(obs, _MAP["unit_type"])
        my_units_lst = sorted(
            [(item["tag"], item["weapon_cooldown"]) for item in my_units],
            key=lambda x: x[0],
        )
        enemy_units = self.get_enemy_units_by_type(obs, _MAP["unit_type"])
        mp = self.get_center_position(obs, "Self", _MAP["unit_type"])
        if len(my_units) > 0 and len(enemy_units) > 0:
            return actions.RAW_FUNCTIONS.Attack_unit(
                "now",
                [item[0] for item in my_units_lst],
                self.choose_nearest_weakest_enemy(mp, enemy_units),
            )
        return actions.RAW_FUNCTIONS.no_op()

    def do_randomly(self, obs):
        my_units = self.get_my_units_by_type(obs, _MAP["unit_type"])
        my_units_lst = sorted(
            [(item["tag"], item["weapon_cooldown"]) for item in my_units],
            key=lambda x: x[0],
        )
        if len(my_units) > 0:
            return actions.RAW_FUNCTIONS.Attack_pt(
                "now", [item[0] for item in my_units_lst], (50, 50)
            )
        return actions.RAW_FUNCTIONS.no_op()

    def action_ATK_nearest(self, obs):
        self.action_lst = []
        my_units = self.get_my_units_by_type(obs, _MAP["unit_type"])
        my_units_lst = sorted(
            [
                (item["tag"], item["x"], item["y"], item["weapon_cooldown"])
                for item in my_units
            ],
            key=lambda x: x[0],
        )
        enemy_units = self.get_enemy_units_by_type(obs, _MAP["unit_type"])
        mp = self.get_center_position(obs, "Self", _MAP["unit_type"])
        if len(my_units) > 0 and len(enemy_units) > 0:
            for unit in my_units_lst:
                self.action_lst.append(
                    actions.RAW_FUNCTIONS.Smart_unit(
                        "now",
                        unit[0],
                        self.get_nearest_enemy((unit[1], unit[2]), enemy_units),
                    )
                )
            return self.action_lst
        return actions.RAW_FUNCTIONS.no_op()

    def action_ATK_clu_nearest(self, obs):
        self.action_lst = []
        my_units = self.get_my_units_by_type(obs, _MAP["unit_type"])
        my_units_lst = sorted(
            [
                (item["tag"], item["x"], item["y"], item["weapon_cooldown"])
                for item in my_units
            ],
            key=lambda x: x[0],
        )
        enemy_units = self.get_enemy_units_by_type(obs, _MAP["unit_type"])
        enemy_units_lst = sorted(
            [
                (
                    item["tag"],
                    item["x"],
                    item["y"],
                    item["health"],
                    item["health_ratio"],
                )
                for item in enemy_units
            ],
            key=lambda x: x[0],
        )
        mp = self.get_center_position(obs, "Self", _MAP["unit_type"])
        if len(my_units) > 0 and len(enemy_units) > 0:
            for clu in self.cluster_result[2]:
                local_enemy_list = self.get_local_enemy(clu[4], enemy_units_lst)
                for unit in clu[4]:
                    self.action_lst.append(
                        actions.RAW_FUNCTIONS.Smart_unit(
                            "now", unit[0], local_enemy_list[0][0][0]
                        )
                    )
            return self.action_lst
        return actions.RAW_FUNCTIONS.no_op()

    def action_ATK_nearest_weakest(self, obs):
        my_units = self.get_my_units_by_type(obs, _MAP["unit_type"])
        my_units_lst = sorted(
            [
                (item["tag"], item["x"], item["y"], item["weapon_cooldown"])
                for item in my_units
            ],
            key=lambda x: x[0],
        )
        enemy_units = self.get_enemy_units_by_type(obs, _MAP["unit_type"])
        mp = self.get_center_position(obs, "Self", _MAP["unit_type"])
        if len(my_units) > 0 and len(enemy_units) > 0:
            return actions.RAW_FUNCTIONS.Smart_unit(
                "now",
                [item[0] for item in my_units_lst],
                self.choose_nearest_weakest_enemy(mp, enemy_units),
            )
        return actions.RAW_FUNCTIONS.no_op()

    def action_ATK_clu_nearest_weakest(self, obs):
        self.action_lst = []
        my_units = self.get_my_units_by_type(obs, _MAP["unit_type"])
        my_units_lst = sorted(
            [
                (item["tag"], item["x"], item["y"], item["weapon_cooldown"])
                for item in my_units
            ],
            key=lambda x: x[0],
        )
        enemy_units = self.get_enemy_units_by_type(obs, _MAP["unit_type"])
        enemy_units_lst = sorted(
            [
                (
                    item["tag"],
                    item["x"],
                    item["y"],
                    item["health"],
                    item["health_ratio"],
                )
                for item in enemy_units
            ],
            key=lambda x: x[0],
        )
        mp = self.get_center_position(obs, "Self", _MAP["unit_type"])
        if len(my_units) > 0 and len(enemy_units) > 0:
            for clu in self.cluster_result[2]:
                local_enemy_list = self.get_local_weak_enemy(clu[4], enemy_units_lst)
                for unit in clu[4]:
                    self.action_lst.append(
                        actions.RAW_FUNCTIONS.Smart_unit(
                            "now", unit[0], local_enemy_list[0][0][0]
                        )
                    )
            return self.action_lst
        return actions.RAW_FUNCTIONS.no_op()

    def action_ATK_threatening(self, obs):
        my_units = self.get_my_units_by_type(obs, _MAP["unit_type"])
        my_units_lst = sorted(
            [
                (item["tag"], item["x"], item["y"], item["weapon_cooldown"])
                for item in my_units
            ],
            key=lambda x: x[0],
        )
        enemy_units = self.get_enemy_units_by_type(obs, _MAP["unit_type"])
        mp = self.get_center_position(obs, "Self", _MAP["unit_type"])
        if len(my_units) > 0 and len(enemy_units) > 0:
            return actions.RAW_FUNCTIONS.Smart_unit(
                "now",
                [item[0] for item in my_units_lst],
                self.choose_threatening_enemy(mp, enemy_units),
            )
        return actions.RAW_FUNCTIONS.no_op()

    def action_DEF_clu_nearest(self, obs):
        self.action_lst = []
        my_units = self.get_my_units_by_type(obs, _MAP["unit_type"])
        my_units_lst = sorted(
            [
                (item["tag"], item["x"], item["y"], item["weapon_cooldown"])
                for item in my_units
            ],
            key=lambda x: x[0],
        )
        enemy_units = self.get_enemy_units_by_type(obs, _MAP["unit_type"])
        enemy_units_lst = sorted(
            [
                (
                    item["tag"],
                    item["x"],
                    item["y"],
                    item["health"],
                    item["health_ratio"],
                )
                for item in enemy_units
            ],
            key=lambda x: x[0],
        )
        mp = self.get_center_position(obs, "Self", _MAP["unit_type"])
        ep = self.get_center_position(obs, "Enemy", _MAP["unit_type"])
        if len(my_units) > 0 and len(enemy_units) > 0:
            for clu in self.cluster_result[2]:
                clu_mp = clu[1]
                clu_enemy_tag_list = [
                    self.get_nearest_enemy((item[1], item[2]), enemy_units)
                    for item in clu[4]
                ]
                clu_ep_lst = [
                    (item[1], item[2])
                    for item in enemy_units_lst
                    if item[0] in clu_enemy_tag_list
                ]
                clu_ep = tuple(sum(x) / len(clu_ep_lst) for x in zip(*clu_ep_lst))
                vec = tuple(3 * x - 3 * y for x, y in zip(clu_mp, clu_ep))
                clu_tp = tuple(map(lambda x, y: min(max((x + y), 0), 128), clu_mp, vec))
                if len(clu) > 0:
                    for unit in clu[4]:
                        self.action_lst.append(
                            actions.RAW_FUNCTIONS.Smart_pt("now", unit[0], clu_tp)
                        )
            return self.action_lst
        return actions.RAW_FUNCTIONS.no_op()

    def action_MIX_gather(self, obs):
        self.action_lst = []
        my_units = self.get_my_units_by_type(obs, _MAP["unit_type"])
        my_units_lst = sorted(
            [
                (item["tag"], item["x"], item["y"], item["weapon_cooldown"])
                for item in my_units
            ],
            key=lambda x: x[0],
        )
        enemy_units = self.get_enemy_units_by_type(obs, _MAP["unit_type"])
        mp = self.get_center_position(obs, "Self", _MAP["unit_type"])
        if len(my_units) > 0 and len(enemy_units) > 0:
            for clu in self.cluster_result[2]:
                if clu[3] < 0.4:
                    self.action_lst.append(
                        actions.RAW_FUNCTIONS.Smart_pt(
                            "now", [unit[0] for unit in clu[4]], clu[1]
                        )
                    )
            for unit in my_units_lst:
                self.action_lst.append(
                    actions.RAW_FUNCTIONS.Smart_unit(
                        "now",
                        unit[0],
                        self.get_nearest_enemy((unit[1], unit[2]), enemy_units),
                    )
                )
            return self.action_lst
        return actions.RAW_FUNCTIONS.no_op()

    def action_MIX_lure(self, obs):
        self.action_lst = []
        my_units = self.get_my_units_by_type(obs, _MAP["unit_type"])
        my_units_lst = sorted(
            [
                (item["tag"], item["x"], item["y"], item["weapon_cooldown"])
                for item in my_units
            ],
            key=lambda x: x[0],
        )
        enemy_units = self.get_enemy_units_by_type(obs, _MAP["unit_type"])
        mp = self.get_center_position(obs, "Self", _MAP["unit_type"])
        ep = self.get_center_position(obs, "Enemy", _MAP["unit_type"])
        if len(my_units) > 0 and len(enemy_units) > 0:
            separation_unit_list = []
            for clu in self.cluster_result[2]:
                if clu[2] < 0.4:
                    for unit in clu[4]:
                        separation_unit_list.append(
                            (unit, distance((unit[1], unit[2]), ep))
                        )
            if len(separation_unit_list) > 1:
                sorted_list = sorted(separation_unit_list, key=lambda x: x[1])
                except_unit_tag = sorted_list[0][0][0]
                self.action_lst.append(
                    actions.RAW_FUNCTIONS.Smart_pt("now", except_unit_tag, ep)
                )
            else:
                except_unit_tag = self.get_nearest_enemy(ep, my_units)
                self.action_lst.append(
                    actions.RAW_FUNCTIONS.Smart_pt("now", except_unit_tag, ep)
                )
            for unit in my_units_lst:
                if unit[0] != except_unit_tag:
                    self.action_lst.append(
                        actions.RAW_FUNCTIONS.Smart_pt("now", unit[0], mp)
                    )
            return self.action_lst
        return actions.RAW_FUNCTIONS.no_op()

    def action_MIX_sacrifice_lure(self, obs):
        self.action_lst = []
        my_units = self.get_my_units_by_type(obs, _MAP["unit_type"])
        my_units_lst = sorted(
            [
                (item["tag"], item["x"], item["y"], item["weapon_cooldown"])
                for item in my_units
            ],
            key=lambda x: x[0],
        )
        enemy_units = self.get_enemy_units_by_type(obs, _MAP["unit_type"])
        mp = (
            self.get_center_position(obs, "Self", _MAP["unit_type"])
            if len(my_units) > 0 and len(enemy_units) > 0
            else (0, 0)
        )
        ep = (
            self.get_center_position(obs, "Enemy", _MAP["unit_type"])
            if len(my_units) > 0 and len(enemy_units) > 0
            else (0, 0)
        )
        if len(my_units) > 0 and len(enemy_units) > 0:
            lure_uid = (
                self.choose_nearest_weakest_enemy(ep, my_units)
                if self.choose_nearest_weakest_enemy(ep, my_units)
                else 0
            )
            back_pt = ((6 / 5 * mp[0] - 1 / 5 * ep[0]), (6 / 5 * mp[1] - 1 / 5 * ep[1]))
            if lure_uid > 0:
                self.action_lst.append(
                    actions.RAW_FUNCTIONS.Smart_unit(
                        "now",
                        lure_uid,
                        self.choose_nearest_weakest_enemy(mp, enemy_units),
                    )
                )
            else:
                self.action_lst.append(
                    actions.RAW_FUNCTIONS.Smart_pt("now", lure_uid, (0, 0))
                )
            if [item[0] for item in my_units_lst if item[0] != lure_uid]:
                self.action_lst.append(
                    actions.RAW_FUNCTIONS.Smart_unit(
                        "queued",
                        [item[0] for item in my_units_lst if item[0] != lure_uid],
                        self.choose_nearest_weakest_enemy(mp, enemy_units),
                    )
                )
                self.action_lst.append(
                    actions.RAW_FUNCTIONS.Move_pt(
                        "now",
                        [item[0] for item in my_units_lst if item[0] != lure_uid],
                        back_pt,
                    )
                )
            return self.action_lst
        return actions.RAW_FUNCTIONS.no_op()

    def get_obs(self, obs):
        my_units = self.get_my_units_by_type(obs, _MAP["unit_type"])
        my_units_lst = sorted(
            [
                (
                    item["tag"],
                    item["x"],
                    item["y"],
                    item["weapon_cooldown"],
                    item["health"],
                    item["health_ratio"],
                )
                for item in my_units
            ],
            key=lambda x: x[0],
        )
        enemy_units = self.get_enemy_units_by_type(obs, _MAP["unit_type"])
        enemy_units_list = sorted(
            [
                (
                    item["tag"],
                    item["x"],
                    item["y"],
                    item["weapon_cooldown"],
                    item["health"],
                    item["health_ratio"],
                )
                for item in enemy_units
            ],
            key=lambda x: x[0],
        )
        self.current_obs["game_loop"] = obs.observation.game_loop[0]
        self.current_obs["my_units_maxH"] = self.score_defense_max
        self.current_obs["enemy_units_maxH"] = self.score_attack_max
        self.current_obs["my_units_lst"] = my_units_lst
        self.current_obs["enemy_units_list"] = enemy_units_list

    def get_short_term_reward(self, previous_obs, current_obs):
        reward = ShortTermReward()
        if previous_obs == {}:
            pass
        else:
            reward.r_kill = (
                len(previous_obs["enemy_units_list"])
                - len(current_obs["enemy_units_list"])
            ) * 5
            reward.r_fall = (
                -(len(previous_obs["my_units_lst"]) - len(current_obs["my_units_lst"]))
                * 5
            )
            if sum(item[5] for item in previous_obs["my_units_lst"]) < sum(
                item[5] for item in previous_obs["enemy_units_list"]
            ):
                if sum(item[5] for item in current_obs["enemy_units_list"]) - sum(
                    item[5] for item in current_obs["my_units_lst"]
                ) < sum(item[5] for item in previous_obs["enemy_units_list"]) - sum(
                    item[5] for item in previous_obs["my_units_lst"]
                ):
                    reward.r_inferior = 10
            if sum(item[5] for item in previous_obs["my_units_lst"]) > sum(
                item[5] for item in previous_obs["enemy_units_list"]
            ):
                if sum(item[5] for item in current_obs["my_units_lst"]) - sum(
                    item[5] for item in current_obs["enemy_units_list"]
                ) < sum(item[5] for item in previous_obs["my_units_lst"]) - sum(
                    item[5] for item in previous_obs["enemy_units_list"]
                ):
                    reward.r_dominant = -10
            reward.r_self_health_loss_ratio = (
                sum(item[5] for item in current_obs["my_units_lst"])
                - sum(item[5] for item in previous_obs["my_units_lst"])
            ) / 10
            reward.r_enemy_health_loss_ratio = (
                sum(item[5] for item in previous_obs["enemy_units_list"])
                - sum(item[5] for item in current_obs["enemy_units_list"])
            ) / 10
            max_previous_enemy_attacks = 0
            max_previous_my_attacks = 0
            for enemy_unit in previous_obs["enemy_units_list"]:
                enemy_attacks = 0
                for my_unit in previous_obs["my_units_lst"]:
                    distance = (
                        (enemy_unit[1] - my_unit[1]) ** 2
                        + (enemy_unit[2] - my_unit[2]) ** 2
                    ) ** 0.5
                    if distance < 5:
                        enemy_attacks += 1
                max_previous_enemy_attacks = max(
                    max_previous_enemy_attacks, enemy_attacks
                )
            for my_unit in previous_obs["my_units_lst"]:
                my_attacks = 0
                for enemy_unit in previous_obs["enemy_units_list"]:
                    distance = (
                        (enemy_unit[1] - my_unit[1]) ** 2
                        + (enemy_unit[2] - my_unit[2]) ** 2
                    ) ** 0.5
                    if distance < 5:
                        my_attacks += 1
                max_previous_my_attacks = max(max_previous_my_attacks, my_attacks)
            max_current_enemy_attacks = 0
            max_current_my_attacks = 0
            for enemy_unit in current_obs["enemy_units_list"]:
                enemy_attacks = 0
                for my_unit in current_obs["my_units_lst"]:
                    distance = (
                        (enemy_unit[1] - my_unit[1]) ** 2
                        + (enemy_unit[2] - my_unit[2]) ** 2
                    ) ** 0.5
                    if distance < _ENV_CONFIG["_UNIT_RADIUS"]:
                        enemy_attacks += 1
                max_current_enemy_attacks = max(
                    max_current_enemy_attacks, enemy_attacks
                )
            for my_unit in current_obs["my_units_lst"]:
                my_attacks = 0
                for enemy_unit in current_obs["enemy_units_list"]:
                    distance = (
                        (enemy_unit[1] - my_unit[1]) ** 2
                        + (enemy_unit[2] - my_unit[2]) ** 2
                    ) ** 0.5
                    if distance < _ENV_CONFIG["_UNIT_RADIUS"]:
                        my_attacks += 1
                max_current_my_attacks = max(max_current_my_attacks, my_attacks)
            if max_current_enemy_attacks > max_previous_enemy_attacks:
                reward.r_fire_coverage = (
                    max_current_enemy_attacks - max_previous_enemy_attacks
                ) * 5
            if max_current_my_attacks < max_previous_my_attacks:
                reward.r_fire_coverage = (
                    max_previous_my_attacks - max_current_my_attacks
                ) * 5
        self.previous_reward = reward
        return reward

    def cul_short_term_reward(self, reward_class: ShortTermReward):
        reward = (
            reward_class.r_kill
            + reward_class.r_fall
            + reward_class.r_inferior
            + reward_class.r_dominant
            + reward_class.r_self_health_loss_ratio
            + reward_class.r_enemy_health_loss_ratio
            + reward_class.r_fire_coverage
            + reward_class.r_covered_in_fire
        )
        return reward

    def new_game(self):
        self.end_game_frames = _ENV_CONFIG["_MAX_STEP"] * _ENV_CONFIG["_STEP_MUL"]
        self.end_game_state = "Dogfall"
        self.end_game_flag = False
        self._termination_signaled = False
        self.score_attack_max = 0
        self.score_defense_max = 0
        self.score_cumulative_attack_last = 0
        self.score_cumulative_defense_last = 0
        self.previous_clu_state = None
        self.previous_clu_action = None
        self.previous_obs = {}
        self.current_obs = {}
        self.action_queue.clear()

    def get_window_im(self, obs):
        my_units = self.get_my_units_by_type(obs, _MAP["unit_type"])
        enemy_units = self.get_enemy_units_by_type(obs, _MAP["unit_type"])
        unit_my_list = sorted(
            [(item["tag"], item["x"], item["y"]) for item in my_units],
            key=lambda x: x[0],
        )
        unit_enemy_list = sorted(
            [(item["tag"], item["x"], item["y"]) for item in enemy_units],
            key=lambda x: x[0],
        )
        influence_map = self.get_influence_map(unit_my_list, unit_enemy_list)
        top, bottom, left, right = self.get_map_boundary(
            influence_map, _ALG_CONFIG["_IM_BOUNDARY_WIDTH"]
        )
        target_gp = self.analyze_influence_map(influence_map)
        target_mp = self.grid_to_map(target_gp[0], target_gp[1])
        window_map = influence_map.T[left:right, top:bottom]
        return window_map

    # 2025-05-22
    def classify_new_state(self, new_state, bktree, threshold=1.0):
        cluster_id = bktree.query(new_state, threshold)
        if cluster_id is not None:
            return cluster_id
        else:
            new_cluster_id = bktree.get_next_cluster_id()
            new_node = ClusterNode(new_state, new_cluster_id)
            bktree.insert(new_node, bktree.root)
            return new_cluster_id

    def get_norm_state(self, obs):
        my_units = self.get_my_units_by_type(obs, _MAP["unit_type"])
        my_units_lst = sorted(
            [
                {
                    "tag": item["tag"],
                    "x": item["x"],
                    "y": item["y"],
                    "hp": item["health"],
                    "weapon_cooldown": item["weapon_cooldown"],
                }
                for item in my_units
            ],
            key=lambda x: x["tag"],
        )
        enemy_units = self.get_enemy_units_by_type(obs, _MAP["unit_type"])
        enemy_units_lst = sorted(
            [
                {
                    "tag": item["tag"],
                    "x": item["x"],
                    "y": item["y"],
                    "hp": item["health"],
                    "weapon_cooldown": item["weapon_cooldown"],
                }
                for item in enemy_units
            ],
            key=lambda x: x["tag"],
        )
        map_resolution = _ENV_CONFIG["_MAP_RESOLUTION"]

        my_units_lst_norm = [
            (
                unit["x"] / (map_resolution / 2) - 1.0,
                1.0 - unit["y"] / (map_resolution / 2),
                unit["hp"] / 45.0,
            )
            for unit in my_units_lst
        ]
        enemy_units_lst_norm = [
            (
                unit["x"] / (map_resolution / 2) - 1.0,
                1.0 - unit["y"] / (map_resolution / 2),
                unit["hp"] / 45.0,
            )
            for unit in enemy_units_lst
        ]

        norm_state = {"red_army": my_units_lst_norm, "blue_army": enemy_units_lst_norm}
        return norm_state

    def get_state_cluster(self, norm_state):
        if self.primary_bktree.root is None:
            self.primary_bktree.root = ClusterNode(norm_state, 1)
            self.secondary_bktree[1].root = ClusterNode(norm_state, 1)
            # self.primary_bktree.root.state_list = [norm_state]
            return (1, 1)
        else:
            new_cluster_id = self.classify_new_state(
                norm_state, self.primary_bktree, threshold=1.0
            )
            if self.secondary_bktree[new_cluster_id].root is None:
                self.secondary_bktree[new_cluster_id].root = ClusterNode(norm_state, 1)
                return (new_cluster_id, 1)
            else:
                new_sub_cluster_id = self.classify_new_state(
                    norm_state, self.secondary_bktree[new_cluster_id], threshold=0.5
                )
                return (new_cluster_id, new_sub_cluster_id)
                # if new_sub_cluster_id != new_cluster_id:
                #     self.secondary_bktree[new_cluster_id].insert(ClusterNode(norm_state, new_sub_cluster_id), self.secondary_bktree[new_cluster_id].root)

            # if new_cluster_id is not None:

    def save_bktree(self):
        def serialize_node(node):
            node_info = {
                "state": node.state,
                "cluster_id": node.cluster_id,
                "children": {},
            }
            for dist, child in node.children.items():
                node_info["children"][dist] = serialize_node(child)
            return node_info

        if self.primary_bktree.root is not None:
            primary_tree_data = serialize_node(self.primary_bktree.root)
            with open(_PATH_CONFIG["_GAME_PRIMARY_BKTREE_PATH"], "w") as f:
                json.dump(primary_tree_data, f, indent=4)

        for cluster_id, bktree in self.secondary_bktree.items():
            if bktree.root is not None:
                secondary_tree_data = serialize_node(bktree.root)
                with open(
                    f"{_PATH_CONFIG['_GAME_SECONDARY_BKTREE_PREFIX']}_{cluster_id}.json",
                    "w",
                ) as f:
                    json.dump(secondary_tree_data, f, indent=4)

    def code_state_clu(self, cluster_list):
        mapped = ""
        if len(cluster_list) > 0:
            for item in cluster_list:
                mapped += "{:01X}".format(int(item[0] * 15.9))
                mapped += "{:01X}".format(int(item[1] * 15.9))
            return mapped
        else:
            return "X"

    def get_state_clu(self, cluster_result):
        cluster_list = [(item[2], item[3]) for item in cluster_result[2]]
        result = self.code_state_clu(cluster_list)
        return result

    def print_cluster_result(self, cluster_result):
        with open(_PATH_CONFIG["_GAME_CLUS_PATH"], "r+") as file:
            existing_data = file.read()
            for item in cluster_result[2]:
                result = (
                    len(item[4]),
                    [(x[1], x[2]) for x in item[4]],
                    item[2],
                    item[3],
                )
                if str(result) not in existing_data:
                    file.write(str(result) + "\n")

    def get_units_health_ratio(self, units):
        if len(units):
            return sum(units) / len(units)
        else:
            return 0.0

    def get_clusters_health(self, clusters, my_units, enemy_units):
        clusters_health_dict = {}
        my_units_lst = sorted(
            [
                (
                    item["tag"],
                    item["x"],
                    item["y"],
                    item["health"],
                    item["health_ratio"],
                )
                for item in my_units
            ]
        )
        enemy_units_lst = sorted(
            [
                (
                    item["tag"],
                    item["x"],
                    item["y"],
                    item["health"],
                    item["health_ratio"],
                )
                for item in enemy_units
            ]
        )
        for cluster in clusters[2]:
            clusters_health_dict.update({cluster[0]: [unit[0] for unit in cluster[4]]})
        new_clusters_health_dict = {}
        for cluster_id, cluster_item in clusters_health_dict.items():
            new_value = [
                (
                    next((unit[0] for unit in my_units_lst if unit[0] == item), None),
                    next((unit[1] for unit in my_units_lst if unit[0] == item), None),
                    next((unit[2] for unit in my_units_lst if unit[0] == item), None),
                    next((unit[3] for unit in my_units_lst if unit[0] == item), None),
                    next((unit[4] for unit in my_units_lst if unit[0] == item), None),
                )
                for item in cluster_item
            ]
            new_clusters_health_dict[cluster_id] = new_value
        new_clusters_health_dict[-1] = [
            (unit[0], unit[1], unit[2], unit[3], unit[4]) for unit in enemy_units_lst
        ]
        return new_clusters_health_dict

    def _end_episode(self, obs):
        TEST_FLAG = obs.get_test_flag()
        plt.close()
        matplotlib.pyplot.figure().clear()
        matplotlib.pyplot.close()
        self.previous_obs = {}
        self.current_obs = {}
        if not TEST_FLAG:
            f = open(_PATH_CONFIG["_GAME_RESULT_PATH"], "a", encoding="UTF-8")
            reward_d = -(self.score_cumulative_attack_now - self.score_attack_max)
            reward_a = self.score_cumulative_defense_now - self.score_defense_max
            f.write(
                f"{self.end_game_state}\t{self.end_game_frames}\t{reward_d}\t{reward_a}\n"
            )
            f.close()
            self.clusters_qtable.q_table.to_csv(
                _PATH_CONFIG["_GAME_QTABLE_PATH"], header=True, index=True, sep=","
            )
            self.end_game_frames = _ENV_CONFIG["_MAX_STEP"] * _ENV_CONFIG["_STEP_MUL"]
            if self.end_game_flag != True:
                self.end_game_state = "Dogfall"
                self.end_game_flag = False
            with open(_PATH_CONFIG["_GAME_ACTION_LOG_PATH"], "a") as file:
                file.write("\n")
                file.close()
            if self.ctx.episode_count == 2 or self.ctx.episode_count % 10 == 0:
                append_qtable_csv(
                    self.ctx.episode_count,
                    self.clusters_qtable.q_table,
                    self.actions,
                )
                save_dataframes_to_csv(
                    self.sub_clusters_qtable_list,
                    _PATH_CONFIG["_GAME_SUB_QTABLE_PATH"],
                )
                self.save_bktree()
            save_node_log(
                self.ctx,
                _PATH_CONFIG["_GAME_STATE_NODE_PATH"],
                _PATH_CONFIG["_GAME_NODE_LOG_PATH"],
            )
        else:
            f = open(_PATH_CONFIG["_GAME_RESULT_TEST_PATH"], "a", encoding="UTF-8")
            reward_d = -(self.score_cumulative_attack_now - self.score_attack_max)
            reward_a = self.score_cumulative_defense_now - self.score_defense_max
            f.write(
                f"{self.end_game_state}\t{self.end_game_frames}\t{reward_d}\t{reward_a}\n"
            )
            f.close()
            self.end_game_frames = _ENV_CONFIG["_MAX_STEP"] * _ENV_CONFIG["_STEP_MUL"]
            if self.end_game_flag != True:
                self.end_game_state = "Dogfall"
                self.end_game_flag = False

        self.previous_clu_state = None
        self.previous_clu_action = None

    def step(self, obs, env):
        super(SmartAgent, self).step(obs, env)

        if obs.last():
            self._end_episode(obs)
            return actions.RAW_FUNCTIONS.no_op()

        TEST_FLAG = obs.get_test_flag()

        if obs.first():
            self._termination_signaled = False
            if not self._initial_spawned:
                unit_list_my = self.get_my_units_by_type(obs, _MAP["unit_type"])
                unit_list_enemy = self.get_enemy_units_by_type(obs, _MAP["unit_type"])
                self._initial_units_my = [(u.x, u.y) for u in unit_list_my]
                self._initial_units_enemy = [(u.x, u.y) for u in unit_list_enemy]
                self._initial_spawned = True
            if not TEST_FLAG:
                unit_list_both = unit_list_my + unit_list_enemy
                dataframe = pd.DataFrame(data=unit_list_both)
                dataframe.to_csv(
                    _PATH_CONFIG["_UNITS_LIST_PATH"], header=True, index=False, sep=","
                )
                merge_units_csv(
                    _PATH_CONFIG["_UNITS_ATTRIBUTE_PATH"],
                    _PATH_CONFIG["_UNITS_LIST_PATH"],
                )
        my_units = self.get_my_units_by_type(obs, _MAP["unit_type"])
        enemy_units = self.get_enemy_units_by_type(obs, _MAP["unit_type"])
        my_units_lst = sorted(
            [(item["tag"], item["weapon_cooldown"]) for item in my_units],
            key=lambda x: x[0],
        )
        if (
            self.action_queue.count("Attack_unit") + self.action_queue.count("Smart_pt")
        ) != 0:
            for tup in [elem for elem in my_units_lst]:
                if tup[1] != 0:
                    self._move_back = True
        if (
            not self._termination_signaled
            and len(enemy_units) == 0
            and obs.observation["score_cumulative"][5]
            == obs.observation["score_cumulative"][3]
        ):
            self.end_game_state = "Win"
            self.end_game_flag = True
            self._termination_signaled = True
            env.f_result = "win"
            if self.end_game_frames > obs.observation.game_loop:
                self.end_game_frames = obs.observation.game_loop
        if not self._termination_signaled and len(my_units) == 0:
            self.end_game_state = "Loss"
            self.end_game_flag = True
            self._termination_signaled = True
            env.f_result = "loss"
            if self.end_game_frames > obs.observation.game_loop:
                self.end_game_frames = obs.observation.game_loop

        # 2025.05.28
        state_norm = self.get_norm_state(obs)
        state_cluster_id = self.get_state_cluster(state_norm)
        # state_im = str(self.get_state(obs))
        state_im = str(state_cluster_id)
        if not TEST_FLAG:
            cluster_item = self.clusters_qtable.choose_action(
                state_im, 1 - 0.5 / math.sqrt(self.ctx.episode_count - 1)
            )
        else:
            cluster_item = self.clusters_qtable.choose_action(state_im, 1)
        cluster_result = getattr(self, cluster_item)(obs)
        cluster_health_result = self.get_clusters_health(
            cluster_result, my_units, enemy_units
        )
        if not TEST_FLAG:
            save_clusters_health_to_csv(
                cluster_health_result,
                _PATH_CONFIG["_GAME_SUB_EPISODE_PATH"],
                self.ctx.episode_count,
                obs.observation.game_loop,
            )
        self.cluster_result = cluster_result
        if not TEST_FLAG:
            self.print_cluster_result(cluster_result)
        self.sub_clusters_qtable_tag = (cluster_result[0], cluster_result[1])
        state_clu = self.get_state_clu(cluster_result)
        if not TEST_FLAG:
            combat_action_item = self.sub_clusters_qtable_list[
                self.sub_clusters_qtable_tag
            ].choose_action(state_clu, 1 - 0.5 / math.sqrt(self.ctx.episode_count - 1))
            with open(_PATH_CONFIG["_GAME_ACTION_LOG_PATH"], "a") as file:
                file.write(
                    str(self.clusters.index(cluster_item))
                    + chr(ord("a") + self.actions.index(combat_action_item))
                )
        else:
            combat_action_item = self.sub_clusters_qtable_list[
                self.sub_clusters_qtable_tag
            ].choose_action(state_clu, 1)

        self.get_obs(obs)
        self.get_short_term_reward(self.previous_obs, self.current_obs)
        self.previous_obs = self.current_obs.copy()
        self.score_cumulative_attack_now = sum([item["health"] for item in enemy_units])
        self.score_cumulative_defense_now = sum([item["health"] for item in my_units])
        reward_attack = -(
            self.score_cumulative_attack_now - self.score_cumulative_attack_last
        )
        reward_defense = (
            self.score_cumulative_defense_now - self.score_cumulative_defense_last
        )
        self.score_cumulative_attack_last = self.score_cumulative_attack_now
        self.score_cumulative_defense_last = self.score_cumulative_defense_now
        if not TEST_FLAG:
            save_short_term_result(
                self.previous_reward,
                _PATH_CONFIG["_GAME_SHORT_TERM_RESULT_PATH"],
                self.ctx.episode_count,
                obs.observation.game_loop,
            )

        reward_cumulative = reward_attack + reward_defense
        if not TEST_FLAG:
            # 2025.05.28
            state_norm = self.get_norm_state(obs)
            state_cluster_id = self.get_state_cluster(state_norm)
            # save_node(str(self.get_state(obs)), sum([item['health'] for item in self.get_my_units_by_type(obs, _MAP["unit_type"])])
            #       - sum([item['health'] for item in self.get_enemy_units_by_type(obs, _MAP["unit_type"])]))
            save_node(
                str(state_cluster_id),
                sum(
                    [
                        item["health"]
                        for item in self.get_my_units_by_type(obs, _MAP["unit_type"])
                    ]
                )
                - sum(
                    [
                        item["health"]
                        for item in self.get_enemy_units_by_type(obs, _MAP["unit_type"])
                    ]
                ),
                self.ctx,
            )
            if self.previous_clu_action is not None:
                self.clusters_qtable.learn(
                    self.previous_clu_state,
                    self.previous_clu_action,
                    # obs.reward,
                    reward_cumulative,
                    "terminal" if obs.last() else state_im,
                )
            if self.previous_combat_action[self.sub_clusters_qtable_tag] is not None:
                self.sub_clusters_qtable_list[self.sub_clusters_qtable_tag].learn(
                    self.previous_combat_state[self.sub_clusters_qtable_tag],
                    self.previous_combat_action[self.sub_clusters_qtable_tag],
                    self.cul_short_term_reward(self.previous_reward),
                    "terminal" if obs.last() else state_clu,
                )
            self.previous_clu_state = state_im
            self.previous_clu_action = cluster_item
            self.previous_sub_tag = self.sub_clusters_qtable_tag
            self.previous_combat_state[self.sub_clusters_qtable_tag] = state_clu
            self.previous_combat_action[self.sub_clusters_qtable_tag] = (
                combat_action_item
            )

        return getattr(self, combat_action_item)(obs)
