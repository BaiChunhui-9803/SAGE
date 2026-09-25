"""
Data Loader module for PredictionRTS
Refactored from data/load_data.py to support lazy loading and Hydra config
"""

import json
import logging
import csv
import re
import os
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from src.structure.bk_tree import (
    BKTreeNode,
    BKTree,
    get_max_cluster_id,
    custom_distance,
)
from src.structure.state_distance import custom_distance as state_custom_distance
from src.utils.path_utils import get_data_paths

logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, cfg):
        self.cfg = cfg
        self.paths = get_data_paths(cfg)

        self._primary_bk_tree = None
        self._secondary_bk_trees = None
        self._state_node_dict = None
        self._reverse_dict = None
        self._state_value = None
        self._state_log = None
        self._state_distance_matrix = None
        self._state_log_distance_matrix = None
        self._game_results = None
        self._action_dict = None
        self._action_log = None
        self._episode_results = None
        self._r_log = None
        self._dt_data = None
        self._action_vocab = None

    @property
    def primary_bk_tree(self):
        if self._primary_bk_tree is None:
            self._primary_bk_tree = self._load_bk_tree_from_file(
                self.paths["primary_bktree"]
            )
        return self._primary_bk_tree

    @property
    def secondary_bk_trees(self):
        if self._secondary_bk_trees is None:
            self._secondary_bk_trees = self._load_all_bktrees(self.primary_bk_tree)
        return self._secondary_bk_trees

    @property
    def state_node_dict(self):
        if self._state_node_dict is None:
            self._state_node_dict, self._reverse_dict, self._state_value = (
                self._read_state_node_file()
            )
        return self._state_node_dict

    @property
    def reverse_dict(self):
        if self._reverse_dict is None:
            self._state_node_dict, self._reverse_dict, self._state_value = (
                self._read_state_node_file()
            )
        return self._reverse_dict

    @property
    def state_value(self):
        if self._state_value is None:
            self._state_node_dict, self._reverse_dict, self._state_value = (
                self._read_state_node_file()
            )
        return self._state_value

    @property
    def state_log(self):
        if self._state_log is None:
            self._state_log = self._read_node_log_file()
        return self._state_log

    @property
    def state_distance_matrix(self):
        if self._state_distance_matrix is None:
            self._state_distance_matrix = self._calculate_distance_matrix()
        return self._state_distance_matrix

    @property
    def game_results(self):
        if self._game_results is None:
            self._game_results = self._read_game_result_file()
        return self._game_results

    @property
    def action_dict(self):
        if self._action_dict is None:
            self._action_dict = self._create_action_dictionary()
        return self._action_dict

    @property
    def action_log(self):
        if self._action_log is None:
            self._action_log = self._read_action_csv()
        return self._action_log

    @property
    def r_log(self):
        if self._r_log is None:
            self._r_log = self._process_rewards()
        return self._r_log

    @property
    def dt_data(self):
        if self._dt_data is None:
            self._dt_data, self._action_vocab = self._preprocess_dt_data()
        return self._dt_data

    @property
    def action_vocab(self):
        if self._action_vocab is None:
            self._dt_data, self._action_vocab = self._preprocess_dt_data()
        return self._action_vocab

    def _load_bk_tree_from_file(self, file_path: str) -> BKTree:
        def deserialize_node(node_data):
            state = {"state": [node_data["state"]]}
            node = BKTreeNode(state, node_data["cluster_id"])
            for dist, child_data in node_data["children"].items():
                child_node = deserialize_node(child_data)
                node.add_child(float(dist), child_node)
            return node

        try:
            with open(file_path, "r") as file:
                tree_data = json.load(file)
            bk_tree = BKTree()
            bk_tree.root = deserialize_node(tree_data)
            logger.debug(f"Loaded BKTree from {file_path}")
            return bk_tree
        except Exception as e:
            logger.error(f"Failed to load BKTree from {file_path}: {e}")
            raise

    def _load_all_bktrees(self, primary_bk_tree: BKTree) -> Dict[int, BKTree]:
        secondary_bk_trees = {}
        logger.info("--- Primary BKTree Analysis Started ---")
        logger.info(f"Root cluster ID: {primary_bk_tree.root.cluster_id}")
        logger.info(f"Root children counts: {len(primary_bk_tree.root.children)}")

        cluster_count = get_max_cluster_id(primary_bk_tree)
        logger.info(f"Detected {cluster_count} clusters to load.")

        for cluster_id in range(1, cluster_count + 1):
            path = f"{self.paths['secondary_bktree_prefix']}_{cluster_id}.json"
            try:
                tree = self._load_bk_tree_from_file(path)
                secondary_bk_trees[cluster_id] = tree
                logger.debug(f"Loaded Secondary BKTree [{cluster_id}/{cluster_count}]")
            except FileNotFoundError:
                logger.warning(f"Secondary BKTree file missing: {path}")
            except Exception as e:
                logger.error(f"Unexpected error on cluster {cluster_id}: {e}")

        logger.info(f"All trees loaded. Total: {len(secondary_bk_trees)}")
        return secondary_bk_trees

    def _read_state_node_file(self) -> Tuple[Dict, Dict, List]:
        state_node_dict = {}
        with open(self.paths["state_node"], "r") as file:
            for line in file:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                key = eval(parts[0])
                id = int(parts[1])
                score = float(parts[2])
                state_node_dict[key] = {"id": id, "score": score}

        reverse_dict = {}
        state_value = []
        for key, value in state_node_dict.items():
            id = value["id"]
            score = value["score"]
            if id not in reverse_dict:
                reverse_dict[id] = {"cluster": key, "score": score}
                state_value.append(score)

        return state_node_dict, reverse_dict, state_value

    def _read_node_log_file(self) -> List[List[int]]:
        result = []
        with open(self.paths["node_log"], "r") as file:
            for line in file:
                parts = line.strip().split()
                parts = [int(part) for part in parts]
                result.append(parts)
        return result

    def _read_game_result_file(self) -> List[List]:
        result = []
        with open(self.paths["game_result"], "r") as file:
            for line in file:
                parts = line.strip().split("\t")
                if len(parts) < 4:
                    continue
                outcome = parts[0]
                steps = int(float(parts[1].strip("[]")))
                score = int(parts[2])
                penalty = int(parts[3])
                result.append([outcome, steps, score, penalty])
        return result

    def _read_action_csv(self) -> List[List[str]]:
        result = []
        with open(self.paths["action_log"], "r") as file:
            reader = csv.reader(file)
            for row in reader:
                if row:
                    actions = [row[0][i : i + 2] for i in range(0, len(row[0]), 2)]
                    action_letters = [a[1] for a in actions if len(a) >= 2]
                    result.append(action_letters)
        return result

    def _create_action_dictionary(self) -> Dict[str, int]:
        action_dict = {}
        action_path = self.paths["action_path"]
        if not os.path.exists(action_path):
            return action_dict

        for filename in os.listdir(action_path):
            if filename.endswith(".csv"):
                action_name = filename.replace(".csv", "")
                action_dict[action_name] = len(action_dict)
        return action_dict

    def _calculate_distance_matrix(self):
        from src.utils.calculate_utils import (
            calculate_and_save_distance_matrix as calc_dm,
        )

        return calc_dm(
            self.reverse_dict,
            state_custom_distance,
            self.secondary_bk_trees,
            self.paths["distance_matrix_folder"],
        )

    def _process_rewards(self):
        episode_results = self._batch_process_episodes()
        _, rewards_only_list = self._process_rewards_from_episodes(episode_results)
        return rewards_only_list

    def _batch_process_episodes(self) -> Dict:
        all_episodes = {}
        directory_path = self.paths["episode_result_path"]

        if not os.path.exists(directory_path):
            logger.error(f"Directory not found: {directory_path}")
            return all_episodes

        files = [f for f in os.listdir(directory_path) if f.endswith(".csv")]
        if not files:
            logger.warning(f"No CSV files found in {directory_path}")
            return all_episodes

        files.sort(
            key=lambda x: int(re.search(r"\d+", x).group())
            if re.search(r"\d+", x)
            else 0
        )

        for filename in files:
            file_path = os.path.join(directory_path, filename)
            episode_id = filename.split(".")[0]
            data = self._parse_csv_content(file_path)
            if data:
                all_episodes[episode_id] = data

        return all_episodes

    def _parse_csv_content(self, file_path: str) -> List:
        steps = []
        current_step = None
        current_cluster_id = None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    step_match = re.search(r"step\[(\d+)\]", line)
                    if step_match:
                        current_step = {
                            "step": int(step_match.group(1)),
                            "self": [],
                            "enemy": [],
                        }
                        steps.append(current_step)
                        continue

                    cluster_match = re.search(r"cluster_(-?\d+):", line)
                    if cluster_match:
                        current_cluster_id = int(cluster_match.group(1))
                        continue

                    if "," in line:
                        if current_step is None:
                            continue
                        raw_units = line.replace(" ", "").strip(";").split(";")
                        for unit in raw_units:
                            if not unit:
                                continue
                            try:
                                vals = [int(v) for v in unit.split(",")]
                                if len(vals) >= 5:
                                    unit_info = {
                                        "unit_id": vals[0],
                                        "x": vals[1],
                                        "y": vals[2],
                                        "hp": vals[3],
                                        "hp_percent": vals[4],
                                    }
                                    if current_cluster_id is not None:
                                        if current_cluster_id >= 0:
                                            current_step["self"].append(unit_info)
                                        else:
                                            current_step["enemy"].append(unit_info)
                            except ValueError:
                                continue
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

        return steps

    def _process_rewards_from_episodes(self, all_data: Dict) -> Tuple[Dict, List]:
        reward_data = {}
        rewards_only_list = []

        sorted_filenames = sorted(
            all_data.keys(),
            key=lambda x: int(re.search(r"\d+", x).group())
            if re.search(r"\d+", x)
            else 0,
        )

        for filename in sorted_filenames:
            steps = all_data[filename]
            steps.sort(key=lambda x: x["step"])

            episode_rewards = []
            last_total_value = None

            for step_entry in steps:
                current_total_value = self._calculate_step_value(step_entry)
                if last_total_value is None:
                    reward_r = 0
                else:
                    reward_r = current_total_value - last_total_value

                last_total_value = current_total_value
                episode_rewards.append(reward_r)

            reward_data[filename] = episode_rewards
            rewards_only_list.append(episode_rewards)

        return reward_data, rewards_only_list

    def _calculate_step_value(self, step_data: Dict) -> int:
        self_hp = sum(unit["hp"] for unit in step_data.get("self", []))
        enemy_hp = sum(unit["hp"] for unit in step_data.get("enemy", []))
        return self_hp - enemy_hp

    def _preprocess_dt_data(self):
        from src.utils.load_utils import preprocess_decision_transformer_data

        return preprocess_decision_transformer_data(
            self.state_log, self.action_log, self.r_log
        )

    def load_all(self):
        self.primary_bk_tree
        self.secondary_bk_trees
        self.state_node_dict
        self.state_log
        self.state_distance_matrix
        self.game_results
        self.action_dict
        self.action_log
        self.r_log
        self.dt_data
        logger.info("All data loaded successfully")
