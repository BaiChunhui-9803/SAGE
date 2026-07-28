from pysc2.lib import actions, features, units


def get_map_config(map_key):
    _MAP_CONFIG = {
        # MM4
        "sce-1": {
            "map_name": "local_enemy_test_1",
            "unit_type_id": 48,
            "unit_type": units.Terran.Marine,
            "unit_scale": 4,
            "expected_spawn_count": 4,
            "expected_unit_hp": 45,
        },
        # MM4m
        "sce-1m": {
            "map_name": "local_enemy_test_1_mirror",
            "unit_type_id": 48,
            "unit_type": units.Terran.Marine,
            "unit_scale": 4,
            "expected_spawn_count": 4,
            "expected_unit_hp": 45,
        },
        # MM4dist
        "sce-2": {
            "map_name": "MarineMicro_MvsM_4_dist",
            "unit_type_id": 48,
            "unit_type": units.Terran.Marine,
            "unit_scale": 4,
            "expected_spawn_count": 4,
            "expected_unit_hp": 45,
        },
        # MM4distm
        "sce-2m": {
            "map_name": "MarineMicro_MvsM_4_dist_mirror",
            "unit_type_id": 48,
            "unit_type": units.Terran.Marine,
            "unit_scale": 4,
            "expected_spawn_count": 4,
            "expected_unit_hp": 45,
        },
        # MM8
        "sce-3": {
            "map_name": "MarineMicro_MvsM_8_far",
            "unit_type_id": 48,
            "unit_type": units.Terran.Marine,
            "unit_scale": 8,
            "expected_spawn_count": 8,
            "expected_unit_hp": 45,
        },
        # MM8m
        "sce-3m": {
            "map_name": "MarineMicro_MvsM_8_far_mirror",
            "unit_type_id": 48,
            "unit_type": units.Terran.Marine,
            "unit_scale": 8,
            "expected_spawn_count": 8,
            "expected_unit_hp": 45,
        },
    }

    _MAP = _MAP_CONFIG[map_key]

    _ENV_CONFIG = {
        "_MAP_RESOLUTION": 128,
        "_STEP_MUL": 10,
        "_UNIT_RADIUS": 5.0,
        "_MAX_EPISODE": 8000,
        "_RUNNER": 10,
        "_RESET_FRAMES": 200,
        "_MAX_STEP": 0,
    }

    # _ENV_CONFIG["_MAX_STEP"] = 250 / _ENV_CONFIG["_STEP_MUL"] * _MAP["unit_scale"] / 4

    _ALG_CONFIG = {
        "_IM_BOUNDARY_WIDTH": 2,
        "_MY_UNIT_INFLUENCE": [16, 9, 4, 1],
        "_ENEMY_UNIT_INFLUENCE": [-16, -9, -4, -1],
        "_MAX_INFLUENCE": 16 * _MAP["unit_scale"],
        "_MIN_INFLUENCE": -16 * _MAP["unit_scale"],
        "learning_rate": 0.1,
        "reward_decay": 0.9,
    }

    _PATH_CONFIG = {
        "_RUNS_PATH": "datas/runs",
        "_DATA_TRANSIT_PATH": "datas/data_for_transit",
        "_UNITS_ATTRIBUTE_PATH": "datas/data_for_overall/units_name.csv",
        "_UNITS_LIST_PATH": "datas/data_for_overall/units_list.csv",
        "_GAME_RESULT_PATH": "datas/data_for_transit/game_result.txt",
        "_GAME_RESULT_TEST_PATH": "datas/data_for_transit/game_result_test.txt",
        "_GAME_QTABLE_PATH": "datas/data_for_transit/q_table.csv",
        "_GAME_CLUS_PATH": "datas/data_for_transit/clusters.csv",
        "_GAME_ACTION_LOG_PATH": "datas/data_for_transit/action_log.csv",
        "_GAME_SUB_QTABLE_PATH": "datas/data_for_transit/sub_q_table",
        "_EPISODE_QTABLE_PATH": "datas/data_for_transit/episode_q_table.csv",
        "_GAME_SUB_EPISODE_PATH": "datas/data_for_transit/sub_episode",
        "_GAME_SHORT_TERM_RESULT_PATH": "datas/data_for_transit/short_term_result",
        "_GAME_ACTION_PATH": "datas/data_for_transit/action.csv",
        "_GAME_GRAPH_PATH": "datas/data_for_transit/graph",
        "_GAME_STATE_NODE_PATH": "datas/data_for_transit/graph/state_node.txt",
        "_GAME_NODE_LOG_PATH": "datas/data_for_transit/graph/node_log.txt",
        "_GAME_BKTREE_PATH": "datas/data_for_transit/bktree",
        "_GAME_PRIMARY_BKTREE_PATH": "datas/data_for_transit/bktree/primary_bktree.json",
        "_GAME_SECONDARY_BKTREE_PREFIX": "datas/data_for_transit/bktree/secondary_bktree",
        "_GAME_AUGMENTED_BKTREE_PATH": "datas/data_for_transit/bktree_augmented/primary_bktree.json",
        "_GAME_AUGMENTED_SECONDARY_PREFIX": "datas/data_for_transit/bktree_augmented/secondary_bktree",
        "_GAME_AUGMENTED_STATE_NODE_PATH": "datas/data_for_transit/graph/state_node_augmented.txt",
    }

    return _MAP_CONFIG, _MAP, _ENV_CONFIG, _ALG_CONFIG, _PATH_CONFIG
