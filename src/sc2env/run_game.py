import time
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from absl import flags, app

from pysc2.env import sc2_env, run_loop, environment
from pysc2.lib import actions, features, units
from s2clientprotocol import debug_pb2 as sc_debug
from s2clientprotocol import common_pb2 as sc_common

from typing import Optional

from src.sc2env.config import get_map_config
from src.sc2env.agent import SmartAgent, Agent
from src.sc2env.utils import GameContext, init_game
from src.sc2env.bridge import GameBridge

_MAP_CONFIG, _MAP, _ENV_CONFIG, _ALG_CONFIG, _PATH_CONFIG = get_map_config("sce-1")
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

flags.DEFINE_string(
    "run_name", None, "Name for this run. Defaults to map_key_YYYYMMDD_HHMMSS"
)
flags.DEFINE_string("map_key", "sce-1", "Map config key")
FLAGS = flags.FLAGS


def _apply_map_config(map_key: str):
    global _MAP_CONFIG, _MAP, _ENV_CONFIG, _ALG_CONFIG, _PATH_CONFIG
    _MAP_CONFIG, _MAP, _ENV_CONFIG, _ALG_CONFIG, _PATH_CONFIG = get_map_config(map_key)

    agent_module = sys.modules.get("src.sc2env.agent")
    if agent_module is not None:
        agent_module._MAP_CONFIG = _MAP_CONFIG
        agent_module._MAP = _MAP
        agent_module._ENV_CONFIG = _ENV_CONFIG
        agent_module._ALG_CONFIG = _ALG_CONFIG
        agent_module._PATH_CONFIG = _PATH_CONFIG

    kg_agent_module = sys.modules.get("src.sc2env.kg_guided_agent")
    if kg_agent_module is not None:
        kg_agent_module._MAP_CONFIG = _MAP_CONFIG
        kg_agent_module._MAP = _MAP
        kg_agent_module._ENV_CONFIG = _ENV_CONFIG
        kg_agent_module._ALG_CONFIG = _ALG_CONFIG
        kg_agent_module._PATH_CONFIG = _PATH_CONFIG

    replay_module = sys.modules.get("src.sc2env.replay_collector")
    if replay_module is not None:
        replay_module._MAP_CONFIG = _MAP_CONFIG
        replay_module._MAP = _MAP
        replay_module._ENV_CONFIG = _ENV_CONFIG
        replay_module._ALG_CONFIG = _ALG_CONFIG
        replay_module._PATH_CONFIG = _PATH_CONFIG


def kill_all_units(env, obs):
    unit_tags = [u.tag for u in obs.raw_units]
    debug_command = [
        sc_debug.DebugCommand(kill_unit=sc_debug.DebugKillUnit(tag=unit_tags))
    ]
    env._controllers[0].debug(debug_command)


def spawn_units(env, agent):
    unit_type_id = _MAP["unit_type_id"]
    debug_commands = []
    for pos in agent._initial_units_my:
        debug_commands.append(
            sc_debug.DebugCommand(
                create_unit=sc_debug.DebugCreateUnit(
                    unit_type=unit_type_id,
                    owner=1,
                    pos=sc_common.Point2D(x=pos[0], y=pos[1]),
                    quantity=1,
                )
            )
        )
    for pos in agent._initial_units_enemy:
        debug_commands.append(
            sc_debug.DebugCommand(
                create_unit=sc_debug.DebugCreateUnit(
                    unit_type=unit_type_id,
                    owner=2,
                    pos=sc_common.Point2D(x=pos[0], y=pos[1]),
                    quantity=1,
                )
            )
        )
    env._controllers[0].debug(debug_commands)


def _move_sc2_window(x=50, y=50, w=640, h=480, timeout=10):
    import ctypes
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        hwnd = ctypes.windll.user32.FindWindowW(None, "StarCraft II")
        if hwnd:
            ctypes.windll.user32.SetWindowPos(hwnd, 0, x, y, w, h, 0x0040)
            return
        time.sleep(0.3)


def run_loop_custom(
    agents,
    env,
    reset_frames=0,
    max_frames=0,
    max_episodes=0,
    bridge: Optional[GameBridge] = None,
):
    """A run loop to have agents and an environment interact.

    When *bridge* is provided, supports:
      - pause/resume via bridge control signals
      - graceful stop via bridge stop signal
    """
    total_frames = 0
    env.total_episodes = 0
    start_time = time.time()
    global_test_flag = False

    env.f_start = True

    observation_spec = env.observation_spec()
    action_spec = env.action_spec()
    for agent, obs_spec, act_spec in zip(agents, observation_spec, action_spec):
        agent.setup(obs_spec, act_spec)

    if bridge:
        bridge.update_status(running=True, paused=False, mode="playing")
        bridge.put_event({"level": "success", "source": "game", "message": "游戏启动"})

    try:
        while not max_episodes or env.total_episodes < max_episodes:
            if bridge and bridge.should_stop():
                bridge.update_status(running=False, mode="stopped")
                bridge.put_event(
                    {"level": "info", "source": "game", "message": "游戏停止"}
                )
                break

            if bridge:
                ctrl = bridge.check_control()
                if ctrl == "step":
                    bridge.send_control("pause")
                    bridge.put_event(
                        {"level": "info", "source": "game", "message": "步进 1 帧"}
                    )
                elif ctrl == "pause":
                    bridge.put_event(
                        {"level": "info", "source": "game", "message": "游戏暂停"}
                    )
                    _resume = bridge.wait_until_resumed()
                    if _resume == "step":
                        bridge.send_control("pause")
                        bridge.put_event(
                            {"level": "info", "source": "game", "message": "步进 1 帧"}
                        )
                    else:
                        bridge.put_event(
                            {"level": "info", "source": "game", "message": "游戏恢复"}
                        )
                        continue
                elif ctrl == "stop":
                    bridge.update_status(running=False, mode="stopped")
                    bridge.put_event(
                        {"level": "info", "source": "game", "message": "游戏停止"}
                    )
                    break

            env.total_episodes += 1
            env.f_result = None
            timesteps = env.reset()
            for a in agents:
                a.reset()
            while True:
                if bridge and bridge.should_stop():
                    bridge.update_status(running=False, mode="stopped")
                    bridge.put_event(
                        {"level": "info", "source": "game", "message": "游戏停止"}
                    )
                    return

                if bridge:
                    ctrl = bridge.check_control()
                    if ctrl == "step":
                        bridge.send_control("pause")
                        bridge.put_event(
                            {"level": "info", "source": "game", "message": "步进 1 帧"}
                        )
                    elif ctrl == "pause":
                        bridge.put_event(
                            {"level": "info", "source": "game", "message": "游戏暂停"}
                        )
                        _resume = bridge.wait_until_resumed()
                        if _resume == "step":
                            bridge.send_control("pause")
                            bridge.put_event(
                                {
                                    "level": "info",
                                    "source": "game",
                                    "message": "步进 1 帧",
                                }
                            )
                        else:
                            bridge.put_event(
                                {
                                    "level": "info",
                                    "source": "game",
                                    "message": "游戏恢复",
                                }
                            )
                            continue
                    elif ctrl == "stop":
                        bridge.update_status(running=False, mode="stopped")
                        bridge.put_event(
                            {"level": "info", "source": "game", "message": "游戏停止"}
                        )
                        return

                total_frames += 1

                timesteps[0].set_test_flag(global_test_flag)

                agent_actions = [
                    agent.step(timestep, env)
                    for agent, timestep in zip(agents, timesteps)
                ]

                if env.f_result == "win" or env.f_result == "loss":
                    result_str = str(env.f_result)
                    obs_now = timesteps[0]
                    my_units = agents[0].get_my_units_by_type(
                        obs_now, _MAP["unit_type"]
                    )
                    enemy_units = agents[0].get_enemy_units_by_type(
                        obs_now, _MAP["unit_type"]
                    )
                    my_hp = sum(u["health"] for u in my_units)
                    enemy_hp = sum(u["health"] for u in enemy_units)
                    score_val = my_hp - enemy_hp
                    try:
                        if bridge:
                            if bridge.check_run_episode():
                                bridge.put_event(
                                    {
                                        "level": "info",
                                        "source": "game",
                                        "message": f"单局结束: {result_str.upper()}, 已暂停",
                                    }
                                )
                                bridge.send_control("pause")
                                _resume = bridge.wait_until_resumed()
                                if _resume == "stop":
                                    bridge.update_status(running=False, mode="stopped")
                                    return
                                continue
                            agents[0].new_game()
                            agents[0].end_game_frames = (
                                _ENV_CONFIG["_MAX_STEP"] * _ENV_CONFIG["_STEP_MUL"]
                            )
                            agents[0].end_game_state = "Dogfall"
                            agents[0].end_game_flag = False
                            agents[0]._termination_signaled = False
                        else:
                            agents[0]._end_episode(timesteps[0])
                            agents[0]._termination_signaled = False
                        kill_all_units(env, timesteps[0].observation)
                        env.f_result = None
                        total_frames = 0
                        spawn_units(env, agents[0])
                        timesteps = env.step(agent_actions, 2)
                        env.total_episodes += 1
                        if bridge:
                            ep_num = (
                                getattr(
                                    agents[0].ctx, "episode_count", env.total_episodes
                                )
                                if hasattr(agents[0], "ctx") and agents[0].ctx
                                else env.total_episodes
                            )
                            bridge.put_event(
                                {
                                    "level": "success"
                                    if result_str == "win"
                                    else "error",
                                    "source": "game",
                                    "message": f"Episode #{ep_num} 判定 {result_str.upper()}, 得分: {score_val:+d} (我方{my_hp} vs 敌方{enemy_hp})",
                                }
                            )
                        if getattr(agents[0], "_done", False):
                            break
                        continue
                    except Exception as e:
                        if bridge:
                            bridge.put_event(
                                {
                                    "level": "error",
                                    "source": "game",
                                    "message": f"kill+spawn 异常: {e}",
                                }
                            )
                if timesteps[0].last():
                    global_test_flag = not global_test_flag
                    break
                if reset_frames > 0 and total_frames > reset_frames:
                    if bridge:
                        if bridge.check_run_episode():
                            bridge.put_event(
                                {
                                    "level": "info",
                                    "source": "game",
                                    "message": "单局结束: 帧数超时, 已暂停",
                                }
                            )
                            bridge.send_control("pause")
                            _resume = bridge.wait_until_resumed()
                            if _resume == "stop":
                                bridge.update_status(running=False, mode="stopped")
                                return
                            continue
                        agents[0].end_game_state = "Dogfall"
                        agents[0].new_game()
                        agents[0].end_game_frames = (
                            _ENV_CONFIG["_MAX_STEP"] * _ENV_CONFIG["_STEP_MUL"]
                        )
                        agents[0]._termination_signaled = False
                    else:
                        agents[0]._end_episode(timesteps[0])
                        agents[0]._termination_signaled = False
                    kill_all_units(env, timesteps[0].observation)
                    env.f_result = None
                    total_frames = 0
                    spawn_units(env, agents[0])
                    timesteps = env.step(agent_actions, 2)
                    env.total_episodes += 1
                    if bridge:
                        ep_num = (
                            getattr(agents[0].ctx, "episode_count", env.total_episodes)
                            if hasattr(agents[0], "ctx") and agents[0].ctx
                            else env.total_episodes
                        )
                        bridge.put_event(
                            {
                                "level": "warn",
                                "source": "game",
                                "message": f"Episode #{ep_num} 结束: Dogfall (超时, frame={total_frames})",
                            }
                        )
                    if getattr(agents[0], "_done", False):
                        break
                    continue
                if max_frames and total_frames >= max_frames:
                    if bridge:
                        bridge.update_status(running=False, mode="stopped")
                    return
                timesteps = env.step(agent_actions)
    except KeyboardInterrupt:
        pass
    finally:
        elapsed_time = time.time() - start_time
        if bridge:
            bridge.update_status(running=False, mode="stopped")
    print(
        "Took %.3f seconds for %s steps: %.3f fps"
        % (elapsed_time, total_frames, total_frames / max(elapsed_time, 1e-6))
    )


def save_run(run_name):
    src = _PATH_CONFIG["_DATA_TRANSIT_PATH"]
    dst = os.path.join(_PATH_CONFIG["_RUNS_PATH"], run_name)
    if os.path.exists(dst):
        dst = dst + f"_dup_{datetime.now().strftime('%H%M%S')}"
    shutil.copytree(src, dst)
    print(f"Run saved to: {os.path.abspath(dst)}")


def run_game(
    map_key,
    run_name,
    bridge: Optional[GameBridge] = None,
    agent_type: str = "smart",
    fallback_action: str = "action_ATK_nearest_weakest",
    window_loc: Optional[tuple] = None,
    data_dir: Optional[str] = None,
    autopilot_mode: str = "multi_step",
    beam_params: Optional[dict] = None,
    replay_actions: Optional[list] = None,
    replay_runs: int = 1,
    kg_file: Optional[str] = None,
    action_strategy: str = "best_beam",
    batch_replay_count: int = 3,
    batch_start: int = 0,
    batch_end: Optional[int] = None,
    batch_output_dir: Optional[str] = None,
    primary_threshold: float = 1.0,
    secondary_threshold: float = 0.5,
    max_episodes: Optional[int] = None,
    override_model_path: Optional[str] = None,
    cf_config: Optional[dict] = None,
    cf_runs: int = 1,
):
    _apply_map_config(map_key)
    steps = _ENV_CONFIG["_MAX_STEP"]
    step_mul = _ENV_CONFIG["_STEP_MUL"]

    agent1 = None
    if agent_type == "batch_replay":
        from src.sc2env.replay_collector import ReplayCollector

        _apply_map_config(map_key)

        action_log_path = ""
        if data_dir:
            action_log_path = os.path.join(data_dir, "action_log.csv")
        agent1 = ReplayCollector(
            bridge=bridge,
            action_log_path=action_log_path,
            replay_count=batch_replay_count,
            batch_start=batch_start,
            batch_end=batch_end,
            output_dir=batch_output_dir,
            primary_threshold=primary_threshold,
            secondary_threshold=secondary_threshold,
        )
    elif agent_type == "kg_guided" and bridge is not None:
        from src.sc2env.kg_guided_agent import KGGuidedAgent

        _apply_map_config(map_key)

        bktree_data = None
        data_bktree_dir = None
        if data_dir:
            data_bktree_dir = Path(data_dir) / "bktree"
            if not data_bktree_dir.exists():
                from src import ROOT_DIR

                data_bktree_dir = ROOT_DIR / data_dir / "bktree"

        primary_bktree_path = ""
        secondary_prefix = ""
        if data_bktree_dir and data_bktree_dir.exists():
            data_primary = data_bktree_dir / "primary_bktree.json"
            if data_primary.exists():
                primary_bktree_path = str(data_primary)
                secondary_prefix = str(data_bktree_dir / "secondary_bktree")

        if not primary_bktree_path:
            primary_bktree_path = _PATH_CONFIG.get("_GAME_PRIMARY_BKTREE_PATH", "")
            aug_primary_path = _PATH_CONFIG.get("_GAME_AUGMENTED_BKTREE_PATH", "")
            if not (primary_bktree_path and os.path.exists(primary_bktree_path)):
                if aug_primary_path and os.path.exists(aug_primary_path):
                    primary_bktree_path = aug_primary_path
                    secondary_prefix = _PATH_CONFIG.get(
                        "_GAME_AUGMENTED_SECONDARY_PREFIX", ""
                    )
        if primary_bktree_path and os.path.exists(primary_bktree_path):
            try:
                import json

                bktree_data = {"primary": None, "secondary": {}}
                with open(primary_bktree_path, "r") as f:
                    bktree_data["primary"] = json.load(f)
                prefix = secondary_prefix or _PATH_CONFIG.get(
                    "_GAME_SECONDARY_BKTREE_PREFIX", ""
                )
                aug_prefix = _PATH_CONFIG.get("_GAME_AUGMENTED_SECONDARY_PREFIX", "")
                if not (prefix and os.path.exists(f"{prefix}_1.json")):
                    prefix = aug_prefix
                if prefix:
                    import glob as _glob

                    for sec_file in _glob.glob(f"{prefix}_*.json"):
                        cid_str = sec_file.rsplit("_", 1)[-1].replace(".json", "")
                        try:
                            with open(sec_file, "r") as sf:
                                bktree_data["secondary"][cid_str] = json.load(sf)
                        except Exception:
                            pass
                print(
                    f"[run_game] Loaded BKTree from {primary_bktree_path} "
                    f"(secondary_prefix={prefix or '-'})"
                )
            except Exception as e:
                print(f"Warning: Failed to load BKTree data: {e}")

        state_id_map = {}
        if data_dir:
            from src import ROOT_DIR
            import ast

            _sn_path = Path(data_dir) / "graph" / "state_node.txt"
            if not _sn_path.exists():
                _sn_path = ROOT_DIR / data_dir / "graph" / "state_node.txt"
        else:
            _sn_path = None
        if _sn_path is None or not _sn_path.exists():
            _aug_sn = _PATH_CONFIG.get("_GAME_AUGMENTED_STATE_NODE_PATH", "")
            if _aug_sn and os.path.exists(_aug_sn):
                _sn_path = Path(_aug_sn)
        if _sn_path and _sn_path.exists():
            try:
                for line in _sn_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        try:
                            ps = ast.literal_eval(parts[0])
                            state_id_map[(int(ps[0]), int(ps[1]))] = int(parts[1])
                        except Exception:
                            pass
                print(
                    f"[run_game] Loaded state_id_map: {len(state_id_map)} entries from {_sn_path}"
                )
            except Exception as e:
                print(f"[run_game] Warning: Failed to load state_node.txt: {e}")
        else:
            _sn_display = _sn_path if _sn_path else "augmented path (not found)"
            print(f"[run_game] Warning: state_node.txt not found ({_sn_display})")

        _kg = None
        _transitions = None
        _dist_matrix = None
        if data_dir:
            from src import ROOT_DIR as _ROOT
            import pickle as _pickle

            _kg_dir = _ROOT / "cache" / "knowledge_graph"
            _kg_file = None
            if kg_file:
                _kg_file = str(_kg_dir / kg_file)
                if not os.path.exists(_kg_file):
                    print(
                        f"[run_game] Warning: Specified KG file not found: {_kg_file}"
                    )
                    _kg_file = None
            if _kg_file is None and _kg_dir.exists():
                for _pkl in sorted(
                    (p for p in _kg_dir.rglob("*.pkl") if "_transitions" not in p.name),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                ):
                    _kg_file = str(_pkl)
                    print(f"[run_game] Auto-discovered KG: {_kg_file}")
                    break
            if _kg_file:
                try:
                    from src.decision.knowledge_graph import DecisionKnowledgeGraph

                    _kg = DecisionKnowledgeGraph.load(_kg_file)
                    print(f"Loaded KG from {_kg_file}")
                except Exception as e:
                    print(f"Warning: Failed to load KG: {e}")
            _trans_path = (
                _kg_file.replace(".pkl", "_transitions.pkl") if _kg_file else ""
            )
            if _trans_path and os.path.exists(_trans_path):
                try:
                    with open(_trans_path, "rb") as f:
                        _transitions = _pickle.load(f)
                    print(f"Loaded transitions from {_trans_path}")
                except Exception as e:
                    print(f"Warning: Failed to load transitions: {e}")
            _dp = Path(data_dir)
            if len(_dp.parts) >= 2:
                _map_id, _data_id = _dp.parts[-2], _dp.parts[-1]
                _npy_dir = _ROOT / "cache" / "npy"
                _dm_path = _npy_dir / f"state_distance_matrix_{_map_id}_{_data_id}.npy"
                if _dm_path.exists():
                    try:
                        import numpy as _np

                        _dist_matrix = _np.load(str(_dm_path), mmap_mode="r")
                        print(
                            f"Loaded dist matrix from {_dm_path} ({_dist_matrix.shape}, mmap)"
                        )
                    except Exception as e:
                        print(f"Warning: Failed to load dist matrix: {e}")
                else:
                    print(f"Warning: Distance matrix not found for data_dir={data_dir}: {_dm_path}")
                    _sparse_candidates = [
                        _npy_dir
                        / f"state_sparse_neighbors_{_map_id}_{_data_id}.pkl",
                    ]
                    if _kg_file:
                        _sparse_candidates.extend(
                            [
                                Path(_kg_file).parent / "sparse_neighbors.pkl",
                                Path(_kg_file).parent / "npy" / "sparse_neighbors.pkl",
                            ]
                        )
                    for _sp_path in _sparse_candidates:
                        if not _sp_path.exists():
                            continue
                        try:
                            from src.decision.sparse_distance_index import (
                                load_sparse_distance_index,
                            )

                            _dist_matrix = load_sparse_distance_index(str(_sp_path))
                            print(
                                f"Loaded sparse distance index from {_sp_path} "
                                f"({len(_dist_matrix.neighbors)} states, top_k={_dist_matrix.top_k})"
                            )
                            break
                        except Exception as e:
                            print(f"Warning: Failed to load sparse distance index: {e}")

        agent1 = KGGuidedAgent(
            bridge=bridge,
            fallback_action=fallback_action,
            initial_bktree_data=bktree_data,
            state_id_map=state_id_map,
            kg=_kg,
            transitions=_transitions,
            dist_matrix=_dist_matrix,
            mode=autopilot_mode,
            beam_params=beam_params or {},
            replay_actions=replay_actions,
            replay_runs=replay_runs,
            action_strategy=action_strategy,
            data_dir=data_dir,
            kg_file=kg_file,
            override_model_path=override_model_path,
            cf_config=cf_config,
            bktree_primary_threshold=primary_threshold,
            bktree_secondary_threshold=secondary_threshold,
        )
    else:
        agent1 = SmartAgent()

    try:
        with sc2_env.SC2Env(
            map_name=_MAP["map_name"],
            players=[
                sc2_env.Agent(sc2_env.Race.terran),
                sc2_env.Bot(sc2_env.Race.terran, sc2_env.Difficulty.very_hard),
            ],
            agent_interface_format=features.AgentInterfaceFormat(
                action_space=actions.ActionSpace.RAW,
                use_raw_units=True,
                raw_resolution=_ENV_CONFIG["_MAP_RESOLUTION"],
            ),
            score_index=-1,
            disable_fog=True,
            step_mul=step_mul,
            game_steps_per_episode=steps * step_mul,
            # realtime=True
        ) as env:
            ctx = GameContext()
            init_game(ctx, _PATH_CONFIG)
            if window_loc:
                import threading

                threading.Thread(
                    target=_move_sc2_window, args=window_loc, daemon=True
                ).start()
            agent2 = Agent()
            agent1.ctx = ctx
            run_loop_custom(
                [agent1, agent2],
                env,
                reset_frames=_ENV_CONFIG["_RESET_FRAMES"],
                max_episodes=max_episodes
                if max_episodes is not None
                else _ENV_CONFIG["_MAX_EPISODE"],
                bridge=bridge,
            )
            if bridge is None:
                save_run(run_name)
    except KeyboardInterrupt:
        if bridge is None:
            save_run(run_name)
        pass


def main(unused_argv):
    map_key = FLAGS.map_key
    run_name = FLAGS.run_name
    if not run_name:
        run_name = f"{map_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"Run name: {run_name}")
    run_game(map_key, run_name)


if __name__ == "__main__":
    app.run(main)
