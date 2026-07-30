#!/usr/bin/env python
"""
Live Game Launcher — 启动实时对局系统

Usage:
    # 一键启动 (游戏进程 + API 服务)
    python scripts/run_live_game.py --map_key sce-1 --etg_file xxx.pkl

    # 分别启动
    python scripts/run_live_game.py --mode game --map_key sce-1
    python scripts/run_live_game.py --mode api --port 8000

    # 查看帮助
    python scripts/run_live_game.py --help
"""

import sys
import argparse
import multiprocessing as mp
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import ROOT_DIR


def _ensure_utf8_stdio():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _run_game_process(bridge, args):
    _ensure_utf8_stdio()
    from absl import flags as absl_flags

    if not absl_flags.FLAGS.is_parsed():
        absl_flags.FLAGS(["run_live_game.py"])

    from src.sc2env.run_game import run_game

    window_loc = None
    if args.window_x is not None and args.window_y is not None:
        window_loc = (args.window_x, args.window_y, args.window_w, args.window_h)

    beam_params = {}
    if args.beam_width is not None:
        beam_params["beam_width"] = args.beam_width
    if args.lookahead_steps is not None:
        beam_params["lookahead_steps"] = args.lookahead_steps
    if args.score_mode is not None:
        beam_params["score_mode"] = args.score_mode
    if args.min_visits is not None:
        beam_params["min_visits"] = args.min_visits
    if args.max_state_revisits is not None:
        beam_params["max_state_revisits"] = args.max_state_revisits
    if args.min_cum_prob is not None:
        beam_params["min_cum_prob"] = args.min_cum_prob
    if args.discount_factor is not None:
        beam_params["discount_factor"] = args.discount_factor
    if args.epsilon is not None:
        beam_params["epsilon"] = args.epsilon
    if args.enable_backup:
        beam_params["enable_backup"] = True
    if args.backup_score_threshold is not None:
        beam_params["backup_score_threshold"] = args.backup_score_threshold
    if args.backup_distance_threshold is not None:
        beam_params["backup_distance_threshold"] = args.backup_distance_threshold
    if args.primary_threshold is not None:
        beam_params["bktree_primary_threshold"] = args.primary_threshold
    if args.secondary_threshold is not None:
        beam_params["bktree_secondary_threshold"] = args.secondary_threshold
    if args.enable_action_tuning:
        beam_params["enable_action_tuning"] = True
    if args.action_tuning_model_path:
        beam_params["action_tuning_model_path"] = args.action_tuning_model_path
    if args.tuning_explore_rate is not None:
        beam_params["tuning_explore_rate"] = args.tuning_explore_rate
    if args.tuning_min_confidence is not None:
        beam_params["tuning_min_confidence"] = args.tuning_min_confidence
    if args.tuning_min_advantage is not None:
        beam_params["tuning_min_advantage"] = args.tuning_min_advantage
    if args.tuning_ucb_c is not None:
        beam_params["tuning_ucb_c"] = args.tuning_ucb_c
    if args.tuning_target_visits is not None:
        beam_params["tuning_target_visits"] = args.tuning_target_visits
    if args.tuning_min_visits is not None:
        beam_params["tuning_min_visits"] = args.tuning_min_visits
    if args.tuning_credit_mode is not None:
        beam_params["tuning_credit_mode"] = args.tuning_credit_mode
    if args.tuning_discount_factor is not None:
        beam_params["tuning_discount_factor"] = args.tuning_discount_factor
    if args.tuning_outcome_bonus is not None:
        beam_params["tuning_outcome_bonus"] = args.tuning_outcome_bonus
    if args.tuning_confidence_return_scale is not None:
        beam_params["tuning_confidence_return_scale"] = args.tuning_confidence_return_scale
    if args.tuning_ood_key_mode is not None:
        beam_params["tuning_ood_key_mode"] = args.tuning_ood_key_mode
    if args.tuning_ood_distance_bucket is not None:
        beam_params["tuning_ood_distance_bucket"] = args.tuning_ood_distance_bucket
    if args.max_nid_fallback_dist is not None:
        beam_params["max_nid_fallback_dist"] = args.max_nid_fallback_dist
    if args.max_nid_fallback_hp_dist is not None:
        beam_params["max_nid_fallback_hp_dist"] = args.max_nid_fallback_hp_dist
    if args.local_result_dir:
        beam_params["local_result_dir"] = args.local_result_dir
    if args.target_episodes is not None:
        beam_params["target_episodes"] = args.target_episodes
    if args.trial_number is not None:
        beam_params["trial_number"] = args.trial_number
    if args.plan_log_path:
        beam_params["plan_log_path"] = args.plan_log_path
    if args.replay_exhaustion_mode:
        beam_params["replay_exhaustion_mode"] = args.replay_exhaustion_mode
    if args.tuning_force_explore:
        beam_params["tuning_force_explore"] = True
    if args.tuning_explore_ood is not None:
        beam_params["tuning_explore_ood"] = bool(args.tuning_explore_ood)
    if args.enable_mechanism_shadow_logging:
        beam_params["enable_mechanism_shadow_logging"] = True
    if args.eval_bktree_normalization:
        beam_params["eval_bktree_normalization"] = args.eval_bktree_normalization
    restart_guard = {
        "enabled": args.restart_guard_enabled,
        "warmup_episodes": args.restart_warmup_episodes,
        "max_ood_ratio": args.restart_guard_max_ood_ratio,
        "max_ood_mc_ratio": args.restart_guard_max_ood_mc_ratio,
        "max_episode_frames": args.restart_guard_max_episode_frames,
        "allow_high_score_ood_update": args.restart_guard_allow_high_score_ood_update,
        "high_score_ood_min_score": args.restart_guard_high_score_ood_min_score,
        "skip_model_update": args.restart_guard_skip_model_update,
        "skip_bad_results": args.restart_guard_skip_bad_results,
        "disable_ood_explore_on_violation": args.restart_guard_disable_ood_explore,
    }
    if restart_guard.get("enabled"):
        beam_params["restart_guard"] = restart_guard
        beam_params["restart_warmup_episodes"] = restart_guard["warmup_episodes"]
    incremental_layer = {
        "enabled": args.enable_incremental_layer,
        "update_bktree": args.incremental_update_bktree,
        "update_etg_delta": args.incremental_update_etg_delta,
        "use_delta_for_planning": args.incremental_use_delta_for_planning,
        "persist_interval_episodes": args.incremental_persist_interval,
        "min_new_state_distance": args.incremental_min_new_state_distance,
        "delta_dir": args.incremental_delta_dir,
    }
    if args.incremental_layer_json:
        try:
            incremental_layer.update(json.loads(args.incremental_layer_json))
        except Exception:
            pass
    if incremental_layer.get("enabled"):
        beam_params["incremental_layer"] = incremental_layer

    if args.beam_params_file:
        try:
            with open(str(args.beam_params_file), "r", encoding="utf-8") as f:
                loaded_params = json.load(f)
            if isinstance(loaded_params, dict):
                beam_params.update(loaded_params)
        except Exception as exc:
            print(f"[run_live_game] failed to load --beam_params_file: {exc}", flush=True)
    if args.beam_params_json:
        try:
            loaded_params = json.loads(args.beam_params_json)
            if isinstance(loaded_params, dict):
                beam_params.update(loaded_params)
        except Exception as exc:
            print(f"[run_live_game] failed to parse --beam_params_json: {exc}", flush=True)

    autopilot_mode = str(beam_params.get("mode") or args.autopilot_mode)
    action_strategy = str(beam_params.get("action_strategy") or args.action_strategy)

    agent_type = (
        "batch_replay" if autopilot_mode == "batch_replay" else "etg_guided"
    )

    cf_config = None
    if args.cf_actions and args.cf_diverge_step is not None:
        cf_config = {
            "original_actions": args.cf_actions.split(","),
            "diverge_step": args.cf_diverge_step,
            "replacement_action": args.cf_replacement or "",
            "cf_run_id": args.cf_run_id or "",
            "cf_runs": args.cf_runs,
        }

    run_game(
        map_key=args.map_key,
        run_name=args.run_name,
        bridge=bridge,
        agent_type=agent_type,
        fallback_action=args.fallback_action,
        window_loc=window_loc,
        data_dir=args.data_dir,
        autopilot_mode=autopilot_mode,
        beam_params=beam_params,
        replay_actions=args.replay_actions.split(",") if args.replay_actions else None,
        replay_runs=args.replay_runs,
        etg_file=args.etg_file,
        action_strategy=action_strategy,
        batch_replay_count=args.replay_count,
        batch_start=args.batch_start,
        batch_end=args.batch_end,
        batch_output_dir=args.batch_output_dir,
        primary_threshold=args.primary_threshold,
        secondary_threshold=args.secondary_threshold,
        max_episodes=args.max_episodes,
        override_model_path=args.override_model_path,
        cf_config=cf_config,
        cf_runs=args.cf_runs,
        load_etg=not args.skip_game_etg,
    )


def _run_api_process(bridge, args):
    _ensure_utf8_stdio()
    from src.sc2env.bridge_server import run_server

    etg_file = None if args.skip_api_etg else args.etg_file
    data_dir = None if args.skip_api_etg else args.data_dir
    run_server(
        bridge,
        host=args.host,
        port=args.port,
        etg_file=etg_file,
        data_dir=data_dir,
    )


def main():
    _ensure_utf8_stdio()
    parser = argparse.ArgumentParser(description="PredictionRTS Live Game System")
    parser.add_argument(
        "--mode",
        choices=["all", "game", "api"],
        default="all",
        help="all=game+api, game=SC2 only, api=server only",
    )
    parser.add_argument("--map_key", default="sce-1", help="Map config key")
    parser.add_argument(
        "--run_name", default=None, help="Run name (auto-generated if None)"
    )
    parser.add_argument(
        "--etg_file", default=None, help="ETG pickle filename in cache/experience_transition_graph/"
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        help="Training data directory (e.g. data/MarineMicro_MvsM_4/6), contains graph/state_node.txt",
    )
    parser.add_argument(
        "--fallback_action",
        default="action_ATK_nearest_weakest",
        help="Default fallback action when no ETG recommendation",
    )
    parser.add_argument("--host", default="0.0.0.0", help="API server host")
    parser.add_argument("--port", type=int, default=8000, help="API server port")
    parser.add_argument(
        "--skip_api_etg",
        action="store_true",
        help="Do not preload ETG in the API process; the game agent still loads it.",
    )
    parser.add_argument(
        "--skip_game_etg",
        action="store_true",
        help="Do not load etg/transitions/distance matrix in the game agent.",
    )
    parser.add_argument(
        "--window_x", type=int, default=None, help="SC2 window X position"
    )
    parser.add_argument(
        "--window_y", type=int, default=None, help="SC2 window Y position"
    )
    parser.add_argument("--window_w", type=int, default=640, help="SC2 window width")
    parser.add_argument("--window_h", type=int, default=480, help="SC2 window height")
    parser.add_argument(
        "--autopilot_mode",
        default="multi_step",
        choices=["single_step", "multi_step", "replay", "batch_replay"],
        help="Autopilot mode",
    )
    parser.add_argument("--beam_width", type=int, default=None, help="Beam width")
    parser.add_argument(
        "--lookahead_steps", type=int, default=None, help="Lookahead steps"
    )
    parser.add_argument("--score_mode", default=None, help="Score mode")
    parser.add_argument("--min_visits", type=int, default=None, help="Min visits")
    parser.add_argument(
        "--max_state_revisits", type=int, default=None, help="Max state revisits"
    )
    parser.add_argument(
        "--min_cum_prob", type=float, default=None, help="Min cumulative probability"
    )
    parser.add_argument(
        "--discount_factor", type=float, default=None, help="Discount factor"
    )
    parser.add_argument(
        "--action_strategy",
        default="best_beam",
        choices=[
            "best_beam",
            "best_subtree_quality",
            "best_subtree_winrate",
            "highest_transition_prob",
            "random_beam",
            "epsilon_greedy",
        ],
        help="Action selection strategy",
    )
    parser.add_argument(
        "--epsilon", type=float, default=None, help="Epsilon for epsilon_greedy"
    )
    parser.add_argument(
        "--enable_backup", action="store_true", help="Enable backup path switching"
    )
    parser.add_argument(
        "--backup_score_threshold",
        type=float,
        default=None,
        help="Backup score threshold",
    )
    parser.add_argument(
        "--backup_distance_threshold",
        type=float,
        default=None,
        help="Backup distance threshold",
    )
    parser.add_argument(
        "--replay_actions", default=None, help="Comma-separated replay action codes"
    )
    parser.add_argument(
        "--replay_runs", type=int, default=1, help="Number of replay runs"
    )
    parser.add_argument(
        "--replay_exhaustion_mode",
        choices=["pause", "end_episode", "fallback", "last_action", "no_op"],
        default=None,
        help="Replay behavior after the recorded action sequence is exhausted",
    )
    parser.add_argument(
        "--local_result_dir",
        default=None,
        help="Directory for local episodes.jsonl/progress.json output",
    )
    parser.add_argument(
        "--target_episodes",
        type=int,
        default=None,
        help="Target episode count recorded in progress.json",
    )
    parser.add_argument(
        "--trial_number",
        type=int,
        default=None,
        help="Trial/repeat identifier written to local result records",
    )
    parser.add_argument(
        "--plan_log_path",
        default=None,
        help="Optional planning/action log path for local evaluation",
    )
    parser.add_argument(
        "--replay_count",
        type=int,
        default=3,
        help="Repeats per sequence (batch replay)",
    )
    parser.add_argument(
        "--batch_start", type=int, default=0, help="Batch replay start index"
    )
    parser.add_argument(
        "--batch_end", type=int, default=None, help="Batch replay end index"
    )
    parser.add_argument(
        "--batch_output_dir",
        default=None,
        help="Base output directory for batch_replay ReplayCollector artifacts",
    )
    parser.add_argument(
        "--primary_threshold",
        type=float,
        default=1.0,
        help="BKTree primary cluster threshold (default: 1.0)",
    )
    parser.add_argument(
        "--secondary_threshold",
        type=float,
        default=0.5,
        help="BKTree secondary cluster threshold (default: 0.5)",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Max episodes per run (overrides config default)",
    )
    parser.add_argument(
        "--override_model_path", default=None, help="Path to ActionOverrideModel pickle"
    )
    parser.add_argument(
        "--cf_actions",
        default=None,
        help="Comma-separated actions for counterfactual replay phase",
    )
    parser.add_argument(
        "--cf_diverge_step",
        type=int,
        default=None,
        help="Step index to inject replacement action in counterfactual mode",
    )
    parser.add_argument(
        "--cf_replacement",
        default=None,
        help="Replacement action code at divergence step",
    )
    parser.add_argument(
        "--cf_run_id", default=None, help="Counterfactual run identifier"
    )
    parser.add_argument(
        "--cf_runs",
        type=int,
        default=1,
        help="Number of counterfactual episodes per divergence point",
    )
    parser.add_argument(
        "--enable_action_tuning",
        action="store_true",
        help="Enable Monte Carlo action tuning router",
    )
    parser.add_argument(
        "--action_tuning_model_path",
        default=None,
        help="Path to ActionTuningModel pickle",
    )
    parser.add_argument("--tuning_explore_rate", type=float, default=None)
    parser.add_argument("--tuning_min_confidence", type=float, default=None)
    parser.add_argument("--tuning_min_advantage", type=float, default=None)
    parser.add_argument("--tuning_ucb_c", type=float, default=None)
    parser.add_argument("--tuning_target_visits", type=int, default=None)
    parser.add_argument("--tuning_min_visits", type=int, default=None)
    parser.add_argument(
        "--tuning_credit_mode",
        choices=["every_visit", "first_visit"],
        default=None,
    )
    parser.add_argument("--tuning_discount_factor", type=float, default=None)
    parser.add_argument("--tuning_outcome_bonus", type=float, default=None)
    parser.add_argument("--tuning_confidence_return_scale", type=float, default=None)
    parser.add_argument("--tuning_ood_key_mode", choices=["aggregate", "exact"], default=None)
    parser.add_argument("--tuning_ood_distance_bucket", type=float, default=None)
    parser.add_argument("--tuning_force_explore", action="store_true")
    parser.add_argument("--tuning_explore_ood", type=int, choices=[0, 1], default=None)
    parser.add_argument(
        "--enable_mechanism_shadow_logging",
        action="store_true",
        help="Record per-frame no-mechanism shadow decisions for final-eval mechanism diagnostics",
    )
    parser.add_argument(
        "--eval_bktree_normalization",
        choices=["pymarl_compatible", "pymarl", "onpolicy", "decision", "live", "predictionrts"],
        default=None,
        help="Normalization used only for final-eval BKTree/state-id recording",
    )
    parser.add_argument("--max_nid_fallback_dist", type=float, default=None)
    parser.add_argument("--max_nid_fallback_hp_dist", type=float, default=None)
    parser.add_argument("--restart_guard_enabled", action="store_true")
    parser.add_argument("--restart_warmup_episodes", type=int, default=10)
    parser.add_argument("--restart_guard_max_ood_ratio", type=float, default=0.30)
    parser.add_argument("--restart_guard_max_ood_mc_ratio", type=float, default=0.30)
    parser.add_argument("--restart_guard_max_episode_frames", type=int, default=80)
    parser.add_argument("--restart_guard_allow_high_score_ood_update", action="store_true")
    parser.add_argument("--restart_guard_high_score_ood_min_score", type=float, default=24.0)
    parser.add_argument("--restart_guard_skip_model_update", action="store_true")
    parser.add_argument("--restart_guard_skip_bad_results", action="store_true")
    parser.add_argument("--restart_guard_disable_ood_explore", action="store_true")
    parser.add_argument("--enable_incremental_layer", action="store_true")
    parser.add_argument("--incremental_update_bktree", action="store_true")
    parser.add_argument("--incremental_update_etg_delta", action="store_true")
    parser.add_argument("--incremental_use_delta_for_planning", action="store_true")
    parser.add_argument("--incremental_delta_dir", default="output/incremental_layer")
    parser.add_argument("--incremental_persist_interval", type=int, default=10)
    parser.add_argument("--incremental_min_new_state_distance", type=float, default=1.0)
    parser.add_argument("--incremental_layer_json", default=None)
    parser.add_argument(
        "--beam_params_file",
        default=None,
        help="JSON file merged into initial ETGGuidedAgent beam params before game startup",
    )
    parser.add_argument(
        "--beam_params_json",
        default=None,
        help="Inline JSON merged into initial ETGGuidedAgent beam params before game startup",
    )
    args = parser.parse_args()

    if args.mode in ("game", "all") and args.etg_file is None:
        print(
            "Warning: --etg_file not specified. ETG predictions will not work until loaded via API."
        )
        print("         You can load it later via: POST /game/load_etg?etg_file=xxx.pkl")

    if args.mode == "all":
        from src.sc2env.bridge import GameBridge

        bridge = GameBridge()

        game_proc = mp.Process(
            target=_run_game_process,
            args=(bridge, args),
            name="sc2_game",
            daemon=True,
        )
        api_proc = mp.Process(
            target=_run_api_process,
            args=(bridge, args),
            name="bridge_api",
            daemon=True,
        )

        game_proc.start()
        api_proc.start()

        print(f"Game process PID: {game_proc.pid}")
        print(f"API server PID: {api_proc.pid}")
        print(f"API endpoint: http://{args.host}:{args.port}")
        print("Press Ctrl+C to stop all processes.")

        try:
            game_proc.join()
        except KeyboardInterrupt:
            print("\nStopping...")
            bridge.request_stop()
            game_proc.join(timeout=5)
            api_proc.join(timeout=5)
            print("All processes stopped.")
        finally:
            bridge.request_stop()
            for proc in (game_proc, api_proc):
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=5)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=2)

    elif args.mode == "game":
        from src.sc2env.bridge import GameBridge

        bridge = GameBridge()
        _run_game_process(bridge, args)

    elif args.mode == "api":
        from src.sc2env.bridge import GameBridge

        bridge = GameBridge()
        print("API mode: no game process. Connect via existing bridge or standalone.")
        print(f"API endpoint: http://{args.host}:{args.port}")
        _run_api_process(bridge, args)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
