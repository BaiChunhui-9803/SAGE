import sys
import socket
import subprocess
import time
import requests
from etg_web.i18n import st
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import ROOT_DIR
from etg_web.constants import (
    BRIDGE_API_URL,
    _ACTION_STRATEGY_LABELS,
    get_bktree_threshold_defaults,
    get_map_key_for_map_id,
)
from etg_web.loaders import load_episode_data
from etg_web.live_game_html import _build_live_game_html


def _render_live_game_sidebar(etg_entry: Optional[Dict] = None):
    etg_file = etg_entry.get("file", "") if etg_entry else ""
    etg_name = etg_entry.get("name", "") if etg_entry else ""
    etg_data_dir = etg_entry.get("data_dir", "") if etg_entry else ""
    if etg_file:
        st.caption(f"ETG: {etg_name}")
        st.caption(f"路径: cache/experience_transition_graph/{etg_file}")
        if etg_data_dir:
            st.caption(f"路径: {etg_data_dir}/bktree/")
            import json, glob as _glob

            _bkt_dir = ROOT_DIR / etg_data_dir / "bktree"
            _pri = _bkt_dir / "primary_bktree.json"
            _pri_cnt = 0
            if _pri.exists():
                try:
                    _d = json.load(open(str(_pri), "r"))
                    _stk = [_d]
                    while _stk:
                        _n = _stk.pop()
                        _pri_cnt += 1
                        _stk.extend(_n.get("children", {}).values())
                except Exception:
                    pass
            _sec_files = sorted(_glob.glob(str(_bkt_dir / "secondary_bktree_*.json")))
            _sec_cnt = len(_sec_files)
            _sn_path = ROOT_DIR / etg_data_dir / "graph" / "state_node.txt"
            _sn_cnt = 0
            if _sn_path.exists():
                try:
                    _sn_cnt = sum(1 for _ in open(str(_sn_path), "r") if _.strip())
                except Exception:
                    pass
            with st.expander("BKTree 详情", expanded=False):
                st.caption(f"Primary 节点: {_pri_cnt}")
                st.caption(f"Secondary 树: {_sec_cnt}")
                st.caption(f"State 映射: {_sn_cnt} 条")
                st.caption(
                    f"map_id: {etg_entry.get('map_id', '-')} | data_id: {etg_entry.get('data_id', '-')}"
                )

    col_port, col_btn = st.columns([3, 2])
    with col_port:
        port = st.number_input(
            "API 端口",
            min_value=1024,
            max_value=65535,
            value=8000,
            key="live_port",
        )
    _start_clicked = False
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(
            "一键启动", type="primary", use_container_width=True, key="live_start"
        ):
            _start_clicked = True

    st.caption("窗口位置")
    wc1, wc2, wc3, wc4 = st.columns(4)
    with wc1:
        st.number_input("X", value=2600, key="live_wx", label_visibility="collapsed")
    with wc2:
        st.number_input("Y", value=50, key="live_wy", label_visibility="collapsed")
    with wc3:
        st.number_input("W", value=640, key="live_ww", label_visibility="collapsed")
    with wc4:
        st.number_input("H", value=480, key="live_wh", label_visibility="collapsed")

    ap_port = int(st.session_state.get("live_port", 8000))
    api_base = f"http://localhost:{ap_port}"

    st.divider()
    start_paused = st.toggle(
        "启动后暂停",
        value=False,
        key="live_start_paused",
        help="开启后服务启动时自动进入暂停状态，需手动恢复",
    )

    map_id = etg_entry.get("map_id", "") if etg_entry else ""
    map_key = get_map_key_for_map_id(map_id)
    st.caption(f"游戏场景: {map_key} ({map_id or '-'})")
    data_id = etg_entry.get("data_id", "") if etg_entry else ""
    ep_data = None
    if map_id and data_id:
        ep_data = load_episode_data(map_id, data_id)
    has_replay = ep_data is not None and ep_data["n_episodes"] > 0

    mode_options = ["单步推演", "多步推演"]
    if has_replay:
        mode_options.append("回放重演")
        mode_options.append("重采样数据集扩张（批量回放重演）")
    st.markdown("**推演模式**")
    live_mode = st.radio(
        "mode",
        options=mode_options,
        index=0,
        horizontal=True,
        key="live_mode",
        label_visibility="collapsed",
    )

    if live_mode == "回放重演" and has_replay:
        ep_outcomes = ep_data["outcomes"]
        ep_actions = ep_data["actions"]
        n_ep = ep_data["n_episodes"]
        replay_options = [
            f"Episode {i} ({ep_outcomes[i] if i < len(ep_outcomes) else '?'}, {len(ep_actions[i]) if i < len(ep_actions) else 0}步)"
            for i in range(n_ep)
        ]
        replay_idx = st.selectbox(
            "选择 Episode",
            options=range(n_ep),
            format_func=lambda i: replay_options[i],
            key="live_replay_ep",
        )
        replay_runs = st.slider("执行次数", 1, 50, 1, key="live_replay_runs")

    batch_start = 0
    batch_end = 0
    replay_count = 3
    threshold_defaults = get_bktree_threshold_defaults(map_id)
    primary_threshold = float(threshold_defaults["primary_threshold"])
    secondary_threshold = float(threshold_defaults["secondary_threshold"])
    if live_mode == "重采样数据集扩张（批量回放重演）" and has_replay:
        n_ep_batch = ep_data["n_episodes"]
        c1, c2, c3 = st.columns(3)
        with c1:
            batch_start = st.number_input(
                "起始 Episode", 0, n_ep_batch - 1, 0, key="live_batch_start"
            )
        with c2:
            batch_end = st.number_input(
                "结束 Episode",
                batch_start,
                n_ep_batch - 1,
                min(n_ep_batch - 1, batch_start + 99),
                key="live_batch_end",
            )
        with c3:
            replay_count = st.slider("每条重复次数", 1, 10, 3, key="live_replay_count")
        total_eps = (batch_end - batch_start + 1) * replay_count
        st.caption(f"预计生成 {total_eps} 条 episode")

        tc1, tc2 = st.columns(2)
        with tc1:
            primary_threshold = st.number_input(
                "主聚类阈值 (primary)",
                0.1,
                5.0,
                float(threshold_defaults["primary_threshold"]),
                0.1,
                key="live_primary_thresh",
                help="坐标分布距离阈值。越小聚类越细、cluster 数越多；越大聚类越粗",
            )
        with tc2:
            secondary_threshold = st.number_input(
                "子聚类阈值 (secondary)",
                0.1,
                3.0,
                float(threshold_defaults["secondary_threshold"]),
                0.1,
                key="live_secondary_thresh",
                help="生命值差异阈值。越小对 HP 变化越敏感",
            )

    else:
        tc1, tc2 = st.columns(2)
        with tc1:
            primary_threshold = st.number_input(
                "BKTree primary",
                0.0,
                5.0,
                float(threshold_defaults["primary_threshold"]),
                0.05,
                key="live_primary_thresh",
                help="实时规划阶段使用的 primary BKTree 最近邻接受阈值。",
            )
        with tc2:
            secondary_threshold = st.number_input(
                "BKTree secondary",
                0.0,
                5.0,
                float(threshold_defaults["secondary_threshold"]),
                0.05,
                key="live_secondary_thresh",
                help="实时规划阶段使用的 secondary BKTree 最近邻接受阈值。",
            )

    live_backup = False
    if live_mode not in ("回放重演", "重采样数据集扩张（批量回放重演）"):
        st.subheader("Beam Search 参数")

        c1, c2 = st.columns(2)
        with c1:
            st.selectbox(
                "评分策略",
                options=["quality", "future_reward", "win_rate"],
                format_func={
                    "quality": "Quality Score",
                    "future_reward": "Future Reward",
                    "win_rate": "Win Rate",
                }.get,
                key="live_sm",
            )
        with c2:
            st.slider("Beam Width", 1, 10, 3, key="live_bw")

        c3, c4 = st.columns(2)
        with c3:
            st.slider("前瞻步数", 1, 15, 5, key="live_la")
        with c4:
            st.slider("最低访问次数", 1, 10, 1, key="live_mv")

        c5, c6 = st.columns(2)
        with c5:
            st.slider(
                "最大状态重复",
                1,
                5,
                2,
                key="live_msr",
                help="同一条推演路径中同一状态最多出现的次数。",
            )
        with c6:
            st.slider(
                "累积概率阈值",
                0.001,
                0.1,
                0.01,
                step=0.001,
                format="%.3f",
                key="live_mcp",
            )

        c7, c8 = st.columns(2)
        with c7:
            st.slider(
                "折扣因子",
                0.5,
                1.0,
                0.9,
                step=0.05,
                format="%.2f",
                key="live_df",
                help="每步累积概率乘以此值的步数次幂。1.0=无折扣。",
            )
        with c8:
            st.selectbox(
                "动作选择",
                options=list(_ACTION_STRATEGY_LABELS.keys()),
                format_func=lambda x: _ACTION_STRATEGY_LABELS[x],
                key="live_as",
            )

        c9, c10 = st.columns(2)
        with c9:
            st.slider(
                "ε",
                0.01,
                0.5,
                0.1,
                step=0.01,
                format="%.2f",
                key="live_eps",
                disabled=(st.session_state.get("live_as") != "epsilon_greedy"),
            )
        with c10:
            st.slider("最大推演步数", 1, 100, 50, key="live_mrs")

        live_backup = st.toggle("启用备选路径", value=False, key="live_backup")
        if live_backup:
            bc1, bc2 = st.columns(2)
            with bc1:
                st.slider(
                    "备选评分阈值",
                    0.0,
                    1.0,
                    0.3,
                    step=0.05,
                    format="%.2f",
                    key="live_backup_st",
                )
            with bc2:
                st.slider(
                    "模糊匹配距离阈值",
                    0.0,
                    1.0,
                    0.2,
                    step=0.05,
                    format="%.2f",
                    key="live_backup_dt",
                )

    if _start_clicked:
        if "live_proc" in st.session_state and st.session_state.live_proc is not None:
            proc = st.session_state.live_proc
            if proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=3)
                except Exception:
                    proc.kill()
        cmd = [
            sys.executable,
            str(ROOT_DIR / "scripts" / "run_live_game.py"),
            "--mode",
            "all",
            "--port",
            str(port),
            "--map_key",
            map_key,
        ]
        if etg_file:
            cmd.extend(["--etg_file", etg_file])
        if etg_data_dir:
            cmd.extend(["--data_dir", etg_data_dir])
        cmd.extend(
            [
                "--window_x",
                str(st.session_state.get("live_wx", 50)),
                "--window_y",
                str(st.session_state.get("live_wy", 50)),
                "--window_w",
                str(st.session_state.get("live_ww", 640)),
                "--window_h",
                str(st.session_state.get("live_wh", 480)),
            ]
        )

        if live_mode == "回放重演":
            cmd.extend(["--autopilot_mode", "replay"])
            if replay_idx < len(ep_data["actions"]):
                cmd.extend(
                    [
                        "--replay_actions",
                        ",".join(ep_data["actions"][replay_idx]),
                    ]
                )
            cmd.extend(["--replay_runs", str(replay_runs)])
        elif live_mode == "重采样数据集扩张（批量回放重演）":
            cmd.extend(["--autopilot_mode", "batch_replay"])
            cmd.extend(["--batch_start", str(batch_start)])
            cmd.extend(["--batch_end", str(batch_end)])
            cmd.extend(["--replay_count", str(replay_count)])
            cmd.extend(["--primary_threshold", str(primary_threshold)])
            cmd.extend(["--secondary_threshold", str(secondary_threshold)])
        elif live_mode == "多步推演":
            cmd.extend(["--autopilot_mode", "multi_step"])
        else:
            cmd.extend(["--autopilot_mode", "single_step"])
        if "--primary_threshold" not in cmd:
            cmd.extend(["--primary_threshold", str(primary_threshold)])
        if "--secondary_threshold" not in cmd:
            cmd.extend(["--secondary_threshold", str(secondary_threshold)])

        if live_mode not in ("回放重演", "重采样数据集扩张（批量回放重演）"):
            cmd.extend(["--beam_width", str(st.session_state.get("live_bw", 3))])
            cmd.extend(["--lookahead_steps", str(st.session_state.get("live_la", 5))])
            cmd.extend(["--score_mode", st.session_state.get("live_sm", "quality")])
            cmd.extend(["--min_visits", str(st.session_state.get("live_mv", 1))])
            cmd.extend(
                ["--max_state_revisits", str(st.session_state.get("live_msr", 2))]
            )
            cmd.extend(["--min_cum_prob", str(st.session_state.get("live_mcp", 0.01))])
            cmd.extend(["--discount_factor", str(st.session_state.get("live_df", 0.9))])
            cmd.extend(
                ["--action_strategy", st.session_state.get("live_as", "best_beam")]
            )
            cmd.extend(["--epsilon", str(st.session_state.get("live_eps", 0.1))])
        if live_backup:
            cmd.append("--enable_backup")
            cmd.extend(
                [
                    "--backup_score_threshold",
                    str(st.session_state.get("live_backup_st", 0.3)),
                ]
            )
            cmd.extend(
                [
                    "--backup_distance_threshold",
                    str(st.session_state.get("live_backup_dt", 0.2)),
                ]
            )
        p = subprocess.Popen(
            cmd,
            cwd=str(ROOT_DIR),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            if sys.platform == "win32"
            else 0,
        )
        st.session_state.live_proc = p
        with st.spinner("等待服务启动..."):
            for _ in range(30):
                time.sleep(0.5)
                try:
                    r = requests.get(f"{api_base}/game/status", timeout=2)
                    if r.status_code == 200:
                        if start_paused:
                            requests.post(
                                f"{api_base}/game/control",
                                json={"command": "pause"},
                                timeout=5,
                            )
                        st.success("服务已启动" + (" (已暂停)" if start_paused else ""))
                        break
                except Exception:
                    continue
            else:
                st.warning("服务启动超时，请检查日志")


def _get_bridge_host():
    if "_bridge_host" not in st.session_state:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            st.session_state["_bridge_host"] = s.getsockname()[0]
            s.close()
        except Exception:
            st.session_state["_bridge_host"] = "localhost"
    return st.session_state["_bridge_host"]


def _render_live_game_content():
    port = int(st.session_state.get("live_port", 8000))
    host = _get_bridge_host()
    html = _build_live_game_html(port, host)
    st.markdown(
        """<style>
.stMainBlockContainer{padding-bottom:1rem !important}
div[data-testid="stVerticalBlock"]>div:has(>div:has(iframe)){margin-bottom:0 !important;padding-bottom:0 !important}
</style>""",
        unsafe_allow_html=True,
    )
    st.components.v1.html(html, height=1250, scrolling=True)
