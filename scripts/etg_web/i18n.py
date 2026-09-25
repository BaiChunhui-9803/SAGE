"""Centralized bilingual presentation support for the ETG web application.

Business values (select-box options, session keys, configuration values) remain
unchanged.  Only the text rendered by Streamlit is localized, so switching the
UI language cannot change planning, training, or live-game behavior.
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Dict

import streamlit as _streamlit


LANGUAGE_KEY = "ui_language"

# Keep every user-facing translation in this single catalogue.  The proxy
# below routes labels, help text, messages, captions, and option display names
# from every ETG web module through ``translate``.
TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "SAGE · 论文与经验图": {"en": "SAGE · Paper & Graph Explorer"},
    "论文复现": {"en": "Paper reproduction"},
    "显示高级实验功能": {"en": "Show advanced experiment tools"},
    "交互式浏览经验转移图 + 图上束搜索规划。": {"en": "Inspect paper evidence, experience graphs, and beam-search plans."},
    "双选": {"en": "Both directions"},
    "功能": {"en": "Features"},
    "界面语言": {"en": "Interface language"},
    "中文": {"en": "Chinese"},
    "English": {"zh": "英文"},
    "项目介绍": {"en": "Project overview"},
    "转移图可视化": {"en": "ETG visualization"},
    "束搜索规划": {"en": "Beam-search planning"},
    "滚动推演": {"en": "Rolling rollout"},
    "原始数据": {"en": "Raw data"},
    "实时对局": {"en": "Live game"},
    "结果分析": {"en": "Result analysis"},
    "参数寻优": {"en": "Parameter optimization"},
    "批量实验": {"en": "Batch experiments"},
    "数据筛选": {"en": "Data filtering"},
    "地图": {"en": "Map"},
    "经验转移图": {"en": "Experience Transition Graph"},
    "类型": {"en": "Type"},
    "窗口": {"en": "Window"},
    "聚焦模式": {"en": "Focus mode"},
    "聚焦状态 ID": {"en": "Focus state ID"},
    "扩展跳数": {"en": "Expansion hops"},
    "扩展方向": {"en": "Expansion direction"},
    "双向": {"en": "Bidirectional"},
    "作为源节点": {"en": "As source node"},
    "作为目标节点": {"en": "As target node"},
    "最小访问次数": {"en": "Minimum visits"},
    "最大节点数": {"en": "Maximum nodes"},
    "高亮终端状态": {"en": "Highlight terminal states"},
    "渲染设置": {"en": "Rendering settings"},
    "边样式": {"en": "Edge style"},
    "直线": {"en": "Straight"},
    "弧线": {"en": "Curved"},
    "弯曲度": {"en": "Curvature"},
    "布局算法": {"en": "Layout algorithm"},
    "渲染": {"en": "Render"},
    "冻结布局（拖拽不回弹）": {"en": "Freeze layout"},
    "起始状态 ID": {"en": "Initial state ID"},
    "评分策略": {"en": "Scoring strategy"},
    "最大步数": {"en": "Maximum steps"},
    "最大状态重复": {"en": "Maximum state repeats"},
    "累计概率阈值": {"en": "Cumulative probability threshold"},
    "折扣因子": {"en": "Discount factor"},
    "开始规划": {"en": "Start planning"},
    "前瞻步数": {"en": "Look-ahead steps"},
    "动作选择": {"en": "Action selection"},
    "状态转移": {"en": "State transition"},
    "推演模式": {"en": "Rollout mode"},
    "单步推演": {"en": "Single-step rollout"},
    "多步推演": {"en": "Multi-step rollout"},
    "启用备选路径": {"en": "Enable backup paths"},
    "开始推演": {"en": "Start rollout"},
    "经验转移图目录为空，请检查 configs/etg_catalog.yaml": {
        "en": "The Experience Transition Graph catalogue is empty; check configs/etg_catalog.yaml."
    },
    "README.md 尚未创建。请在项目根目录放置 README.md 文件。": {
        "en": "README.md was not found. Place it in the project root."
    },
}

TRANSLATIONS.update({zh: {"en": en} for zh, en in {
    "搜索": "Search", "输入状态 ID": "Enter state ID", "状态搜索": "State search",
    "状态节点": "State nodes", "动作种类": "Action types", "总访问次数": "Total visits",
    "构建图谱...": "Building graph...", "当前渲染": "Rendered graph", "个节点": "nodes", "条边": "edges",
    "没有满足条件的节点，请放宽筛选条件。": "No nodes match these filters. Widen the filters.",
    "渲染可视化（首次可能需要几秒）...": "Rendering graph (the first load may take several seconds)...",
    "节点统计表（按访问量排序）": "Node statistics (sorted by visits)",
    "边统计表（按质量排序 Top 50）": "Edge statistics (top 50 by quality)",
    "该状态的所有动作统计": "Action statistics for this state",
    "该状态在当前 ETG 中无动作记录。": "This state has no action records in this ETG.",
    "该状态的转移概率": "Transition probabilities for this state",
    "不存在于当前经验转移图中。请选择一个有效状态。": "is absent from this ETG. Choose a valid state.",
    "不存在于经验转移图中": "is absent from the ETG", "状态 ": "State ",
    "最低访问次数": "Minimum visits", "累积概率阈值": "Cumulative probability threshold",
    "Quality Score 范围": "Quality Score range", "默认": "default",
    "数据来源": "Data source", "仅展示指定状态的邻居子图": "Show the neighbourhood of the selected state",
    "过滤低频 state-action 对": "Filter infrequent state-action pairs",
    "过滤 state-action 对的 Quality Score 范围": "Filter state-action pairs by their quality scores",
    "限制渲染规模，避免卡顿": "Limit rendering size to keep the interface responsive",
    "Win终端=绿色, Loss终端=红色": "Winning terminals are green; losing terminals are red",
    "直线: 简洁清晰; 弧线: 同一对节点间的多条边自动扇形分散": "Straight edges are compact; curved parallel edges fan out",
    "手动弧线模式下，弧线弯曲的基础倍率": "Base curvature for manual arcs",
    "双选：同时扩展前后；源节点：仅后继；目标节点：仅前驱": "Both: predecessors and successors; source: successors; target: predecessors",
    "不同力导向算法影响节点分布方式": "Choose the force-directed node layout",
    "每步累积概率乘以此值的步数次幂。1.0=无折扣，越低越惩罚过深路径。": "Discount cumulative probability by depth; 1.0 disables discounting",
    "每条 beam 路径中同一状态最多出现的次数。设为 1 则完全禁止重复访问。": "Maximum visits to a state within a beam; 1 forbids revisits",
    "同一条推演路径中同一状态最多出现的次数。设为 1 则完全禁止重复访问。": "Maximum visits to a state within a rollout; 1 forbids revisits",
    "单步规划参数（每步束搜索）": "Per-step beam-search parameters", "滚动推演参数": "Rollout parameters",
    "最大推演步数": "Maximum rollout steps", "备选评分阈值": "Backup score threshold",
    "模糊匹配距离阈值": "Fuzzy matching distance threshold", "概率采样": "Probability sampling", "最高概率": "Highest probability",
    "从指定起始状态出发，用 Beam Search 在转移图上进行多步规划搜索。": "Plan from the selected initial state with beam search over the ETG.",
    "从指定起始状态出发，按策略逐步滚动推演至终端状态。": "Roll out from the selected state toward a terminal state.",
    "无法规划：该状态无可用动作。": "No plan: this state has no eligible actions.",
    "无搜索结果。": "No search results.", "最优动作推荐": "Action recommendation",
    "推荐动作": "Recommended action", "预期累积奖励": "Expected cumulative reward", "预期胜率": "Expected win rate",
    "最优路径累积概率": "Best-path cumulative probability", "最优路径步数": "Best-path length",
    "节点颜色对应不同 Beam；鼠标悬停查看详情。": "Colours identify beams; hover over nodes for details.",
    "没有转移数据，无法进行规划。": "has no transition data for planning.",
    "生成路径树...": "Building planning tree...", "路径树图": "Planning tree", "束搜索追溯": "Beam-search trace",
    "路径推荐": "Recommended paths", "路径详情": "Path details", "片段匹配": "Trajectory matching",
    "匹配原始对局片段...": "Matching archived trajectories...",
    "无匹配结果。原始对局数据可能未加载。": "No matches. Full training trajectories are not included for the augmented graph.",
    "无路径数据。": "No path data.", "导出 HTML": "Export HTML", "导出 JSON": "Export JSON",
    "推演无结果：该状态无可用动作或为终端状态。": "No rollout: this state has no eligible actions or is terminal.",
    "推演摘要": "Rollout summary", "总步数": "Total steps", "终止原因": "Termination reason", "起止状态": "Start/end state",
    "累积概率过低": "Cumulative probability too low", "终端状态": "Terminal state",
    "达到最大推演步数": "Maximum rollout length reached", "无可用动作": "No eligible action", "无搜索路径": "No search path",
    "胜率/质量趋势": "Win-rate / quality trend", "推演轨迹": "Rollout trajectory", "初始规划": "Initial plan",
    "平均胜率": "Mean win rate", "平均奖励": "Mean reward", "平均Quality": "Mean quality", "平均 Quality": "Mean quality",
    "综合评分": "Composite score", "综合置信度": "Composite confidence", "首步动作": "First action",
    "首步目标": "First target", "末状态": "End state", "末Quality": "End quality", "末胜率": "End win rate",
    "置信度": "Confidence", "长度": "Length", "段类型": "Segment type", "位置": "Position",
    "状态相似度": "State similarity", "动作匹配率": "Action agreement", "得分": "Score", "结果": "Result",
}.items()})


def language() -> str:
    """Return the active UI language without modifying business state."""
    return _streamlit.session_state.get(LANGUAGE_KEY, "zh")


def translate(value: Any, lang: str = None) -> Any:
    """Translate a display value while leaving non-string values untouched."""
    if not isinstance(value, str):
        return value
    lang = lang or language()
    exact = TRANSLATIONS.get(value, {}).get(lang)
    if exact is not None:
        return exact
    # Decorated labels and concise dynamic captions share the same vocabulary.
    if lang == "en":
        for source in sorted(TRANSLATIONS, key=len, reverse=True):
            if len(source) > 1 and source in value and "en" in TRANSLATIONS[source]:
                value = value.replace(source, TRANSLATIONS[source]["en"])
    return value


def _translate_first_argument(args: tuple[Any, ...]) -> tuple[Any, ...]:
    if args and isinstance(args[0], str):
        return (translate(args[0]), *args[1:])
    return args


def _translate_options(kwargs: Dict[str, Any], has_options: bool) -> None:
    """Localize option display names without changing the underlying values."""
    if not has_options:
        return
    original = kwargs.get("format_func")
    selected_language = language()
    if original is None:
        kwargs["format_func"] = lambda value: translate(value, selected_language)
    else:
        kwargs["format_func"] = lambda value: translate(original(value), selected_language)


_FIRST_ARGUMENT_TEXT = {
    "title",
    "header",
    "subheader",
    "caption",
    "markdown",
    "write",
    "text",
    "info",
    "success",
    "warning",
    "error",
    "exception",
    "button",
    "download_button",
    "checkbox",
    "toggle",
    "radio",
    "selectbox",
    "multiselect",
    "segmented_control",
    "number_input",
    "slider",
    "select_slider",
    "text_input",
    "text_area",
    "date_input",
    "time_input",
    "file_uploader",
    "color_picker",
    "expander",
    "metric",
    "spinner",
    "toast",
}

_KEYWORD_TEXT = {"label", "help", "placeholder", "caption"}
_OPTION_WIDGETS = {"radio", "selectbox", "multiselect", "segmented_control", "select_slider"}


class _LocalizedStreamlit:
    """A transparent Streamlit proxy used by every ETG web module."""

    def __init__(self, backend=_streamlit):
        self._backend = backend

    def __enter__(self):
        self._backend.__enter__()
        return self

    def __exit__(self, *args):
        return self._backend.__exit__(*args)

    def __getattr__(self, name: str) -> Any:
        target = getattr(self._backend, name)
        if name in {"columns", "tabs"}:
            return lambda *args, **kwargs: [_LocalizedStreamlit(item) for item in target(*args, **kwargs)]
        if not callable(target) or name not in _FIRST_ARGUMENT_TEXT:
            return target

        @wraps(target)
        def localized(*args: Any, **kwargs: Any) -> Any:
            if name in _FIRST_ARGUMENT_TEXT:
                args = _translate_first_argument(args)
            for key in _KEYWORD_TEXT:
                if key in kwargs:
                    kwargs[key] = translate(kwargs[key])
            if name == "metric" and len(args) > 1:
                args = (args[0], translate(args[1]), *args[2:])
            if name in {"radio", "selectbox"} and len(args) < 3 and kwargs.get("key") in _streamlit.session_state:
                options = kwargs.get("options", args[1] if len(args) > 1 else [])
                current = _streamlit.session_state[kwargs["key"]]
                try:
                    kwargs["index"] = list(options).index(current)
                except ValueError:
                    pass
            if name in _OPTION_WIDGETS:
                _translate_options(kwargs, "options" in kwargs or len(args) > 1)
            return target(*args, **kwargs)

        return localized


st = _LocalizedStreamlit()


def render_language_switcher() -> None:
    """Render the Chinese/English switcher above the sidebar feature selector."""
    _streamlit.session_state.setdefault(LANGUAGE_KEY, "zh")
    selected = _streamlit.segmented_control(
        "界面语言",
        options=("zh", "en"),
        default=_streamlit.session_state[LANGUAGE_KEY],
        format_func=lambda code: "中文" if code == "zh" else "English",
        key="language_switcher",
        label_visibility="collapsed",
    )
    if selected in {"zh", "en"}:
        _streamlit.session_state[LANGUAGE_KEY] = selected
