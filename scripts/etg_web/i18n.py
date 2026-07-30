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


def language() -> str:
    """Return the active UI language without modifying business state."""
    return _streamlit.session_state.get(LANGUAGE_KEY, "zh")


def translate(value: Any) -> Any:
    """Translate a display value while leaving non-string values untouched."""
    if not isinstance(value, str):
        return value
    return TRANSLATIONS.get(value, {}).get(language(), value)


def _translate_first_argument(args: tuple[Any, ...]) -> tuple[Any, ...]:
    if args and isinstance(args[0], str):
        return (translate(args[0]), *args[1:])
    return args


def _translate_options(kwargs: Dict[str, Any], has_options: bool) -> None:
    """Localize option display names without changing the underlying values."""
    if not has_options:
        return
    original = kwargs.get("format_func")
    if original is None:
        kwargs["format_func"] = translate
    else:
        kwargs["format_func"] = lambda value: translate(original(value))


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
}

_KEYWORD_TEXT = {"label", "help", "placeholder", "caption"}
_OPTION_WIDGETS = {"radio", "selectbox", "multiselect", "segmented_control", "select_slider"}


class _LocalizedStreamlit:
    """A transparent Streamlit proxy used by every ETG web module."""

    def __getattr__(self, name: str) -> Any:
        target = getattr(_streamlit, name)
        if not callable(target) or name not in _FIRST_ARGUMENT_TEXT:
            return target

        @wraps(target)
        def localized(*args: Any, **kwargs: Any) -> Any:
            if name in _FIRST_ARGUMENT_TEXT:
                args = _translate_first_argument(args)
            for key in _KEYWORD_TEXT:
                if key in kwargs:
                    kwargs[key] = translate(kwargs[key])
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
