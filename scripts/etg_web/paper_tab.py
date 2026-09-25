"""Paper results, figures, and asset navigation for the Streamlit application."""
from pathlib import Path

import pandas as pd

from etg_web.i18n import language, st
from paper_assets.aiide26_sage.evidence import PACK, ROOT


def text(zh, en):
    return zh if language() == "zh" else en


@st.cache_data
def _table_inputs():
    return pd.read_csv(PACK / "reference/camera_ready_tables.csv")


def render_asset_browser(t):
    st.markdown(t(
        "`paper_assets/aiide26_sage/` 按论文图表组织数据与绘图代码。`raw/` 保存评估日志和运行参数，`data/` 保存图表输入，`reference/` 保存论文图表，`scripts/` 保存分析与绘图工具。",
        "`paper_assets/aiide26_sage/` organizes the paper's data and plotting code. `raw/` holds evaluation logs and run settings, `data/` figure and table inputs, `reference/` paper figures and tables, and `scripts/` analysis and plotting tools."))
    folders = ["."] + sorted(p.relative_to(PACK).as_posix() for p in PACK.rglob("*")
        if p.is_dir() and "__pycache__" not in p.parts and not p.name.startswith("."))
    selected = st.selectbox(t("目录树", "Directory tree"), folders,
        format_func=lambda p: "paper_assets/aiide26_sage/" if p == "." else "  " * p.count("/") + "└─ " + p + "/",
        key="paper_asset_folder")
    folder = PACK if selected == "." else PACK / selected
    children = sorted((p for p in folder.iterdir() if p.name != "__pycache__" and not p.name.startswith(".")),
        key=lambda p: (not p.is_dir(), p.name))
    st.code("paper_assets/aiide26_sage/" + ("" if selected == "." else selected + "/") + "\n" +
        "\n".join(("└── " if i == len(children)-1 else "├── ") + p.name + ("/" if p.is_dir() else "")
                  for i, p in enumerate(children)), language=None)
    files = [p for p in children if p.is_file()]
    if files:
        st.dataframe(pd.DataFrame([{t("文件", "File"): p.name, t("大小（字节）", "Size (bytes)"): p.stat().st_size}
                                  for p in files]), hide_index=True, use_container_width=True)
    st.caption(t("选择目录可查看其中的文件。evidence.py 读取图表数据，plots.py 提供绘图入口，provenance.json 保存文件来源及标识。",
                 "Select a directory to inspect its files. evidence.py reads table and figure data, plots.py provides the plotting entry points, and provenance.json records file sources and identities."))


@st.cache_data
def _pdf_preview(path, modified):
    import fitz
    with fitz.open(path) as doc:
        return doc[0].get_pixmap(matrix=fitz.Matrix(2.2, 2.2), alpha=False).tobytes("png")


def render_paper_tab():
    local_language = language()
    t = lambda zh, en: zh if local_language == "zh" else en
    st.caption(t("浏览论文结果、配图及对应数据目录。", "Browse the paper's results, figures, and data directories."))
    section = st.radio(t("查看内容", "Explore"), ["tables", "figures", "sources"], horizontal=True,
        format_func=lambda key: {"tables": t("论文表格", "Paper tables"),
                                "figures": t("论文配图", "Paper figures"),
                                "sources": t("论文资产目录", "Paper assets")}[key], key="paper_section")
    if section == "tables":
        number = st.radio(t("论文表格", "Paper table"), [1, 2, 3], horizontal=True,
            format_func=lambda n: t(f"表 {n}", f"Table {n}"), key="paper_table")
        if number == 3:
            st.caption(t("论文表 3：实时决策性能。", "Table 3: real-time decision performance reported in the paper."))
            st.dataframe(pd.read_csv(PACK / "reference/camera_ready_table_03.csv"), hide_index=True, use_container_width=True)
            return
        tables = _table_inputs()
        part = tables[tables.table.eq(number)].copy()
        part["score"] = part.apply(lambda r: f"{r.score_mean:.2f} ± {r.score_std:.2f}", axis=1)
        view = part[["scenario", "method", "win_rate_pct", "score"]].rename(columns={
            "scenario": t("场景", "Scenario"), "method": t("方法", "Method"),
            "win_rate_pct": t("胜率（%）", "Win rate (%)"), "score": t("得分：均值 ± 标准差", "Score: mean ± SD")})
        st.dataframe(view, hide_index=True, use_container_width=True)
        st.download_button(t("下载表格 CSV", "Download table CSV"), part.drop(columns="score").to_csv(index=False).encode("utf-8"),
                           file_name=f"paper_table_{number:02}.csv", mime="text/csv")
        st.caption(t(f"论文表 {number} 中报告的结果。", f"Results reported in Table {number} of the paper."))
    elif section == "figures":
        catalog = {
            "1": ("fig1_architecture.png", "reference", t("方法结构示意（静态原图）", "Method schematic (static original)")),
            "2a": ("fig2_sce1.pdf", "reference", t("4v4 场景示意（静态原图）", "4v4 scenarios (static original)")),
            "2b": ("fig2_sce2.pdf", "reference", t("位移场景示意（静态原图）", "Shifted scenarios (static original)")),
            "2c": ("fig2_sce3.pdf", "reference", t("8v8 场景示意（静态原图）", "8v8 scenarios (static original)")),
            "3": ("fig3_parameter_correlation.pdf", "figure_03_parameter_correlation", t("规划参数相关性", "Planning-parameter correlations")),
            "4": ("fig4_backup_switching.pdf", "figure_04_backup_switch", t("高质量参数组条件分布", "Conditional distributions of retained parameter groups")),
            "5a": ("fig5a_score_distribution.pdf", "figure_05_06_gate_exploration", t("所选对局得分", "Selected episode scores")),
            "5b": ("fig5b_decision_trace.pdf", "figure_05_06_gate_exploration", t("真实对局分支轨迹", "Decision trace from recorded episodes")),
            "6": ("fig6_state_quality.pdf", "figure_05_06_gate_exploration", t("状态质量与对局进度", "State quality and episode progress")),
            "7": ("fig7_state_diagnostics.pdf", "figure_07_state_diagnostics", t("sce-2 状态空间诊断", "sce-2 state-space diagnostics")),
        }
        key = st.selectbox(t("配图", "Figure"), list(catalog), format_func=lambda k: f"{k} — {catalog[k][2]}", key="paper_figure")
        filename, data_dir, description = catalog[key]
        version = st.radio(t("图像来源", "Image source"), ["reference", "reproduced"], horizontal=True,
            format_func=lambda v: t("定稿原图（归档）", "Camera-ready original (archived)") if v == "reference" else t("从数据重绘", "Redrawn from data"), key="paper_figure_source")
        path = PACK / "reference" / filename if version == "reference" else ROOT / "output/paper_reproduction/figures" / filename
        if key in ["1", "2a", "2b", "2c"] and version == "reproduced":
            st.info(t("此图为静态示意，不属于数据绘图脚本的输出。", "This is a static illustration, not a data-driven plot."))
        elif not path.exists():
            st.info(t("尚未生成重绘结果，请先运行下方命令。", "No redraw exists yet. Run the command below first."))
        else:
            preview = _pdf_preview(str(path), path.stat().st_mtime_ns) if path.suffix == ".pdf" else str(path)
            st.image(preview, caption=description, use_container_width=True)
            st.download_button(t("下载当前图", "Download this figure"), path.read_bytes(), file_name=path.name,
                               mime="application/pdf" if path.suffix == ".pdf" else "image/png")
        if key in ["3", "4", "5a", "5b", "6", "7"]:
            st.code(f"python scripts/reproduce_paper.py --figure {key}", language="bash")
            st.code(f"paper_assets/aiide26_sage/data/{data_dir}/", language=None)
    else:
        render_asset_browser(t)
