from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def rerun(at):
    # Streamlit 1.40's AppTest represents segmented controls as ButtonGroups.
    # Its serializer expects a list even for the real widget's scalar value.
    for group in at.button_group:
        if isinstance(group.value, str):
            group.set_value([group.value])
    at.run()
    assert not at.exception, [item.value for item in at.exception]


def test_language_switch_preserves_table_selection_and_paper_navigation():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "scripts/visualize_etg_web.py"), default_timeout=30).run()
    assert not at.exception
    assert "数据筛选" not in str(at.button_group(key="tab_selector").options)
    at.radio(key="paper_table").set_value(2)
    rerun(at)
    at.button_group(key="language_switcher").set_value(["en"])
    rerun(at)
    assert at.radio(key="paper_section").label == "Explore"
    assert at.radio(key="paper_table").value == 2
    assert at.dataframe[0].value.shape[0] == 36
    assert list(at.dataframe[0].value.columns) == ["Scenario", "Method", "Win rate (%)", "Score: mean ± SD"]
    at.button_group(key="language_switcher").set_value(["zh"])
    rerun(at)
    assert at.radio(key="paper_table").value == 2


def test_beam_cache_is_separate_for_different_graph_files():
    from etg_web.prediction_tab import _cached_beam_results
    _cached_beam_results.clear()
    results = []
    for folder in ["MarineMicro_MvsM_4_augmented", "MarineMicro_MvsM_4_mirror_augmented"]:
        results.append(_cached_beam_results(
            folder + "/etg_simple.pkl", folder + "/etg_simple_transitions.pkl",
            0, 3, 5, 1, .01, "quality", 2, .9)["action"])
    assert results == ["4b", "1b"]


def test_english_pages_render_and_execute_planning_and_rollout():
    import re
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "scripts/visualize_etg_web_en.py"), default_timeout=90).run()

    def check():
        assert not at.exception, [item.value for item in at.exception]
        for kind in ["title", "caption", "markdown", "info", "warning", "success", "error", "code"]:
            for item in getattr(at, kind):
                assert not re.search(r"[\u3400-\u9fff]", str(item.value)), (kind, item.value)
        for kind in ["radio", "selectbox", "button", "number_input", "slider", "checkbox"]:
            for item in getattr(at, kind):
                assert not re.search(r"[\u3400-\u9fff]", item.label), item.label
                if hasattr(item, "options"):
                    assert not re.search(r"[\u3400-\u9fff]", str(item.options)), item.options
        for item in at.dataframe:
            assert not re.search(r"[\u3400-\u9fff]", item.value.to_string()), item.value

    check()
    at.radio(key="paper_table").set_value(3).run()
    check()
    assert at.dataframe[0].value.shape[0] == 8
    assert not at.warning
    at.radio(key="paper_section").set_value("sources").run()
    check()
    assert "raw/evaluations" in at.selectbox(key="paper_asset_folder").value or at.selectbox(key="paper_asset_folder").value == "."
    at.selectbox(key="paper_asset_folder").set_value("reference").run()
    check()
    assert "camera_ready_tables.csv" in at.dataframe[0].value["File"].tolist()
    assert not at.json
    assert not at.info
    for page in ["Project overview", "Fresh experiments", "Experience Transition Graph", "Beam-search planning", "Graph rollout"]:
        at.radio(key="english_page").set_value(page).run()
        check()
        if page == "Experience Transition Graph":
            assert {item.label: item.value for item in at.slider}["Neighborhood hops"] == 1
            assert {item.label: item.value for item in at.slider}["Neighbors per state"] == 6
            assert {item.label: item.value for item in at.slider}["Maximum rendered nodes"] == 24
            assert not any(item.key == "english_method" for item in at.selectbox)
        if page == "Beam-search planning":
            at.button[0].click().run()
            check()
            assert "Recommended" in at.success[0].value
        if page == "Graph rollout":
            at.button[0].click().run()
            check()
            assert at.dataframe[0].value.shape[0] > 1
