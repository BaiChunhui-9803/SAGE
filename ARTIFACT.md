# SAGE installation and experiments

Run all commands from the repository root. Start with the Git LFS checkout described in [README.md](README.md#quick-start-paper-evidence-and-web).

## Paper figures and Web

Use **Python 3.12** for this CPU workflow; StarCraft II and PyTorch are not required.

```bash
python -m venv .venv-artifact
# Windows: .venv-artifact\Scripts\activate
# Linux/macOS: source .venv-artifact/bin/activate
python -m pip install -r requirements-artifact.txt
python scripts/reproduce_paper.py --figure all
python -m streamlit run scripts/visualize_etg_web_en.py
```

Figures are written to `output/paper_reproduction/figures/`. To draw one figure, replace `all` with `3`, `4`, `5a`, `5b`, `6`, or `7`.

Open the local URL printed by Streamlit:

- **Paper results:** view tables and figures; **Paper assets** browses their input directories.
- **Experience Transition Graph:** inspect one of six scenario graphs. Defaults are state **1**, **two hops**, six neighbors per state, at most 24 nodes and 60 action transitions.
- **Beam-search planning** and **Graph rollout:** select a scenario and recorded configuration, then run the planner or rollout.
- **Fresh experiments:** obtain commands for new Graph-only or Full SAGE runs.

See [paper assets](paper_assets/aiide26_sage/README.md) for the figure/table inputs and scripts. Stop Streamlit with Ctrl+C.

## Fresh SC2 evaluations

On Windows, install **StarCraft II** and use a separate **Python 3.8.10** environment. Set `SC2PATH` to your game directory if PySC2 cannot locate it. The launcher loads all six bundled maps from `assets/maps/`.

```powershell
py -3.8 -m venv .venv-sc2
.venv-sc2\Scripts\activate
python -m pip install -r requirements-sc2.txt
python scripts/run_paper_sage.py --scenario sce-1 --method graph-only --episodes 2 --run
python scripts/run_paper_sage.py --scenario sce-1 --method full --episodes 2 --run
```

Run these source scripts directly in the SC2 environment; do not install the Python 3.12 project package into it. Omit `--run` to inspect the prepared configuration without starting the game. Use `--episodes 300` for a full-length evaluation.

Scenario keys are `sce-1`, `sce-1m`, `sce-2`, `sce-2m`, `sce-3`, and `sce-3m`. Each run writes to a new directory under `output/paper_live/`:

- `episodes.jsonl`: episode scores and results.
- `completion.json`: completion status and requested/recorded episode counts.
- `startup_beam_params.json` and `run_manifest.json`: settings and launch command.

Full SAGE updates a run-local copy of its supplied initial model. After an interruption, keep the incomplete output and launch a new run.

## Optional parameter search

In the SC2 environment, use a new output directory:

```bash
python -m pip install -r requirements-sc2-tools.txt
python scripts/parameter_learner.py --trials 2 --episodes 2 --run_dir output/search_smoke
```

The default searches Graph-only planning parameters on `sce-1`. Results are in `study_summary.json` and `trials/`. Use `--config` with a copy of `configs/learner_config.yaml` to customize the search, or `--resume` with the same run directory to continue a study.
