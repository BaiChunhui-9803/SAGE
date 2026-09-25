# SAGE

**Switch-Aware Graph Planning Integrated with Gated Exploration on Unseen States for RTS Micromanagement** — AIIDE 2026.

[Artifact and experiment guide](ARTIFACT.md) · [Paper figure/table inputs](paper_assets/aiide26_sage/README.md)

SAGE stores offline experience in an **Experience Transition Graph (ETG)**. **Graph-only SAGE** uses switch-aware graph planning. **Full SAGE** adds a local action-value model and evidence gates for uncertain or unseen observations. This artifact provides both implementations, frozen runtime assets, archived SAGE evaluations, and a local Streamlit explorer.

External methods are supplied as explicitly attributed **paper-reported reference values**. These values are stored alongside the SAGE evaluation data in the paper assets directory.

## Quick start: paper evidence and Web

Use Python **3.12**. This workflow runs on CPU and needs neither StarCraft II nor PyTorch. Use a **source checkout with all Git LFS objects**; a wheel or `pip install .` alone is not the artifact.

```bash
git lfs install
git clone https://github.com/BaiChunhui-9803/SAGE.git
cd SAGE
# Check out the artifact revision supplied with the submission, then:
git lfs pull
python -m venv .venv-artifact
# Windows: .venv-artifact\Scripts\activate
# Linux/macOS: source .venv-artifact/bin/activate
python -m pip install -r requirements-artifact.txt
python scripts/reproduce_paper.py --figure all
python -m streamlit run scripts/visualize_etg_web_en.py
```

The plotting command writes Figures 3–7 to `output/paper_reproduction/figures/`. The explorer provides paper tables and figures, a directory browser for their inputs, graph inspection, beam planning, graph rollouts, and experiment commands.

Choose **Experience Transition Graph** to inspect each of the six scenarios separately. The default view expands one hop with up to six neighbors per state; rendering is capped at 24 nodes and 60 edges. Use the controls to explore a larger neighborhood.

## Fresh Graph-only and Full SAGE experiments

Install the SC2 environment separately with **Python 3.8.10** and `requirements-sc2.txt`; see [the complete instructions](ARTIFACT.md#fresh-sc2-evaluations). The launcher reads the six maps directly from `assets/maps/`, without editing PySC2 or registering custom maps in its package.

```bash
# In the SC2 environment, verify two episodes and the reset between them:
python scripts/run_paper_sage.py --scenario sce-1 --method graph-only --episodes 2 --run
python scripts/run_paper_sage.py --scenario sce-1 --method full --episodes 2 --run
# Omit --run to inspect the prepared settings without starting SC2.
```

Use `--episodes 300` for a new full-length evaluation. Scenario keys are `sce-1`, `sce-1m`, `sce-2`, `sce-2m`, `sce-3`, and `sce-3m`. The wrapper preserves recorded settings, normalizes historical routing tags, and copies Full SAGE's initial model before online updates. Each run has its own logs, parameters, and `completion.json` under `output/paper_live/`.

## Layout and resources

```text
paper_assets/aiide26_sage/     SAGE logs, selected IDs, models, plot inputs and provenance
  data/external_methods/     External-method paper references with attribution
  reference/                 Camera-ready figures and table transcriptions
scripts/reproduce_paper.py    Offline evidence checks and figure redraws
scripts/run_paper_sage.py     Fresh Graph-only / Full SAGE evaluations
scripts/visualize_etg_web_en.py Streamlit explorer entry
scripts/etg_web/              Shared graph tools and presentation modules
src/decision/                ETG, beam search, switch-aware rollout and local models
src/sc2env/                  State/action adapter, SAGE agent, maps and optional Web API
src/data/, src/structure/    State abstraction, BK-Trees and distance support
cache/experience_transition_graph/ Frozen graphs and distance assets
data/                        Augmented BK-Trees/state maps and older trajectory examples
assets/maps/                 Six scenario maps
tests/                       Evidence, launcher, asset and Web checks
```

The curated manifest covers about **1.98 GB**. Four dense distance archives expand once to about **2.26 GB** of ignored local cache; 8v8 scenarios use sparse indices. Reserve at least **8 GB free disk space**, plus SC2 and Python environments, and preferably **8 GB RAM**. Offline checks take a few minutes depending on CPU/storage.

Project code uses the [MIT license](LICENSE). External software and maps retain their applicable terms. Cite the accepted SAGE paper and the exact artifact revision used.
