# SAGE

**Switch-Aware Graph Planning Integrated with Gated Exploration on Unseen States for RTS Micromanagement**

SAGE is a research implementation for interpretable real-time strategy (RTS) micromanagement in StarCraft II. It converts offline trajectories into an Experience Transition Graph (ETG), reuses reliable graph evidence through switch-aware beam planning, and invokes gated local exploration only when an observation is uncertain or unseen.

This implementation accompanies the AIIDE 2026 submission *SAGE: Switch-Aware Graph Planning Integrated with Gated Exploration on Unseen States for RTS Micromanagement*.

## Highlights

- **Explicit experience graph**: abstracts trajectories with a cluster-centric BK-Tree and stores state-action transition evidence in an ETG.
- **Switch-aware planning**: performs inter-track beam search and can switch to compatible historical states when exact reuse is not appropriate.
- **Gated exploration**: keeps online evidence separate from the base graph and permits UCB-style action correction only after sufficient local support.
- **Reproducible scenarios**: includes the six Marine micromanagement maps and lightweight trajectory/ETG artifacts used by the supplied configurations.
- **Interactive inspection**: provides a Streamlit interface for graph, planning, rollout, data, live-game, and optimisation views.

## Repository layout

```text
SAGE/
├── assets/maps/              # Six SC2 scenario maps used in the paper
├── configs/                  # Hydra, ETG catalogue, and learner settings
├── data/                     # Lightweight baseline data for all six scenarios
├── cache/knowledge_graph/    # Corresponding baseline ETG artifacts
├── scripts/                  # Collection, ETG construction, evaluation, and UI entry points
├── src/                      # SAGE algorithms, SC2 environment, and utilities
├── ARCHITECTURE.md           # Module and data-flow reference
└── requirements.txt          # Runtime dependencies
```

Large raw collections, generated distance matrices, optimisation runs, plots, and intermediate paper material are intentionally excluded. They are not required to inspect the method or run the included baseline artifacts.

## Setup

SAGE has been developed for Python 3.8+ and StarCraft II with PySC2.

```bash
git clone https://github.com/BaiChunhui-9803/SAGE.git
cd SAGE

python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
pip install -e ".[live,sc2]"
```

The live SC2 environment also requires a local StarCraft II installation. Copy the supplied maps from `assets/maps/` into the installation's `Maps/` directory (or a subdirectory recognised by your PySC2 setup). See [assets/maps/README.md](assets/maps/README.md) for the scenario mapping.

## Quick start

Start the interactive explorer:

```bash
streamlit run scripts/visualize_kg_web.py
```

The release configuration defaults to the included `sce-1` (4v4 Marine) baseline. To run a live game with another included scenario, choose one of `sce-1`, `sce-1m`, `sce-2`, `sce-2m`, `sce-3`, or `sce-3m`:

```bash
python scripts/run_live_game.py --mode all --map_key sce-1 \
  --kg_file MarineMicro_MvsM_4/kg_simple.pkl \
  --data_dir data/MarineMicro_MvsM_4/6
```

Build an ETG from your own collected trajectories:

```bash
python scripts/build_from_collected.py \
  --input output/collected_data/<run> \
  --bktree-dir output/collected_data/<run> \
  --output-dir cache/knowledge_graph/<name>
```

## Included scenarios

| Key | StarCraft II map | Units | Variant |
| --- | --- | ---: | --- |
| `sce-1` | `local_enemy_test_1` | 4 vs 4 | standard |
| `sce-1m` | `local_enemy_test_1_mirror` | 4 vs 4 | mirrored |
| `sce-2` | `MarineMicro_MvsM_4_dist` | 4 vs 4 | distance-shifted |
| `sce-2m` | `MarineMicro_MvsM_4_dist_mirror` | 4 vs 4 | mirrored, distance-shifted |
| `sce-3` | `MarineMicro_MvsM_8_far` | 8 vs 8 | larger, far-start |
| `sce-3m` | `MarineMicro_MvsM_8_far_mirror` | 8 vs 8 | mirrored, far-start |

The map constants are defined in `src/sc2env/config.py`; the ETG catalogue is in `configs/kg_catalog.yaml`.

## Reproducibility notes

- The included `data/` and `cache/knowledge_graph/` directories are compact baseline artifacts, suitable for code inspection and end-to-end examples.
- Full-scale training/evaluation outputs are excluded from Git because they are generated artifacts and substantially exceed practical repository size.
- The repository does not redistribute StarCraft II itself. Please ensure that use of the bundled scenario maps complies with the game license and your institution's policies.

## Citation

The accompanying manuscript is under anonymous review. A formal citation will be added after the paper is publicly available. Until then, please cite this repository by its URL and state the commit hash used in your experiments.

## License

This project is released under the [MIT License](LICENSE).
