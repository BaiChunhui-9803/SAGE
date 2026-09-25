# AIIDE 2026 SAGE paper assets

Run `python scripts/reproduce_paper.py --figure all` from the source checkout. Outputs go to `output/paper_reproduction/`, outside these frozen inputs.

| Paper item | Inputs here | Operation |
|---|---|---|
| Figures 1–2 | `reference/fig1_architecture.png`, `fig2_sce*.pdf` | Static camera-ready illustrations |
| Figure 3 | `data/figure_03_parameter_correlation/`: trials, studies, correlations | Recompute 300 correlations; `--figure 3` |
| Figure 4 | `data/figure_04_backup_switch/`: switching comparisons and selected groups | Recompute 124 filtered rows; `--figure 4` |
| Figure 5a | `data/figure_05_06_gate_exploration/selection/`, `raw/evaluations/` | Frozen episode IDs; `--figure 5a` |
| Figure 5b | Same logs and `branch_case/` | sce-2 Graph-only episode 272 / Full episode 23; `--figure 5b` |
| Figure 6 | Same logs, selection and `analysis_ready/` | Rebuild state aggregates; `--figure 6` |
| Figure 7 | `data/figure_07_state_diagnostics/upstream_sce2/` | Redraw archived MDS coordinates and node/edge tables; `--figure 7` |
| Table 1, SAGE columns | Selected SAGE logs | Recompute win %, mean score and sample SD |
| Tables 1–2, external methods | `data/external_methods/paper_reported_results.csv` | Results reported in Tables 1–2 |
| Table 3 | `reference/camera_ready_table_03.csv` | Real-time decision results reported in Table 3 |

`raw/evaluations/<run>/` retains 300 episodes per run, frame diagnostics, evaluation summaries, fixed parameters, and the evaluation-start model for Full SAGE. Twelve runs cover the two methods and six scenarios. Historical names `ETG-only` / `etg-only` mean Graph-only SAGE; `Synergy` means Full SAGE.

Selected lists contain 100 episodes per method/scenario. Their `(experiment_id, episode_id)` membership defines the sample. `evidence.py` reads the selected scores, outcomes, frame counts and state aggregates from the logs.

`reference/` contains the final figures and table values. `data/external_methods/` stores results reported for the comparison methods. Figure 7 also uses archived plot coordinates, sequences and state-transition tables.

`provenance.json` records hashes and upstream identity. Source paths in historical metadata are not runtime dependencies. Adapted source files retain their original `source_sha256` and a separate release hash. Active historical plotting helpers are in `scripts/legacy/`; call the portable CLI, which supplies repository-relative inputs, rather than their old standalone machine-specific commands.

See [ARTIFACT.md](../../ARTIFACT.md) for installation, figure generation and experiment commands.
