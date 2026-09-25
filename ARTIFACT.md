# SAGE artifact and experiment guide

The artifact is a source repository with Git LFS data, runnable SAGE implementations, a local Streamlit explorer, and this guide. A hosted website is not required. All commands run from the repository root. The submitted camera-ready remains unchanged.

## Workflows

| Workflow | Environment | Entry |
|---|---|---|
| Paper tables, figures and asset browsing | Python 3.12, CPU | Streamlit explorer |
| Figures 3–7 from supplied inputs | Python 3.12, CPU | `scripts/reproduce_paper.py` |
| Graph inspection, beam search and graph rollouts | Python 3.12, CPU | Explorer graph/planning pages |
| Graph-only / Full SAGE evaluations | Python 3.8.10, StarCraft II | `scripts/run_paper_sage.py` |
| Bayesian planning-parameter search | SC2 environment plus optional tools | `scripts/parameter_learner.py` |

The six scenario graphs, distance assets and initial models are supplied with the repository. Tables 1–3 are available under `paper_assets/aiide26_sage/reference/`; SAGE evaluation logs and figure inputs are organized under `raw/` and `data/`.

## Installation and basic assets

Use the revision identified in the artifact submission, not an unspecified future branch tip:

```bash
git lfs install
git clone https://github.com/BaiChunhui-9803/SAGE.git
cd SAGE
# Check out the artifact revision specified in the submission.
git lfs pull
git lfs fsck
```

An LFS pointer is not usable graph data. The supported delivery includes both the Git revision and its LFS objects, or an equivalent complete source-and-data archive. A wheel is insufficient. The repository does not redistribute the SC2 client.

| File | Purpose |
|---|---|
| `requirements-artifact.txt` | Pinned direct dependencies for evidence, plots and Web |
| `requirements.txt` | Alias of the offline installation |
| `requirements-artifact-lock.txt` | Full resolved Windows/Python 3.12 environment, including pytest |
| `requirements-sc2.txt` | Minimal standard-PySC2 live evaluation environment |
| `requirements-sc2-tools.txt` | Optional FastAPI/Optuna tools for new parameter searches |
| `requirements-sc2-lock.txt` | Resolved Windows/Python 3.8 environment including optional tools |
| `paper_assets/aiide26_sage/provenance.json` | Curated assets, SHA-256 and upstream identity |

The six scenarios each need matching augmented ETG, transition counts, state-ID map, primary/secondary BK-Trees and distance assets. Full SAGE also needs its scenario-specific evaluation-start model. Four 4v4 scenarios use compressed dense distances; two 8v8 scenarios use sparse indices. `assets/maps/` contains all six maps. Never combine state IDs from different datasets.

Reserve at least 8 GB free disk space plus the game and Python environments. Four dense matrices expand once to about 2.26 GB of ignored local NPY cache; interrupted extraction is retryable. At least 8 GB RAM is recommended. The frozen manifest covers about 1.98 GB.

## Paper figures and graph assets

Install Python 3.12 (validated with 3.12.12), create a virtual environment, and activate it:

```bash
python -m venv .venv-artifact
# Windows: .venv-artifact\Scripts\activate
# Linux/macOS: source .venv-artifact/bin/activate
python -m pip install -r requirements-artifact.txt
python scripts/reproduce_paper.py --figure all
```

The plotting command reads the supplied figure inputs and writes PDF/PNG files under `output/paper_reproduction/figures/`. `paper_assets/aiide26_sage/README.md` maps each figure and table to its input directory and script.

Individual redraws use the same command with `--figure 3`, `4`, `5a`, `5b`, `6`, or `7`. For example:

```bash
python scripts/reproduce_paper.py --figure 5a
```

Outputs are kept outside frozen inputs. Avoid using old plotting scripts as standalone commands: the portable wrapper supplies their current paths and layout settings.

## Web walkthrough

```bash
python -m streamlit run scripts/visualize_etg_web_en.py
```

Open the local address printed by Streamlit. Stop a server with Ctrl+C before reusing its port, or use `--server.port` to choose another port.

1. **Paper results:** select a table or figure. Figures offer the paper original and a version generated from the supplied inputs. Run the plotting CLI to generate the latter. Download the displayed table or figure directly.
2. **Paper assets:** browse the directory tree under `paper_assets/aiide26_sage/`, select a folder, and inspect its files. The page describes the roles of the logs, plot inputs, reference figures and scripts.
3. **Experience Transition Graph:** select one of the six scenarios and an initial abstract state. Each scenario loads its own stored graph. The default expands one outgoing hop and up to six neighbors per state, with at most 24 nodes and 60 edges. Neighbors are ranked by recorded transition count. Layout physics is disabled; drag or zoom to inspect. Controls permit up to three hops, 12 neighbors per state and 100 nodes, with the 60-edge cap retained. Display filters keep the original transition probabilities.
4. **Beam-search planning:** choose recorded Graph-only or Full parameters, optionally change width/depth/visits, then Run beam search. Inspect the recommended graph action and candidate paths. Adjust filters if no admissible action is found.
5. **Graph rollout:** choose single-step replanning or multi-step plan following, backup switching, a next-state rule and a rollout length. Run graph rollout and inspect/download the trace. Graph sampling uses seed 42.
6. **Fresh experiments:** obtain the preparation/run commands for the selected method and scenario.

The explorer uses the SAGE graph, planning and rollout modules. Live gated exploration runs through the SC2 launcher described below.

## Fresh SC2 evaluations

The validated live platform is **Windows with Python 3.8.10**, standard PyPI PySC2 4.0.0, and a locally installed SC2 client (tested executable build 97425). Install the client separately. `SC2PATH` is optional if PySC2 finds the normal installation; otherwise set it to your own client directory.

Create a separate environment using Python 3.8.10:

```bash
# Windows Python launcher; use your Python 3.8.10 executable if py is unavailable.
py -3.8 -m venv .venv-sc2
.venv-sc2\Scripts\activate
python -m pip install -r requirements-sc2.txt
```

Do not install the Python 3.12 project metadata into this environment. Run the source launchers directly. `src/sc2env/maps.py` passes bundled map bytes to PySC2, so no editing of site-packages or custom-map installation is needed. Evaluation flags are maintained by the SAGE agent; stock PySC2 TimeStep/BaseAgent APIs are used.

First prepare configurations without starting the game (also works in the offline environment):

```bash
python scripts/run_paper_sage.py --scenario sce-1 --method graph-only --episodes 2
python scripts/run_paper_sage.py --scenario sce-1 --method full --episodes 2
```

Then run in the SC2 environment:

```bash
python scripts/run_paper_sage.py --scenario sce-1 --method graph-only --episodes 2 --run
python scripts/run_paper_sage.py --scenario sce-1 --method full --episodes 2 --run
```

Each invocation creates a unique directory under `output/paper_live/`. To name it, use `--output output/paper_live/my_run`; existing directories are rejected. A prepared directory is an inspection record; invoking again prepares a separate run. To evaluate all six scenarios, substitute the scenario key in the same two commands:

| Scenario | Scene | Runtime data directory |
|---|---|---|
| sce-1 | 4v4 | `data/MarineMicro_MvsM_4/augmented_1` |
| sce-1m | Mirrored 4v4 | `data/MarineMicro_MvsM_4_mirror/augmented_1` |
| sce-2 | Shifted 4v4 | `data/MarineMicro_MvsM_4_dist/augmented_1` |
| sce-2m | Mirrored shifted 4v4 | `data/MarineMicro_MvsM_4_dist_mirror/augmented_1` |
| sce-3 | 8v8 | `data/MarineMicro_MvsM_8/augmented_1` |
| sce-3m | Mirrored 8v8 | `data/MarineMicro_MvsM_8_mirror/augmented_1` |

For a new full evaluation, use `--episodes 300`. Two episodes exercise startup and the between-episode reset. The wrapper reads the complete recorded startup settings for the selected method/scenario; it does not approximate Full by toggling one Graph-only flag. Changes are output paths, copied model paths, historical `kg_*` to `etg_*` routing names, and the requested episode count.

Validate each run:

- `completion.json`: `passed: true`, process exit 0 and exactly the requested records.
- `episodes.jsonl`: one record per completed episode with score, result and frame diagnostics.
- `progress.json`: matching completed/target counts and branch counters.
- `startup_beam_params.json` and `run_manifest.json`: exact parameters, source evaluation and command.
- Full SAGE's `action_tuning_model.pkl`: a run-local copy; the frozen initial model remains unchanged.

New trajectories and scores depend on game randomness. Each evaluation writes to its own output directory. If interrupted, retain the incomplete directory and start a new run; the launcher creates a separate run directory on each invocation.

## New parameter-search experiments

In the SC2 environment install the optional tools, then run a small study in a new directory:

```bash
python -m pip install -r requirements-sc2-tools.txt
python scripts/parameter_learner.py --trials 2 --episodes 2 --run_dir output/search_smoke
```

The default configuration searches Graph-only planning parameters on sce-1. It starts one game and a local API, changes trial parameters, and pauses after each requested episode count. Verify `study_summary.json` reports two completed trials and each `user_attrs.num_episodes` equals 2; raw records are under `trials/trial_*/`. Use `--trials 50 --episodes 100` for a larger new study. `--resume` with the same `--run_dir` resumes its Optuna study.

Edit a copy of `configs/learner_config.yaml` and pass `--config` to change the search space, objective, scenario thresholds or optional exploration phases. For other scenarios, set `game.map_key`, `game.etg_file`, `game.data_dir`, and both `bktree` thresholds consistently. The prepared paper configuration gives each scenario's recorded thresholds. Do not point a mutable training model at the frozen paper model. Use `run_paper_sage.py --method full` to evaluate Full SAGE with its supplied initial model and recorded configuration.

## Data organization

`paper_assets/aiide26_sage/raw/evaluations/` contains SAGE episode and frame logs, recorded parameters and Full SAGE initial models. The selection lists in `data/figure_05_06_gate_exploration/selection/` specify the 100 episode IDs used for each method/scenario. SAGE score standard deviations use `ddof=1`.

External-method results reported in the paper are stored in `data/external_methods/paper_reported_results.csv`. The `reference/` directory contains the paper's figures and table values. Figure 7 reads MDS coordinates, selected sequences and state-transition tables from `data/figure_07_state_diagnostics/`.

`provenance.json` records asset paths, file identities and source locations. The plotting wrapper resolves inputs relative to this checkout. Generated figures and new experiment outputs are written under `output/`.

## Development checks

In the offline environment:

```bash
python -m pip install pytest==8.3.5
python -m pytest tests -q
```

The source checkout and its Git LFS objects form the distributed artifact. Environment folders, runtime output and local author notes are excluded by `.gitignore`.
