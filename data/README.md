# Runtime state assets and historical examples

The six `MarineMicro_MvsM_*/augmented_1/` directories contain the BK-Trees and state-ID maps used by the paper SAGE configurations. Pair each with the same map's `cache/experience_transition_graph/*_augmented/` graph, transitions and distances. Do not mix state IDs between scenarios or datasets. Fresh evaluations write new state records under their output directory.

| Scenario | Map directory | Older example ID |
|---|---|---|
| sce-1 | MarineMicro_MvsM_4 | 6 |
| sce-1m | MarineMicro_MvsM_4_mirror | 3 |
| sce-2 | MarineMicro_MvsM_4_dist | 1 |
| sce-2m | MarineMicro_MvsM_4_dist_mirror | 3 |
| sce-3 | MarineMicro_MvsM_8 | 1 |
| sce-3m | MarineMicro_MvsM_8_mirror | 1 |

Older examples contain action/state trajectories, outcomes and BK-Trees for graph inspection. Use the `augmented_1/` assets with the paper evaluation configurations.

New experiment logs, distance extraction caches and author-side intermediate calculations remain outside the tracked inputs.
