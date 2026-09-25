# SAGE scenario maps

The live launcher uses `src/sc2env/maps.py` to pass bundled map bytes directly to standard PySC2. No custom registration in site-packages or copying into the SC2 installation is required. Install StarCraft II separately and set `SC2PATH` if PySC2 cannot locate it.

| Key | Bundled file |
|---|---|
| sce-1 | local_enemy_test_1.SC2Map |
| sce-1m | local_enemy_test_1_mirror.SC2Map |
| sce-2 | MarineMicro_MvsM_4_dist.SC2Map |
| sce-2m | MarineMicro_MvsM_4_dist_mirror.SC2Map |
| sce-3 | MarineMicro_MvsM_8_far.SC2Map |
| sce-3m | MarineMicro_MvsM_8_far_mirror.SC2Map |

Fetch Git LFS objects before launching. Map identity is covered by the paper-asset provenance manifest. The StarCraft II client is not redistributed; its applicable license and the maps' applicable terms continue to apply.
