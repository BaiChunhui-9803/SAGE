"""Fuse per-episode state trajectories from multiple BKTree spaces.

Input is the JSONL produced by ``scripts/export_state_sequence_records.py``.
For each map, this script:

1. resolves every ``(source bktree_path, source state_id)`` to its stored
   normalized SC2 state,
2. inserts those states into one unified BKTree,
3. writes episode records with both original and unified state id sequences.

The source archives are not modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.structure.BKTree_sc2 import BKTree, ClusterNode, classify_new_state, get_max_cluster_id
from src.structure.custom_distance_sc2 import CustomDistance


Cluster = Tuple[int, int]
MAP_PRIMARY_THRESHOLDS = {
    "MarineMicro_MvsM_4": 0.7,
    "MarineMicro_MvsM_4_mirror": 0.7,
    "MarineMicro_MvsM_4_dist": 0.7,
    "MarineMicro_MvsM_4_dist_mirror": 0.7,
    "MarineMicro_MvsM_8": 1.0,
    "MarineMicro_MvsM_8_mirror": 1.0,
}
DEFAULT_SECONDARY_THRESHOLD = 0.5


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        with open(str(path), "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(str(path), "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception:
                continue
            if isinstance(record, dict):
                yield record


def _serialize_node(node: Optional[ClusterNode]) -> Optional[Dict[str, Any]]:
    if node is None:
        return None
    return {
        "state": node.state,
        "cluster_id": node.cluster_id,
        "children": {str(dist): _serialize_node(child) for dist, child in node.children.items()},
    }


def _deserialize_node(node_data: Optional[Mapping[str, Any]]) -> Optional[ClusterNode]:
    if node_data is None:
        return None
    if "state" not in node_data or "cluster_id" not in node_data:
        return None
    node = ClusterNode(node_data["state"], int(node_data["cluster_id"]))
    for dist_key, child_data in node_data.get("children", {}).items():
        try:
            dist_val: Any = int(dist_key)
        except ValueError:
            dist_val = float(dist_key)
        child = _deserialize_node(child_data)
        if child is not None:
            node.children[dist_val] = child
    return node


def _load_bktree(path: Path, distance_index: int, threshold: float) -> BKTree:
    tree = BKTree(CustomDistance(threshold=threshold).multi_distance, distance_index=distance_index)
    if not path.exists():
        return tree
    tree.root = _deserialize_node(_read_json(path))
    if tree.root is not None:
        tree.next_cluster_id = get_max_cluster_id(tree) + 1
    return tree


def _parse_state_node(path: Path) -> Dict[int, Cluster]:
    mapping: Dict[int, Cluster] = {}
    if not path.exists():
        return mapping
    with open(str(path), "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            cluster_text = parts[0].strip().strip("()")
            cluster_parts = [part.strip() for part in cluster_text.split(",")]
            if len(cluster_parts) != 2:
                continue
            try:
                cluster = (int(cluster_parts[0]), int(cluster_parts[1]))
                state_id = int(float(parts[1]))
            except ValueError:
                continue
            mapping[state_id] = cluster
    return mapping


def _resolve_bktree_dir(raw_path: Any, input_root: Path, cwd: Path) -> Optional[Path]:
    if raw_path in (None, ""):
        return None
    raw = Path(str(raw_path).replace("\\", "/"))
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend([cwd / raw, input_root / raw])
        if raw.parts and raw.parts[0] == "pymarl":
            candidates.append(input_root / raw)
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate.resolve()
    return None


class SourceBKTreeLookup:
    def __init__(self, bktree_dir: Path, primary_threshold: float, secondary_threshold: float):
        self.bktree_dir = bktree_dir.resolve()
        self.primary_threshold = primary_threshold
        self.secondary_threshold = secondary_threshold
        self.state_to_cluster = self._load_state_to_cluster()
        self.secondary_trees: Dict[int, BKTree] = {}
        self.state_cache: Dict[int, Optional[Dict[str, Any]]] = {}

    def _load_state_to_cluster(self) -> Dict[int, Cluster]:
        candidates = [
            self.bktree_dir / "state_node.txt",
            self.bktree_dir.parent / "graph" / "state_node.txt",
        ]
        for path in candidates:
            mapping = _parse_state_node(path)
            if mapping:
                return mapping
        return {}

    def _secondary(self, primary_id: int) -> BKTree:
        if primary_id not in self.secondary_trees:
            self.secondary_trees[primary_id] = _load_bktree(
                self.bktree_dir / f"secondary_bktree_{int(primary_id)}.json",
                distance_index=1,
                threshold=self.secondary_threshold,
            )
        return self.secondary_trees[primary_id]

    def state_for_id(self, state_id: Any) -> Optional[Dict[str, Any]]:
        try:
            sid = int(state_id)
        except (TypeError, ValueError):
            return None
        if sid in self.state_cache:
            return self.state_cache[sid]
        cluster = self.state_to_cluster.get(sid)
        if cluster is None:
            self.state_cache[sid] = None
            return None
        primary_id, secondary_id = cluster
        tree = self._secondary(primary_id)
        node = tree.find_node_by_cluster_id(int(secondary_id)) if tree.root is not None else None
        if node is None:
            self.state_cache[sid] = None
            return None
        self.state_cache[sid] = node.state
        return self.state_cache[sid]


class UnifiedBKTreeBuilder:
    def __init__(self, primary_threshold: float, secondary_threshold: float):
        self.primary_threshold = primary_threshold
        self.secondary_threshold = secondary_threshold
        self.primary = BKTree(
            CustomDistance(threshold=secondary_threshold).multi_distance,
            distance_index=0,
        )
        self.secondary: Dict[int, BKTree] = {}
        self.cluster_to_state_id: Dict[Cluster, int] = {}
        self.next_state_id = 0

    def _new_secondary_tree(self) -> BKTree:
        return BKTree(
            CustomDistance(threshold=self.secondary_threshold).multi_distance,
            distance_index=1,
        )

    def insert_or_query(self, state: Mapping[str, Any]) -> Tuple[int, Cluster]:
        if self.primary.root is None:
            self.primary.root = ClusterNode(dict(state), 1)
            self.primary.next_cluster_id = 2
            sec = self._new_secondary_tree()
            sec.root = ClusterNode(dict(state), 1)
            sec.next_cluster_id = 2
            self.secondary[1] = sec
            return self._state_id_for_cluster((1, 1)), (1, 1)

        primary_id = int(classify_new_state(dict(state), self.primary, threshold=self.primary_threshold))
        sec = self.secondary.get(primary_id)
        if sec is None:
            sec = self._new_secondary_tree()
            sec.root = ClusterNode(dict(state), 1)
            sec.next_cluster_id = 2
            self.secondary[primary_id] = sec
            return self._state_id_for_cluster((primary_id, 1)), (primary_id, 1)

        if sec.root is None:
            sec.root = ClusterNode(dict(state), 1)
            sec.next_cluster_id = 2
            return self._state_id_for_cluster((primary_id, 1)), (primary_id, 1)

        secondary_id = int(classify_new_state(dict(state), sec, threshold=self.secondary_threshold))
        return self._state_id_for_cluster((primary_id, secondary_id)), (primary_id, secondary_id)

    def _state_id_for_cluster(self, cluster: Cluster) -> int:
        if cluster not in self.cluster_to_state_id:
            self.cluster_to_state_id[cluster] = self.next_state_id
            self.next_state_id += 1
        return self.cluster_to_state_id[cluster]

    def save(self, output_dir: Path) -> None:
        bktree_dir = output_dir / "bktree"
        bktree_dir.mkdir(parents=True, exist_ok=True)
        with open(str(bktree_dir / "primary_bktree.json"), "w", encoding="utf-8") as f:
            json.dump(_serialize_node(self.primary.root), f, ensure_ascii=False, indent=2)
        for primary_id, tree in sorted(self.secondary.items()):
            if tree.root is None:
                continue
            with open(str(bktree_dir / f"secondary_bktree_{primary_id}.json"), "w", encoding="utf-8") as f:
                json.dump(_serialize_node(tree.root), f, ensure_ascii=False, indent=2)
        with open(str(bktree_dir / "state_node.txt"), "w", encoding="utf-8") as f:
            for cluster, state_id in sorted(self.cluster_to_state_id.items(), key=lambda item: item[1]):
                f.write(f"({cluster[0]}, {cluster[1]})\t{state_id}\n")
        with open(str(bktree_dir / "bktree_config.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "primary_threshold": self.primary_threshold,
                    "secondary_threshold": self.secondary_threshold,
                    "state_count": self.next_state_id,
                    "primary_count": len(self.secondary),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )


def _map_output_dir(output_root: Path, map_id: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in map_id or "unknown_map")
    return output_root / safe


def _thresholds_for_map(
    map_id: str,
    primary_override: Optional[float],
    secondary_override: Optional[float],
) -> Tuple[float, float]:
    primary = (
        float(primary_override)
        if primary_override is not None
        else MAP_PRIMARY_THRESHOLDS.get(map_id, 1.0)
    )
    secondary = (
        float(secondary_override)
        if secondary_override is not None
        else DEFAULT_SECONDARY_THRESHOLD
    )
    return primary, secondary


def _load_records(records_path: Path, map_ids: Optional[set[str]]) -> Dict[str, List[Dict[str, Any]]]:
    by_map: Dict[str, List[Dict[str, Any]]] = {}
    for record in _iter_jsonl(records_path):
        map_id = str(record.get("map_id") or "")
        if not map_id:
            continue
        if map_ids and map_id not in map_ids:
            continue
        by_map.setdefault(map_id, []).append(record)
    return by_map


def _fuse_map(
    map_id: str,
    records: List[Dict[str, Any]],
    input_root: Path,
    output_root: Path,
    primary_threshold: float,
    secondary_threshold: float,
    limit_records: int,
    anchor_first_state_to_zero: bool,
) -> Dict[str, Any]:
    cwd = Path.cwd()
    out_dir = _map_output_dir(output_root, map_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    builder = UnifiedBKTreeBuilder(primary_threshold, secondary_threshold)
    lookups: Dict[str, SourceBKTreeLookup] = {}
    source_state_map: Dict[Tuple[str, int], Dict[str, Any]] = {}
    episode_count = 0
    missing_states = 0
    reconstructed_states = 0
    total_steps = 0

    records_out = out_dir / "state_sequence_records_unified.jsonl"
    source_map_out = out_dir / "source_state_map.jsonl"
    with open(str(records_out), "w", encoding="utf-8") as f_records:
        for record in records[:limit_records or None]:
            bktree_dir = _resolve_bktree_dir(record.get("bktree_path"), input_root, cwd)
            source_key = str(bktree_dir) if bktree_dir is not None else ""
            lookup = lookups.get(source_key)
            if lookup is None and bktree_dir is not None:
                lookup = SourceBKTreeLookup(bktree_dir, primary_threshold, secondary_threshold)
                lookups[source_key] = lookup

            unified_sequence: List[int] = []
            unified_status_sequence: List[str] = []
            unified_source_sequence: List[Any] = []
            source_sequence = record.get("state_id_sequence")
            if not isinstance(source_sequence, list):
                source_sequence = []
            frame_state_sequence = record.get("frame_state_sequence")
            if isinstance(frame_state_sequence, list) and frame_state_sequence:
                state_refs = frame_state_sequence
            else:
                state_refs = [{"source_state_id": source_state_id} for source_state_id in source_sequence]

            for state_ref in state_refs:
                if not isinstance(state_ref, Mapping):
                    missing_states += 1
                    continue
                source_state_id = state_ref.get("source_state_id")
                state_key = state_ref.get("state_key", source_state_id)
                state = None
                source_status = ""

                if source_state_id is not None and lookup is not None:
                    old_id = None
                    map_key = None
                    try:
                        old_id = int(source_state_id)
                        map_key = (source_key, old_id)
                        cached = source_state_map.get(map_key)
                        if cached is not None:
                            unified_id = int(cached["unified_state_id"])
                            if anchor_first_state_to_zero and not unified_sequence:
                                unified_id = 0
                            unified_sequence.append(unified_id)
                            unified_status_sequence.append("source_bktree")
                            unified_source_sequence.append(source_state_id)
                            continue
                    except (TypeError, ValueError):
                        pass

                    state = lookup.state_for_id(source_state_id)
                    source_status = "source_bktree"
                    if state is not None:
                        unified_id, unified_cluster = builder.insert_or_query(state)
                        if anchor_first_state_to_zero and not unified_sequence:
                            unified_id = 0
                            source_status = "source_bktree_start_anchor"
                        unified_sequence.append(unified_id)
                        unified_status_sequence.append(source_status)
                        unified_source_sequence.append(source_state_id)
                        if map_key is not None and old_id is not None:
                            source_state_map[map_key] = {
                                "map_id": map_id,
                                "source_bktree_path": source_key,
                                "source_state_id": old_id,
                                "unified_state_id": unified_id,
                                "unified_cluster": list(unified_cluster),
                                "source_kind": source_status,
                            }
                        continue

                reconstructed_state = state_ref.get("reconstructed_state")
                if isinstance(reconstructed_state, Mapping):
                    unified_id, unified_cluster = builder.insert_or_query(reconstructed_state)
                    source_status = "ood_reconstructed"
                    if anchor_first_state_to_zero and not unified_sequence:
                        unified_id = 0
                        source_status = "ood_reconstructed_start_anchor"
                    unified_sequence.append(unified_id)
                    unified_status_sequence.append(source_status)
                    unified_source_sequence.append(state_key)
                    reconstructed_states += 1
                    source_state_map[(source_key, -reconstructed_states)] = {
                        "map_id": map_id,
                        "source_bktree_path": source_key,
                        "source_state_id": None,
                        "source_state_key": state_key,
                        "frame_index": state_ref.get("frame_index"),
                        "unified_state_id": unified_id,
                        "unified_cluster": list(unified_cluster),
                        "source_kind": source_status,
                    }
                    continue

                missing_states += 1

            out_record = dict(record)
            out_record["unified_bktree_path"] = str((out_dir / "bktree").resolve())
            out_record["unified_state_id_sequence"] = unified_sequence
            out_record["unified_state_id_sequence_length"] = len(unified_sequence)
            out_record["unified_state_source_sequence"] = unified_source_sequence
            out_record["unified_state_status_sequence"] = unified_status_sequence
            out_record["unified_reconstructed_state_count"] = sum(
                1 for status in unified_status_sequence if status == "ood_reconstructed"
            )
            out_record["unified_missing_state_count"] = max(0, len(state_refs) - len(unified_sequence))
            f_records.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            episode_count += 1
            total_steps += len(unified_sequence)

    with open(str(source_map_out), "w", encoding="utf-8") as f_map:
        for item in source_state_map.values():
            f_map.write(json.dumps(item, ensure_ascii=False) + "\n")

    builder.save(out_dir)
    summary = {
        "map_id": map_id,
        "episodes": episode_count,
        "unified_states": builder.next_state_id,
        "source_bktrees": len([key for key in lookups if key]),
        "anchor_first_state_to_zero": bool(anchor_first_state_to_zero),
        "total_unified_steps": total_steps,
        "missing_states": missing_states,
        "reconstructed_states": reconstructed_states,
        "output_dir": str(out_dir),
        "records_file": str(records_out),
        "source_state_map_file": str(source_map_out),
    }
    with open(str(out_dir / "fusion_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse state sequences into one BKTree per map")
    parser.add_argument(
        "--records",
        default=str(Path("output") / "learner_results" / "all_data" / "state_sequence_records.jsonl"),
        help="JSONL produced by export_state_sequence_records.py",
    )
    parser.add_argument(
        "--input-root",
        default=str(Path("output") / "learner_results" / "all_data"),
        help="Root used to resolve relative PyMARL bktree paths",
    )
    parser.add_argument(
        "--output-root",
        default=str(Path("output") / "learner_results" / "all_data" / "fused_bktree"),
        help="Output root; one subdirectory per map_id",
    )
    parser.add_argument("--map-id", action="append", default=[], help="Limit to a map_id; may be repeated")
    parser.add_argument(
        "--primary-threshold",
        type=float,
        default=None,
        help="Override primary threshold; default follows the source map family",
    )
    parser.add_argument(
        "--secondary-threshold",
        type=float,
        default=None,
        help="Override secondary threshold; default is 0.5",
    )
    parser.add_argument("--limit-records", type=int, default=0, help="Debug limit per map; 0 means all")
    parser.add_argument(
        "--anchor-first-state-to-zero",
        action="store_true",
        help="Visualization-only option: force each episode's first resolved state to unified state 0.",
    )
    args = parser.parse_args()

    records_path = Path(args.records)
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    map_ids = set(args.map_id) if args.map_id else None
    by_map = _load_records(records_path, map_ids)
    output_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for map_id, records in sorted(by_map.items()):
        primary_threshold, secondary_threshold = _thresholds_for_map(
            map_id, args.primary_threshold, args.secondary_threshold
        )
        summaries.append(
            _fuse_map(
                map_id,
                records,
                input_root,
                output_root,
                primary_threshold,
                secondary_threshold,
                int(args.limit_records),
                bool(args.anchor_first_state_to_zero),
            )
        )

    overall = {
        "records": str(records_path),
        "output_root": str(output_root),
        "anchor_first_state_to_zero": bool(args.anchor_first_state_to_zero),
        "maps": summaries,
    }
    with open(str(output_root / "fusion_summary.json"), "w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)
    print(json.dumps(overall, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
