#!/usr/bin/env python
"""
Deploy augmented ETG data to system standard paths.

Usage:
    python scripts/deploy_augmented.py \
        --collected-dir output/collected_data/ep0-3999_r10_p0.7_s0.5 \
        --etg-dir cache/experience_transition_graph/MarineMicro_MvsM_4_augmented \
        --map-id MarineMicro_MvsM_4 --data-id augmented_1

Copies:
  1. BKTree JSON  -> datas/data_for_transit/bktree_augmented/
  2. state_node   -> datas/data_for_transit/graph/state_node_augmented.txt
  3. dist_matrix  -> cache/npy/state_distance_matrix_{map_id}_{data_id}.npy
  4. sparse index -> cache/npy/state_sparse_neighbors_{map_id}_{data_id}.pkl
"""

import sys
import os
import shutil
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import ROOT_DIR

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BASE_DIR = ROOT_DIR


def deploy_bktree(collected_dir: Path, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)

    primary_src = collected_dir / "primary_bktree.json"
    if primary_src.exists():
        shutil.copy2(primary_src, dest_dir / "primary_bktree.json")
        logger.info(f"  primary_bktree.json -> {dest_dir}")
    else:
        logger.warning(f"  primary_bktree.json not found in {collected_dir}")

    count = 0
    for src in sorted(collected_dir.glob("secondary_bktree_*.json")):
        shutil.copy2(src, dest_dir / src.name)
        count += 1
    logger.info(f"  {count} secondary_bktree_*.json -> {dest_dir}")


def deploy_state_node(etg_dir: Path, dest_path: Path):
    src = etg_dir / "state_node.txt"
    if src.exists():
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest_path)
        logger.info(f"  state_node.txt -> {dest_path}")
    else:
        logger.warning(f"  state_node.txt not found in {etg_dir}")


def deploy_dist_matrix(etg_dir: Path, dest_path: Path):
    src = etg_dir / "npy" / "state_distance_matrix.npy"
    if not src.exists():
        logger.warning(f"  state_distance_matrix.npy not found in {etg_dir / 'npy'}")
        return
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest_path)
    logger.info(f"  state_distance_matrix.npy -> {dest_path}")


def deploy_sparse_neighbors(etg_dir: Path, dest_path: Path):
    candidates = [
        etg_dir / "sparse_neighbors.pkl",
        etg_dir / "npy" / "sparse_neighbors.pkl",
    ]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        logger.warning(f"  sparse_neighbors.pkl not found in {etg_dir}")
        return
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest_path)
    logger.info(f"  sparse_neighbors.pkl -> {dest_path}")


def deploy_data_dir(collected_dir: Path, etg_dir: Path, map_id: str, data_id: str):
    data_dir = BASE_DIR / "data" / map_id / data_id
    bktree_dest = data_dir / "bktree"
    state_node_dest = data_dir / "graph" / "state_node.txt"

    logger.info(f"Deploying data_dir structure: {data_dir}")
    deploy_bktree(collected_dir, bktree_dest)
    deploy_state_node(etg_dir, state_node_dest)


def main():
    parser = argparse.ArgumentParser(
        description="Deploy augmented ETG data to standard paths"
    )
    parser.add_argument(
        "--collected-dir",
        required=True,
        help="Collected data directory containing BKTree JSON files",
    )
    parser.add_argument(
        "--etg-dir",
        required=True,
        help="ETG build output directory (contains state_node.txt, npy/)",
    )
    parser.add_argument(
        "--map-id",
        default="MarineMicro_MvsM_4",
        help="Map identifier for dist_matrix naming (default: MarineMicro_MvsM_4)",
    )
    parser.add_argument(
        "--data-id",
        default="augmented_1",
        help="Data identifier for dist_matrix naming (default: augmented_1)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without copying files",
    )
    args = parser.parse_args()

    collected_dir = Path(args.collected_dir)
    etg_dir = Path(args.etg_dir)

    if not collected_dir.exists():
        logger.error(f"Collected dir not found: {collected_dir}")
        sys.exit(1)
    if not etg_dir.exists():
        logger.error(f"ETG dir not found: {etg_dir}")
        sys.exit(1)

    bktree_dest = BASE_DIR / "datas" / "data_for_transit" / "bktree_augmented"
    state_node_dest = (
        BASE_DIR / "datas" / "data_for_transit" / "graph" / "state_node_augmented.txt"
    )
    dist_matrix_dest = (
        BASE_DIR
        / "cache"
        / "npy"
        / f"state_distance_matrix_{args.map_id}_{args.data_id}.npy"
    )
    sparse_neighbors_dest = (
        BASE_DIR
        / "cache"
        / "npy"
        / f"state_sparse_neighbors_{args.map_id}_{args.data_id}.pkl"
    )

    logger.info("=" * 50)
    logger.info("DEPLOY AUGMENTED ETG")
    logger.info(f"  Collected dir : {collected_dir}")
    logger.info(f"  ETG dir        : {etg_dir}")
    logger.info("=" * 50)

    if args.dry_run:
        logger.info("[DRY RUN] No files will be modified")
        logger.info(f"  BKTree dest   : {bktree_dest}")
        logger.info(f"  State node    : {state_node_dest}")
        logger.info(f"  Dist matrix   : {dist_matrix_dest}")
        logger.info(f"  Sparse index  : {sparse_neighbors_dest}")
        logger.info(f"  Data dir      : data/{args.map_id}/{args.data_id}/")
        return

    logger.info("Deploying BKTree...")
    deploy_bktree(collected_dir, bktree_dest)

    logger.info("Deploying state_node.txt...")
    deploy_state_node(etg_dir, state_node_dest)

    logger.info("Deploying distance matrix...")
    deploy_dist_matrix(etg_dir, dist_matrix_dest)

    logger.info("Deploying sparse neighbor index...")
    deploy_sparse_neighbors(etg_dir, sparse_neighbors_dest)

    logger.info("Deploying data_dir structure...")
    deploy_data_dir(collected_dir, etg_dir, args.map_id, args.data_id)

    logger.info("=" * 50)
    logger.info("DEPLOY COMPLETE")
    logger.info(f"  data_dir for etg_catalog.yaml: data/{args.map_id}/{args.data_id}")


if __name__ == "__main__":
    main()
