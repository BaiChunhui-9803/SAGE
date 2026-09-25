"""Portable CLI for the frozen AIIDE 2026 SAGE evidence package."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paper_assets.aiide26_sage.evidence import PACK, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Verify SAGE measurements, paper-reference transcriptions and figure inputs")
    parser.add_argument("--from-raw", action="store_true", help="Rebuild selected episode/frame/state tables from JSONL")
    parser.add_argument("--check-hashes", action="store_true", help="Verify every file listed in provenance.json")
    parser.add_argument("--figure", choices=["3", "4", "5a", "5b", "6", "7", "all"])
    parser.add_argument("--output", type=Path, default=ROOT / "output/paper_reproduction")
    args = parser.parse_args()
    output = args.output.resolve()
    if output == PACK or PACK in output.parents:
        parser.error("Outputs must be outside the frozen paper_assets package")
    if args.verify or args.from_raw or args.check_hashes or not args.figure:
        report = verify(output, from_raw=args.from_raw, hashes=args.check_hashes)
        print(json.dumps(report, indent=2))
        if not report["passed"]:
            return 1
    if args.figure:
        from paper_assets.aiide26_sage.plots import reproduce
        reproduce(args.figure, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
