"""List or kill orphan PredictionRTS multiprocessing child processes.

This is intended for Windows runs where SC2 launch fails with WinError 1455
because previous run_live_game child processes kept large etg/SC2 memory alive.
By default it only lists candidates. Pass --kill to terminate them.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys


def _iter_candidates():
    try:
        import psutil
    except Exception as exc:
        print(f"psutil is required: {exc}")
        return []

    current_pid = os.getpid()
    candidates = []
    for proc in psutil.process_iter(["pid", "name", "cmdline", "ppid", "memory_info"]):
        try:
            pid = int(proc.info.get("pid") or 0)
            if pid == current_pid:
                continue
            name = (proc.info.get("name") or "").lower()
            cmdline = " ".join(proc.info.get("cmdline") or [])
            if "python" not in name:
                continue
            if "PredictionRTS" not in cmdline or "multiprocessing.spawn" not in cmdline:
                continue
            if "streamlit" in cmdline.lower():
                continue
            rss = getattr(proc.info.get("memory_info"), "rss", 0) or 0
            candidates.append((rss, pid, proc.info.get("ppid"), cmdline))
        except Exception:
            continue
    return sorted(candidates, reverse=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kill", action="store_true", help="terminate candidate processes")
    args = parser.parse_args()

    candidates = _iter_candidates()
    if not candidates:
        print("No orphan PredictionRTS multiprocessing child process found.")
        return 0

    total_rss = sum(rss for rss, _, _, _ in candidates)
    print(f"Candidates: {len(candidates)}, total RSS={total_rss / 1024 / 1024:.1f} MB")
    for rss, pid, ppid, cmdline in candidates:
        print(f"pid={pid:6d} ppid={ppid!s:6s} rss={rss / 1024 / 1024:8.1f} MB  {cmdline[:180]}")

    if not args.kill:
        print("Dry run only. Re-run with --kill to terminate these processes.")
        return 0

    for _, pid, _, _ in candidates:
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"terminated pid={pid}")
        except Exception as exc:
            print(f"failed pid={pid}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
