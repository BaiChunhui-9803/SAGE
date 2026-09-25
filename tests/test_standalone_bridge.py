"""A standalone evaluation must not wait for an absent Web queue consumer."""
from pathlib import Path
import subprocess
import sys


def test_standalone_bridge_exits_after_large_episode_payload():
    code = """
from src.sc2env.bridge import GameBridge
bridge = GameBridge(publish=False)
payload = {'frames': 'x' * 1000000}
bridge.put_history(payload)
bridge.update_status(observation=payload)
bridge.put_event(payload)
bridge.confirm_params(1)
assert bridge.get_histories() == []
"""
    subprocess.run([sys.executable, "-B", "-c", code],
        cwd=Path(__file__).resolve().parents[1], check=True, timeout=10)
