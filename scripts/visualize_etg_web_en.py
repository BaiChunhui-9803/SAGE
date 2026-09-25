"""English artifact explorer using the original SAGE graph and planning modules."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from etg_web.english_app import main

if __name__ == "__main__":
    main()
