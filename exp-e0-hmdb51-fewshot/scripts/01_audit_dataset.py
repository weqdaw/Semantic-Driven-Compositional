"""Audit official annotations, identities and training decode eligibility."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hmdb51_e0.cli import main

if __name__ == "__main__":
    raise SystemExit(main(1))
