"""E1.2 support-only scaling and concept-audit diagnostics."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from hmdb51_e0.e12_cli import main

if __name__=='__main__':
    raise SystemExit(main())
