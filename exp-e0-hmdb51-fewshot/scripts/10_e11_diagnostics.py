"""E1.1 geometry, fixed fusion and frame audit entry point."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from hmdb51_e0.e11_cli import main

if __name__=='__main__':
    raise SystemExit(main())
