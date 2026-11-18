"""E1 concept baselines and matched random-projection controls."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from hmdb51_e0.e1_cli import main

if __name__=='__main__':
    raise SystemExit(main())
