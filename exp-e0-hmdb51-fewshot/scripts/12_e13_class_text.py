"""E1.3 class text and support-prototype references."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from hmdb51_e0.e13_cli import main

if __name__=='__main__':
    raise SystemExit(main())
