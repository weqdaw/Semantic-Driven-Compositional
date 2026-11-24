"""Run the controlled E1.5 instance-cache workflow explicitly."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from hmdb51_e0.e15_cli import main

if __name__ == '__main__':
    raise SystemExit(main())
