"""Second-round source grouping and paired V0/V1 development commands."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from hmdb51_e0.round2_cli import main

if __name__ == '__main__':
    raise SystemExit(main())
