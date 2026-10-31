"""Compatibility entry; use run_small.py for the Small-only GPTQ workflow."""
from pathlib import Path
import sys
root = next(p for p in Path(__file__).resolve().parents if (p / 'run_small.py').is_file())
sys.path.insert(0, str(root / 'pipeline'))
from small_gptq.cli import main
if __name__ == '__main__':
    raise SystemExit(main())
