"""Run the DeepSeek-VL2-Small fixed GPTQ workflow from any working directory."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / 'pipeline'))
from small_gptq.cli import main

if __name__ == '__main__':
    raise SystemExit(main())
