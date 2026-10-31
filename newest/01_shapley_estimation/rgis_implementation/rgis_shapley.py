"""Compatibility import for the sole configuration estimator."""
from pathlib import Path
import sys
root = next(p for p in Path(__file__).resolve().parents if (p / 'run_small.py').is_file())
sys.path.insert(0, str(root / 'pipeline'))
from small_gptq.estimator import ConfigOwenEstimator, enumerate_menus, menu_to_key, key_to_menu
RGISShapleyEstimator = ConfigOwenEstimator
