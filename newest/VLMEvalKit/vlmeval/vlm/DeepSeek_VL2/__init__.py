"""DeepSeek_VL2 backend with shared DeepSeek support modules."""

from pathlib import Path as _Path

# Resolve backend-specific modules here first, then shared support modules.
__path__.append(str(_Path(__file__).resolve().parent.parent / "deepseek_model"))
del _Path
