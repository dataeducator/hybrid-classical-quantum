"""pytest configuration – makes the numbered package importable."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

# Add codes/ to sys.path so `import 1_data_harmonization` can be
# resolved via importlib even though the name is not a valid identifier.
_codes_dir = str(Path(__file__).resolve().parent.parent)
if _codes_dir not in sys.path:
    sys.path.insert(0, _codes_dir)
