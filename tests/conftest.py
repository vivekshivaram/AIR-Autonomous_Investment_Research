"""Makes the repository root importable so tests can use `src` and `tests.helpers`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
