# tests/conftest.py

from pathlib import Path
import sys

# Ensure the repository root is on sys.path for tests.
# This makes `import src` work even when pytest is invoked from editors/CI
# without an editable install.
ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
    # Insert at position 0 to prefer local repo copy over installed packages
    sys.path.insert(0, ROOT_STR)
