# conftest.py — top-level pytest configuration for oil-price-demo.
# Ensures runtime/ and workflows/ are on sys.path for all tests so
# activities and scripts can be imported without AGENTSMITH_DIR being set.
import sys
from pathlib import Path

_repo = Path(__file__).resolve().parent
for p in (_repo / "runtime", _repo / "workflows", _repo / "scripts"):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))
