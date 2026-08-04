from __future__ import annotations

import sys
from pathlib import Path


def ensure_src_path() -> None:
    """Make the repo-root src/ (phase0 package) importable from code_rag_platform modules."""
    repo_root = Path(__file__).resolve().parents[3]
    src_path = repo_root / "src"
    src_str = str(src_path)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
