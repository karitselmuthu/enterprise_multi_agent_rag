from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APPS_PATH = ROOT / "apps"
SRC_PATH = ROOT / "src"
if str(APPS_PATH) not in sys.path:
    sys.path.insert(0, str(APPS_PATH))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from code_rag_cli.main import run


if __name__ == "__main__":
    run()
