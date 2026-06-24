"""Compatibility wrapper for the simplified generator CLI."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from .generate import main
except ImportError:
    datasets_root = Path(__file__).resolve().parents[1]
    if str(datasets_root) not in sys.path:
        sys.path.insert(0, str(datasets_root))
    from create_synthetic_dataset.generate import main


if __name__ == "__main__":
    if not any(arg.startswith("--epoch") or arg in {"--start-epoch", "--end-epoch"} for arg in sys.argv[1:]):
        sys.argv.append("--epoch-folders")
    main()
