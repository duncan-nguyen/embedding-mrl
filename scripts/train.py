#!/usr/bin/env python3
"""Run an experiment without installing the package.

python scripts/train.py --config configs/mipic/bgem3.yaml
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embedding_mrl.cli import main

if __name__ == "__main__":
    sys.exit(main())
