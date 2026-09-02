#!/usr/bin/env python3
"""Root entrypoint wrapper for MedGraphRAG inference pipeline.

Delegates execution to scripts/run_pipeline.py.
Usage:
    python main.py
    python main.py llm.temperature=0.3
"""

import sys
from pathlib import Path

# Add scripts directory to path and execute run_pipeline main
scripts_dir = Path(__file__).resolve().parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from run_pipeline import main

if __name__ == "__main__":
    main()
