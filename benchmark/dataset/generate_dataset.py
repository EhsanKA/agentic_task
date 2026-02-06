#!/usr/bin/env python
"""CLI entry point for dataset generation.

Usage:
    cd benchmark/dataset && python generate_dataset.py
"""

import os
import sys

# Add repo root to path for package imports
_repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, _repo_root)

from benchmark.data.generate import save_dataset

if __name__ == "__main__":
    save_dataset(output_dir=os.path.dirname(os.path.abspath(__file__)) or ".")
