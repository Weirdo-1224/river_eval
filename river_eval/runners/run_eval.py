"""Compatibility wrapper for ``python -m river_eval.runners.run_eval``."""

from __future__ import annotations

from benchmark.runners.run_eval import run


if __name__ == "__main__":
    run()
