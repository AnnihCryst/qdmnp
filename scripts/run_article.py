#!/usr/bin/env python
"""Entry point for the article computation chain.

The orchestration itself lives in :mod:`qdmnp.pipeline` so that its selection,
ranking and validation logic stays importable and testable; this file only
forwards the command line.
"""
from qdmnp.pipeline import main

if __name__ == "__main__":
    raise SystemExit(main())
