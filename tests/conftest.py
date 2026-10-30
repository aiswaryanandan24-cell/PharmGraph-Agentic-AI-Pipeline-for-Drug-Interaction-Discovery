"""
Shared pytest configuration for PharmGraph tests.

Usage:
    pytest tests/ -v                    # run all tests
    pytest tests/ -v -k "brand"         # only brand resolution tests
    pytest tests/ -v -k "not retriever" # skip tests that need the knowledge graph
    pytest tests/ --tb=short            # short traceback on failure
"""

import sys
import os

# Ensure project root is on the path for all test modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
