"""
pytest configuration for the churn-prediction test suite.

- Adds the project root to sys.path so `src.*` imports work in all tests.
- Loads .env so GROQ_API_KEY is available if present (not required for most tests).
- Registers the `integration` marker to avoid "unknown mark" warnings.
"""

import sys
from pathlib import Path

import pytest

# Add the project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load .env (silently ignored if the file doesn't exist)
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: marks tests that require heavy ML dependencies "
        "(faiss, sentence-transformers, Groq API). "
        "Run with: pytest -m integration",
    )
