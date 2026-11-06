"""Tests locaux synthétiques ; aucun GPU, calcul complet ou envoi réseau."""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    with patch.dict(os.environ, dict(os.environ), clear=True):
        yield
