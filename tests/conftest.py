import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

SR = 16000


@pytest.fixture
def tono():
    def _tono(dur_s, f=220.0):
        t = np.arange(int(dur_s * SR)) / SR
        return (0.5 * np.sin(2 * np.pi * f * t)).astype(np.float32)
    return _tono


@pytest.fixture
def silencio():
    return lambda dur_s: np.zeros(int(dur_s * SR), np.float32)
