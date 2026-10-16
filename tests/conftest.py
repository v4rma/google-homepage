import numpy as np
import pytest

from contemplative.ingest.base import FS, Packet
from contemplative.scorer import load_spec

FS_INT = int(FS)


def pink_noise(n: int, rng: np.random.Generator, exponent: float = 1.0) -> np.ndarray:
    """Gaussian noise shaped to a 1/f^exponent power spectrum (Fourier method)."""
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    f = np.fft.rfftfreq(n, 1.0 / FS)
    f[0] = f[1]
    spec *= f ** (-exponent / 2.0)
    x = np.fft.irfft(spec, n)
    return x / x.std()


def eeg_packets(x: np.ndarray, t0: float = 0.0, fs: float = FS) -> list[Packet]:
    """(4, n) array -> one eeg Packet per sample."""
    return [Packet("eeg", t0 + i / fs, [float(v) for v in x[:, i]]) for i in range(x.shape[1])]


@pytest.fixture
def spec():
    return load_spec()


@pytest.fixture
def fallback_spec():
    from contemplative.scorer.features import FALLBACK_SPEC

    return load_spec(FALLBACK_SPEC)


@pytest.fixture
def rng():
    return np.random.default_rng(1234)
