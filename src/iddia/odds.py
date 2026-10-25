"""Bahis oranlarından olasılık çıkarma ve marj temizleme."""
import numpy as np
from scipy.optimize import brentq


def implied(odds: np.ndarray) -> np.ndarray:
    """Ham ters oranlar; satır toplamı 1'den büyüktür (fark marjdır)."""
    return 1.0 / np.asarray(odds, dtype=float)


def margin(odds: np.ndarray) -> np.ndarray:
    return implied(odds).sum(axis=1) - 1.0


def proportional(odds: np.ndarray) -> np.ndarray:
    """Ters oranları toplamlarına bölerek normalleştirir."""
    inv = implied(odds)
    return inv / inv.sum(axis=1, keepdims=True)


def _shin_row(inv: np.ndarray) -> np.ndarray:
    total = inv.sum()

    def probs(z: float) -> np.ndarray:
        return (np.sqrt(z**2 + 4 * (1 - z) * inv**2 / total) - z) / (2 * (1 - z))

    z = brentq(lambda z: probs(z).sum() - 1.0, 0.0, 0.4)
    return probs(z)


def shin(odds: np.ndarray) -> np.ndarray:
    """Shin (1993) yöntemi: marjın favori-sürpriz eğilimini hesaba katar."""
    inv = implied(odds)
    return np.vstack([_shin_row(row) for row in inv])
