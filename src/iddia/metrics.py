"""Olasılık tahminleri için skorlar. Sonuç sırası her yerde H, D, A."""
import numpy as np
import pandas as pd

OUTCOMES = ["H", "D", "A"]


def one_hot(results: pd.Series | np.ndarray) -> np.ndarray:
    results = np.asarray(results)
    return np.column_stack([results == o for o in OUTCOMES]).astype(float)


def rps(probs: np.ndarray, results) -> np.ndarray:
    """Ranked Probability Score, maç başına (düşük daha iyi)."""
    cum_p = np.cumsum(probs, axis=1)[:, :2]
    cum_o = np.cumsum(one_hot(results), axis=1)[:, :2]
    return ((cum_p - cum_o) ** 2).sum(axis=1) / 2


def log_loss(probs: np.ndarray, results) -> np.ndarray:
    p = (probs * one_hot(results)).sum(axis=1)
    return -np.log(np.clip(p, 1e-12, 1))


def brier(probs: np.ndarray, results) -> np.ndarray:
    return ((probs - one_hot(results)) ** 2).sum(axis=1)


def accuracy(probs: np.ndarray, results) -> np.ndarray:
    return (np.asarray(OUTCOMES)[probs.argmax(axis=1)] == np.asarray(results)).astype(float)


def summary(probs: np.ndarray, results) -> dict:
    return {
        "n": len(results),
        "rps": rps(probs, results).mean(),
        "log_loss": log_loss(probs, results).mean(),
        "brier": brier(probs, results).mean(),
        "accuracy": accuracy(probs, results).mean(),
    }


def calibration(probs: np.ndarray, results, bins: int = 10) -> pd.DataFrame:
    """Üç sonucun olasılıklarını birlikte kovalara ayırır: tahmin ortalaması ve gerçekleşme oranı."""
    p = probs.ravel()
    o = one_hot(results).ravel()
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    df = pd.DataFrame({"bin": idx, "p": p, "o": o})
    out = df.groupby("bin").agg(predicted=("p", "mean"), observed=("o", "mean"), n=("p", "size"))
    out.index = [f"{edges[i]:.1f}-{edges[i + 1]:.1f}" for i in out.index]
    return out


def ece(probs: np.ndarray, results, bins: int = 10) -> float:
    """Beklenen kalibrasyon hatası: kovalar arası ağırlıklı |tahmin - gerçekleşen|."""
    table = calibration(probs, results, bins)
    return float((table["n"] * (table["predicted"] - table["observed"]).abs()).sum() / table["n"].sum())
