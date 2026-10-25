"""Kıyas modelleri: lig frekansları ve kapanış oranları."""
import numpy as np
import pandas as pd

from iddia import odds as odds_mod
from iddia.metrics import OUTCOMES


def frequencies(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Her maça, eğitim dönemindeki ligin H/D/A oranlarını verir."""
    freq = train.groupby("league")["ftr"].value_counts(normalize=True).unstack().reindex(columns=OUTCOMES)
    return freq.loc[test["league"]].to_numpy()


def closing_odds(test: pd.DataFrame, method: str = "shin") -> np.ndarray:
    """Marjı temizlenmiş kapanış oranı olasılıkları; oranı olmayan satırlar NaN."""
    o = test[["odds_h", "odds_d", "odds_a"]].to_numpy(dtype=float)
    out = np.full(o.shape, np.nan)
    ok = np.isfinite(o).all(axis=1) & (o > 1).all(axis=1)
    fn = odds_mod.shin if method == "shin" else odds_mod.proportional
    if ok.any():
        out[ok] = fn(o[ok])
    return out
