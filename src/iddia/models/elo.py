"""Gol farkına duyarlı Elo ve sıralı lojistik ile 1X2 olasılıkları."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit


@dataclass
class Elo:
    k: float = 20.0
    home_adv: float = 60.0
    carryover: float = 0.8      # yeni sezonda puanlar lig ortalamasına bu oranda çekilir
    new_team: float = -100.0    # ligde ilk kez görülen takımın ortalamaya göre başlangıcı
    ratings: dict = field(default_factory=dict)

    def run(self, matches: pd.DataFrame) -> pd.Series:
        """Maçları sırayla oynatır; her maç için maç ÖNCESİ puan farkını döndürür."""
        diffs = np.empty(len(matches))
        season_seen: dict[str, str] = {}
        first_season: dict[str, str] = {}
        for i, row in enumerate(matches.itertuples(index=False)):
            table = self.ratings.setdefault(row.league, {})
            if season_seen.get(row.league) != row.season:
                if table:
                    mean = np.mean(list(table.values()))
                    for team in table:
                        table[team] = mean + self.carryover * (table[team] - mean)
                season_seen[row.league] = row.season
                first_season.setdefault(row.league, row.season)
            # İlk sezonda herkes eşit başlar; sonra gelen takım ligin yeni üyesidir.
            base = np.mean(list(table.values())) if table else 1500.0
            start = base + (self.new_team if row.season != first_season[row.league] else 0.0)
            rh = table.setdefault(row.home, start)
            ra = table.setdefault(row.away, start)
            diff = rh + self.home_adv - ra
            diffs[i] = diff
            expected = 1 / (1 + 10 ** (-diff / 400))
            actual = 1.0 if row.fthg > row.ftag else 0.5 if row.fthg == row.ftag else 0.0
            gd = abs(row.fthg - row.ftag)
            change = self.k * (np.log(gd + 1) + 1) * (actual - expected)
            table[row.home] = rh + change
            table[row.away] = ra - change
        return pd.Series(diffs, index=matches.index, name="elo_diff")


@dataclass
class OrderedLogit:
    """P(A) = σ(c1 - b·x), P(A veya D) = σ(c2 - b·x)."""
    b: float = 0.0
    c1: float = 0.0
    c2: float = 0.0

    def predict(self, x: np.ndarray) -> np.ndarray:
        p_a = expit(self.c1 - self.b * x)
        p_ad = expit(self.c2 - self.b * x)
        return np.column_stack([1 - p_ad, p_ad - p_a, p_a])

    def fit(self, x: np.ndarray, results) -> "OrderedLogit":
        idx = pd.Series(results).map({"H": 0, "D": 1, "A": 2}).to_numpy()
        x = np.asarray(x) / 100.0

        def nll(params):
            b, c1, log_gap = params
            self.b, self.c1, self.c2 = b, c1, c1 + np.exp(log_gap)
            p = self.predict(x)[np.arange(len(x)), idx]
            return -np.log(np.clip(p, 1e-12, 1)).sum()

        res = minimize(nll, x0=[0.5, -1.0, 0.0], method="L-BFGS-B")
        b, c1, log_gap = res.x
        self.b, self.c1, self.c2 = b / 100.0, c1, c1 + np.exp(log_gap)
        return self
