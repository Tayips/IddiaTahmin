"""Dixon ve Coles (1997): zaman ağırlıklı Poisson skor modeli, düşük skor düzeltmesiyle."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

MAX_GOALS = 10


def tau(x, y, lam, mu, rho):
    """Düşük skorlardaki (0-0, 1-0, 0-1, 1-1) bağımlılık düzeltmesi."""
    out = np.ones_like(lam, dtype=float)
    out = np.where((x == 0) & (y == 0), 1 - lam * mu * rho, out)
    out = np.where((x == 0) & (y == 1), 1 + lam * rho, out)
    out = np.where((x == 1) & (y == 0), 1 + mu * rho, out)
    out = np.where((x == 1) & (y == 1), 1 - rho, out)
    return out


@dataclass
class DixonColes:
    xi: float = 0.0019          # günlük zaman sönümü (yarı ömür ≈ 1 yıl)
    l2: float = 0.01            # parametreleri küçük tutan hafif ceza, az maçlı takımlar için
    teams: list = field(default_factory=list)
    attack: dict = field(default_factory=dict)
    defence: dict = field(default_factory=dict)
    home: float = 0.25
    rho: float = -0.05
    _x0: np.ndarray | None = None

    def fit(self, matches: pd.DataFrame, as_of: pd.Timestamp) -> "DixonColes":
        teams = sorted(set(matches["home"]) | set(matches["away"]))
        index = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        hi = matches["home"].map(index).to_numpy()
        ai = matches["away"].map(index).to_numpy()
        x = matches["fthg"].to_numpy()
        y = matches["ftag"].to_numpy()
        age = (as_of - matches["date"]).dt.days.to_numpy()
        w = np.exp(-self.xi * age)

        def unpack(p):
            att = p[:n] - p[:n].mean()   # hücum parametrelerinin ortalaması 0
            return att, p[n:2 * n], p[2 * n], p[2 * n + 1]

        def nll(p):
            att, dfn, home, rho = unpack(p)
            lam = np.exp(att[hi] + dfn[ai] + home)
            mu = np.exp(att[ai] + dfn[hi])
            t = np.clip(tau(x, y, lam, mu, rho), 1e-10, None)
            ll = np.log(t) + poisson.logpmf(x, lam) + poisson.logpmf(y, mu)
            return -(w * ll).sum() + self.l2 * (p[:2 * n] ** 2).sum()

        x0 = np.zeros(2 * n + 2)
        x0[2 * n], x0[2 * n + 1] = 0.25, -0.05
        if self._x0 is not None and self.teams:
            for t, i in index.items():
                if t in self.attack:
                    x0[i], x0[n + i] = self.attack[t], self.defence[t]
            x0[2 * n], x0[2 * n + 1] = self.home, self.rho
        bounds = [(None, None)] * (2 * n) + [(None, None), (-0.3, 0.3)]
        res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds)
        att, dfn, self.home, self.rho = unpack(res.x)
        self.teams = teams
        self.attack = dict(zip(teams, att))
        self.defence = dict(zip(teams, dfn))
        self._x0 = res.x
        return self

    def _params(self, team: str) -> tuple[float, float]:
        if team in self.attack:
            return self.attack[team], self.defence[team]
        # Hiç verisi olmayan (ör. yeni yükselen) takım: en zayıf üç takımın ortalaması.
        strength = {t: self.attack[t] - self.defence[t] for t in self.teams}
        weakest = sorted(strength, key=strength.get)[:3]
        return (np.mean([self.attack[t] for t in weakest]), np.mean([self.defence[t] for t in weakest]))

    def score_matrix(self, home: str, away: str) -> np.ndarray:
        ah, dh = self._params(home)
        aa, da = self._params(away)
        lam = np.exp(ah + da + self.home)
        mu = np.exp(aa + dh)
        goals = np.arange(MAX_GOALS + 1)
        m = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mu))
        gx, gy = np.meshgrid(goals, goals, indexing="ij")
        m *= tau(gx, gy, np.full(m.shape, lam), np.full(m.shape, mu), self.rho)
        return m / m.sum()

    def predict(self, home: str, away: str) -> np.ndarray:
        m = self.score_matrix(home, away)
        return np.array([np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()])
