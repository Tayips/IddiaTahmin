"""Model deneyleri: kalibrasyon, Dixon-Coles, model birleşimi ve rakip gücüne göre form.

Temel modeller sezon sezon ileri yürüyerek tahmin üretir (her sezon yalnızca
önceki sezonlarla). Kalibrasyon ve birleşim ağırlıkları bir test sezonu için
yalnızca o sezondan ÖNCEKİ sezonların bu dış-örneklem tahminlerinden öğrenilir;
test sezonu hiçbir seçimde kullanılmaz.
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from iddia import metrics
from iddia.compare import prepare
from iddia.evaluate import predict_dixon_coles
from iddia.features import FEATURES_NO_MARKET, HALF_LIFE, MARKET_FEATURES
from iddia.models.baseline import closing_odds
from iddia.models.ml import Model

REPORT_DIR = Path("reports")
BASE = FEATURES_NO_MARKET + MARKET_FEATURES
ADJ = ["adj_form_diff"]


def adjusted_form(df: pd.DataFrame, elo_diff: pd.Series, beta: float) -> pd.Series:
    """Rakip gücüne göre düzeltilmiş form: gol farkı eksi Elo farkının beklediği gol farkı.

    Her maçta yalnızca önceki maçların artıkları kullanılır (shift).
    """
    resid = (df["fthg"] - df["ftag"]) - beta * elo_diff / 100
    long = pd.concat([
        pd.DataFrame({"idx": df.index, "league": df["league"], "team": df["home"], "date": df["date"], "side": "h", "r": resid}),
        pd.DataFrame({"idx": df.index, "league": df["league"], "team": df["away"], "date": df["date"], "side": "a", "r": -resid}),
    ]).sort_values(["date", "idx"]).reset_index(drop=True)
    long["form"] = long.groupby(["league", "team"])["r"].transform(
        lambda s: s.shift(1).ewm(halflife=HALF_LIFE, ignore_na=True).mean())
    wide = long.pivot(index="idx", columns="side", values="form").reindex(df.index)
    return (wide["h"] - wide["a"]).rename("adj_form_diff")


def base_predictions(df, X, seasons) -> dict[str, np.ndarray]:
    out = {k: np.full((len(df), 3), np.nan) for k in ["lr", "lr_adj", "elo"]}
    has_mk = X["mk_h"].notna()
    for season in seasons:
        train, test = (df["season"] < season) & has_mk, (df["season"] == season).to_numpy()
        out["lr"][test] = Model("lr", BASE).fit(X[train], df.loc[train, "ftr"]).predict(X[test])
        out["lr_adj"][test] = Model("lr", BASE + ADJ).fit(X[train], df.loc[train, "ftr"]).predict(X[test])
        elo_only = Model("lr", ["elo_diff", "is_t1"]).fit(X[df["season"] < season], df.loc[df["season"] < season, "ftr"])
        out["elo"][test] = elo_only.predict(X[test])
    out["dc"] = predict_dixon_coles(df, df["season"].isin(seasons))
    out["piyasa"] = X[["mk_h", "mk_d", "mk_a"]].to_numpy()
    out["kapanis"] = closing_odds(df)
    return out


def logloss(p, y) -> float:
    return float(metrics.log_loss(p, y).mean())


def platt(p_val, y_val, p_test):
    """Çok sınıflı Platt: log olasılıklar üzerinde lojistik regresyon."""
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(np.log(np.clip(p_val, 1e-6, 1)), y_val)
    q = lr.predict_proba(np.log(np.clip(p_test, 1e-6, 1)))
    return q[:, [list(lr.classes_).index(o) for o in metrics.OUTCOMES]]


def isotonic(p_val, y_val, p_test):
    """Her sonuç için ayrı isotonic regresyon, sonra toplam 1'e ölçekleme."""
    onehot = metrics.one_hot(y_val)
    q = np.column_stack([
        IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1).fit(p_val[:, k], onehot[:, k]).predict(p_test[:, k])
        for k in range(3)])
    return q / q.sum(axis=1, keepdims=True)


def best_weights(parts: list[np.ndarray], y, step=0.1) -> tuple:
    """Doğrulama verisinde log loss'u en düşük yapan ağırlıklar (toplamı 1, adımlarla)."""
    grid = np.round(np.arange(0, 1 + step / 2, step), 2)
    best = None
    for w in itertools.product(grid, repeat=len(parts) - 1):
        if sum(w) > 1 + 1e-9:
            continue
        w = (*w, round(1 - sum(w), 2))
        score = logloss(sum(wi * p for wi, p in zip(w, parts)), y)
        if best is None or score < best[0]:
            best = (score, w)
    return best[1]


def stacked(preds, df, ok, test_seasons) -> tuple[dict, dict]:
    """Her test sezonu için kalibrasyon ve birleşimi önceki sezonlarla kurar."""
    out = {k: np.full((len(df), 3), np.nan) for k in ["lr_platt", "lr_isotonic", "birlesim_lr_dc", "birlesim_lr_dc_elo"]}
    chosen = {}
    for season in test_seasons:
        val = (ok & (df["season"] < season)).to_numpy()
        test = (ok & (df["season"] == season)).to_numpy()
        y = df.loc[val, "ftr"]
        out["lr_platt"][test] = platt(preds["lr"][val], y, preds["lr"][test])
        out["lr_isotonic"][test] = isotonic(preds["lr"][val], y, preds["lr"][test])
        w2 = best_weights([preds["lr"][val], preds["dc"][val]], y)
        out["birlesim_lr_dc"][test] = w2[0] * preds["lr"][test] + w2[1] * preds["dc"][test]
        w3 = best_weights([preds["lr"][val], preds["dc"][val], preds["elo"][val]], y)
        out["birlesim_lr_dc_elo"][test] = sum(w * preds[k][test] for w, k in zip(w3, ["lr", "dc", "elo"]))
        chosen[season] = {"lr+dc": w2, "lr+dc+elo": w3}
    return out, chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-seasons", type=int, default=5)
    parser.add_argument("--warmup-seasons", type=int, default=2, help="ağırlık öğrenmek için ek dış-örneklem sezonları")
    args = parser.parse_args()
    df, X = prepare()
    full = sorted(df["season"].unique())[:-1]                    # devam eden sezon hariç
    oos = full[-(args.test_seasons + args.warmup_seasons):]
    test_seasons = full[-args.test_seasons:]

    # Rakip gücüne göre form: katsayı yalnızca dış-örneklem dönemi öncesinden
    early = df["season"] < oos[0]
    elo = X["elo_diff"] * 100
    beta = float(np.polyfit(elo[early] / 100, (df["fthg"] - df["ftag"])[early], 1)[0])
    X = X.join(adjusted_form(df, elo, beta))

    preds = base_predictions(df, X, oos)
    ok = df["season"].isin(oos)
    for k in ["lr", "lr_adj", "elo", "dc", "piyasa", "kapanis"]:
        ok &= np.isfinite(preds[k]).all(axis=1)
    extra, chosen = stacked(preds, df, ok, test_seasons)
    preds.update(extra)

    mask = (ok & df["season"].isin(test_seasons)).to_numpy()
    rows = []
    for name, p in preds.items():
        for group, gm in [("Hepsi", mask)] + [(lg, mask & (df["league"] == lg).to_numpy()) for lg in sorted(df["league"].unique())]:
            s = metrics.summary(p[gm], df.loc[gm, "ftr"])
            s["ece"] = metrics.ece(p[gm], df.loc[gm, "ftr"])
            rows.append({"lig": group, "model": name, **s})
    table = pd.DataFrame(rows)
    REPORT_DIR.mkdir(exist_ok=True)
    table.to_csv(REPORT_DIR / "deneyler_skorlar.csv", index=False)

    # Sezon sezon RPS farkı: iyileşme istikrarlı mı?
    per_season = pd.DataFrame({
        name: [float(metrics.rps(preds[name][mask & (df["season"] == s).to_numpy()],
                                 df.loc[mask & (df["season"] == s).to_numpy(), "ftr"]).mean()) for s in test_seasons]
        for name in preds}, index=test_seasons)
    per_season.to_csv(REPORT_DIR / "deneyler_sezonluk.csv")

    pd.set_option("display.width", 160)
    print(f"Rakip düzeltmesi katsayısı: {beta:.3f} gol / 100 Elo")
    print(f"Test sezonları: {', '.join(test_seasons)}  ({int(mask.sum())} maç)")
    print(table[table["lig"] == "Hepsi"].sort_values("rps").round(4).to_string(index=False))
    print("\nSezonluk RPS:\n" + per_season.round(4).to_string())
    print("\nSeçilen ağırlıklar:")
    for s, w in chosen.items():
        print(s, w)


if __name__ == "__main__":
    main()
