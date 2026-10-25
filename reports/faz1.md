# Faz 1 sonuçları: temel modeller

Veri: Süper Lig (T1) ve Premier Lig (E0), 2014/15–2025/26. Hükmen sonuçlanan 29 maç (T1 2022/23) çıkarıldı.
Test: son 5 sezon (2021/22–2025/26), 3621 maç. Her model yalnızca tahmin anından önceki maçları gördü.
Yeniden üretmek için: `python -m iddia.evaluate`

## Skorlar (iki lig birlikte)

| Model | RPS ↓ | Log loss ↓ | Brier ↓ | İsabet ↑ | ECE ↓ |
|---|---|---|---|---|---|
| Lig frekansları (taban) | 0.2298 | 1.0662 | 0.6445 | %44.9 | 0.001 |
| Elo + sıralı lojistik | 0.1998 | 0.9793 | 0.5834 | %53.5 | 0.010 |
| Dixon-Coles | 0.2020 | 0.9899 | 0.5877 | %52.9 | 0.008 |
| Kapanış oranları (Shin) | **0.1921** | **0.9553** | **0.5667** | **%55.6** | 0.012 |

Lig bazında: Premier Lig'de Elo 0.2001 / oranlar 0.1936; Süper Lig'de Elo 0.1994 / oranlar 0.1904.

## Güven eşiğine göre isabet

| Model | ≥ %50 | ≥ %60 | ≥ %70 |
|---|---|---|---|
| Elo | %61.9 (1864 maç) | %69.7 (990) | %74.9 (451) |
| Dixon-Coles | %63.1 (1776) | %70.3 (933) | %76.4 (385) |
| Kapanış oranları | %66.1 (1862) | %72.9 (1003) | %80.0 (479) |

## Yorum

- Yalnızca skorlarla kurulan modeller tabanı açık farkla geçiyor, ama kapanış oranlarının yaklaşık 0.008 RPS gerisinde.
- Elo, ayarsız haliyle Dixon-Coles'tan biraz daha iyi. Dixon-Coles'un zaman sönümü ve pencere uzunluğu henüz ayarlanmadı.
- Modelin "%70+" dediği maçlarda isabet %75 civarında; bu maçlar test maçlarının sekizde biri.
- Sıradaki adım (Faz 2): Elo, Dixon-Coles parametreleri, form ve dinlenme gibi özelliklerle gradient boosting, ardından kalibrasyon ve topluluk.
