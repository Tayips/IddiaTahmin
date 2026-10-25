# Ek deneyler: daha fazla ligle eğitim ve puan durumu

Test: son 5 tam sezon (2021/22–2025/26), yalnızca Süper Lig + Premier Lig, 3621 maç.
Her test sezonu yalnızca önceki sezonlarla eğitilen modellerle tahmin edildi.
Yeniden üretmek için: `python -m iddia.deney_ek --indir` (ya da bir `deney-*` dalına push: `.github/workflows/deney.yml`)

Ek ligler (football-data.co.uk): E1, D1, I1, SP1, F1, N1, P1, B1, SC0, G1 — toplam ~41.000 maç daha.
Puan durumu özellikleri: küme düşme hattına ve lidere uzaklık (kalan maçlarda alınabilecek puana oranla), maç başı puan, sezonun ilerleyişi ve bunların sezon sonuna doğru ağırlıklanmış hâli.

## 1-X-2

| Model | RPS ↓ | Log loss ↓ | İsabet |
|---|---|---|---|
| A: E0+T1 ile eğitim (önceki model) | 0.1934 | 0.9606 | %55.5 |
| **B: tüm liglerle eğitim** | **0.1931** | **0.9586** | %55.4 |
| C: E0+T1 + puan durumu | 0.1936 | 0.9617 | %55.1 |
| D: tüm ligler + puan durumu | 0.1931 | 0.9583 | %55.3 |
| Kapanış oranı (referans, deneyler.md) | 0.1921 | 0.9553 | %55.6 |

B, A'yı 5 sezonun 4'ünde geçti (RPS farkı −0.0007 ile +0.0001 arası). Puan durumu özellikleri tek başına kötüleştirdi (C), tüm liglerle birlikteyken (D) B'ye bir şey katmadı.

## Gol pazarları (log loss ↓)

| Model | Üst/Alt 2.5 | KG |
|---|---|---|
| A | 0.6744 | 0.6838 |
| **B** | **0.6740** | **0.6831** |
| C | 0.6756 | 0.6850 |
| D | 0.6743 | 0.6832 |

Üst/Alt'ta modelin %65+ emin olduğu tahminler A'da 603 maçın %69.3'ünde, B'de 485 maçın %72.2'sinde tuttu.

## Sonuç

Günlük model artık tüm liglerle eğitiliyor (B). Kazanç küçük ama üç pazarda da aynı yönde ve sezonlara yayılmış. Puan durumu özellikleri eklenmedi: oranlar bu bilgiyi zaten taşıyor. İsabet oranı neredeyse değişmedi; iyileşen, olasılıkların kalitesi.
