# Faz 2 sonuçları: form, şut ve piyasa özellikleri

Test: son 5 tam sezon (2021/22–2025/26), Süper Lig + Premier Lig, 3621 maç. Her sezon yalnızca önceki sezonlarla eğitilen modelle tahmin edildi.
Yeniden üretmek için: `python -m iddia.compare`

## Özellikler

- Elo farkı (Faz 1)
- Her takımın son maçlarına üstel ağırlıklı (yarı ömür 6 maç) ortalamaları: atılan/yenen gol, şut, isabetli şut
- Dinlenme günü (en fazla 14)
- Maç öncesi (kapanıştan önceki) piyasa ortalaması oranlarından Shin yöntemiyle olasılık. Canlı tahminde de elimizde olan oran bu.

xG yalnızca 2026/27 sezonunda var, eğitimde kullanılamadı.

## Skorlar

| Model | RPS ↓ | Log loss ↓ | İsabet ↑ |
|---|---|---|---|
| Elo (Faz 1) | 0.1998 | 0.9793 | %53.5 |
| Lojistik regresyon, form (piyasasız) | 0.1972 | 0.9710 | %54.0 |
| Gradient boosting, form | 0.2034 | 1.0033 | %51.5 |
| Piyasa, maç öncesi oran | 0.1935 | 0.9599 | %55.3 |
| **Lojistik regresyon, form + piyasa** | **0.1934** | 0.9606 | **%55.5** |
| Gradient boosting, form + piyasa | 0.1993 | 0.9908 | %52.4 |
| Piyasa, kapanış oranı (referans) | 0.1921 | 0.9553 | %55.6 |

## Güven eşiğine göre isabet

| Model | ≥ %50 | ≥ %60 | ≥ %70 |
|---|---|---|---|
| Elo | %61.9 (1864) | %69.7 (990) | %74.9 (451) |
| Form (piyasasız) | %62.5 (1960) | %70.0 (1098) | %76.4 (500) |
| Form + piyasa | %65.0 (1930) | %71.8 (1114) | %77.5 (559) |

## Yorum

- Form ve şut bilgisi Elo'yu geçiyor (+0.5 puan isabet, RPS 0.1998 → 0.1972).
- Piyasa oranını eklemek en büyük sıçrama (+2 puan). Ortaya çıkan model maç öncesi piyasayla aynı seviyede, kapanış oranına çok yakın; piyasayı belirgin biçimde geçmiyor.
- Gradient boosting bu veri boyutunda ezberliyor; lojistik regresyon daha iyi.
- Günlük sistem artık oranı olan maçlarda "form + piyasa", olmayanlarda "form" modelini kullanıyor.
