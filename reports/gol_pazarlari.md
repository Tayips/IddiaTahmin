# Gol pazarları: Üst/Alt 2.5 ve Karşılıklı Gol

Test: son 5 tam sezon (2021/22–2025/26), Süper Lig + Premier Lig, 3621 maç. Her sezon yalnızca önceki sezonlarla eğitildi.
Yeniden üretmek için: `python -m iddia.goals`

| Pazar | Model | İsabet | Brier ↓ | %60+ emin: isabet (maç) | %65+ emin: isabet (maç) |
|---|---|---|---|---|---|
| Üst/Alt 2.5 | Taban (eğitimdeki Üst oranı) | %55.5 | 0.2478 | – | – |
| Üst/Alt 2.5 | Form | %55.9 | 0.2439 | %66.4 (669) | %70.6 (211) |
| Üst/Alt 2.5 | **Form + piyasa** | **%57.1** | 0.2409 | %64.9 (1206) | %69.3 (603) |
| Üst/Alt 2.5 | Piyasa (maç öncesi oran) | %56.5 | 0.2408 | %66.3 (878) | %72.1 (362) |
| KG | Taban | %55.8 | 0.2473 | – | – |
| KG | Form | %55.0 | 0.2469 | %60.4 (525) | %61.6 (73) |
| KG | **Form + piyasa** | **%55.9** | 0.2453 | %62.8 (615) | %60.7 (117) |

## Yorum

- Üst/Alt 2.5'te model piyasayla aynı seviyede: isabeti biraz yüksek, Brier skoru eşit. Piyasayı belirgin biçimde geçmiyor.
- Karşılıklı Gol tahmin etmesi zor bir pazar. Model taban oranın ancak biraz üstünde; bu pazar için oran verisi de yok.
- Günlük sistem iki pazar için de "form + piyasa" modelini kullanıyor ve sonuçları sayfada ayrı sayıyor.
