# Model deneyleri: kalibrasyon, Dixon-Coles, birleşim, rakip gücüne göre form

Test: son 5 tam sezon (2021/22–2025/26), Süper Lig + Premier Lig, 3621 maç.
Yeniden üretmek için: `python -m iddia.deneyler` (yaklaşık 4 dakika)

Temel modeller her sezonu yalnızca önceki sezonlarla eğitilerek tahmin etti. Kalibrasyon ve birleşim ağırlıkları, bir test sezonu için yalnızca ondan önceki sezonların dış-örneklem tahminlerinden öğrenildi (2019/20 ve 2020/21 bu iş için ek olarak tahmin edildi). Yani test sezonu hiçbir seçimde kullanılmadı.

## Skorlar

| Model | RPS ↓ | Log loss ↓ | Brier ↓ | İsabet ↑ | ECE ↓ |
|---|---|---|---|---|---|
| Kapanış oranı (referans) | 0.1921 | 0.9553 | 0.5667 | %55.6 | 0.012 |
| **Lojistik regresyon, form + piyasa (şu anki model)** | **0.1934** | 0.9606 | 0.5703 | %55.5 | 0.006 |
| + rakip gücüne göre form | 0.1934 | 0.9605 | 0.5703 | %55.4 | 0.007 |
| Maç öncesi piyasa | 0.1935 | 0.9599 | 0.5699 | %55.3 | 0.010 |
| Birleşim: lojistik + Dixon-Coles | 0.1940 | 0.9618 | 0.5712 | %55.1 | 0.007 |
| Birleşim: lojistik + Dixon-Coles + Elo | 0.1941 | 0.9621 | 0.5715 | %55.1 | 0.008 |
| Lojistik + Platt kalibrasyonu | 0.1946 | 0.9645 | 0.5726 | %54.9 | 0.011 |
| Lojistik + isotonic kalibrasyon | 0.1952 | 0.9712 | 0.5745 | %54.5 | 0.005 |
| Elo | 0.1997 | 0.9788 | 0.5831 | %53.5 | 0.008 |
| Dixon-Coles | 0.2020 | 0.9899 | 0.5877 | %52.9 | 0.008 |

Sezon sezon RPS ayrıntısı: `reports/deneyler_sezonluk.csv`. Hiçbir deney şu anki modeli hiçbir sezonda belirgin biçimde geçmedi.

## Yorum

- **Kalibrasyon:** Şu anki model zaten iyi kalibre (ECE 0.006, kapanış oranından bile düşük). Platt ve isotonic düzeltmeleri skoru kötüleştirdi, çünkü düzeltilecek bir sapma yok ve düzeltme yalnızca gürültü ekliyor. Sayfadaki kalibrasyon tablosunda görülen sapma, az sayıda maçtan gelen tesadüf.
- **Dixon-Coles ve birleşim:** Dixon-Coles tek başına Elo'nun da gerisinde. Birleşimde ağırlıklar sezonlar ilerledikçe kendiliğinden lojistik modele kaydı (0.7 → 0.9). Yani Dixon-Coles'un modele kattığı yeni bilgi yok.
- **Rakip gücüne göre form:** Katsayı 0.48 gol / 100 Elo puanı. Fark sıfıra yakın; Elo farkı ve piyasa oranı bu bilgiyi zaten taşıyor.
- **Sonuç:** Günlük sistemde değişiklik yapılmadı. Mevcut verilerden (sonuçlar, şutlar, oranlar) çıkarılabilecek bilgi büyük ölçüde tükendi. Piyasaya yaklaşmanın ötesine geçmek için piyasanın geç fiyatladığı yeni bilgi gerekiyor: kesinleşmiş kadrolar, sakatlıklar ve xG.
