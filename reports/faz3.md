# Faz 3: xG ve sakat/cezalı oyuncu verisi

Yeniden üretmek için:

```bash
python -m iddia.understat                       # Premier Lig xG, 2014/15'ten bugüne (~1 dakika)
API_FOOTBALL_KEY=... python -m iddia.apifootball --seasons 2022 2023 2024   # sakat/cezalı listeleri
python -m iddia.compare                         # Faz 2 tablosu + ek veri katkısı (reports/faz3_ek_veri.csv)
```

## Veri

- **xG (Understat):** Premier Lig 2014/15–2026/27, 4630 maçın tamamı football-data maç tablosuyla eşleşti (lig + takım çifti, en fazla 1 gün tarih farkı). Maç başına xG, penaltısız xG, rakip ceza sahası yakınına pas (deep) ve PPDA (pres yoğunluğu).
- Understat Süper Lig'i kapsamıyor. football-data 2026/27'den itibaren iki lig için de xG veriyor; bu sezon ikisi Premier Lig'de 0.95 korelasyonlu. Understat'ın olduğu maçlarda onunki kullanılıyor.
- **Sakat/cezalı (API-Football):** her maç için "eksik" ve "şüpheli" oyuncu sayısı. Kod hazır, anahtar ortama eklenince ölçülecek.

## xG'nin katkısı

Test: Premier Lig, 2021/22–2025/26, 1896 maç. İki model aynı maçlarla eğitilip test edildi; tek fark xG özelliği.

| Temel model | RPS temel | RPS + xG | İsabet temel | İsabet + xG |
|---|---|---|---|---|
| Form (piyasasız) | 0.1988 | 0.1983 | %54.2 | %54.7 |
| Form + piyasa | 0.1958 | 0.1956 | %55.3 | %55.7 |

- Kazanç çok küçük: RPS farkı form modelinde −0.0006 (standart hata 0.0004), piyasalı modelde −0.0002 (standart hata 0.0002). İstatistiksel olarak anlamlı değil.
- Takım bazlı xG ortalamalarının hepsini (xG, penaltısız xG, deep, PPDA; 12 özellik) eklemek modeli kötüleştirdi. Farklı yarı ömürler (6, 10, 20, 38 maç) de sonucu değiştirmedi. Modelde yalnızca tek bir özellik kaldı: iki takımın xG averajı farkı.
- Yorum: xG takım gücünü goldan daha iyi ölçüyor, ama bu bilgi şut/isabetli şut formunda ve özellikle piyasa oranında zaten var. Oran, xG'yi izleyen binlerce kişinin görüşünü içeriyor.
- Günlük model bu yüzden değişmedi.

## Sakat/cezalı oyuncu

Bekliyor: `API_FOOTBALL_KEY`. Ücretsiz plan günde 100 istek veriyor; lig ve sezon başına 2 istek (fikstür + sakatlık listesi) yeterli.
Kadro (ilk 11) maçtan yaklaşık 1 saat önce açıklanıyor, günlük tahmin saatlerinde (08:00 ve 17:00) henüz yok; bu yüzden geçmiş testine katılmadı.
