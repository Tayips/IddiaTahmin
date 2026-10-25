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
- **Sakat/cezalı (API-Football):** her maç için "eksik" ve "şüpheli" oyuncu sayısı, 2022/23–2024/25.

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

## Sakat/cezalı oyuncunun katkısı

API-Football ücretsiz planı yalnızca 2022/23–2024/25 sezonlarını açıyor. Bu üç sezon için iki ligde 14.759 sakatlık/ceza kaydı indirildi (GitHub Actions'ta, depo sırrıyla). Her maç için iki takımın "eksik" ve "şüpheli" oyuncu sayısı ve eksik farkı modele eklendi.

Test: 2023/24 ve 2024/25, iki lig, 1482 maç (eğitim 2022/23'ten başlıyor). Yine aynı maçlarda "temel" ile "temel + sakatlık" kıyası:

| Temel model | RPS temel | RPS + sakatlık | İsabet temel | İsabet + sakatlık |
|---|---|---|---|---|
| Form (piyasasız) | 0.1939 | 0.1943 | %55.5 | %55.3 |
| Form + piyasa | 0.1887 | 0.1886 | %56.6 | %56.6 |

- Katkı yok. Piyasasız modelde biraz kötüleşiyor bile.
- Sebep: eksik oyuncu sayısı kaba bir ölçü (yedek kaleci ile en golcü oyuncu aynı sayılıyor) ve oranlar sakatlık haberlerini zaten fiyatlıyor.
- Oyuncu önemine göre ağırlıklandırmak (dakika, gol katkısı) her sezon için onlarca ek istek gerektiriyor.
- **Asıl engel:** ücretsiz plan bu sezonu (2026/27) hiç açmıyor. Yani sakatlık verisi günlük tahminde ücretli plan olmadan kullanılamıyor. Bu sonuçlarla ücretli plana geçmeyi önermiyorum.

Kadro (ilk 11) maçtan yaklaşık 1 saat önce açıklanıyor, günlük tahmin saatlerinde (08:00 ve 17:00) henüz yok; bu yüzden teste katılmadı.

Yeniden çalıştırmak için: **Actions → Faz 3 ölçüm → Run workflow** (iş akışı main'e birleştikten sonra görünür).
