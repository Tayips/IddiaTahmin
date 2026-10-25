# IddiaTahmini

Futbol maç sonuçları için olasılık tahmini. Plan ve gereksinimler: proje araştırma sayfası.

## Faz 0: veri

```bash
pip install -r requirements.txt
python -m iddia.download --leagues T1 E0 --from 2014   # football-data.co.uk CSV'lerini data/raw altına indirir
python -m iddia.build                                   # data/processed/matches.csv üretir
pytest
```

`src` dizinini `PYTHONPATH`'e ekleyin veya `pip install -e .` kullanın.

Lig kodları football-data.co.uk ile aynıdır: `T1` Süper Lig, `E0` Premier Lig.

## Faz 1: temel modeller

```bash
pip install -e . scipy
python -m iddia.evaluate --test-seasons 5   # yaklaşık 2-3 dakika
```

Sonuçlar ve yorum: [reports/faz1.md](reports/faz1.md).

## Günlük tahmin sayfası

`python -m iddia.daily` her çalıştığında:

1. Geçmiş sezonları, bu sezonu ve yaklaşan maçları (`fixtures.csv`) indirir.
2. Elo modelini tüm geçmişle kurar ve yaklaşan Süper Lig ve Premier Lig maçlarına 1/X/2 olasılığı verir.
3. Tahminleri `data/predictions.csv` dosyasına yazar. Bir maçın tahmini başlama saatine kadar güncellenir, sonra değişmez.
4. Biten maçları puanlar ve `docs/index.html` sayfasını ile `docs/data.json` dosyasını üretir.

İlk çalıştırmada bu sezonun oynanmış maçları geriye dönük doldurulur: her maç yalnızca kendinden önceki sonuçlarla tahmin edilir. Sayfada bu maçlar "canlı" takipten ayrı sayılır.

GitHub Actions (`.github/workflows/daily.yml`) bunu her gün 05:00 ve 14:00 UTC'de çalıştırır. Sayfayı yayınlamak için bir kez: **Settings → Pages → Build and deployment → Source: Deploy from a branch → Branch: `main`, klasör `/docs` → Save**. Sayfa `https://tayips.github.io/IddiaTahmin/` adresinde açılır.

Elle çalıştırmak için: **Actions → Günlük tahmin → Run workflow**.

## Faz 2: özellikler

```bash
python -m iddia.compare   # yaklaşık 20 saniye
```

Sonuçlar: [reports/faz2.md](reports/faz2.md). Günlük sistem bu fazdan itibaren `form-piyasa-v2` modelini kullanır.
