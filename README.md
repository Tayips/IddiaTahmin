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
