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
