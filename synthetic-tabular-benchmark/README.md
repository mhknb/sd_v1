## Synthetic Tabular Benchmark

Bu depo, sentetik tablo verisi üreten modelleri (ör. CTGAN) farklı veri kümeleri üzerinde karşılaştırmalı olarak değerlendirmek için modüler bir kıyaslama altyapısı sağlar.

### Hızlı Başlangıç

```bash
python run_benchmark.py --model ctgan --dataset adult
```

### Dizin Yapısı

```
synthetic-tabular-benchmark/
├── configs/
├── src/
├── experiments/
├── results/
├── notebooks/
└── tests/
```

### Özellikler
- Modüler model ve veri seti kayıt sistemi
- YAML tabanlı yapılandırma
- TSTR ve istatistiksel benzerlik metrikleri

### Kurulum

```bash
pip install -r requirements.txt
```


