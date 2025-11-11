# 🧹 GitHub Cleanup List

## ✅ БЕЗОПАСНО УДАЛИТЬ (старая документация)

### Устаревшие MD/TXT файлы (186K total):
- [ ] `PROJECT_INVENTORY_DAY7.txt` (147K)
- [ ] `FILE_INVENTORY.md` (15K)
- [ ] `COMMIT_CHECKLIST.md` (11K)
- [ ] `DAY12_FINAL_SUMMARY.md` (9.5K)
- [ ] `GITHUB_STATUS_REPORT.md` (11K)
- [ ] `HYBRID_COMPONENTS_STATUS.md` (8.2K)

**Причина:** Устарели, актуальная документация в README.md и HYBRID_SYSTEM_ARCHITECTURE.md

---

## ✅ БЕЗОПАСНО УДАЛИТЬ (версионные скрипты)

### Старые версии prepare_data (заменены prepare_data_v4.py):
- [ ] `scripts/prepare_data_v2.py` (6.3K)
- [ ] `scripts/prepare_data_v3.py` (8.1K)
- [ ] `scripts/prepare_data_v5_triple.py` (0 bytes - пустой!)
- [ ] `scripts/prepare_data_v4_plan.py` (790 bytes - план, не код)
- [ ] `scripts/prepare_data_with_features_v2.py` (7.4K)

### Старые версии train_model (заменены train_xgboost.py):
- [ ] `scripts/train_model_v2.py` (3.6K)
- [ ] `scripts/train_model_v5_final.py` (0 bytes - пустой!)
- [ ] `scripts/train_model_enriched_v2.py` (5.7K)
- [ ] `scripts/train_v5_improved.py` (9.7K)

### Старые версии backtest:
- [ ] `scripts/run_backtest_v2.py` (2.3K)
- [ ] `scripts/run_backtest_enriched_v2.py` (4.9K)
- [ ] `scripts/optimize_threshold_by_pf_v2.py` (9.8K)

**Причина:** Заменены актуальными версиями, v2/v3/v5 больше не используются

**KEEP:** `scripts/prepare_data_v4.py` (используется в текущей системе!)

---

## ✅ БЕЗОПАСНО УДАЛИТЬ (старые отчеты)

### Огромные CSV (1.4M!):
- [ ] `reports/day11_forensics/all_trades_detailed.csv` (1.4M)

### Устаревшие PNG (900K total):
- [ ] `reports/backtest_enriched_v2_results.png` (132K)
- [ ] `reports/backtest_v2_results.png` (127K)
- [ ] `reports/xgboost_046_equity.png` (96K)
- [ ] `reports/threshold_optimization_v2.png` (314K)
- [ ] `reports/btc_full_history.png` (71K)

**Причина:** Старые результаты, актуальные в `reports/hybrid_backtest/`

**KEEP:**
- `reports/regime_detection_analysis.png` (493K) - актуален
- `reports/final_xgboost_equity.png` (96K) - финальная версия
- `reports/xgboost_backtest_equity_curve.png` (89K) - актуален

---

## ⚠️ ПРОВЕРИТЬ ПЕРЕД УДАЛЕНИЕМ

### Неиспользуемые модели (?):
- [ ] `models/best_cnn_model.pth` (165K) - CNN модель
- [ ] `models/logistic_enriched_v2_metadata.json` (244 bytes)

**Действие:** Проверить используются ли CNN и Logistic модели

---

## 📊 ИТОГО К УДАЛЕНИЮ:

- **Документация:** ~186 KB
- **Старые скрипты:** ~47 KB
- **Старые отчеты:** ~2.1 MB
- **Модели (?):** ~165 KB

**Всего освободим:** ~2.5 MB в Git истории
**Файлов:** ~30 штук

---

## 🚀 КОМАНДЫ ДЛЯ УДАЛЕНИЯ:

```bash
# Старая документация
git rm PROJECT_INVENTORY_DAY7.txt FILE_INVENTORY.md COMMIT_CHECKLIST.md DAY12_FINAL_SUMMARY.md GITHUB_STATUS_REPORT.md HYBRID_COMPONENTS_STATUS.md

# Версионные скрипты
git rm scripts/prepare_data_v2.py scripts/prepare_data_v3.py scripts/prepare_data_v5_triple.py scripts/prepare_data_v4_plan.py scripts/prepare_data_with_features_v2.py
git rm scripts/train_model_v2.py scripts/train_model_v5_final.py scripts/train_model_enriched_v2.py scripts/train_v5_improved.py
git rm scripts/run_backtest_v2.py scripts/run_backtest_enriched_v2.py scripts/optimize_threshold_by_pf_v2.py

# Старые отчеты
git rm reports/day11_forensics/all_trades_detailed.csv
git rm reports/backtest_enriched_v2_results.png reports/backtest_v2_results.png reports/xgboost_046_equity.png reports/threshold_optimization_v2.png reports/btc_full_history.png
```
