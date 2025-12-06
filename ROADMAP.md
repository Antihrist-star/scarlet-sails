# ScArlet-Sails Roadmap

## Vision

**Что мы строим:**
Систему, где quant-стратегии дают численные сигналы, LLM Council интерпретирует контекст, а человек принимает финальное решение.

**Зачем:**
Не найти "идеальный edge", а построить процесс принятия решений, который:
- учится на своих ошибках,
- контролирует риск,
- генерирует данные для исследований.

**Философия:**
Council — не автопилот, а "вторая голова". Человек несёт риск и ответственность. Система помогает видеть то, что человек может пропустить.

---

## Success Definition

### Через 6 месяцев — три слоя успеха:

**1. Практика (P&L):**
- Paper trading показывает стабильную положительную доходность
- Дневная просадка <= заданного лимита
- Предсказуемое поведение, не "один раз выстрелили"

**2. Research:**
- Датасет решений: P_rb, P_ml, P_hyb, Council, человек, исходы
- Dispersion analysis с p-value
- Черновик выводов уровня технического отчёта

**3. Система:**
- Команда работает по процессу
- RAG пополняется системно
- Есть каденция: ретро, разбор ошибок, улучшение

### "Не зря" — минимум:
- Архитектура задокументирована
- Датасет: S(t), решения, исходы
- Прокачан процесс принятия решений

---

## Current State

### Готово ✅

**Инфраструктура:**
- 59 parquet файлов (14 монет × 4 TF)
- Feature engine v2 (74 признака)
- DVC версионирование

**Quant:**
- Rule-Based Strategy (P_rb)
- XGBoost ML Strategy (P_ml)
- OpportunityScorer, AdvancedRiskPenalty

**Council каркас:**
- CanonicalState (core/canonical_state.py)
- RAG Retriever (rag/retriever.py)
- BaseAgent (council/base_agent.py)

**Документация:**
- README.md, ARCHITECTURE.md
- Configs (council.yaml, risk_limits.yaml)

### Не готово ❌

- Hybrid Strategy (P_hyb)
- Council contracts (Stage 0/1/2/3)
- Quant Aggregator
- Human Interface
- Position sizing по формуле
- Kill-switch
- Dispersion analysis module
