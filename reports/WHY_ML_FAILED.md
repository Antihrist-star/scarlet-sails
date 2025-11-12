# WHY ML MODEL FAILED - ROOT CAUSE ANALYSIS

**Date:** 2025-11-12
**Status:** 🔴 INVESTIGATING
**Priority:** CRITICAL

---

## 🚨 SYMPTOMS

### 1. EXCESSIVE SIGNALS
```
BTC_15m:  266,491 trades (Rule-Based: 24,734) = 10.8x more
ETH_15m:  266,491 trades (Rule-Based: 26,895) = 9.9x more
LTC_15m:  266,485 trades (Rule-Based: 26,040) = 10.2x more
```

**SUSPICIOUS:** All three assets generate ~266K trades (almost identical!)

### 2. HIGH OUT-OF-DISTRIBUTION RATIO
```
BTC_15m:  OOD 98.3%
ETH_15m:  OOD 98.3%
LTC_15m:  OOD 98.3%

BUT:

BTC_1h:   OOD 0.0%
BTC_4h:   OOD 0.0%
BTC_1d:   OOD 0.0%
```

**QUESTION:** Why only 15m timeframe has 98% OOD?

### 3. POOR PERFORMANCE
```
ML Model:
- Win Rate: 38.2% (worse than Rule-Based 46.1%)
- Profit Factor: 1.09 (barely profitable)
- Total trades: 1.7M (5x more than Rule-Based)
```

### 4. NORMALIZED MODEL DIDN'T HELP
```
Old model (absolute features):  OOD ~58%, 0 trades
New model (normalized features): OOD 98.3%, 1.7M trades
```

**PARADOX:** Normalized model has HIGHER OOD!

---

## 🔍 HYPOTHESES

### Hypothesis 1: Threshold Too Low
- Current threshold: 0.65
- Maybe model outputs probabilities ~0.66-0.70 for most bars
- Solution: Raise threshold to 0.85-0.90

**Likelihood:** MEDIUM
**Test:** Run diagnostic on probability distribution

---

### Hypothesis 2: Model Trained on Wrong Data
- Model trained on 2018-2020 (bear market)
- Testing on 2024-2025 (bull market)
- Even normalized features differ in bull vs bear

**Likelihood:** HIGH
**Test:** Check scaler statistics, compare training vs test distributions

---

### Hypothesis 3: Data Leakage During Training
- Forward-looking features accidentally included
- Model "knows" future outcomes
- Result: Overconfident predictions

**Likelihood:** MEDIUM
**Test:** Review training code for lookahead bias

---

### Hypothesis 4: Normalization Bug
- Some features still absolute (not normalized)
- Or normalization done wrong (e.g., on wrong axis)
- Result: Scaler sees test data as outliers

**Likelihood:** HIGH
**Test:** Inspect `retrain_xgboost_normalized.py` line by line

---

### Hypothesis 5: 15m Data Has Different Structure
- 15m has more bars → more noise
- Higher frequency = different statistical properties
- Scaler trained mostly on 1h/4h data?

**Likelihood:** MEDIUM
**Test:** Check training data composition

---

### Hypothesis 6: Model Collapsed
- XGBoost overfitted to training set
- Output probabilities clustered around few values
- Result: Almost all bars look "good"

**Likelihood:** HIGH
**Test:** Check unique probability values (should be >10,000)

---

## 🧪 TESTS TO RUN

### ✅ Test 1: Probability Distribution Analysis
**Script:** `diagnose_ml_deep.py`
**What:** Check how probabilities are distributed
**Expected:** Normal distribution around 0.5, wide range
**If failed:** Model collapsed or overfit

---

### ⏳ Test 2: Feature Importance Analysis
**Script:** TODO
**What:** Which features drive predictions most?
**Expected:** Balanced importance across features
**If failed:** Model relies on 1-2 features (overfitting)

---

### ⏳ Test 3: Training Data Inspection
**Script:** TODO
**What:** What data was model trained on?
**Expected:** Representative mix of bull/bear/sideways
**If failed:** Training data not diverse enough

---

### ⏳ Test 4: Normalization Code Review
**Script:** Review `retrain_xgboost_normalized.py`
**What:** Check normalization logic line by line
**Expected:** All features properly normalized
**If failed:** Bug in normalization code

---

### ⏳ Test 5: Cross-Timeframe Test
**Script:** Test same model on BTC only across all TF
**What:** Does OOD pattern hold?
**Expected:** Similar OOD across all TF
**If failed:** 15m is special case

---

## 📊 FINDINGS

### Finding 1: [PENDING]
**Test:** Probability distribution
**Result:** [Run diagnose_ml_deep.py first]
**Conclusion:** [TBD]

---

### Finding 2: [PENDING]
**Test:** Feature importance
**Result:** [TBD]
**Conclusion:** [TBD]

---

## ✅ ROOT CAUSE

**Status:** 🔴 NOT YET IDENTIFIED

**Leading theory:** [TBD after tests]

---

## 🔧 SOLUTION

**Proposed fix:** [TBD after root cause identified]

**Alternative approaches:**
1. Retrain model from scratch with better data
2. Use different model (LightGBM, CatBoost)
3. Ensemble multiple models
4. Fall back to Rule-Based only

---

## 📝 NOTES

- Normalized model was supposed to solve OOD problem
- Instead, OOD got WORSE (58% → 98.3%)
- This suggests fundamental issue with training data or model architecture
- Need to go back to basics and verify every step

---

## 🎯 ACTION ITEMS

- [ ] Run `diagnose_ml_deep.py` on BTC_15m
- [ ] Analyze probability distribution
- [ ] Check feature importance
- [ ] Review normalization code
- [ ] Inspect training data
- [ ] Test on single timeframe first
- [ ] Document all findings here
- [ ] Decide: fix or retrain from scratch?

---

**Last updated:** 2025-11-12 (initial creation)
