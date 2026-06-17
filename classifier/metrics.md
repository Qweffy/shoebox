# Expense-category classifier

TF-IDF (1-2 grams) + LinearSVC over SROIE receipt text. **Vendor-grouped, stratified 5-fold CV** (no vendor appears in both train and test → leakage-safe). n=987 receipts, classes=['dining', 'fuel', 'groceries', 'other', 'pharmacy', 'retail', 'services'].

> **Labels are weak** (keyword-derived; see `labels.py`). This measures whether TF-IDF generalizes the weak signal under honest CV, not gold-validated accuracy.

- **Macro-F1: 0.487**
- Majority-class baseline macro-F1: 0.088

## Class distribution

- dining: 183
- fuel: 21
- groceries: 218
- other: 440
- pharmacy: 22
- retail: 81
- services: 22

## Per-class report

```
              precision    recall  f1-score   support

      dining       0.81      0.43      0.56       183
        fuel       1.00      0.76      0.86        21
   groceries       0.70      0.46      0.56       218
       other       0.55      0.88      0.68       440
    pharmacy       0.00      0.00      0.00        22
      retail       0.33      0.05      0.09        81
    services       1.00      0.50      0.67        22

    accuracy                           0.61       987
   macro avg       0.63      0.44      0.49       987
weighted avg       0.62      0.61      0.57       987

```

## Confusion matrix

| gt \ pred | dining | fuel | groceries | other | pharmacy | retail | services |
|---|---|---|---|---|---|---|---|
| **dining** | 78 | 0 | 2 | 103 | 0 | 0 | 0 |
| **fuel** | 0 | 16 | 0 | 5 | 0 | 0 | 0 |
| **groceries** | 2 | 0 | 100 | 114 | 0 | 2 | 0 |
| **other** | 15 | 0 | 30 | 389 | 0 | 6 | 0 |
| **pharmacy** | 0 | 0 | 2 | 20 | 0 | 0 | 0 |
| **retail** | 0 | 0 | 7 | 70 | 0 | 4 | 0 |
| **services** | 1 | 0 | 1 | 9 | 0 | 0 | 11 |

