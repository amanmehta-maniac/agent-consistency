# Combined 40-Question Analysis Summary

## Dataset
- **Total questions**: 40
- **Hard questions**: 20
- **Easier questions**: 20

## Behavioral Consistency Categories

| Category | Count | Hard | Easier |
|----------|-------|------|--------|
| Consistent-Correct | 14 | 0 | 14 |
| Consistent-Wrong | 4 | 4 | 0 |
| Inconsistent | 22 | 16 | 6 |

## Step Progression (Layer 32)

| Step | Pearson r | p-value | n |
|------|-----------|---------|---|
| 1 | 0.0332 | 0.8388 | 40 |
| 2 | 0.0551 | 0.7358 | 40 |
| 3 | -0.1596 | 0.3254 | 40 |
| 4 | -0.4363** | 0.0049 | 40 |
| 5 | -0.2981 | 0.1162 | 29 |

## Layer-wise Correlation at Step 3

| Layer | Pearson r | p-value | n |
|-------|-----------|---------|---|
| 0 | -0.0247 | 0.8799 | 40 |
| 8 | -0.3208** | 0.0436 | 40 |
| 16 | -0.2574 | 0.1088 | 40 |
| 24 | -0.1382 | 0.3952 | 40 |
| 32 | -0.1596 | 0.3254 | 40 |
| 40 | -0.2368 | 0.1413 | 40 |
| 48 | -0.2658 | 0.0974 | 40 |
| 56 | -0.2929 | 0.0666 | 40 |
| 64 | -0.3155** | 0.0473 | 40 |
| 72 | -0.3325** | 0.0361 | 40 |
| 80 | -0.3563** | 0.0240 | 40 |

## Consistent-Correct vs Inconsistent Comparison

| Metric | Value |
|--------|-------|
| CC Mean Similarity | 0.9551 ± 0.0214 |
| Inc Mean Similarity | 0.9613 ± 0.0333 |
| T-statistic | -0.5986 |
| P-value | 0.5534 |
| Cohen's d | -0.2200 |

## Threshold Classifier Performance

| Metric | Value |
|--------|-------|
| ROC AUC | 0.2995 |
| Optimal Threshold | 0.8980 |
| Accuracy | 0.4000 |
| Precision | 0.3684 |
| Recall | 1.0000 |
| F1 Score | 0.5385 |

## Key Findings

1. **Step 3 peak**: Not significant (r=-0.160)
2. **Consistent-correct vs inconsistent**: No significant difference (p=0.5534)
3. **Classifier performance**: AUC=0.299, suggesting poor predictive ability

---
*Generated from combined analysis of 40 HotpotQA questions*
