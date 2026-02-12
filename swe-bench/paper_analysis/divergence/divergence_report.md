# Divergence Step Analysis

## Summary

**Key Finding**: Claude's runs diverge **2.2× later** than Llama's.

| Model | Mean Divergence Step | Median | % at Step 1 | % by Step 5 |
|-------|---------------------|--------|-------------|-------------|
| Claude | 3.1 | 3 | 20% | 90% |
| Llama | 1.4 | 1 | 60% | 100% |

## Statistical Significance

- **t-test**: t = 2.57, p = 0.0192
- **Mann-Whitney U**: U = 82.0, p = 0.0062
- **Effect size**: Cohen's d = 1.15 (large)

## Interpretation

1. **Claude is more consistent early**: Only 20% of Claude tasks diverge at Step 1, vs 60% for Llama.

2. **Llama diverges immediately**: 100% of Llama tasks diverge by Step 5; runs quickly take different paths.

3. **Claude's early consistency**: 90% of Claude tasks diverge by Step 5, but with a later start (Step 3 median).

4. **This explains the CV difference**: Claude's higher consistency (CV 15% vs 47%) is rooted in consistent early strategy, not late convergence.

## First Divergent Actions

Most common divergence patterns:
- **Claude**: EXPLORE vs UNDERSTAND (agents chose different investigation approaches)
- **Llama**: EXPLORE vs LOCATE (agents chose different search strategies)

## Implications for Paper

> **Variance originates in early interpretation, not late execution.**
>
> Claude's runs share a consistent strategy for the first 3 steps on average, 
> then diverge in execution details. Llama's runs diverge immediately, 
> explaining higher variance throughout.

## Figures

- `fig_divergence_distribution.png`: Histogram of divergence steps
- `fig_divergence_heatmap.png`: Per-task, per-step divergence visualization
- `fig_sequence_alignment.png`: Action sequence comparison for astropy-13236

## Per-Task Results

| Task | Claude Div. | Llama Div. | Difference |
|------|-------------|------------|------------|
| astropy-12907 | 1 | 1 | 0 |
| astropy-13033 | 4 | 2 | +2 |
| astropy-13236 | 1 | 1 | 0 |
| astropy-13398 | 8 | 2 | +6 |
| astropy-13453 | 2 | 1 | +1 |
| astropy-13579 | 3 | 1 | +2 |
| astropy-13977 | 3 | 2 | +1 |
| astropy-14096 | 3 | 1 | +2 |
| astropy-14182 | 4 | 1 | +3 |
| astropy-14309 | 2 | 2 | 0 |

## Conclusion

Claude's runs share a common early strategy, diverging only in later execution.
This supports **Claim 1** (Claude is 3.1× more consistent) and explains the mechanism:
consistency comes from **early agreement on approach**, not late convergence.
