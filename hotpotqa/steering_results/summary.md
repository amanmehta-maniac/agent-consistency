# Activation Steering Experiment Results

## Overview

- **Questions tested**: 5
- **Steering scales**: [0.5, 1.0, 2.0]
- **Intervention**: Layer 72, Step 4

## Summary Table

| Question ID | Orig CV | Baseline CV | Scale 0.5 CV | Scale 1.0 CV | Scale 2.0 CV | Baseline Acc | Best Steered Acc |
|-------------|---------|-------------|--------------|--------------|--------------|--------------|------------------|
| 5ab3b0bf5542... | 0.542 | 0.306 | 0.000 | 0.000 | 0.000 | 0.75 | 0.00 |
| 5ab3e4565542... | 0.377 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 | 0.00 |
| 5ae0d4c95542... | 0.353 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 | 0.00 |
| 5ab56e325542... | 0.352 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 | 0.00 |
| 5a85ea095542... | 0.324 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 | 0.00 |

## Aggregate Statistics

- **Mean baseline CV**: 0.061 (std=0.122)
- **Mean CV at scale 0.5**: 0.000 (std=0.000)
- **Mean CV at scale 1.0**: 0.000 (std=0.000)
- **Mean CV at scale 2.0**: 0.000 (std=0.000)

## CV Change Analysis

| Question | Best Scale | CV Reduction | % Change |
|----------|------------|--------------|----------|
| 5ab3b0bf5542... | 0.5 | 0.306 | 100.0% |
| 5ab3e4565542... | None | (no improvement) | - |
| 5ae0d4c95542... | None | (no improvement) | - |
| 5ab56e325542... | None | (no improvement) | - |
| 5a85ea095542... | None | (no improvement) | - |

## Key Findings

1. **Consistency Impact**: [To be filled based on results]
2. **Accuracy Impact**: [To be filled based on results]
3. **Optimal Scale**: [To be filled based on results]

## Plots

- `cv_comparison.png`: CV across conditions
- `accuracy_comparison.png`: Accuracy across conditions
- `dose_response.png`: CV vs steering scale