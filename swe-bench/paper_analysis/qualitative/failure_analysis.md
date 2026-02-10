# RQ3: Failure Mode Analysis

## Summary

| Metric | Claude | Llama |
|--------|--------|-------|
| **Total Runs** | 50 | 50 |
| **Successes** | 29 (58%) | 2 (4%) |
| **Failures** | 21 (42%) | 48 (96%) |

## Failure Mode Distribution

| Failure Mode | Claude | Llama | Description |
|--------------|--------|-------|-------------|
| **WRONG_FIX** | 21 (100%) | 37 (77%) | Submitted patch with incorrect logic |
| **EMPTY_PATCH** | 0 (0%) | 10 (21%) | No actual code changes made |
| **LOOP_DEATH** | 0 (0%) | 1 (2%) | Stuck in repetitive action loop |

## Key Findings

### 1. Claude's Failure Pattern: "Wrong but Thorough"
- **100% of Claude failures are WRONG_FIX**
- Claude always makes an attempt, never gives up
- Average steps in failed runs: 47.0 (similar to successful runs)
- Failure is due to incorrect understanding, not lack of effort

### 2. Llama's Failure Pattern: "Fast and Careless"
- **21% of failures are EMPTY_PATCH** (submitted nothing)
- Llama sometimes gives up before trying
- Average steps in failed runs: 16.3 (very fast)
- Failure is often due to insufficient exploration

### 3. The "Effort Gap"

| Metric | Claude Failures | Llama Failures |
|--------|-----------------|----------------|
| **Avg Steps** | 46.6 | 17.2 |
| **Min Steps** | 29 | 7 |
| **Max Steps** | 64 | 55 |

## Interpretation

1. **Claude fails despite trying hard**: All failures involve substantive patches that simply don't fix the bug correctly. This suggests the issue is understanding the problem, not effort.

2. **Llama fails by not trying**: 21% of failures involve no code changes at all. Llama rushes to submit without making changes. This is a behavioral issue, not a capability issue.

3. **Speed kills accuracy**: Llama's average of 7-36 steps is insufficient for complex bugs. Claude's 29-64 steps allows for more thorough exploration.

## Task-Level Analysis

### Tasks where Claude fails consistently (0% accuracy):
- **astropy-13236**: All 5 runs fail (WRONG_FIX)
- **astropy-13398**: All 5 runs fail (WRONG_FIX)  
- **astropy-13977**: All 5 runs fail (WRONG_FIX)

These are likely **hard tasks** that require specific domain knowledge.

### Tasks where Llama fails consistently:
- **astropy-12907**: 0/5 (Claude: 5/5) - Llama rushes, Claude succeeds
- **astropy-13453**: 0/5 (Claude: 5/5) - Same pattern
- **astropy-14096**: 0/5 (Claude: 5/5) - Same pattern

## Conclusion

**Claude's failures are "quality failures"** — it tries but gets the logic wrong.
**Llama's failures are "effort failures"** — it often doesn't try hard enough.

This explains why Claude's higher step count correlates with higher accuracy: 
more steps = more exploration = better understanding = correct fixes.
