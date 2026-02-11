# Claude's 21 Failures — Pattern Analysis

Claude failed **21/50 runs (42%)**. This analysis examines whether failures are random or systematic.

---

## 1. Failure Distribution by Task

| Task | Failures | Successes | Success Rate | Pattern |
|------|----------|-----------|--------------|---------|
| astropy-12907 | 0 | 5 | 100% | ALL_SUCCESS |
| astropy-13033 | 2 | 3 | 60% | MIXED |
| astropy-13236 | 5 | 0 | 0% | ALL_FAIL |
| astropy-13398 | 5 | 0 | 0% | ALL_FAIL |
| astropy-13453 | 0 | 5 | 100% | ALL_SUCCESS |
| astropy-13579 | 0 | 5 | 100% | ALL_SUCCESS |
| astropy-13977 | 5 | 0 | 0% | ALL_FAIL |
| astropy-14096 | 0 | 5 | 100% | ALL_SUCCESS |
| astropy-14182 | 4 | 1 | 20% | MIXED |
| astropy-14309 | 0 | 5 | 100% | ALL_SUCCESS |

### Summary:
- **ALL_FAIL tasks**: 3 (astropy-13236, astropy-13398, astropy-13977) → 15 failures
- **MIXED tasks**: 2 (astropy-13033, astropy-14182) → 6 failures
- **ALL_SUCCESS tasks**: 5 → 0 failures

**Key insight**: 71% of failures (15/21) come from just 3 tasks where Claude failed every run.

---

## 2. MIXED Result Tasks (Some Success, Some Fail)

### astropy-13033 (3/5 Success)

| Run | Steps | Success | Pattern |
|-----|-------|---------|---------|
| 1 | 35 | ❌ | +319 lines, 11 files |
| 2 | 46 | ✓ | +445 lines, 11 files |
| 3 | 38 | ✓ | +554 lines, 11 files |
| 4 | 44 | ❌ | +408 lines, 12 files |
| 5 | 50 | ✓ | +600 lines, 15 files |

**Observation**: All runs had similar approach. Success/failure appears **random** — small differences in implementation led to different test outcomes.

### astropy-14182 (1/5 Success)

| Run | Steps | Success | Pattern |
|-----|-------|---------|---------|
| 1 | 49 | ❌ | +480 lines, 11 files |
| 2 | 64 | ❌ | +372 lines, 12 files |
| 3 | 54 | ✓ | +489 lines, 10 files |
| 4 | 52 | ❌ | +334 lines, 11 files |
| 5 | 47 | ❌ | +16 lines, 1 file |

**Observation**: Run 3 succeeded with similar approach to failures. Run 5 was minimal but wrong. Success appears **close but wrong** — minor implementation differences matter.

---

## 3. ALL_FAIL Tasks (0/5 Success)

### astropy-13236 (5/5 Fail)
- **Pattern**: CONSISTENT_WRONG
- **Interpretation**: Claude added deprecation warnings instead of removing the behavior
- **Steps**: [31, 56, 29, 43, 44]
- **All 5 runs**: Same wrong approach (ADD_WARNING)

### astropy-13398 (5/5 Fail)
- **Pattern**: CONSISTENT_WRONG
- **Steps**: [48, 46, 55, 45, 45]
- **All 5 runs**: Similar wrong fix approach

### astropy-13977 (5/5 Fail)
- **Pattern**: CONSISTENT_WRONG
- **Steps**: [41, 53, 48, 59, 44]
- **All 5 runs**: Similar wrong fix approach

**Key insight**: On ALL_FAIL tasks, Claude makes the **same wrong interpretation every run**. This is systematic, not random.

---

## 4. Failure Categorization

| Category | Count | Tasks | Description |
|----------|-------|-------|-------------|
| **CONSISTENT_WRONG** | 15 (71%) | 13236, 13398, 13977 | Same wrong interpretation every run |
| **CLOSE_BUT_WRONG** | 4 (19%) | 14182 | Almost correct, minor implementation errors |
| **RANDOM_FAIL** | 2 (10%) | 13033 | Random variance in otherwise correct approach |

### Interpretation:

**71% of failures are CONSISTENT_WRONG** — Claude has systematic blind spots on certain task types.

These are not random failures. Claude consistently:
1. Misinterprets "remove X" as "add warning before X"
2. Over-engineers simple fixes
3. Maintains backward compatibility when tests expect breaking changes

---

## 5. VERIFY Analysis

### VERIFY Steps vs Success

| Metric | Successful Runs | Failed Runs |
|--------|-----------------|-------------|
| **Avg VERIFY steps** | 15.1 | 13.5 |

| VERIFY Steps | Successes | Failures | Success Rate |
|--------------|-----------|----------|--------------|
| 5 | 1 | 0 | 100% |
| 6 | 2 | 1 | 67% |
| 7 | 0 | 2 | 0% |
| 8 | 1 | 2 | 33% |
| 9 | 1 | 4 | 20% |

### Interpretation:

**Weak positive correlation** between VERIFY steps and success.

However, VERIFY steps don't help when the interpretation is wrong:
- All 5 runs of astropy-13236 had VERIFY steps
- All 5 still failed because they tested the wrong fix

**Testing helps execution, not interpretation.**

---

## 6. Key Insights

### The Two Failure Modes

1. **Interpretation Failures (71%)**: Claude understands the task wrong
   - No amount of testing fixes this
   - Same wrong approach across all runs
   - Example: astropy-13236 (add warning vs remove code)

2. **Execution Failures (29%)**: Claude understands correctly but makes errors
   - Testing can help catch these
   - Random variance between runs
   - Example: astropy-13033 (60% success rate)

### Why Consistency is High Despite Failures

Claude's CV is low (15.2%) because even on failed tasks:
- Same interpretation every run
- Similar step counts
- Similar approaches

**Consistency ≠ Correctness**. Claude is consistently wrong on hard tasks.

### Task Characteristics That Predict Failure

| Characteristic | Success Rate |
|----------------|--------------|
| "Add feature X" | High |
| "Fix bug by modifying X" | High |
| "Remove deprecated behavior" | **Low** |
| "Change default behavior" | **Low** |

Claude's bias toward backward compatibility makes it fail on "remove/change" tasks.

---

## Summary Table

| Pattern | Tasks | Failures | Key Issue |
|---------|-------|----------|-----------|
| CONSISTENT_WRONG | 3 | 15 (71%) | Wrong interpretation, same every run |
| CLOSE_BUT_WRONG | 1 | 4 (19%) | Right idea, minor implementation errors |
| RANDOM_FAIL | 1 | 2 (10%) | Random variance, mostly works |

**Conclusion**: Claude's failures are not random. 71% stem from systematic misinterpretation of certain task types. Improving testing won't help — Claude needs better task understanding.
