# Llama's 2 Successful Runs — Deep Analysis

Llama achieved only **2/50 successful runs (4% accuracy)**. These 2 successes are critical to understand.

---

## 1. Basic Info: Which runs succeeded?

| Task ID | Run # | Steps |
|---------|-------|-------|
| astropy-14309 | Run 4 | 8 steps |
| astropy-13236 | Run 5 | 13 steps |

**Key observation**: Both successes came from **later runs** (Run 4 and Run 5), not early runs.

---

## 2. Compare to failures on same task

### astropy-14309 (ECSV format identification)

| Run | Steps | Status |
|-----|-------|--------|
| 1 | 11 | ❌ FAIL |
| 2 | 31 | ❌ FAIL |
| 3 | 16 | ❌ FAIL |
| **4** | **8** | **✓ SUCCESS** |
| 5 | 7 | ❌ FAIL |

- **Mean**: 14.6 steps, **Std**: 9.8, **CV**: 67.2%
- **Successful run took FEWER steps than average** (-6.6 steps)
- Paradox: Run 5 took even fewer steps (7) but failed!

### astropy-13236 (NdarrayMixin subclass warning)

| Run | Steps | Status |
|-----|-------|--------|
| 1 | 7 | ❌ FAIL |
| 2 | 21 | ❌ FAIL |
| 3 | 11 | ❌ FAIL (empty patch) |
| 4 | 14 | ❌ FAIL |
| **5** | **13** | **✓ SUCCESS** |

- **Mean**: 13.2 steps, **Std**: 5.1, **CV**: 38.8%
- **Successful run was near average** (-0.2 steps)

---

## 3. Trajectory Comparison

### astropy-14309 Run 4 (SUCCESS, 8 steps)

```
1. ls -la
2. cat astropy/io/registry/base.py
3. grep -n "identify_format" astropy/io/registry/base.py
4. sed -n '280,290p' astropy/io/registry/base.py
5. grep -n "args\[" astropy/io/registry/base.py
6. cat astropy/io/fits/connect.py
7. sed -i 's/return isinstance(args\[0\], (HDUList, TableHDU, BinTableHDU, GroupsHDU...
8. cat <<'EOF' > test.py
   from astropy.io.registry import identify_format
   ...
```

**Pattern**: Explore → Read → Locate issue → ONE targeted edit → Verify

### astropy-14309 Run 1 (FAIL, 11 steps)

```
1. ls -la
2. ls -la astropy
3. ls -la astropy/io
4. ls -la astropy/io/registry
5. cat astropy/io/registry/base.py
6. grep -n "def identify_format" astropy/io/registry/base.py
7. sed -n '282,300p' astropy/io/registry/base.py
8. grep -n "def is_fits" astropy/io/fits/connect.py
9. sed -n '48,55p' astropy/io/fits/connect.py
10. sed -n '72p' astropy/io/fits/connect.py
11. sed -i '72s/return/isinstance(args[0], (HDUList, TableHDU, BinTableHDU, GroupsHD...
```

**Pattern**: Over-explored (4× `ls`) → Read correctly → Made WRONG edit

**Divergence point**: Step 7. Success run did `sed -n '280,290p'`, failure did `sed -n '282,300p'`. Slightly different context led to different understanding.

### astropy-13236 Run 5 (SUCCESS, 13 steps)

```
1. grep -r "if (not isinstance(data, Column) and not data_is_mixin" .
2. sed -i '/if (not isinstance(data, Column) and not data_is_mixin/a \        import warnings...'
3. grep "FutureWarning" astropy/table/table.py
4. nl -ba astropy/table/table.py | sed -n '/FutureWarning/,/if (not isinstance...'
5. sed -i '/if (not isinstance(data, Column) and not data_is_mixin/a \        # ' ...
6. nl -ba astropy/table/table.py | sed -n '/if (not isinstance(data, Column)...'
7. sed -i '/data = data.view(NdarrayMixin)/a \        # ' ...
8. nl -ba astropy/table/table.py | sed -n '/if (not isinstance(data, Column)...'
9. echo "import numpy as np; from astropy.table import Table; data = np.array([(1, ..."
10. sed -i '1246s/^/        /' astropy/table/table.py
11. python test.py
12. sed -i '/if (not isinstance(data, Column) and not data_is_mixin/a \        import warnings...'
13. python test.py
```

**Pattern**: Locate → Edit → Check → Edit → **Test → Iterate until pass**

### astropy-13236 Run 1 (FAIL, 7 steps)

```
1. find . -type f -name "*Table*" -o -name "*Column*"
2. grep -r "Table" * | grep -v "test"
3. grep -r "NdarrayMixin" *
4. cat astropy/table/ndarray_mixin.py astropy/table/table.py
5. sed -i '/isinstance(data, np.ndarray) and len(data.dtype) > 1/i \        warning...'
6. sed -i '/warnings.warn("In the future, structured arrays will be added as Column...'
7. git diff
```

**Pattern**: Locate → Read → Edit → Edit → **Submit without testing**

**Divergence point**: Run 5 **tested after edits** (steps 11, 13). Run 1 **never tested**.

---

## 4. Behavioral Analysis

### Phase breakdown for astropy-14309:

| Run | Status | EXPLORE | LOCATE | UNDERSTAND | EDIT | VERIFY |
|-----|--------|---------|--------|------------|------|--------|
| 1 | FAIL | 4 | 2 | 4 | 1 | 0 |
| 2 | FAIL | 23 | 0 | 7 | 0 | 0 |
| 3 | FAIL | 1 | 2 | 1 | 6 | 6 |
| **4** | **SUCCESS** | **1** | **2** | **4** | **1** | **0** |
| 5 | FAIL | 0 | 1 | 3 | 1 | 2 |

**Insight**: Success run had BALANCED phases — 1 explore, 2 locate, 4 understand, 1 edit.
Failure Run 2 had 23 explore steps (over-explored, never edited!)

### Phase breakdown for astropy-13236:

| Run | Status | EXPLORE | LOCATE | UNDERSTAND | EDIT | VERIFY |
|-----|--------|---------|--------|------------|------|--------|
| 1 | FAIL | 0 | 3 | 1 | 2 | 0 |
| 2 | FAIL | 0 | 3 | 7 | 5 | 6 |
| 3 | FAIL | 0 | 3 | 6 | 2 | 0 |
| 4 | FAIL | 0 | 1 | 8 | 5 | 0 |
| **5** | **SUCCESS** | **0** | **2** | **3** | **6** | **2** |

**Insight**: Success run had **VERIFY steps** (python test.py). Runs 1, 3, 4 had ZERO verify steps.

### Key patterns in successful runs:

1. **Read before edit**: Both successes had UNDERSTAND steps before EDIT
2. **Test after edit**: astropy-13236 success tested with `python test.py`
3. **Focused edits**: Both made 1-6 targeted edits, not random attempts
4. **Not too much exploration**: Avoid getting lost (unlike Run 2's 23 explore steps)

---

## 5. Compare to Claude

### Performance on the same 2 tasks:

| Task | Claude | Llama | Winner |
|------|--------|-------|--------|
| astropy-14309 | 5/5 (100%) | 1/5 (20%) | **Claude** |
| astropy-13236 | 0/5 (0%) | 1/5 (20%) | **Llama** |

**Surprising finding**: Llama beat Claude on astropy-13236!
- Claude: 0% accuracy despite 40.6 avg steps
- Llama: 20% accuracy with only 13.2 avg steps

### Why did Llama beat Claude on astropy-13236?

**Claude's approach** (all 5 runs failed):
- Spent 40+ steps investigating
- Made complex, multi-part fixes
- Over-engineered the solution

**Llama's successful approach** (Run 5):
- Went straight to the issue with `grep -r`
- Made simple, direct edit
- Tested and iterated

**This is an example where Claude's thoroughness backfired.**

### Trajectory comparison: astropy-14309

**Claude Run 1 (SUCCESS, 48 steps):**
```
1. find /testbed -type f -name "*.py" | grep -E "(registry|connect)" | head -20
2. cat /testbed/astropy/io/fits/connect.py
3. sed -n '50,75p' /testbed/astropy/io/fits/connect.py
4. sed -n '300,320p' /testbed/astropy/io/registry/base.py
5. cat <<'EOF' > /testbed/test_issue.py
6. cd /testbed && python test_issue.py
7. cd /testbed && git log --oneline --all | grep -i "2a0c5c6" | head -5
8. cd /testbed && git show 2a0c5c6f5 --stat
9. cd /testbed && git show 2a0c5c6f5 -- astropy/io/fits/connect.py
10. sed -i '72s/return isinstance(args\[0\], (HDUList, TableHDU, BinTableHDU...
... (38 more steps)
```

**Llama Run 4 (SUCCESS, 8 steps):**
```
1. ls -la
2. cat astropy/io/registry/base.py
3. grep -n "identify_format" astropy/io/registry/base.py
4. sed -n '280,290p' astropy/io/registry/base.py
5. grep -n "args\[" astropy/io/registry/base.py
6. cat astropy/io/fits/connect.py
7. sed -i 's/return isinstance(args\[0\], (HDUList, TableHDU, BinTableHDU...
8. cat <<'EOF' > test.py
```

**Both succeeded, but Llama was 6× faster** (8 vs 48 steps).

---

## 6. Key Insights

### Why did these 2 Llama runs succeed?

1. **Right task complexity**: Both tasks had relatively simple fixes
2. **Luck in file discovery**: First search hit the right file
3. **Testing behavior**: astropy-13236 success tested after edits
4. **Balanced approach**: Not too much exploration, not too little

### Why is Llama's success rate so low (4%)?

1. **No testing**: Most runs never run `python` or `pytest`
2. **Premature submission**: Submits after 7-10 steps without verification
3. **Wrong file targeting**: Often edits wrong file
4. **Empty patches**: 21% of failures had no code changes

### The "Lucky Path" hypothesis

Llama can succeed when:
- ✓ Task has simple, localized fix
- ✓ Bug is in obvious location
- ✓ First search hits right file
- ✓ Agent happens to test after editing

But on complex tasks requiring:
- ✗ Multi-file changes
- ✗ Understanding class hierarchies
- ✗ Non-obvious bug locations

...Llama's fast-but-shallow approach fails 96% of the time.

---

## Summary

| Metric | Llama Successes | Llama Failures |
|--------|-----------------|----------------|
| Avg steps | 10.5 | 17.5 |
| VERIFY steps | 1.0 | 0.8 |
| Read before edit | Yes | Sometimes |
| Test after edit | 50% | ~10% |

**Conclusion**: Llama's 2 successes were "lucky runs" — right task, right file, right approach. The 96% failure rate reflects Llama's inability to handle tasks requiring systematic exploration and verification.
