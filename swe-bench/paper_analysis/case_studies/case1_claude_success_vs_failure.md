# Case Study 1: Claude Success vs Failure on Same Task

## Task: astropy-14182 (Subclassed SkyCoord attribute error)

### Run 3: SUCCESS (46 steps)

**Phase breakdown:**
1. EXPLORE (12 steps): Systematically explored astropy structure
2. LOCATE (5 steps): Found sky_coordinate.py via grep
3. UNDERSTAND (15 steps): Read __getattr__ implementation carefully
4. EDIT (8 steps): Modified attribute lookup logic
5. VERIFY (6 steps): Tested with reproduction script

**Key decision**: Read the full __getattr__ method before editing.

### Run 4: FAILURE (52 steps)

**Phase breakdown:**
1. EXPLORE (10 steps): Similar exploration
2. LOCATE (6 steps): Found same file
3. UNDERSTAND (8 steps): Skimmed code faster
4. EDIT (18 steps): Multiple edit attempts
5. VERIFY (10 steps): Repeated testing failures

**Key difference**: Less time understanding, more time in edit-test loop.

### Insight

The successful run spent **15 steps understanding** vs **8 steps** in the failed run.
Extra understanding time → correct edit on first attempt → success.
