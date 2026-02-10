# Case Study 2: Llama's Loop Death (astropy-14182 Run 4)

## Task: astropy-14182

### Run Details: 55 steps, FAILURE

**Action sequence (last 15 steps):**
```
Step 41: python test.py  # Test fails
Step 42: cat sky_coordinate.py | head -100
Step 43: sed -i 's/...'  # Minor edit
Step 44: python test.py  # Test fails
Step 45: cat sky_coordinate.py | head -100
Step 46: sed -i 's/...'  # Another edit
Step 47: python test.py  # Test fails
...
Step 55: submit  # Gave up, submitted broken code
```

### Pattern: Edit-Test-Edit Loop

Llama fell into a repetitive loop:
1. Make small edit
2. Run test
3. Test fails
4. Read same code section
5. Make another small edit
6. Repeat

### Why This Happened

- Insufficient initial exploration (only 2 steps)
- Never read the full method context
- Each edit was a guess, not informed by understanding
- No strategy change despite repeated failures

### Contrast with Claude

Claude's approach on same task:
1. Explored 10+ files to understand class hierarchy
2. Read __getattr__ in context of parent classes
3. Made one correct edit
4. Verified once, passed

### Insight

Loop death is a symptom of insufficient understanding.
More upfront exploration prevents trial-and-error loops.
