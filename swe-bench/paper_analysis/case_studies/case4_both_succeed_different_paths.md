# Case Study 4: Both Models Succeed, Different Paths

## Task: astropy-14309 (ECSV format identification)

### Claude Run 1: 45 steps, SUCCESS

**Approach: Systematic exploration**
```
Steps 1-8: Explore io/ascii directory structure
Steps 9-15: Read ecsv.py and related files
Steps 16-20: Understand format identification logic
Steps 21-30: Study identify_format function
Steps 31-40: Implement fix with full context
Steps 41-45: Verify and submit
```

**Key characteristics:**
- Read 6 related files before editing
- Understood the registration system
- Made comprehensive fix

### Llama Run 4: 8 steps, SUCCESS

**Approach: Direct targeting**
```
Step 1: grep -r "identify_format" 
Step 2: cat io/ascii/ecsv.py | head -100
Step 3: sed -i 's/...'  # Direct edit
Step 4: python -c "..."  # Quick test
Step 5-7: Minor adjustments
Step 8: submit
```

**Key characteristics:**
- Went directly to the obvious file
- Made minimal edit
- Lucky that simple fix worked

### Comparison

| Metric | Claude | Llama |
|--------|--------|-------|
| Steps | 45 | 8 |
| Files read | 6 | 1 |
| Edits made | 1 | 2 |
| Tests run | 3 | 2 |

### Insight

**Same outcome, different reliability:**

Claude's thorough approach works consistently across runs (5/5 success).
Llama's fast approach only works sometimes (1/3 success on this task).

When the task is simple enough, both paths lead to success.
But Claude's extra steps provide insurance against harder variations.
