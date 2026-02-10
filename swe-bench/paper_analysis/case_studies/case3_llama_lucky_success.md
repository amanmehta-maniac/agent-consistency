# Case Study 3: Llama's Lucky Success

## Task: astropy-13236 Run 3

### One of Llama's 2 Successful Runs (out of 50)

**Summary:** 11 steps, resolved

**Action sequence:**
```
Step 1: find . -name "*.py" -path "*wcs*"
Step 2: cat astropy/wcs/wcsapi/sliced_low_level_wcs.py | head -200
Step 3: grep -n "world_to_pixel" sliced_low_level_wcs.py
Step 4: cat -n sliced_low_level_wcs.py | sed -n '150,200p'
Step 5: sed -i '175s/old/new/'  # Direct edit
Step 6: python -c "from astropy.wcs..."  # Quick test
Step 7: cat sliced_low_level_wcs.py | grep -A5 "def world"
Step 8: sed -i '180s/old/new/'  # Another edit
Step 9: python test_script.py  # Test passes!
Step 10: git diff
Step 11: submit
```

### Why This Worked

1. **Lucky file find**: First search found the right file
2. **Simple bug**: This task had a relatively simple fix
3. **Correct heuristic**: The bug was near "world_to_pixel"
4. **Fast verification**: Quick test confirmed fix

### Why This is Rare

- Required hitting correct file on first try
- Required bug being in obvious location
- Required fix being simple enough for trial-and-error

### Contrast with Other Llama Runs

On harder tasks, Llama's fast approach fails:
- Wrong file selected
- Bug not in expected location
- Fix requires understanding context

### Insight

Llama can succeed when:
1. Task is simple
2. Bug location is obvious
3. Fix is straightforward

But on complex tasks requiring understanding, speed is counterproductive.
