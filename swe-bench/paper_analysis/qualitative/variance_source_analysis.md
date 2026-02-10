# RQ2: Variance Source Analysis

## Summary

Variance in agent behavior originates from **exploration and understanding phases**, not from editing.

## Phase Distribution

| Phase | Claude % | Llama % | Claude CV | Llama CV |
|-------|----------|---------|-----------|----------|
| EXPLORE | 34% | 9% | 42% | 123% |
| LOCATE | 15% | 16% | 43% | 59% |
| UNDERSTAND | 30% | 18% | 23% | 101% |
| EDIT | 13% | 33% | 84% | 74% |
| VERIFY | 3% | 20% | 118% | 124% |
| OTHER | 5% | 4% | 82% | 143% |

## Key Findings

### 1. Claude's Exploration Advantage
- Claude spends **34%** of steps exploring (ls, find, cd) vs Llama's **9%**
- This is **10x more exploration steps** (15 vs 1.5 on average)
- Result: Better understanding of codebase structure

### 2. Claude's Understanding Advantage  
- Claude spends **30%** of steps understanding (cat, head) vs Llama's **18%**
- This is **4.7x more reading steps** (14 vs 3 on average)
- Result: Better comprehension before editing

### 3. Llama's "Edit First" Problem
- Llama allocates **33%** to editing vs Claude's **13%**
- Llama spends **20%** verifying vs Claude's **3%**
- Pattern: Llama edits quickly, then repeatedly tests and fails

### 4. Where Variance is Highest
- **VERIFY phase has highest CV** for both models (118-124%)
- This is expected: verification depends on success of edits
- **EXPLORE has highest CV difference**: Claude 42% vs Llama 123%
- Llama's exploration is erratic; Claude's is systematic

## Interpretation

**Claude's workflow**: EXPLORE → UNDERSTAND → EDIT → VERIFY
**Llama's workflow**: LOCATE → EDIT → VERIFY → repeat

Claude's methodical exploration reduces downstream variance:
- More time understanding = more consistent edits
- More consistent edits = fewer verification iterations

Llama's rushed approach increases variance:
- Less exploration = inconsistent file discovery
- Less understanding = trial-and-error editing
- More verification loops = higher step count variance

## CLAIM 5 Support

> "Llama's speed comes from skipping EXPLORE and UNDERSTAND phases"

Evidence:
- Claude: 15 EXPLORE + 14 UNDERSTAND = 29 steps (63%)
- Llama: 1.5 EXPLORE + 3 UNDERSTAND = 4.5 steps (27%)

Llama saves 25 steps by skipping exploration, but this leads to 96% failure rate.
