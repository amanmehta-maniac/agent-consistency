#!/usr/bin/env python3
"""Select easier questions from HotpotQA for consistent-correct category."""
import json
import os
os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import load_dataset

# Load full validation set
print("Loading HotpotQA validation set...")
ds = load_dataset("hotpot_qa", "distractor", split="validation")

# Get IDs from pilot to exclude
pilot_file = "pilot_questions.json"
pilot_ids = set()
if os.path.exists(pilot_file):
    with open(pilot_file) as f:
        pilot_ids = {q["id"] for q in json.load(f)}
    print(f"Excluding {len(pilot_ids)} pilot questions")

# Score questions for easiness
# Criteria: yes/no questions, shorter questions, comparison questions
def score_question(item):
    score = 0
    q = item["question"].lower()
    answer = item["answer"].lower()
    
    # Yes/no questions are easier (binary answer)
    if answer in ["yes", "no"]:
        score += 3
    
    # Comparison questions ("Are both X and Y...") tend to be more straightforward
    if "are both" in q or "were both" in q:
        score += 2
    if "which" not in q:  # "Which is older" can be tricky
        score += 1
    
    # Shorter questions tend to be simpler
    if len(q) < 80:
        score += 1
    
    return score

# Score all questions
print("Scoring questions for easiness...")
scored = []
for item in ds:
    if item["id"] in pilot_ids:
        continue
    
    score = score_question(item)
    if score >= 5:  # Only high-score questions
        context = {}
        for title, sentences in zip(item["context"]["title"], item["context"]["sentences"]):
            context[title] = " ".join(sentences)
        
        scored.append({
            "id": item["id"],
            "question": item["question"],
            "answer": item["answer"],
            "context": context,
            "easiness_score": score
        })

# Sort by score (highest first)
scored.sort(key=lambda x: -x["easiness_score"])

# Take top 20
selected = scored[:20]

print(f"\nFound {len(scored)} easier questions, selecting top 20:")
print("-" * 60)
for i, q in enumerate(selected, 1):
    print(f"{i:2}. [{q['easiness_score']}] {q['id'][:20]}...")
    print(f"    Q: {q['question'][:60]}...")
    print(f"    A: {q['answer']}")
    print()

# Save to file
output_file = "easier_questions_selection.json"
with open(output_file, "w") as f:
    json.dump(selected, f, indent=2)

print(f"Saved {len(selected)} questions to {output_file}")
