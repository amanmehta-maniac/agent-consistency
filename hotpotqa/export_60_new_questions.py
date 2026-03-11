#!/usr/bin/env python3
"""
Export 60 new HotpotQA questions (30 hard + 30 easy) for the 600-run experiment.
Excludes the 40 questions already used in the pilot and easier sets.

Easy criteria: "Are both X and Y..." comparison questions, yes/no answers,
               direct entity lookups.
Hard criteria: Multi-hop requiring 2+ retrieval steps, answers not directly
               stated in a single document.
"""

import json
import os
import re
os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import load_dataset


def load_excluded_ids():
    excluded = set()
    for path in ["pilot_questions.json", "easier_questions_selection.json"]:
        if os.path.exists(path):
            with open(path) as f:
                for q in json.load(f):
                    excluded.add(q["id"])
    return excluded


EASY_PATTERNS = [
    r"^are both\b",
    r"^is .+ (the same|also|both)\b",
    r"^do both\b",
    r"^were both\b",
    r"^did both\b",
]

def is_easy_question(item):
    q = item["question"].lower().strip()
    if item["answer"].lower().strip() not in ("yes", "no"):
        return False
    for pat in EASY_PATTERNS:
        if re.search(pat, q):
            return True
    if q.startswith(("is ", "are ", "was ", "were ", "does ", "do ", "did ", "has ", "have ")):
        if item["answer"].lower().strip() in ("yes", "no"):
            if len(item["context"]["title"]) <= 4:
                return True
    return False


def is_hard_question(item):
    if item["answer"].lower().strip() in ("yes", "no"):
        return False
    n_docs = len(item["context"]["title"])
    if n_docs < 4:
        return False
    return True


def format_context(item):
    context = {}
    for title, sentences in zip(item["context"]["title"], item["context"]["sentences"]):
        context[title] = " ".join(sentences)
    return context


def main():
    excluded_ids = load_excluded_ids()
    print(f"Excluding {len(excluded_ids)} already-used questions")

    print("Loading HotpotQA validation split...")
    ds = load_dataset("hotpot_qa", "distractor", split="validation")
    print(f"Total validation examples: {len(ds)}")

    easy_candidates = []
    hard_candidates = []

    for item in ds:
        if item["id"] in excluded_ids:
            continue

        if is_easy_question(item):
            easy_candidates.append(item)
        elif is_hard_question(item):
            hard_candidates.append(item)

    print(f"\nEasy candidates: {len(easy_candidates)}")
    print(f"Hard candidates: {len(hard_candidates)}")

    import random
    random.seed(42)
    random.shuffle(easy_candidates)
    random.shuffle(hard_candidates)

    easy_selected = easy_candidates[:30]
    hard_selected = hard_candidates[:30]

    print(f"\nSelected: {len(easy_selected)} easy, {len(hard_selected)} hard")

    questions = []

    for item in easy_selected:
        questions.append({
            "id": item["id"],
            "question": item["question"],
            "answer": item["answer"],
            "context": format_context(item),
            "difficulty": "easy",
        })

    for item in hard_selected:
        questions.append({
            "id": item["id"],
            "question": item["question"],
            "answer": item["answer"],
            "context": format_context(item),
            "difficulty": "hard",
        })

    random.shuffle(questions)

    out_path = "new_60_questions.json"
    with open(out_path, "w") as f:
        json.dump(questions, f, indent=2)

    print(f"\nExported {len(questions)} questions to {out_path}")
    print(f"\nSample easy: {easy_selected[0]['question'][:80]}...")
    print(f"Sample hard: {hard_selected[0]['question'][:80]}...")

    n_easy = sum(1 for q in questions if q["difficulty"] == "easy")
    n_hard = sum(1 for q in questions if q["difficulty"] == "hard")
    n_yes_no = sum(1 for q in questions if q["answer"].lower() in ("yes", "no"))
    print(f"\nBreakdown: {n_easy} easy, {n_hard} hard, {n_yes_no} yes/no answers")


if __name__ == "__main__":
    main()
