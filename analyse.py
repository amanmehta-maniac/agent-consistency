python3 -c "
import json
from pathlib import Path

results_dir = Path('results')
for f in sorted(results_dir.glob('*.json')):
    data = json.load(open(f))
    question = data['question'][:60] + '...'
    ground_truth = data['answer']
    
    answers = [r['final_answer'] for r in data['runs']]
    successes = [r['success'] for r in data['runs']]
    steps = [len(r['steps']) for r in data['runs']]
    
    unique_answers = len(set(str(a) for a in answers))
    success_rate = sum(successes) / len(successes)
    correct = sum(1 for a in answers if a and ground_truth.lower() in str(a).lower())
    
    print(f'Q: {question}')
    print(f'   Ground truth: {ground_truth}')
    print(f'   Answers: {answers}')
    print(f'   Unique answers: {unique_answers}/5')
    print(f'   Correct: {correct}/5')
    print(f'   Steps per run: {steps}')
    print()
"