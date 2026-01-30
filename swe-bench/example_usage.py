"""
Example usage of SWE-bench consistency runner.

Example task: Fix a simple Python bug.
"""

from runner import SWEBenchRunner

# Example task
task = """
Fix the bug in the following code:

```python
def add(a, b):
    return a - b  # Bug: should be a + b
```

Create a test file to verify the fix works.
"""

task_id = "example_add_function"

# Initialize runner
runner = SWEBenchRunner(
    model_name="gpt-4o",
    provider="openai",
    results_dir="results",
    temperature=0.7,
    max_steps=20,
)

# Run 10 times
print(f"Running task '{task_id}' 10 times...")
runs = runner.run_single_task(
    task=task,
    task_id=task_id,
    n_runs=10,
)

# Save results
runner.save_task_results(task, task_id, runs)

print(f"\n✓ Completed {len(runs)} runs")
print(f"  Results saved to: {runner.results_dir / f'{task_id}.json'}")
