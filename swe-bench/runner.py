"""
SWE-bench consistency experiment runner.
Runs the same task N times, tracks action sequences, saves results to JSON.

Usage:
    # Simple test
    python runner.py \
        --model gpt-4o \
        --provider openai \
        --task "Create a file called hello.py that prints 'Hello World'" \
        --task-id test_001 \
        --n-runs 3

    # With SWE-bench dataset
    python runner.py \
        --model gpt-4o \
        --provider openai \
        --swebench \
        --n-tasks 10 \
        --n-runs 10
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime
from collections import Counter
import sys

# Add mini-swe-agent to path
MINI_SWE_PATH = Path(__file__).parent / "mini-swe-agent" / "src"
sys.path.insert(0, str(MINI_SWE_PATH))

import yaml
from minisweagent.agents.default import (
    DefaultAgent, 
    Submitted, 
    LimitsExceeded, 
    FormatError, 
    ExecutionTimeoutError,
    NonTerminatingException,
    TerminatingException,
)
from minisweagent.environments.local import LocalEnvironment
from minisweagent.environments.docker import DockerEnvironment
from minisweagent.models.litellm_model import LitellmModel
from minisweagent.models.snowflake_cortex_model import SnowflakeCortexModel


# ============================================================================
# SWE-bench Docker Helpers
# ============================================================================

def get_swebench_docker_image_name(instance_id: str) -> str:
    """Get the Docker image name for a SWE-bench instance.
    
    Based on mini-swe-agent's implementation in run/extra/swebench.py
    """
    # Docker doesn't allow double underscore, so replace with magic token
    id_docker_compatible = instance_id.replace("__", "_1776_")
    return f"docker.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class SWEStep:
    """Structured representation of a single agent step."""
    step_number: int
    timestamp: str
    thought: str          # Reasoning before bash block
    action: str           # Bash command executed
    observation: str      # Command output
    returncode: int
    raw_response: str     # Full LLM response


@dataclass
class SWERun:
    """Complete agent run with all steps."""
    run_id: str
    task_id: str
    task: str
    model: str
    provider: str
    temperature: float
    steps: List[SWEStep] = field(default_factory=list)
    action_sequence: List[str] = field(default_factory=list)  # Just the bash commands
    final_output: Optional[str] = None
    exit_status: Optional[str] = None
    success: bool = False
    error: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    total_cost: float = 0.0
    n_calls: int = 0
    n_steps: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """All runs for a single task (matches HotpotQA format)."""
    task_id: str
    task: str
    model: str
    provider: str
    temperature: float
    n_runs: int
    runs: List[dict] = field(default_factory=list)
    # Consistency metrics (computed after runs)
    unique_sequences: int = 0
    success_consistency: float = 0.0  # Renamed from answer_consistency
    avg_steps: float = 0.0
    step_variance: float = 0.0


# ============================================================================
# Tracked Agent
# ============================================================================

INTERPRETATION_GUARD_PROMPT = (
    "STOP. Before making any code changes, you must first state your task interpretation.\n"
    "Output the following structured block (plain text, no bash command):\n\n"
    "TASK INTERPRETATION:\n"
    "(a) Expected behavior change: <one sentence describing what should change>\n"
    "(b) Likely files/modules: <1-3 file paths or module names>\n"
    "(c) Verification plan: <1-2 checks or tests to confirm the fix>\n\n"
    "After outputting this block, continue with your edit in the next step."
)


def _is_edit_action(action: str) -> bool:
    """Check if a bash command is an EDIT-class action."""
    action_stripped = action.strip()
    if action_stripped.startswith(("sed ", "sed\t", "awk ", "awk\t", "patch ")):
        return True
    if action_stripped.startswith(("echo ", "printf ")) and (">" in action_stripped):
        return True
    if action_stripped.startswith("cat ") and ("<<" in action_stripped or ">" in action_stripped):
        return True
    if action_stripped.startswith(("tee ", "mv ", "cp ")) and ">" in action_stripped:
        return True
    # python/python3 one-liners writing to files
    if action_stripped.startswith(("python -c", "python3 -c")) and ">" in action_stripped:
        return True
    # perl in-place editing
    if action_stripped.startswith(("perl -i", "perl -pi")):
        return True
    return False


class TrackedAgent(DefaultAgent):
    """Wrapper around DefaultAgent that tracks action sequences."""
    
    def __init__(self, *args, interpretation_guard: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracked_steps: List[SWEStep] = []
        self.step_count = 0
        self.interpretation_guard = interpretation_guard
        self._guard_fired = False
        self.guard_text: Optional[str] = None
        self.guard_step_index: Optional[int] = None
        self.guard_intercepted_action: Optional[str] = None
    
    def step(self) -> dict:
        """Override step() to track actions. Follows original flow exactly."""
        self.step_count += 1
        step_start = datetime.now().isoformat()
        
        # Query model (same as parent)
        response = self.query()
        raw_response = response.get("content", "")
        thought = self._extract_thought(raw_response)
        
        # Parse action
        action_dict = self.parse_action(response)
        action = action_dict["action"]
        
        # --- Interpretation Guard ---
        # Before the first EDIT action, inject a guard step
        if (self.interpretation_guard
                and not self._guard_fired
                and _is_edit_action(action)):
            self._fire_guard(step_start, thought, action, raw_response)
            # Re-query to get the actual edit command.
            # If parse_action raises FormatError, it propagates to run()
            # which handles it via _track_error_step (with its own increment).
            step_start = datetime.now().isoformat()
            response = self.query()
            raw_response = response.get("content", "")
            thought = self._extract_thought(raw_response)
            action_dict = self.parse_action(response)
            action = action_dict["action"]
            # Increment only after parse succeeds to avoid double-count on FormatError
            self.step_count += 1
        
        # Execute action
        output = self.execute_action(action_dict)
        observation = output.get("output", "")
        returncode = output.get("returncode", -1)
        
        # Add observation to messages (what get_observation normally does)
        obs_text = self.render_template(self.config.action_observation_template, output=output)
        self.add_message("user", obs_text)
        
        # Track step
        step = SWEStep(
            step_number=self.step_count,
            timestamp=step_start,
            thought=thought,
            action=action,
            observation=observation,
            returncode=returncode,
            raw_response=raw_response,
        )
        self.tracked_steps.append(step)
        
        return output
    
    def _fire_guard(self, step_start: str, thought: str, action: str, raw_response: str):
        """Inject the interpretation guard before the first EDIT action."""
        self._guard_fired = True
        self.guard_step_index = self.step_count
        self.guard_intercepted_action = action

        # Track the intercepted edit step as a guard step
        guard_step = SWEStep(
            step_number=self.step_count,
            timestamp=step_start,
            thought=thought,
            action=f"[GUARD_INTERCEPTED] {action}",
            observation="Interpretation guard triggered before first edit.",
            returncode=0,
            raw_response=raw_response,
        )
        self.tracked_steps.append(guard_step)

        # Inject guard prompt into conversation
        self.add_message("user", INTERPRETATION_GUARD_PROMPT)

        # Query LLM for interpretation (uses self.query() for limit checks)
        guard_response = self.query()
        guard_content = guard_response.get("content", "")
        self.guard_text = guard_content

        # Track the guard response as a step
        self.step_count += 1
        guard_reply_step = SWEStep(
            step_number=self.step_count,
            timestamp=datetime.now().isoformat(),
            thought=guard_content,
            action="[INTERPRETATION_GUARD]",
            observation="Guard interpretation recorded.",
            returncode=0,
            raw_response=guard_content,
        )
        self.tracked_steps.append(guard_reply_step)

        # Now prompt the agent to continue with its edit
        self.add_message("user",
            "Thank you. Now proceed with your planned code change. "
            "Provide exactly one bash command."
        )

    def _extract_thought(self, response: str) -> str:
        """Extract reasoning/thought from response (text before bash block)."""
        bash_match = response.find("```bash")
        if bash_match > 0:
            thought = response[:bash_match].strip()
            # Remove common prefixes
            for prefix in ["THOUGHT:", "Thought:", "REASONING:", "Reasoning:"]:
                if thought.startswith(prefix):
                    thought = thought[len(prefix):].strip()
            return thought
        return response.strip()  # If no bash block, whole response is "thought"
    
    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Override run to handle exceptions while preserving tracking."""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.tracked_steps = []
        self.step_count = 0
        self._guard_fired = False
        self.guard_text = None
        self.guard_step_index = None
        self.guard_intercepted_action = None
        
        self.add_message("system", self.render_template(self.config.system_template))
        self.add_message("user", self.render_template(self.config.instance_template))
        
        while True:
            try:
                self.step()
            except NonTerminatingException as e:
                # Track error step
                self._track_error_step(str(e))
                self.add_message("user", str(e))
            except TerminatingException as e:
                self.add_message("user", str(e))
                return type(e).__name__, str(e)
    
    def _track_error_step(self, error_msg: str):
        """Track a step that resulted in an error."""
        self.step_count += 1
        step = SWEStep(
            step_number=self.step_count,
            timestamp=datetime.now().isoformat(),
            thought="",
            action="[ERROR]",
            observation=error_msg,
            returncode=-1,
            raw_response="",
        )
        self.tracked_steps.append(step)


# ============================================================================
# Runner
# ============================================================================

class SWEBenchRunner:
    """Runner for SWE-bench consistency experiments."""
    
    def __init__(
        self,
        model_name: str,
        provider: str,
        results_dir: str = "results",
        temperature: float = 0.7,
        max_steps: int = 250,
        timeout: int = 30,
        use_docker: bool = False,
        interpretation_guard: bool = False,
    ):
        self.model_name = model_name
        self.provider = provider
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.temperature = temperature
        self.max_steps = max_steps
        self.timeout = timeout
        self.use_docker = use_docker
        self.interpretation_guard = interpretation_guard
        
        # Load agent config - use SWE-bench config for Docker, default for local
        if use_docker:
            # SWE-bench config has templates designed for Docker (no system info vars)
            config_path = MINI_SWE_PATH / "minisweagent" / "config" / "extra" / "swebench.yaml"
        else:
            # Default config works with LocalEnvironment (has system info vars)
            config_path = MINI_SWE_PATH / "minisweagent" / "config" / "default.yaml"
        
        self.agent_config = yaml.safe_load(config_path.read_text())["agent"]
        self.agent_config["step_limit"] = max_steps
    
    def _get_litellm_model_name(self) -> str:
        """Convert provider/model to litellm format."""
        if self.provider == "openai":
            return self.model_name  # gpt-4o, gpt-4o-mini
        elif self.provider == "anthropic":
            return f"anthropic/{self.model_name}"  # anthropic/claude-3-5-sonnet-20241022
        elif self.provider == "together":
            return f"together_ai/{self.model_name}"  # together_ai/meta-llama/...
        elif self.provider == "snowflake":
            return self.model_name  # Handled by SnowflakeCortexModel directly
        else:
            return self.model_name
    
    def _create_agent(self, task_data: Optional[Dict[str, str]] = None) -> tuple[TrackedAgent, Any]:
        """Create a new TrackedAgent instance.
        
        Args:
            task_data: Optional dict with task_id, repo, base_commit for SWE-bench tasks
            
        Returns:
            Tuple of (agent, environment) - environment returned for cleanup
        """
        # Select model based on provider
        if self.provider == "snowflake":
            model = SnowflakeCortexModel(
                model_name=self.model_name,
                model_kwargs={
                    "temperature": self.temperature,
                    "max_tokens": 4096,
                },
                cost_tracking="ignore_errors",  # Snowflake doesn't return exact costs
            )
        else:
            model = LitellmModel(
                model_name=self._get_litellm_model_name(),
                model_kwargs={
                    "temperature": self.temperature,
                    "timeout": 120,  # 2 min timeout per API call
                    "num_retries": 2,  # Retry on transient failures
                    "max_tokens": 4096,  # Limit response length to prevent hangs
                },
            )
        
        # Use DockerEnvironment for SWE-bench tasks, LocalEnvironment for simple tasks
        if self.use_docker and task_data and task_data.get("task_id"):
            instance_id = task_data["task_id"]
            image_name = get_swebench_docker_image_name(instance_id)
            print(f"    [Docker] Using image: {image_name}")
            env = DockerEnvironment(
                image=image_name,
                cwd="/testbed",  # SWE-bench repos are mounted at /testbed
                timeout=self.timeout,
                pull_timeout=300,  # Allow more time for image pull
            )
        else:
            env = LocalEnvironment(timeout=self.timeout)
        
        agent = TrackedAgent(model=model, env=env,
                             interpretation_guard=self.interpretation_guard,
                             **self.agent_config)
        return agent, env
    
    def run_single(self, task: str, task_id: str, run_number: int, task_data: Optional[Dict[str, str]] = None) -> SWERun:
        """Run a single task once.
        
        Args:
            task: The task/problem statement
            task_id: Unique identifier for the task
            run_number: Which run number this is (1-indexed)
            task_data: Full task data dict (for SWE-bench with repo/commit info)
        """
        run_id = f"{task_id}_run_{run_number:02d}"
        start_time = datetime.now().isoformat()
        env = None
        
        try:
            agent, env = self._create_agent(task_data)
            exit_status, message = agent.run(task)
            end_time = datetime.now().isoformat()
            
            # Extract action sequence (just real bash commands, exclude guard/error pseudo-actions)
            action_sequence = [s.action for s in agent.tracked_steps
                               if s.action
                               and not s.action.startswith("[")]
            
            # Determine success
            success = exit_status == "Submitted"
            final_output = message if success else None
            
            # Collect guard metadata if applicable
            metadata = {}
            if self.interpretation_guard:
                metadata["interpretation_guard_enabled"] = True
                metadata["interpretation_guard_text"] = agent.guard_text
                metadata["interpretation_guard_step"] = agent.guard_step_index
                metadata["interpretation_guard_intercepted_action"] = agent.guard_intercepted_action
            
            return SWERun(
                run_id=run_id,
                task_id=task_id,
                task=task,
                model=self.model_name,
                provider=self.provider,
                temperature=self.temperature,
                steps=agent.tracked_steps,
                action_sequence=action_sequence,
                final_output=final_output,
                exit_status=exit_status,
                success=success,
                error=None if success else message,
                start_time=start_time,
                end_time=end_time,
                total_cost=agent.model.cost,
                n_calls=agent.model.n_calls,
                n_steps=len(agent.tracked_steps),
                metadata=metadata,
            )
            
        except Exception as e:
            return SWERun(
                run_id=run_id,
                task_id=task_id,
                task=task,
                model=self.model_name,
                provider=self.provider,
                temperature=self.temperature,
                steps=[],
                action_sequence=[],
                success=False,
                error=f"Exception: {str(e)}",
                start_time=start_time,
                end_time=datetime.now().isoformat(),
            )
        finally:
            # Cleanup Docker container if using Docker
            if env and hasattr(env, 'cleanup'):
                try:
                    env.cleanup()
                except Exception:
                    pass  # Ignore cleanup errors
    
    def run_task(self, task: str, task_id: str, n_runs: int = 10, task_data: Optional[Dict[str, str]] = None) -> TaskResult:
        """Run the same task N times and compute consistency metrics.
        
        Args:
            task: The task/problem statement
            task_id: Unique identifier for the task
            n_runs: Number of times to run the task
            task_data: Full task data dict (for SWE-bench with repo/commit info)
        """
        print(f"\n{'='*60}")
        print(f"Task: {task_id}")
        print(f"Description: {task[:100]}...")
        if self.use_docker and task_data:
            print(f"Repo: {task_data.get('repo', 'N/A')}")
        print(f"{'='*60}")
        
        runs: List[SWERun] = []
        
        for i in range(1, n_runs + 1):
            print(f"  Run {i}/{n_runs}...", end=" ", flush=True)
            run = self.run_single(task, task_id, i, task_data)
            runs.append(run)
            
            status = "✓" if run.success else "✗"
            print(f"{status} ({run.n_steps} steps, {run.exit_status})")
        
        # Compute consistency metrics
        action_sequences = [tuple(r.action_sequence) for r in runs]
        unique_sequences = len(set(action_sequences))
        
        # Success consistency (% runs with same success/failure outcome)
        success_states = [r.success for r in runs]
        most_common_count = Counter(success_states).most_common(1)[0][1]
        success_consistency = most_common_count / len(runs)
        
        # Step statistics
        step_counts = [r.n_steps for r in runs]
        avg_steps = sum(step_counts) / len(step_counts) if step_counts else 0
        if len(step_counts) > 1 and avg_steps > 0:
            step_variance = (max(step_counts) - min(step_counts)) / avg_steps
        else:
            step_variance = 0.0
        
        result = TaskResult(
            task_id=task_id,
            task=task,
            model=self.model_name,
            provider=self.provider,
            temperature=self.temperature,
            n_runs=n_runs,
            runs=[asdict(r) for r in runs],
            unique_sequences=unique_sequences,
            success_consistency=success_consistency,
            avg_steps=avg_steps,
            step_variance=step_variance,
        )
        
        # Print summary
        print(f"\n  Summary:")
        print(f"    Unique sequences: {unique_sequences}/{n_runs}")
        print(f"    Success consistency: {success_consistency:.1%}")
        print(f"    Avg steps: {avg_steps:.1f} (variance ratio: {step_variance:.2f})")
        
        return result
    
    def save_result(self, result: TaskResult):
        """Save task result to JSON."""
        output_file = self.results_dir / f"{result.task_id}.json"
        with open(output_file, "w") as f:
            json.dump(asdict(result), f, indent=2, default=str)
        print(f"  Saved to: {output_file}")
    
    def run_experiment(self, tasks: List[Dict[str, str]], n_runs: int = 10):
        """Run experiment on multiple tasks."""
        print(f"\n{'#'*60}")
        print(f"SWE-bench Consistency Experiment")
        print(f"{'#'*60}")
        print(f"  Model: {self.model_name} ({self.provider})")
        print(f"  Temperature: {self.temperature}")
        print(f"  Environment: {'Docker' if self.use_docker else 'Local'}")
        print(f"  Tasks: {len(tasks)}")
        print(f"  Runs per task: {n_runs}")
        print(f"  Total runs: {len(tasks) * n_runs}")
        print(f"  Results dir: {self.results_dir}")
        
        results = []
        skipped = 0
        for task_data in tasks:
            task_id = task_data["task_id"]
            result_path = self.results_dir / f"{task_id}.json"
            
            # Skip if already completed (resume support)
            if result_path.exists():
                print(f"\n[Skip] {task_id} - already completed (results exist)")
                skipped += 1
                # Load existing result for summary stats
                with open(result_path) as f:
                    existing = json.load(f)
                    results.append(TaskResult(
                        task_id=existing["task_id"],
                        task=existing["task"],
                        model=existing["model"],
                        provider=existing["provider"],
                        temperature=existing["temperature"],
                        n_runs=existing["n_runs"],
                        runs=[],  # Don't need to reload runs for summary
                        unique_sequences=existing["unique_sequences"],
                        success_consistency=existing["success_consistency"],
                        avg_steps=existing["avg_steps"],
                        step_variance=existing["step_variance"],
                    ))
                continue
            
            result = self.run_task(
                task=task_data["task"],
                task_id=task_id,
                n_runs=n_runs,
                task_data=task_data,  # Pass full task data for Docker setup
            )
            self.save_result(result)
            results.append(result)
        
        if skipped > 0:
            print(f"\n[Resume] Skipped {skipped} already-completed tasks")
        
        # Print final summary
        print(f"\n{'#'*60}")
        print(f"Experiment Complete!")
        print(f"{'#'*60}")
        avg_unique = sum(r.unique_sequences for r in results) / len(results)
        avg_consistency = sum(r.success_consistency for r in results) / len(results)
        print(f"  Avg unique sequences: {avg_unique:.1f}")
        print(f"  Avg success consistency: {avg_consistency:.1%}")
        
        return results


# ============================================================================
# SWE-bench Dataset Loading
# ============================================================================

def load_swebench_tasks(n_tasks: int = 10, split: str = "test") -> List[Dict[str, str]]:
    """Load tasks from SWE-bench dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets library...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "datasets"], check=True)
        from datasets import load_dataset
    
    print(f"Loading SWE-bench Verified dataset...")
    dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split=split)
    
    tasks = []
    for i, item in enumerate(dataset):
        if i >= n_tasks:
            break
        tasks.append({
            "task_id": item["instance_id"],
            "task": item["problem_statement"],
            "repo": item.get("repo", ""),
            "base_commit": item.get("base_commit", ""),
        })
    
    print(f"Loaded {len(tasks)} tasks")
    return tasks


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run SWE-bench consistency experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple test with custom task
  python runner.py --model gpt-4o --provider openai \\
      --task "Create hello.py that prints Hello World" \\
      --task-id test_001 --n-runs 3

  # Run on SWE-bench dataset
  python runner.py --model gpt-4o --provider openai \\
      --swebench --n-tasks 10 --n-runs 10

  # With Claude
  python runner.py --model claude-sonnet-4-20250514 --provider anthropic \\
      --swebench --n-tasks 5 --n-runs 10

  # With Snowflake Cortex
  python runner.py --model claude-3-5-sonnet --provider snowflake \\
      --swebench --n-tasks 5 --n-runs 10
        """
    )
    
    # Model settings
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--provider", required=True, choices=["openai", "anthropic", "together", "snowflake"])
    parser.add_argument("--temperature", type=float, default=0.7)
    
    # Task settings (either --task or --swebench)
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--task", help="Custom task description")
    task_group.add_argument("--swebench", action="store_true", help="Use SWE-bench dataset")
    
    parser.add_argument("--task-id", help="Task ID (required with --task)")
    parser.add_argument("--n-tasks", type=int, default=10, help="Number of SWE-bench tasks")
    
    # Run settings
    parser.add_argument("--n-runs", type=int, default=10, help="Runs per task")
    parser.add_argument("--max-steps", type=int, default=250, help="Max steps per run")
    parser.add_argument("--results-dir", default="results", help="Output directory")
    
    # Intervention flags
    parser.add_argument("--interpretation-guard", action="store_true", default=False,
                        help="Enable interpretation guard: inject a structured interpretation "
                             "checkpoint before the first EDIT action in each run")
    
    args = parser.parse_args()
    
    # Validate
    if args.task and not args.task_id:
        parser.error("--task-id is required when using --task")
    
    # Create runner (use Docker for SWE-bench tasks)
    runner = SWEBenchRunner(
        model_name=args.model,
        provider=args.provider,
        results_dir=args.results_dir,
        temperature=args.temperature,
        max_steps=args.max_steps,
        use_docker=args.swebench,  # Use Docker for SWE-bench, Local for simple tasks
        interpretation_guard=args.interpretation_guard,
    )
    
    # Get tasks
    if args.swebench:
        tasks = load_swebench_tasks(n_tasks=args.n_tasks)
    else:
        tasks = [{"task_id": args.task_id, "task": args.task}]
    
    # Run experiment
    runner.run_experiment(tasks, n_runs=args.n_runs)


if __name__ == "__main__":
    main()