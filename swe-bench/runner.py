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


def get_swebench_docker_image(instance_id: str) -> str:
    """Get the Docker image name for a SWE-bench instance.
    
    Docker doesn't allow double underscore, so we replace them with a magic token.
    Image pattern: docker.io/swebench/sweb.eval.x86_64.{id}:latest
    """
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
    answer_consistency: float = 0.0
    avg_steps: float = 0.0
    step_variance: float = 0.0


# ============================================================================
# Tracked Agent
# ============================================================================

class TrackedAgent(DefaultAgent):
    """Wrapper around DefaultAgent that tracks action sequences."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracked_steps: List[SWEStep] = []
        self.step_count = 0
    
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
        max_steps: int = 50,
        timeout: int = 60,
        use_docker: bool = False,
    ):
        self.model_name = model_name
        self.provider = provider
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.temperature = temperature
        self.max_steps = max_steps
        self.timeout = timeout
        self.use_docker = use_docker
        
        # Load agent config - use swebench config for Docker, default otherwise
        if use_docker:
            config_path = MINI_SWE_PATH / "minisweagent" / "config" / "extra" / "swebench.yaml"
        else:
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
        else:
            return self.model_name
    
    def _create_agent(self, task_metadata: Optional[Dict] = None) -> tuple[TrackedAgent, Any]:
        """Create a new TrackedAgent instance.
        
        Args:
            task_metadata: Optional dict with 'instance_id', 'repo', 'base_commit' for Docker setup
            
        Returns:
            Tuple of (agent, environment) - environment returned for cleanup
        """
        model = LitellmModel(
            model_name=self._get_litellm_model_name(),
            model_kwargs={"temperature": self.temperature},
        )
        
        if self.use_docker and task_metadata:
            # Use DockerEnvironment for SWE-bench tasks
            instance_id = task_metadata.get("instance_id", task_metadata.get("task_id", ""))
            image = get_swebench_docker_image(instance_id)
            print(f"    Using Docker image: {image}")
            
            env = DockerEnvironment(
                image=image,
                cwd="/testbed",  # SWE-bench standard working directory
                timeout=self.timeout,
                env={
                    "PAGER": "cat",
                    "MANPAGER": "cat",
                    "LESS": "-R",
                    "PIP_PROGRESS_BAR": "off",
                    "TQDM_DISABLE": "1",
                },
            )
        else:
            # Use LocalEnvironment for simple tasks
            env = LocalEnvironment(timeout=self.timeout)
        
        agent = TrackedAgent(model=model, env=env, **self.agent_config)
        return agent, env
    
    def run_single(self, task: str, task_id: str, run_number: int, task_metadata: Optional[Dict] = None) -> SWERun:
        """Run a single task once.
        
        Args:
            task: Task description/problem statement
            task_id: Unique task identifier
            run_number: Which run number this is (1-indexed)
            task_metadata: Optional dict with 'instance_id', 'repo', 'base_commit' for Docker
        """
        run_id = f"{task_id}_run_{run_number:02d}"
        start_time = datetime.now().isoformat()
        env = None
        
        try:
            agent, env = self._create_agent(task_metadata)
            exit_status, message = agent.run(task)
            end_time = datetime.now().isoformat()
            
            # Extract action sequence (just bash commands)
            action_sequence = [s.action for s in agent.tracked_steps if s.action and s.action != "[ERROR]"]
            
            # Determine success
            success = exit_status == "Submitted"
            final_output = message if success else None
            
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
            # Cleanup Docker container if used
            if env is not None and hasattr(env, 'cleanup'):
                env.cleanup()
    
    def run_task(self, task: str, task_id: str, n_runs: int = 10, task_metadata: Optional[Dict] = None) -> TaskResult:
        """Run the same task N times and compute consistency metrics.
        
        Args:
            task: Task description/problem statement
            task_id: Unique task identifier  
            n_runs: Number of times to run the task
            task_metadata: Optional dict with SWE-bench instance info for Docker setup
        """
        print(f"\n{'='*60}")
        print(f"Task: {task_id}")
        print(f"Description: {task[:100]}...")
        if self.use_docker:
            print(f"Environment: Docker")
        print(f"{'='*60}")
        
        runs: List[SWERun] = []
        
        for i in range(1, n_runs + 1):
            print(f"  Run {i}/{n_runs}...", end=" ", flush=True)
            run = self.run_single(task, task_id, i, task_metadata)
            runs.append(run)
            
            status = "✓" if run.success else "✗"
            print(f"{status} ({run.n_steps} steps, {run.exit_status})")
        
        # Compute consistency metrics
        action_sequences = [tuple(r.action_sequence) for r in runs]
        unique_sequences = len(set(action_sequences))
        
        # Answer consistency (most common final output)
        # Success consistency (% runs with same success/failure outcome)
        from collections import Counter
        success_states = [r.success for r in runs]
        most_common_count = Counter(success_states).most_common(1)[0][1]
        answer_consistency = most_common_count / len(runs)
        
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
            answer_consistency=answer_consistency,
            avg_steps=avg_steps,
            step_variance=step_variance,
        )
        
        # Print summary
        print(f"\n  Summary:")
        print(f"    Unique sequences: {unique_sequences}/{n_runs}")
        print(f"    Answer consistency: {answer_consistency:.1%}")
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
        print(f"  Tasks: {len(tasks)}")
        print(f"  Runs per task: {n_runs}")
        print(f"  Total runs: {len(tasks) * n_runs}")
        print(f"  Environment: {'Docker' if self.use_docker else 'Local'}")
        print(f"  Results dir: {self.results_dir}")
        
        results = []
        for task_data in tasks:
            # Pass full task_data as metadata for Docker setup
            result = self.run_task(
                task=task_data["task"],
                task_id=task_data["task_id"],
                n_runs=n_runs,
                task_metadata=task_data if self.use_docker else None,
            )
            self.save_result(result)
            results.append(result)
        
        # Print final summary
        print(f"\n{'#'*60}")
        print(f"Experiment Complete!")
        print(f"{'#'*60}")
        avg_unique = sum(r.unique_sequences for r in results) / len(results)
        avg_consistency = sum(r.answer_consistency for r in results) / len(results)
        print(f"  Avg unique sequences: {avg_unique:.1f}")
        print(f"  Avg answer consistency: {avg_consistency:.1%}")
        
        return results


# ============================================================================
# SWE-bench Dataset Loading
# ============================================================================

def load_swebench_tasks(n_tasks: int = 10, split: str = "test") -> List[Dict[str, str]]:
    """Load tasks from SWE-bench dataset.
    
    Returns list of dicts with:
        - task_id: instance_id from SWE-bench
        - instance_id: same as task_id (used for Docker image name)
        - task: problem_statement  
        - repo: repository name (e.g., 'django/django')
        - base_commit: commit hash to checkout
    """
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
            "instance_id": item["instance_id"],  # Used for Docker image name
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
  # Simple test with custom task (local environment)
  python runner.py --model gpt-4o --provider openai \\
      --task "Create hello.py that prints Hello World" \\
      --task-id test_001 --n-runs 3

  # Run on SWE-bench with Docker (required for real SWE-bench tasks)
  python runner.py --model gpt-4o --provider openai \\
      --swebench --use-docker --n-tasks 10 --n-runs 10

  # With Claude on SWE-bench
  python runner.py --model claude-sonnet-4-5-20250929 --provider anthropic \\
      --swebench --use-docker --n-tasks 5 --n-runs 10

  # With Llama via Together AI
  python runner.py --model meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo --provider together \\
      --swebench --use-docker --n-tasks 5 --n-runs 10

Note: --use-docker requires Docker to be installed and the SWE-bench images
      (docker.io/swebench/sweb.eval.x86_64.*) to be available.
        """
    )
    
    # Model settings
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--provider", required=True, choices=["openai", "anthropic", "together"])
    parser.add_argument("--temperature", type=float, default=0.7)
    
    # Task settings (either --task or --swebench)
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--task", help="Custom task description")
    task_group.add_argument("--swebench", action="store_true", help="Use SWE-bench dataset")
    
    parser.add_argument("--task-id", help="Task ID (required with --task)")
    parser.add_argument("--n-tasks", type=int, default=10, help="Number of SWE-bench tasks")
    
    # Run settings
    parser.add_argument("--n-runs", type=int, default=10, help="Runs per task")
    parser.add_argument("--max-steps", type=int, default=50, help="Max steps per run")
    parser.add_argument("--results-dir", default="results", help="Output directory")
    parser.add_argument("--use-docker", action="store_true", 
                        help="Use Docker environment (required for real SWE-bench tasks)")
    parser.add_argument("--timeout", type=int, default=60, help="Command timeout in seconds")
    
    args = parser.parse_args()
    
    # Validate
    if args.task and not args.task_id:
        parser.error("--task-id is required when using --task")
    
    # Create runner
    runner = SWEBenchRunner(
        model_name=args.model,
        provider=args.provider,
        results_dir=args.results_dir,
        temperature=args.temperature,
        max_steps=args.max_steps,
        timeout=args.timeout,
        use_docker=args.use_docker,
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