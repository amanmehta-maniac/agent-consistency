"""
ReAct agent implementation for HotpotQA.
Supports OpenAI, Together AI, Anthropic APIs, and Llama on SPCS with hidden state extraction.
"""

import json
import asyncio
from typing import Dict, List, Optional, Any, Literal, Callable
from dataclasses import dataclass, asdict, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import openai
from together import Together
from anthropic import AsyncAnthropic


@dataclass
class AgentStep:
    """Structured representation of a single agent step."""
    step_number: int
    timestamp: str
    thought: str
    action: str
    action_input: Dict[str, Any]
    observation: str
    raw_response: str
    hidden_states: Optional[Dict[str, Any]] = None  # Hidden states from SPCS Llama


@dataclass
class AgentRun:
    """Complete agent run with all steps."""
    run_id: str
    task_id: str
    model: str
    provider: Literal["openai", "together", "anthropic", "llama_spcs", "llama_k8s"]
    question: str
    steps: List[AgentStep]
    final_answer: Optional[str] = None
    success: bool = False
    error: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None


class ReActAgent:
    """
    ReAct-style agent that alternates between reasoning and acting.
    Supports OpenAI and Together AI APIs.
    """
    
    def __init__(
        self,
        model: str,
        provider: Literal["openai", "together", "anthropic", "llama_spcs", "llama_k8s"],
        openai_api_key: Optional[str] = None,
        together_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        spcs_connection: Optional[str] = None,
        k8s_endpoint: Optional[str] = None,
        max_steps: int = 15,
        temperature: float = 0.7,
        search_fn: Optional[Callable[[str], Any]] = None,
        retrieve_fn: Optional[Callable[[str], Any]] = None,
    ):
        """
        Initialize the ReAct agent.
        
        Args:
            model: Model name (e.g., "gpt-4o", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo")
            provider: "openai", "together", "anthropic", "llama_spcs", or "llama_k8s"
            openai_api_key: OpenAI API key (required if provider is "openai")
            together_api_key: Together AI API key (required if provider is "together")
            anthropic_api_key: Anthropic API key (required if provider is "anthropic")
            spcs_connection: Snowflake connection name (required if provider is "llama_spcs")
            k8s_endpoint: Llama 70B k8s endpoint URL (required if provider is "llama_k8s")
            max_steps: Maximum number of reasoning-action steps
            temperature: Sampling temperature
            search_fn: Optional custom async search function (query: str) -> str
            retrieve_fn: Optional custom async retrieve function (title: str) -> str
        """
        self.model = model
        self.provider = provider
        self.max_steps = max_steps
        self.temperature = temperature
        self._last_hidden_states = None  # Store hidden states from last LLM call
        
        # Initialize API clients
        self.executor = None
        if provider == "openai":
            if not openai_api_key:
                raise ValueError("openai_api_key required for OpenAI provider")
            self.client = openai.AsyncOpenAI(api_key=openai_api_key)
        elif provider == "together":
            if not together_api_key:
                raise ValueError("together_api_key required for Together AI provider")
            self.client = Together(api_key=together_api_key)
            # Use thread pool executor for Together AI since it's sync-only
            self.executor = ThreadPoolExecutor(max_workers=1)
        elif provider == "anthropic":
            if not anthropic_api_key:
                raise ValueError("anthropic_api_key required for Anthropic provider")
            self.client = AsyncAnthropic(api_key=anthropic_api_key)
        elif provider == "llama_spcs":
            if not spcs_connection:
                raise ValueError("spcs_connection required for llama_spcs provider")
            self.spcs_connection = spcs_connection
            # Use thread pool executor for Snowflake calls
            self.executor = ThreadPoolExecutor(max_workers=1)
        elif provider == "llama_k8s":
            if not k8s_endpoint:
                raise ValueError("k8s_endpoint required for llama_k8s provider")
            self.k8s_endpoint = k8s_endpoint
            # Use thread pool executor for HTTP calls
            self.executor = ThreadPoolExecutor(max_workers=1)
        else:
            raise ValueError(f"Unknown provider: {provider}")
        
        # Use custom tools if provided, otherwise use default mock tools
        self._custom_search = search_fn
        self._custom_retrieve = retrieve_fn
        
        # Available tools for HotpotQA
        self.tools = {
            "Search": self._search,
            "Retrieve": self._retrieve,
            "Finish": self._finish,
        }
    
    async def _call_llm(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
    ) -> str:
        """Call the LLM via the appropriate provider."""
        if system_prompt:
            full_messages = [{"role": "system", "content": system_prompt}] + messages
        else:
            full_messages = messages
        
        if self.provider == "openai":
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                temperature=self.temperature,
            )
            return response.choices[0].message.content
        
        elif self.provider == "together":
            # Together AI client is sync-only, so we run it in a thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self.executor,
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=full_messages,
                    temperature=self.temperature,
                )
            )
            return response.choices[0].message.content
        
        elif self.provider == "anthropic":
            # Anthropic uses a Messages API; map OpenAI-style messages.
            system = None
            user_messages: List[Dict[str, str]] = []
            for m in full_messages:
                role = m.get("role")
                content = m.get("content", "")
                if role == "system" and system is None:
                    system = content
                else:
                    user_messages.append({"role": role, "content": content})

            # Anthropic expects alternating user/assistant messages; we pass as-is.
            response = await self.client.messages.create(
                model=self.model,
                system=system,
                max_tokens=1024,
                temperature=self.temperature,
                messages=[
                    {"role": m["role"], "content": m["content"]} for m in user_messages
                ],
                stop_sequences=["Observation:"],
            )
            # Concatenate text content blocks
            parts = []
            for block in response.content:
                if getattr(block, "type", None) == "text":
                    parts.append(block.text)
            return "\n".join(parts).strip()
        
        elif self.provider == "llama_spcs":
            # Call Llama on SPCS via Snowflake SQL
            import subprocess
            
            loop = asyncio.get_event_loop()
            
            def call_spcs():
                # Build SQL query to call the chat completion function
                # Escape for SQL: double single quotes and escape backslashes
                messages_json = json.dumps(full_messages)
                # For Snowflake SQL strings: escape backslashes first, then single quotes
                messages_escaped = messages_json.replace("\\", "\\\\").replace("'", "''")
                
                sql = f"""
                USE WAREHOUSE XSMALL;
                SELECT GPU_NOTEBOOK_DB.GPU_NOTEBOOK_SCHEMA.CHAT_COMPLETION_HS(
                    PARSE_JSON('{messages_escaped}'),
                    512,
                    {self.temperature}
                ) AS response
                """
                
                result = subprocess.run(
                    ["snow", "sql", "-c", self.spcs_connection, "-q", sql, "--format", "JSON"],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode != 0:
                    raise RuntimeError(f"Snowflake SQL failed: {result.stderr}")
                
                # Parse the JSON output
                # Output is list of statement results: [[USE WAREHOUSE result], [SELECT result]]
                # Each statement result is a list of rows
                output = json.loads(result.stdout)
                # Get the SELECT result (second statement), first row, RESPONSE column
                select_result = output[1][0]  # Second statement, first row
                response_str = select_result["RESPONSE"]
                response_data = json.loads(response_str) if isinstance(response_str, str) else response_str
                
                # Store hidden states for later retrieval
                self._last_hidden_states = response_data.get("hidden_states")
                
                return response_data.get("content", "")
            
            return await loop.run_in_executor(self.executor, call_spcs)
        
        elif self.provider == "llama_k8s":
            # Call Llama 70B on k8s via HTTP
            import requests
            
            loop = asyncio.get_event_loop()
            
            def call_k8s():
                # Build request payload
                payload = {
                    "messages": [{"role": m["role"], "content": m["content"]} for m in full_messages],
                    "max_new_tokens": 512,
                    "temperature": self.temperature,
                    "stop_sequences": ["Observation:"],  # Stop after Action Input so agent loop can inject real observations
                    "return_hidden_states": True,
                    "hidden_state_pooling": "last"
                }
                
                response = requests.post(
                    f"{self.k8s_endpoint}/v1/chat/completions",
                    json=payload,
                    timeout=300
                )
                
                if response.status_code != 200:
                    raise RuntimeError(f"K8s endpoint failed: {response.status_code} {response.text}")
                
                response_data = response.json()
                
                # Store hidden states for later retrieval
                self._last_hidden_states = response_data.get("hidden_states")
                
                return response_data.get("content", "")
            
            return await loop.run_in_executor(self.executor, call_k8s)
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for ReAct agent."""
        return """You are a helpful assistant that answers questions using a ReAct (Reasoning + Acting) approach.

You have access to the following tools:
1. Search(query): Search for relevant information. Returns ONLY document titles (not content).
2. Retrieve(title): Get the full text of a document. You MUST use this to read document content.
3. Finish(answer): Submit your final answer.

CRITICAL: Search returns titles only. To read a document, you must call Retrieve with the exact title.

Format:
Thought: [reasoning]
Action: [Search/Retrieve/Finish]
Action Input: {"key": "value"}

Complete Example:
Question: What is the capital of the country where the Eiffel Tower is located?

Thought: I need to find where the Eiffel Tower is located.
Action: Search
Action Input: {"query": "Eiffel Tower"}

Observation: Found 3 relevant document(s): Eiffel Tower, Paris architecture, Gustave Eiffel

Thought: Search returned document titles. I need to retrieve "Eiffel Tower" to read its content.
Action: Retrieve
Action Input: {"title": "Eiffel Tower"}

Observation: The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France...

Thought: The document says the Eiffel Tower is in Paris, France. The capital of France is Paris.
Action: Finish
Action Input: {"answer": "Paris"}

Now answer the user's question using this same pattern: Search -> Retrieve -> Finish.
"""
    
    def _parse_response(self, response: str) -> tuple[str, str, Dict[str, Any]]:
        """
        Parse LLM response to extract thought, action, and action_input.
        Returns (thought, action, action_input_dict)
        """
        thought = ""
        action = ""
        action_input = {}
        
        lines = response.strip().split('\n')
        current_section = None
        thought_lines = []
        action_input_lines = []
        
        for line in lines:
            line_lower = line.lower().strip()
            if line_lower.startswith('thought:'):
                current_section = 'thought'
                thought_lines.append(line.split(':', 1)[1].strip())
            elif line_lower.startswith('action:'):
                current_section = 'action'
                action = line.split(':', 1)[1].strip()
            elif line_lower.startswith('action input:'):
                current_section = 'action_input'
                action_input_str = line.split(':', 1)[1].strip()
                action_input_lines.append(action_input_str)
            elif current_section == 'thought':
                thought_lines.append(line.strip())
            elif current_section == 'action_input':
                action_input_lines.append(line.strip())
        
        thought = ' '.join(thought_lines).strip()
        
        # Try to parse action_input as JSON
        if action_input_lines:
            action_input_str = ' '.join(action_input_lines)
            # Try to extract JSON from the string
            try:
                # Look for JSON object in the string
                start_idx = action_input_str.find('{')
                end_idx = action_input_str.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    action_input = json.loads(action_input_str[start_idx:end_idx])
                else:
                    # Fallback: treat as simple string value
                    action_input = {"value": action_input_str}
            except json.JSONDecodeError:
                # If JSON parsing fails, wrap in a dict
                action_input = {"value": action_input_str}
        
        return thought, action, action_input
    
    async def _search(self, query: str) -> str:
        """Search tool - uses custom function if provided, otherwise returns mock results."""
        if self._custom_search:
            return await self._custom_search(query)
        # Default mock behavior
        return f"Search results for '{query}': [Document 1, Document 2, Document 3]"
    
    async def _retrieve(self, title: str) -> str:
        """Retrieve tool - uses custom function if provided, otherwise returns mock content."""
        if self._custom_retrieve:
            return await self._custom_retrieve(title)
        # Default mock behavior
        return f"Document content for '{title}': [This is mock content. In production, this would fetch the actual document from HotpotQA.]"
    
    async def _finish(self, answer: str) -> str:
        """Finish tool - signals the agent is done."""
        return f"Task completed. Final answer: {answer}"
    
    async def run(
        self,
        question: str,
        task_id: str,
        run_id: Optional[str] = None,
    ) -> AgentRun:
        """
        Run the agent on a question.
        
        Args:
            question: The question to answer
            task_id: Identifier for the task
            run_id: Optional identifier for this specific run
            
        Returns:
            AgentRun object with all steps and final answer
        """
        if run_id is None:
            run_id = f"{task_id}_{datetime.now().isoformat()}"
        
        start_time = datetime.now().isoformat()
        steps = []
        conversation_history = []
        final_answer = None
        success = False
        error = None
        
        try:
            # Initial question
            conversation_history.append({
                "role": "user",
                "content": f"Question: {question}"
            })
            
            for step_num in range(1, self.max_steps + 1):
                # Get agent response
                try:
                    response = await self._call_llm(
                        messages=conversation_history,
                        system_prompt=self._get_system_prompt(),
                    )
                except Exception as e:
                    error = f"LLM call failed at step {step_num}: {str(e)}"
                    break
                
                # Parse response
                thought, action, action_input = self._parse_response(response)
                
                # Validate action
                if action not in self.tools:
                    error = f"Invalid action '{action}' at step {step_num}"
                    break
                
                # Execute action
                try:
                    if action == "Finish":
                        # Extract answer from action_input
                        if isinstance(action_input, dict):
                            final_answer = action_input.get("answer", str(action_input))
                        else:
                            final_answer = str(action_input)
                        observation = await self.tools[action](final_answer)
                        success = True
                    else:
                        # Extract input parameter (usually "query" or "title")
                        if isinstance(action_input, dict):
                            # Try common parameter names
                            param_value = (
                                action_input.get("query") or
                                action_input.get("title") or
                                action_input.get("value") or
                                list(action_input.values())[0] if action_input else ""
                            )
                        else:
                            param_value = str(action_input)
                        
                        observation = await self.tools[action](param_value)
                except Exception as e:
                    observation = f"Error executing action: {str(e)}"
                    error = observation
                    break
                
                # Log step (include hidden states if available from llama_spcs)
                hidden_states = None
                if self.provider in ("llama_spcs", "llama_k8s") and self._last_hidden_states:
                    hidden_states = self._last_hidden_states
                    self._last_hidden_states = None  # Clear after capturing
                
                step = AgentStep(
                    step_number=step_num,
                    timestamp=datetime.now().isoformat(),
                    thought=thought,
                    action=action,
                    action_input=action_input,
                    observation=observation,
                    raw_response=response,
                    hidden_states=hidden_states,
                )
                steps.append(step)
                
                # Add to conversation history
                conversation_history.append({
                    "role": "assistant",
                    "content": response
                })
                conversation_history.append({
                    "role": "user",
                    "content": f"Observation: {observation}"
                })
                
                # Check if finished
                if action == "Finish":
                    break
            
            if not success and not error:
                error = f"Reached maximum steps ({self.max_steps}) without finishing"
        
        except Exception as e:
            error = f"Unexpected error: {str(e)}"
        
        end_time = datetime.now().isoformat()
        
        return AgentRun(
            run_id=run_id,
            task_id=task_id,
            model=self.model,
            provider=self.provider,
            question=question,
            steps=steps,
            final_answer=final_answer,
            success=success,
            error=error,
            start_time=start_time,
            end_time=end_time,
        )
    
    def to_json(self, run: AgentRun) -> Dict[str, Any]:
        """Convert AgentRun to JSON-serializable dict."""
        return asdict(run)
    
    def save_run(self, run: AgentRun, filepath: str):
        """Save a run to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_json(run), f, indent=2, ensure_ascii=False)
    
    def log_step(self, step: AgentStep) -> Dict[str, Any]:
        """
        Convert a single step to JSON-serializable dict.
        Useful for incremental logging during execution.
        """
        return asdict(step)
    
    def log_step_to_file(self, step: AgentStep, filepath: str, append: bool = True):
        """
        Log a single step to a JSON file (one JSON object per line).
        Useful for streaming logs during long-running experiments.
        """
        mode = 'a' if append else 'w'
        with open(filepath, mode) as f:
            json.dump(self.log_step(step), f, ensure_ascii=False)
            f.write('\n')  # Newline-delimited JSON


async def run_agent_async(
    question: str,
    task_id: str,
    model: str,
    provider: Literal["openai", "together", "anthropic", "llama_spcs", "llama_k8s"],
    openai_api_key: Optional[str] = None,
    together_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
    spcs_connection: Optional[str] = None,
    k8s_endpoint: Optional[str] = None,
    run_id: Optional[str] = None,
    **kwargs,
) -> AgentRun:
    """
    Convenience function to run an agent asynchronously.
    
    Args:
        question: The question to answer
        task_id: Identifier for the task
        model: Model name
        provider: "openai", "together", "anthropic", "llama_spcs", or "llama_k8s"
        openai_api_key: OpenAI API key
        together_api_key: Together AI API key
        anthropic_api_key: Anthropic API key
        spcs_connection: Snowflake connection name (for llama_spcs)
        k8s_endpoint: Llama 70B k8s endpoint URL (for llama_k8s)
        run_id: Optional run identifier
        **kwargs: Additional arguments passed to ReActAgent
        
    Returns:
        AgentRun object
    """
    agent = ReActAgent(
        model=model,
        provider=provider,
        openai_api_key=openai_api_key,
        together_api_key=together_api_key,
        anthropic_api_key=anthropic_api_key,
        spcs_connection=spcs_connection,
        k8s_endpoint=k8s_endpoint,
        **kwargs,
    )
    return await agent.run(question=question, task_id=task_id, run_id=run_id)


async def run_agent_n_times(
    question: str,
    task_id: str,
    n: int,
    model: str,
    provider: Literal["openai", "together", "anthropic", "llama_spcs", "llama_k8s"],
    openai_api_key: Optional[str] = None,
    together_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
    spcs_connection: Optional[str] = None,
    k8s_endpoint: Optional[str] = None,
    run_id_prefix: Optional[str] = None,
    batch_size: Optional[int] = None,
    batch_delay: float = 0.0,
    **kwargs,
) -> List[AgentRun]:
    """
    Run an agent N times in parallel on the same question.
    Each run gets a unique run_id.
    
    Args:
        question: The question to answer
        task_id: Identifier for the task
        n: Number of runs
        model: Model name
        provider: "openai", "together", "anthropic", "llama_spcs", or "llama_k8s"
        openai_api_key: OpenAI API key
        together_api_key: Together AI API key
        anthropic_api_key: Anthropic API key
        spcs_connection: Snowflake connection name (for llama_spcs)
        k8s_endpoint: Llama 70B k8s endpoint URL (for llama_k8s)
        run_id_prefix: Optional prefix for run IDs (defaults to task_id)
        batch_size: Optional batch size for throttling (used mainly for OpenAI)
        batch_delay: Delay in seconds between batches
        **kwargs: Additional arguments passed to ReActAgent
        
    Returns:
        List of AgentRun objects, one per run
    """
    if run_id_prefix is None:
        run_id_prefix = task_id
    
    # Create tasks
    tasks: List[asyncio.Task] = []
    for i in range(n):
        run_id = f"{run_id_prefix}_run_{i+1:04d}"
        task = run_agent_async(
            question=question,
            task_id=task_id,
            model=model,
            provider=provider,
            openai_api_key=openai_api_key,
            together_api_key=together_api_key,
            anthropic_api_key=anthropic_api_key,
            spcs_connection=spcs_connection,
            k8s_endpoint=k8s_endpoint,
            run_id=run_id,
            **kwargs,
        )
        tasks.append(task)
    
    # Execute runs, optionally in throttled batches (for OpenAI rate limits)
    results: List[Any] = []
    if provider == "openai" and batch_size and batch_size > 0 and batch_size < n:
        for start in range(0, n, batch_size):
            batch_tasks = tasks[start:start + batch_size]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            results.extend(batch_results)
            # Avoid sleeping after the last batch
            if start + batch_size < n and batch_delay > 0:
                await asyncio.sleep(batch_delay)
    else:
        # Default: fire all in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle any exceptions
    runs = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            # Create an error run
            error_run = AgentRun(
                run_id=f"{run_id_prefix}_run_{i+1:04d}",
                task_id=task_id,
                model=model,
                provider=provider,
                question=question,
                steps=[],
                success=False,
                error=f"Exception during execution: {str(result)}",
                start_time=datetime.now().isoformat(),
                end_time=datetime.now().isoformat(),
            )
            runs.append(error_run)
        else:
            runs.append(result)
    
    return runs


# --- Demo config (python agent.py) ---
# We are NOT running full HotpotQA here. This is a demo with:
#   - A single fixed question (change DEMO_QUESTION to experiment)
#   - Mock Search/Retrieve tools (no HotpotQA corpus yet)
#   - Option 2: Together AI for parallel runs (so you can run many times without burning OpenAI credits)
# Full HotpotQA (200 examples, real search/retrieve) → use runner.py + data/ when built.
DEMO_QUESTION = "What is the capital of France?"
DEMO_N_PARALLEL = 3
DEMO_MODEL_TOGETHER = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"


if __name__ == "__main__":
    import os

    async def example():
        print("=== Agent demo (NOT full HotpotQA) ===")
        print(f"Question: {DEMO_QUESTION}")
        print("Tools: mock Search/Retrieve (HotpotQA not wired yet)\n")

        # Example 1: Single run with OpenAI
        if os.getenv("OPENAI_API_KEY"):
            agent = ReActAgent(
                model="gpt-4o",
                provider="openai",
                openai_api_key=os.getenv("OPENAI_API_KEY"),
            )
            run = await agent.run(
                question=DEMO_QUESTION,
                task_id="example_1",
            )
            print(json.dumps(agent.to_json(run), indent=2))

        # Example 2: Single run with Together AI
        if os.getenv("TOGETHER_API_KEY"):
            agent = ReActAgent(
                model=DEMO_MODEL_TOGETHER,
                provider="together",
                together_api_key=os.getenv("TOGETHER_API_KEY"),
            )
            run = await agent.run(
                question=DEMO_QUESTION,
                task_id="example_2",
            )
            print(json.dumps(agent.to_json(run), indent=2))

        # Example 3: Run N times in parallel (option 2 = Together AI, for repeated runs)
        if os.getenv("TOGETHER_API_KEY"):
            print(f"\nRunning {DEMO_N_PARALLEL} parallel runs (Together AI)...")
            runs = await run_agent_n_times(
                question=DEMO_QUESTION,
                task_id="example_parallel",
                n=DEMO_N_PARALLEL,
                model=DEMO_MODEL_TOGETHER,
                provider="together",
                together_api_key=os.getenv("TOGETHER_API_KEY"),
            )
            print(f"Completed {len(runs)} runs")
            for r in runs:
                print(f"  {r.run_id}: success={r.success}, steps={len(r.steps)}")
        elif os.getenv("OPENAI_API_KEY"):
            print("\nTOGETHER_API_KEY not set; skipping parallel runs. Set it to use option 2.")

    asyncio.run(example())

