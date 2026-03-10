"""
Llama 3.1 8B Instruct server with hidden state extraction.
Exposes both standard completion and hidden states via FastAPI.
"""
import os
import json
import torch

# Fix for CUBLAS_STATUS_INVALID_VALUE on H200 GPUs with PyTorch 2.10+cu128
# The default cublas library has issues with fp16 on these GPUs
torch.backends.cuda.preferred_blas_library('cublaslt')

import numpy as np
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Llama Hidden States API", version="1.0.0")

# Global model and tokenizer
model = None
tokenizer = None
device = None


class CompletionRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    return_hidden_states: bool = False
    hidden_state_layers: Optional[List[int]] = None  # None = all layers


class CompletionResponse(BaseModel):
    text: str
    prompt_tokens: int
    completion_tokens: int
    hidden_states: Optional[Dict[str, Any]] = None


class HiddenStatesRequest(BaseModel):
    text: str
    layers: Optional[List[int]] = None  # None = all layers
    pooling: str = "last"  # "last", "mean", "all"


class HiddenStatesResponse(BaseModel):
    hidden_states: Dict[str, List[List[float]]]  # layer -> [seq_len, hidden_dim] or pooled
    num_layers: int
    hidden_dim: int
    sequence_length: int


class ChatMessage(BaseModel):
    role: str  # "system", "user", "assistant"
    content: str


class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stop_sequences: Optional[List[str]] = None
    return_hidden_states: bool = True  # Default True for our use case
    hidden_state_pooling: str = "last"  # "last", "mean"


class ChatCompletionResponse(BaseModel):
    content: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str  # "stop", "length"
    hidden_states: Optional[Dict[str, Any]] = None  # All 33 layers from input context


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model_name: str


class SteeredChatCompletionRequest(BaseModel):
    """Chat completion with activation steering intervention."""
    messages: List[ChatMessage]
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stop_sequences: Optional[List[str]] = None
    return_hidden_states: bool = True
    hidden_state_pooling: str = "last"
    # Steering parameters
    steering_vector: List[float]  # Shape: (hidden_dim,) - centroid or target vector
    steering_layer: int = 72  # Layer to apply steering (72 or 32)
    steering_scale: float = 1.0  # Multiplier for steering vector
    steering_mode: str = "add"  # "add" (direct add), "push" (state - centroid), "pull" (centroid - state)


class SteeredChatCompletionResponse(BaseModel):
    """Response with steering metadata."""
    content: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    hidden_states: Optional[Dict[str, Any]] = None
    steering_applied: Dict[str, Any] = None  # Metadata about the steering


@app.on_event("startup")
async def load_model():
    global model, tokenizer, device
    
    model_name = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
    hf_token = os.getenv("HF_TOKEN")
    use_nnsight = os.getenv("USE_NNSIGHT", "false").lower() == "true"
    use_pipeline_parallel = os.getenv("USE_PIPELINE_PARALLEL", "false").lower() == "true"
    
    logger.info(f"Loading model: {model_name}")
    logger.info(f"Using nnsight: {use_nnsight}")
    logger.info(f"Using pipeline parallelism: {use_pipeline_parallel}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    if use_nnsight:
        # Use nnsight for steering interventions
        from nnsight import LanguageModel
        logger.info("Loading model with nnsight (dispatch=True for GPU inference)...")
        model = LanguageModel(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            token=hf_token,
            trust_remote_code=True,
            dispatch=True,  # Required to actually load weights on GPU
        )
        tokenizer = model.tokenizer
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token,
            trust_remote_code=True
        )
        
        # Determine device map
        if use_pipeline_parallel:
            # Use "sequential" for true pipeline parallelism without tensor sharding
            # This places entire modules on devices sequentially, avoiding cross-device ops
            device_map = "sequential"
            logger.info("Using sequential device map for pipeline parallelism")
        else:
            device_map = "auto"
            
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True,
        )
        model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Model loaded successfully on {device}")
    num_layers = model.config.num_hidden_layers if hasattr(model, 'config') else model.model.config.num_hidden_layers
    hidden_size = model.config.hidden_size if hasattr(model, 'config') else model.model.config.hidden_size
    logger.info(f"Model has {num_layers} layers, hidden_dim={hidden_size}")


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy" if model is not None else "loading",
        model_loaded=model is not None,
        device=str(device) if device else "unknown",
        model_name=os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
    )


@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(request: CompletionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    inputs = tokenizer(request.prompt, return_tensors="pt").to(device)
    prompt_tokens = inputs.input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature if request.temperature > 0 else None,
            top_p=request.top_p,
            do_sample=request.temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            output_hidden_states=request.return_hidden_states,
            return_dict_in_generate=True,
        )
    
    generated_ids = outputs.sequences[0][prompt_tokens:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    hidden_states_data = None
    if request.return_hidden_states and hasattr(outputs, 'hidden_states'):
        hidden_states_data = _extract_hidden_states(
            outputs.hidden_states,
            request.hidden_state_layers
        )
    
    return CompletionResponse(
        text=generated_text,
        prompt_tokens=prompt_tokens,
        completion_tokens=len(generated_ids),
        hidden_states=hidden_states_data
    )


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """
    Chat completion endpoint with hidden state extraction.
    
    Extracts hidden states from the INPUT context BEFORE generation,
    capturing the model's internal state before it decides what to do.
    Returns all 33 layers (embedding + 32 transformer layers).
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Convert messages to Llama chat format
    messages_for_template = [
        {"role": m.role, "content": m.content} for m in request.messages
    ]
    
    # Apply chat template - this formats messages properly for Llama 3.1
    prompt = tokenizer.apply_chat_template(
        messages_for_template,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_tokens = inputs.input_ids.shape[1]
    
    # Step 1: Extract hidden states from INPUT context (before generation)
    hidden_states_data = None
    if request.return_hidden_states:
        with torch.no_grad():
            input_outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )
        
        # Extract all 33 layers (layer 0 = embeddings, layers 1-32 = transformer)
        all_hidden_states = input_outputs.hidden_states
        num_layers = len(all_hidden_states) - 1  # 32 transformer layers
        seq_len = all_hidden_states[0].shape[1]
        hidden_dim = all_hidden_states[0].shape[2]
        
        hidden_states_data = {
            "layers": {},
            "metadata": {
                "num_layers": num_layers,
                "hidden_dim": int(hidden_dim),
                "input_sequence_length": seq_len,
                "pooling": request.hidden_state_pooling,
                "extraction_point": "input_context_before_generation"
            }
        }
        
        # Extract all 33 layers
        for layer_idx in range(num_layers + 1):
            hidden = all_hidden_states[layer_idx][0]  # [seq_len, hidden_dim]
            
            if request.hidden_state_pooling == "last":
                # Last token representation (most relevant for next-token prediction)
                pooled = hidden[-1].cpu().float().numpy().tolist()
                hidden_states_data["layers"][f"layer_{layer_idx}"] = pooled
            elif request.hidden_state_pooling == "mean":
                # Mean pooling across sequence
                pooled = hidden.mean(dim=0).cpu().float().numpy().tolist()
                hidden_states_data["layers"][f"layer_{layer_idx}"] = pooled
    
    # Step 2: Generate response
    stop_token_ids = []
    if request.stop_sequences:
        for seq in request.stop_sequences:
            tokens = tokenizer.encode(seq, add_special_tokens=False)
            if tokens:
                stop_token_ids.append(tokens[0])
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature if request.temperature > 0 else None,
            top_p=request.top_p,
            do_sample=request.temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode generated tokens (excluding prompt)
    generated_ids = outputs[0][prompt_tokens:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Determine finish reason
    finish_reason = "length"
    if len(generated_ids) < request.max_new_tokens:
        finish_reason = "stop"
    
    # Check for stop sequences and truncate if needed
    if request.stop_sequences:
        for seq in request.stop_sequences:
            if seq in generated_text:
                generated_text = generated_text.split(seq)[0]
                finish_reason = "stop"
                break
    
    return ChatCompletionResponse(
        content=generated_text,
        prompt_tokens=prompt_tokens,
        completion_tokens=len(generated_ids),
        finish_reason=finish_reason,
        hidden_states=hidden_states_data
    )


@app.post("/v1/chat/completions_steered", response_model=SteeredChatCompletionResponse)
async def chat_completions_steered(request: SteeredChatCompletionRequest):
    """
    Chat completion with activation steering using nnsight.
    
    Applies a steering vector to the hidden states at a specified layer during
    generation. This allows causal intervention experiments to test if modifying
    internal representations affects behavioral consistency.
    
    Uses nnsight library for safe interventions on multi-GPU accelerate models.
    
    Args:
        steering_vector: Direction to steer (shape: hidden_dim)
        steering_layer: Which transformer layer to intervene (0-79 for 70B, 0-31 for 8B)
        steering_scale: Multiplier for the steering vector (e.g., 0.5, 1.0, 2.0)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    use_nnsight = os.getenv("USE_NNSIGHT", "false").lower() == "true"
    
    # Validate steering layer
    num_layers = model.config.num_hidden_layers
    if request.steering_layer < 0 or request.steering_layer >= num_layers:
        raise HTTPException(
            status_code=400,
            detail=f"steering_layer must be between 0 and {num_layers-1}"
        )
    
    # Validate steering vector dimension
    hidden_dim = model.config.hidden_size
    if len(request.steering_vector) != hidden_dim:
        raise HTTPException(
            status_code=400,
            detail=f"steering_vector must have {hidden_dim} dimensions, got {len(request.steering_vector)}"
        )
    
    # Convert steering vector to tensor
    steering_tensor = torch.tensor(request.steering_vector, dtype=torch.float16)
    
    # Convert messages to Llama chat format
    messages_for_template = [
        {"role": m.role, "content": m.content} for m in request.messages
    ]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages_for_template,
        tokenize=False,
        add_generation_prompt=True
    )
    
    steering_mode = getattr(request, 'steering_mode', 'add')
    steering_applied_info = {
        "layer": request.steering_layer,
        "scale": request.steering_scale,
        "mode": steering_mode,
        "vector_norm": float(torch.norm(steering_tensor).item()),
        "intervention_type": "nnsight_additive_last_token" if use_nnsight else "hook_additive_last_token"
    }
    
    if use_nnsight:
        # ============ HOOK-BASED STEERING WITH NEW TENSOR CREATION ============
        # Create new tensors instead of in-place modification to avoid CUDA errors
        logger.info(f"Using hook-based steering (new tensor) at layer {request.steering_layer}, mode={steering_mode}")
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        # Move to first GPU (model handles distribution)
        inputs = {k: v.cuda() for k, v in inputs.items()}
        prompt_tokens = inputs["input_ids"].shape[1]
        
        def steering_hook(module, input, output):
            """
            Hook that creates new tensors to avoid corrupting accelerate metadata.
            Returns a new tuple with modified hidden states.
            
            Supports three steering modes:
            - "add": new_state = state + scale * steering_vector (direct add)
            - "push": new_state = state + scale * (state - centroid) (push away from centroid)
            - "pull": new_state = state + scale * (centroid - state) (pull toward centroid)
            """
            hidden_states = output[0]  # [batch, seq, hidden_dim]
            
            # Get device and dtype from the tensor
            target_device = hidden_states.device
            target_dtype = hidden_states.dtype
            
            # Clone the hidden states to create a NEW tensor not linked to accelerate
            new_hidden_states = hidden_states.clone()
            
            # Get current state at last token position
            current_state = new_hidden_states[:, -1, :]  # [batch, hidden_dim]
            
            # Move centroid/target to correct device and dtype
            centroid = steering_tensor.to(device=target_device, dtype=target_dtype)
            
            # Compute steering delta based on mode
            if steering_mode == "push":
                # Push away from centroid: add (state - centroid) direction
                direction = current_state - centroid
                steering_delta = request.steering_scale * direction
            elif steering_mode == "pull":
                # Pull toward centroid: add (centroid - state) direction
                direction = centroid - current_state
                steering_delta = request.steering_scale * direction
            else:  # "add" mode (default, backward compatible)
                # Direct add: just add the steering vector
                steering_delta = request.steering_scale * centroid
            
            # Modify the CLONE (not the original)
            new_hidden_states[:, -1, :] = current_state + steering_delta
            
            # Return new tuple with the modified clone
            # LlamaDecoderLayer output: (hidden_states, self_attn, present_key_value)
            return (new_hidden_states,) + output[1:]
        
        # nnsight.LanguageModel structure (empirically determined):
        # model = nnsight LanguageModel wrapper (has .generate())
        # model.model = LlamaModel (the inner transformer, NOT LlamaForCausalLM)
        # model.model.layers = the decoder layers
        # 
        # For generation: use model._model (the underlying HF model) which is LlamaForCausalLM
        target_layer = model.model.layers[request.steering_layer]
        hook_handle = target_layer.register_forward_hook(steering_hook)
        
        try:
            hidden_states_data = None
            
            # Generate with steering hook active
            # Use model._model which is the actual LlamaForCausalLM with .generate()
            with torch.no_grad():
                outputs = model._model.generate(
                    **inputs,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature if request.temperature > 0 else None,
                    top_p=request.top_p,
                    do_sample=request.temperature > 0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
        except Exception as e:
            logger.error(f"Steering generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Steering failed: {str(e)}")
        finally:
            hook_handle.remove()
        
        generated_ids = outputs[0][prompt_tokens:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        logger.info(f"Hook steering succeeded, output: {generated_text[:50]}...")
        
    else:
        # ============ HOOK-BASED STEERING (PIPELINE PARALLEL COMPATIBLE) ============
        # With pipeline parallelism, entire layers are on single GPUs, so hooks work
        logger.info(f"Using hook-based steering at layer {request.steering_layer}, mode={steering_mode}")
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        # Move to first GPU (model handles distribution)
        inputs = {k: v.cuda() for k, v in inputs.items()}
        prompt_tokens = inputs["input_ids"].shape[1]
        
        def steering_hook(module, input, output):
            """
            Post-hook to modify hidden states after layer processing.
            With pipeline parallelism, tensors are not sharded so this works.
            
            Supports three steering modes:
            - "add": new_state = state + scale * steering_vector (direct add)
            - "push": new_state = state + scale * (state - centroid) (push away from centroid)
            - "pull": new_state = state + scale * (centroid - state) (pull toward centroid)
            """
            hidden_states = output[0]  # [batch, seq, hidden_dim]
            
            # Get device and dtype from the tensor
            target_device = hidden_states.device
            target_dtype = hidden_states.dtype
            
            # Clone to avoid in-place issues
            new_hidden_states = hidden_states.clone()
            
            # Get current state at last token position
            current_state = new_hidden_states[:, -1, :]  # [batch, hidden_dim]
            
            # Move centroid/target to correct device and dtype
            centroid = steering_tensor.to(device=target_device, dtype=target_dtype)
            
            # Compute steering delta based on mode
            if steering_mode == "push":
                # Push away from centroid: add (state - centroid) direction
                direction = current_state - centroid
                steering_delta = request.steering_scale * direction
            elif steering_mode == "pull":
                # Pull toward centroid: add (centroid - state) direction
                direction = centroid - current_state
                steering_delta = request.steering_scale * direction
            else:  # "add" mode (default, backward compatible)
                # Direct add: just add the steering vector
                steering_delta = request.steering_scale * centroid
            
            # Apply steering
            new_hidden_states[:, -1, :] = current_state + steering_delta
            
            # Return new output tuple
            return (new_hidden_states,) + output[1:]
        
        # Register post-hook on the target layer
        target_layer = model.model.layers[request.steering_layer]
        hook_handle = target_layer.register_forward_hook(steering_hook)
        
        try:
            hidden_states_data = None
            
            # Generate with steering hook active
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature if request.temperature > 0 else None,
                    top_p=request.top_p,
                    do_sample=request.temperature > 0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
        except Exception as e:
            logger.error(f"Steering generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Steering failed: {str(e)}")
        finally:
            hook_handle.remove()
        
        generated_ids = outputs[0][prompt_tokens:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        logger.info(f"Hook steering succeeded, output: {generated_text[:50]}...")
    
    # Determine finish reason
    finish_reason = "length"
    if len(generated_ids) < request.max_new_tokens:
        finish_reason = "stop"
    
    # Check for stop sequences and truncate if needed
    if request.stop_sequences:
        for seq in request.stop_sequences:
            if seq in generated_text:
                generated_text = generated_text.split(seq)[0]
                finish_reason = "stop"
                break
    
    logger.info(f"Steered completion: layer={request.steering_layer}, scale={request.steering_scale}, "
                f"output_len={len(generated_ids)}, method={'nnsight' if use_nnsight else 'hook'}")
    
    return SteeredChatCompletionResponse(
        content=generated_text,
        prompt_tokens=prompt_tokens,
        completion_tokens=len(generated_ids),
        finish_reason=finish_reason,
        hidden_states=hidden_states_data,
        steering_applied=steering_applied_info
    )


@app.post("/v1/chat/completions_sf")
async def chat_completions_snowflake(request: dict):
    """
    Snowflake service function compatible chat completion endpoint.
    Expects {"data": [[row_id, messages_array, max_new_tokens, temperature], ...]}
    Returns {"data": [[row_id, result], ...]}
    
    Each message in messages_array should be {"role": "...", "content": "..."}
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    for row in request.get("data", []):
        row_id = row[0]
        messages_raw = row[1] if len(row) > 1 else []
        max_new_tokens = row[2] if len(row) > 2 and row[2] else 512
        temperature = row[3] if len(row) > 3 and row[3] is not None else 0.7
        
        try:
            # Convert messages to ChatMessage format
            messages = [ChatMessage(role=m["role"], content=m["content"]) for m in messages_raw]
            
            # Create request object
            chat_request = ChatCompletionRequest(
                messages=messages,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                return_hidden_states=True,
                hidden_state_pooling="last"
            )
            
            # Call the main chat completion function
            response = await chat_completions(chat_request)
            results.append([row_id, response.model_dump()])
        except Exception as e:
            results.append([row_id, {"error": str(e)}])
    
    return {"data": results}


@app.post("/v1/hidden_states", response_model=HiddenStatesResponse)
async def get_hidden_states(request: HiddenStatesRequest):
    """
    Extract hidden states from a forward pass (no generation).
    Useful for analysis of specific text representations.
    """
    return _extract_hidden_states_for_text(request.text, request.layers, request.pooling)


@app.post("/v1/hidden_states_sf")
async def get_hidden_states_snowflake(request: dict):
    """
    Snowflake service function compatible endpoint.
    Expects {"data": [[row_id, text, layers, pooling], ...]}
    Returns {"data": [[row_id, result], ...]}
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    for row in request.get("data", []):
        row_id = row[0]
        text = row[1]
        layers = row[2] if len(row) > 2 and row[2] else None
        pooling = row[3] if len(row) > 3 and row[3] else "last"
        
        try:
            response = _extract_hidden_states_for_text(text, layers, pooling)
            results.append([row_id, response.model_dump()])
        except Exception as e:
            results.append([row_id, {"error": str(e)}])
    
    return {"data": results}


def _extract_hidden_states_for_text(text: str, layers: Optional[List[int]], pooling: str) -> HiddenStatesResponse:
    """Core hidden state extraction logic."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )
    
    # outputs.hidden_states is tuple of (num_layers+1) tensors, each [batch, seq, hidden]
    all_hidden_states = outputs.hidden_states
    num_layers = len(all_hidden_states) - 1  # -1 for embedding layer
    seq_len = all_hidden_states[0].shape[1]
    hidden_dim = all_hidden_states[0].shape[2]
    
    # Select layers
    layers_to_extract = layers if layers else list(range(num_layers + 1))
    
    result = {}
    for layer_idx in layers_to_extract:
        if layer_idx < 0 or layer_idx > num_layers:
            continue
        
        hidden = all_hidden_states[layer_idx][0]  # [seq_len, hidden_dim]
        
        if pooling == "last":
            # Last token representation
            pooled = hidden[-1].cpu().numpy().tolist()
            result[f"layer_{layer_idx}"] = [pooled]
        elif pooling == "mean":
            # Mean pooling across sequence
            pooled = hidden.mean(dim=0).cpu().numpy().tolist()
            result[f"layer_{layer_idx}"] = [pooled]
        else:
            # Return all token representations
            result[f"layer_{layer_idx}"] = hidden.cpu().numpy().tolist()
    
    return HiddenStatesResponse(
        hidden_states=result,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        sequence_length=seq_len
    )


def _extract_hidden_states(
    hidden_states_tuple,
    layers: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Extract hidden states from generation output.
    hidden_states_tuple is a tuple of (num_generated_tokens) tuples,
    each containing (num_layers+1) tensors of shape [batch, 1, hidden_dim].
    """
    if not hidden_states_tuple:
        return None
    
    # Get dimensions from first token's hidden states
    first_token_hidden = hidden_states_tuple[0]
    num_layers = len(first_token_hidden) - 1
    hidden_dim = first_token_hidden[0].shape[-1]
    
    layers_to_extract = layers if layers else list(range(num_layers + 1))
    
    result = {"layers": {}, "metadata": {
        "num_layers": num_layers,
        "hidden_dim": int(hidden_dim),
        "num_generated_tokens": len(hidden_states_tuple)
    }}
    
    for layer_idx in layers_to_extract:
        if layer_idx < 0 or layer_idx > num_layers:
            continue
        
        # Collect hidden states for this layer across all generated tokens
        layer_hidden = []
        for token_hidden in hidden_states_tuple:
            # token_hidden[layer_idx] is [batch, 1, hidden_dim]
            h = token_hidden[layer_idx][0, 0].cpu().numpy().tolist()
            layer_hidden.append(h)
        
        result["layers"][f"layer_{layer_idx}"] = layer_hidden
    
    return result


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
