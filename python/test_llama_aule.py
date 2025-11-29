#!/usr/bin/env python3
"""
Test aule-attention with real LLaMA model inference.

This script loads a LLaMA model using HuggingFace transformers
and patches its attention mechanism to use aule-attention via OpenCL.

NO ROCm/CUDA required - runs on MI300X via Mesa Rusticl!

Usage:
    python test_llama_aule.py

Requirements:
    pip install transformers accelerate sentencepiece
"""

import os
import time
import numpy as np
from typing import Optional, Tuple

# Force CPU for PyTorch (we'll use aule for attention)
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Try to import required packages
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    print("ERROR: PyTorch required. Install with: pip install torch")
    exit(1)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    from transformers.models.llama.modeling_llama import LlamaAttention, LlamaConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("ERROR: transformers required. Install with: pip install transformers")
    exit(1)

# Import aule-attention
try:
    import aule_unified as aule
    AULE_AVAILABLE = True
except ImportError:
    try:
        from aule import attention as aule_attention_func, Attention
        AULE_AVAILABLE = True
    except ImportError:
        print("ERROR: aule-attention not found")
        exit(1)


class AuleLlamaAttention(torch.nn.Module):
    """
    LLaMA attention layer using aule-attention backend.

    Replaces the standard PyTorch attention with OpenCL-accelerated
    FlashAttention-2 via aule.
    """

    def __init__(self, original_attention: LlamaAttention, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = getattr(config, 'num_key_value_heads', self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        # Copy projection weights from original
        self.q_proj = original_attention.q_proj
        self.k_proj = original_attention.k_proj
        self.v_proj = original_attention.v_proj
        self.o_proj = original_attention.o_proj

        # Initialize aule backend (will auto-select OpenCL on MI300X without ROCm)
        self.aule_attn = None
        self._init_aule()

    def _init_aule(self):
        """Initialize aule attention backend."""
        try:
            self.aule_attn = aule.Attention(backend='auto')
            print(f"  AuleLlamaAttention using: {self.aule_attn.backend_name}")
        except Exception as e:
            print(f"  Warning: Failed to init aule ({e}), falling back to PyTorch")
            self.aule_attn = None

    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat KV heads for GQA."""
        if n_rep == 1:
            return hidden_states
        batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_kv_heads, n_rep, seq_len, head_dim
        )
        return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to [batch, heads, seq, dim]
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Handle KV cache
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # Repeat KV for GQA
        key_states = self._repeat_kv(key_states, self.num_key_value_groups)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups)

        # Use aule attention if available
        if self.aule_attn is not None:
            # Convert to numpy for aule
            Q_np = query_states.detach().cpu().numpy().astype(np.float32)
            K_np = key_states.detach().cpu().numpy().astype(np.float32)
            V_np = value_states.detach().cpu().numpy().astype(np.float32)

            # Run attention via aule (causal=True for autoregressive)
            output_np = self.aule_attn.forward(Q_np, K_np, V_np, causal=True)

            # Convert back to torch
            attn_output = torch.from_numpy(output_np).to(hidden_states.device, dtype=hidden_states.dtype)
        else:
            # Fallback to PyTorch SDPA
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states,
                is_causal=True,
            )

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        # Return (attn_output, past_key_value) for newer transformers
        # The decoder layer unpacks as: hidden_states, _ = self.self_attn(...)
        return attn_output, past_key_value


def patch_llama_attention(model):
    """
    Patch all attention layers in a LLaMA model to use aule-attention.
    """
    print("\nPatching LLaMA attention layers with aule-attention...")
    patched_count = 0

    for name, module in model.named_modules():
        if isinstance(module, LlamaAttention):
            # Get parent module
            parent_name = '.'.join(name.split('.')[:-1])
            attr_name = name.split('.')[-1]

            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model

            # Replace with AuleLlamaAttention
            config = model.config
            new_attn = AuleLlamaAttention(module, config)
            setattr(parent, attr_name, new_attn)
            patched_count += 1

    print(f"Patched {patched_count} attention layers")
    return model


def test_llama_inference():
    """
    Test LLaMA inference with aule-attention.
    """
    print("=" * 70)
    print("AULE-ATTENTION + LLAMA TEST")
    print("Testing real LLM inference without ROCm/CUDA")
    print("=" * 70)

    # Check aule backends
    print("\n--- Available Backends ---")
    backends = aule.get_available_backends()
    print(f"Backends: {backends}")
    aule.print_backend_info()

    # Use a small model for testing
    # Options: "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "meta-llama/Llama-2-7b-hf"
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    print(f"\n--- Loading Model: {model_name} ---")
    print("(This will download ~2GB on first run)")

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model on CPU (we're using aule for GPU acceleration of attention only)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use FP32 for CPU
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        model.eval()

        print(f"Model loaded: {model.config.num_hidden_layers} layers, "
              f"{model.config.num_attention_heads} heads, "
              f"{model.config.hidden_size} hidden size")

    except Exception as e:
        print(f"Failed to load model: {e}")
        print("\nTrying with a minimal config for testing...")

        # Create minimal model for testing
        config = LlamaConfig(
            vocab_size=32000,
            hidden_size=512,
            intermediate_size=1024,
            num_hidden_layers=2,
            num_attention_heads=8,
            max_position_embeddings=512,
        )
        model = AutoModelForCausalLM.from_config(config)
        tokenizer = None
        print("Created minimal test model")

    # Patch attention layers
    model = patch_llama_attention(model)

    # Test prompts
    prompts = [
        "The capital of France is",
        "def fibonacci(n):",
        "Once upon a time",
    ]

    print("\n--- Running Inference ---")

    for prompt in prompts:
        print(f"\nPrompt: {prompt}")

        if tokenizer:
            inputs = tokenizer(prompt, return_tensors="pt")
        else:
            # Fake input for minimal model
            inputs = {"input_ids": torch.randint(0, 32000, (1, 10))}

        # Time the generation
        start_time = time.perf_counter()

        with torch.no_grad():
            if tokenizer:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
                generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                # Just run forward pass for minimal model
                outputs = model(**inputs)
                generated = f"[Forward pass output shape: {outputs.logits.shape}]"

        elapsed = time.perf_counter() - start_time

        print(f"Output: {generated}")
        print(f"Time: {elapsed*1000:.1f}ms")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

    return True


def benchmark_attention_comparison():
    """
    Benchmark aule-attention vs PyTorch on realistic LLaMA dimensions.
    """
    print("\n" + "=" * 70)
    print("ATTENTION BENCHMARK: AULE vs PYTORCH")
    print("=" * 70)

    # LLaMA-7B dimensions
    configs = [
        {"name": "TinyLlama-1.1B", "batch": 1, "heads": 32, "seq": 512, "dim": 64},
        {"name": "LLaMA-7B prefill", "batch": 1, "heads": 32, "seq": 2048, "dim": 128},
        {"name": "LLaMA-7B batch", "batch": 4, "heads": 32, "seq": 512, "dim": 128},
    ]

    for cfg in configs:
        print(f"\n--- {cfg['name']} ---")
        print(f"Shape: [{cfg['batch']}, {cfg['heads']}, {cfg['seq']}, {cfg['dim']}]")

        # Create random tensors
        Q = np.random.randn(cfg['batch'], cfg['heads'], cfg['seq'], cfg['dim']).astype(np.float32)
        K = np.random.randn(cfg['batch'], cfg['heads'], cfg['seq'], cfg['dim']).astype(np.float32)
        V = np.random.randn(cfg['batch'], cfg['heads'], cfg['seq'], cfg['dim']).astype(np.float32)

        # Warmup and benchmark aule
        try:
            with aule.Attention(backend='auto') as attn:
                # Warmup
                _ = attn.forward(Q, K, V, causal=True)

                # Benchmark
                times = []
                for _ in range(5):
                    start = time.perf_counter()
                    output = attn.forward(Q, K, V, causal=True)
                    elapsed = time.perf_counter() - start
                    times.append(elapsed)

                avg_time = np.mean(times) * 1000

                # Calculate TFLOPS
                # FlashAttention: ~4 * batch * heads * seq^2 * dim FLOPs
                flops = 4 * cfg['batch'] * cfg['heads'] * cfg['seq']**2 * cfg['dim']
                tflops = flops / (np.mean(times) * 1e12)

                print(f"Aule ({attn.backend_name}): {avg_time:.2f}ms, {tflops:.3f} TFLOPS")
        except Exception as e:
            print(f"Aule: FAILED - {e}")

        # Benchmark PyTorch CPU
        Q_torch = torch.from_numpy(Q)
        K_torch = torch.from_numpy(K)
        V_torch = torch.from_numpy(V)

        # Warmup
        _ = F.scaled_dot_product_attention(Q_torch, K_torch, V_torch, is_causal=True)

        times = []
        for _ in range(5):
            start = time.perf_counter()
            output = F.scaled_dot_product_attention(Q_torch, K_torch, V_torch, is_causal=True)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_time = np.mean(times) * 1000
        tflops = flops / (np.mean(times) * 1e12)
        print(f"PyTorch CPU: {avg_time:.2f}ms, {tflops:.3f} TFLOPS")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark_attention_comparison()
    else:
        test_llama_inference()
        print("\n")
        benchmark_attention_comparison()
