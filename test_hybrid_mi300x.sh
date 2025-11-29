#!/bin/bash
# Test aule-attention hybrid backend on MI300X
#
# Tests the new optimized OpenCL kernel with vectorization
# and compares against the previous implementation.
#
# Usage: ./test_hybrid_mi300x.sh <droplet-ip>

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <droplet-ip>"
    exit 1
fi

DROPLET_IP="$1"
REMOTE="root@$DROPLET_IP"

echo "============================================================"
echo "HYBRID BACKEND TEST ON MI300X"
echo "Testing optimized OpenCL kernel with vectorization"
echo "============================================================"
echo ""

# Copy updated files
echo "[1/3] Copying updated files..."
cd /home/yab/Sndr
scp python/aule_opencl.py python/aule_hybrid.py python/aule_unified.py "$REMOTE:~/aule-attention/python/"

# Run benchmark
echo "[2/3] Running hybrid backend benchmark..."
ssh "$REMOTE" << 'EOF'
cd ~/aule-attention/python

# Set environment
export CUDA_VISIBLE_DEVICES=""
export HIP_VISIBLE_DEVICES=""

echo ""
echo "=== System Info ==="
python3 -c "
import pyopencl as cl
for p in cl.get_platforms():
    for d in p.get_devices():
        print(f'Device: {d.name}')
        print(f'  Compute units: {d.max_compute_units}')
        print(f'  Global memory: {d.global_mem_size / 1e9:.1f} GB')
        print(f'  Local memory: {d.local_mem_size / 1024:.0f} KB')
        print(f'  Max workgroup: {d.max_work_group_size}')
"

echo ""
echo "=== Running Hybrid Backend Benchmark ==="
python3 aule_hybrid.py

echo ""
echo "=== Comparison: Old vs New Kernel ==="
python3 << 'PYEOF'
import numpy as np
import time

# Test configurations
configs = [
    {"name": "TinyLlama", "batch": 1, "heads": 32, "seq": 512, "dim": 64},
    {"name": "LLaMA-7B", "batch": 1, "heads": 32, "seq": 2048, "dim": 128},
]

print("\n--- Old OpenCL Kernel (baseline) ---")
try:
    from aule_opencl import OpenCLAttention

    with OpenCLAttention() as attn:
        for cfg in configs:
            Q = np.random.randn(cfg['batch'], cfg['heads'], cfg['seq'], cfg['dim']).astype(np.float32)
            K = np.random.randn(cfg['batch'], cfg['heads'], cfg['seq'], cfg['dim']).astype(np.float32)
            V = np.random.randn(cfg['batch'], cfg['heads'], cfg['seq'], cfg['dim']).astype(np.float32)

            # Warmup
            _ = attn.forward(Q, K, V, causal=True)

            # Benchmark
            times = []
            for _ in range(5):
                start = time.perf_counter()
                output = attn.forward(Q, K, V, causal=True)
                elapsed = time.perf_counter() - start
                times.append(elapsed)

            avg_ms = np.mean(times) * 1000
            flops = 4 * cfg['batch'] * cfg['heads'] * cfg['seq']**2 * cfg['dim']
            tflops = flops / (np.mean(times) * 1e12)

            print(f"{cfg['name']}: {avg_ms:.2f}ms, {tflops:.4f} TFLOPS")
except Exception as e:
    print(f"Failed: {e}")

print("\n--- New Hybrid Backend (optimized) ---")
try:
    from aule_hybrid import HybridAttention

    with HybridAttention(backend='opencl', verbose=False) as attn:
        for cfg in configs:
            Q = np.random.randn(cfg['batch'], cfg['heads'], cfg['seq'], cfg['dim']).astype(np.float32)
            K = np.random.randn(cfg['batch'], cfg['heads'], cfg['seq'], cfg['dim']).astype(np.float32)
            V = np.random.randn(cfg['batch'], cfg['heads'], cfg['seq'], cfg['dim']).astype(np.float32)

            # Warmup
            _ = attn.forward(Q, K, V, causal=True)

            # Benchmark
            times = []
            for _ in range(5):
                start = time.perf_counter()
                output = attn.forward(Q, K, V, causal=True)
                elapsed = time.perf_counter() - start
                times.append(elapsed)

            avg_ms = np.mean(times) * 1000
            flops = 4 * cfg['batch'] * cfg['heads'] * cfg['seq']**2 * cfg['dim']
            tflops = flops / (np.mean(times) * 1e12)

            print(f"{cfg['name']}: {avg_ms:.2f}ms, {tflops:.4f} TFLOPS")
except Exception as e:
    print(f"Failed: {e}")

print("\n--- Correctness Check ---")
try:
    from aule_opencl import OpenCLAttention
    from aule_hybrid import HybridAttention

    Q = np.random.randn(1, 8, 64, 64).astype(np.float32)
    K = np.random.randn(1, 8, 64, 64).astype(np.float32)
    V = np.random.randn(1, 8, 64, 64).astype(np.float32)

    with OpenCLAttention() as old_attn:
        old_output = old_attn.forward(Q, K, V, causal=True)

    with HybridAttention(backend='opencl', verbose=False) as new_attn:
        new_output = new_attn.forward(Q, K, V, causal=True)

    max_diff = np.abs(old_output - new_output).max()
    print(f"Max difference between old and new kernels: {max_diff:.6f}")

    if max_diff < 1e-4:
        print("✓ Results match!")
    else:
        print("✗ Results differ (may be precision issue)")
except Exception as e:
    print(f"Comparison failed: {e}")

PYEOF
EOF

echo ""
echo "[3/3] Test LLaMA inference with hybrid backend..."
ssh "$REMOTE" << 'EOF'
cd ~/aule-attention/python

export CUDA_VISIBLE_DEVICES=""
export HIP_VISIBLE_DEVICES=""

echo ""
echo "=== LLaMA Inference Test ==="
python3 << 'PYEOF'
import numpy as np
import time
import sys

# Check if transformers is available
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("transformers not installed, skipping LLaMA test")
    sys.exit(0)

# Force CPU for PyTorch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Import hybrid backend
from aule_hybrid import HybridAttention

print("Testing with TinyLlama model...")
print("(Using hybrid OpenCL backend for attention)")

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()

    print(f"Model loaded: {model.config.num_hidden_layers} layers")

    # Patch attention with hybrid backend
    from transformers.models.llama.modeling_llama import LlamaAttention

    class HybridLlamaAttention(torch.nn.Module):
        def __init__(self, original_attention, config):
            super().__init__()
            self.config = config
            self.hidden_size = config.hidden_size
            self.num_heads = config.num_attention_heads
            self.head_dim = self.hidden_size // self.num_heads
            self.num_key_value_heads = getattr(config, 'num_key_value_heads', self.num_heads)
            self.num_key_value_groups = self.num_heads // self.num_key_value_heads

            self.q_proj = original_attention.q_proj
            self.k_proj = original_attention.k_proj
            self.v_proj = original_attention.v_proj
            self.o_proj = original_attention.o_proj

            # Use hybrid backend
            self.attn = HybridAttention(backend='opencl', verbose=False)

        def _repeat_kv(self, hidden_states, n_rep):
            if n_rep == 1:
                return hidden_states
            batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
            hidden_states = hidden_states[:, :, None, :, :].expand(
                batch, num_kv_heads, n_rep, seq_len, head_dim
            )
            return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)

        def forward(self, hidden_states, attention_mask=None, position_ids=None,
                    past_key_value=None, output_attentions=False, use_cache=False,
                    cache_position=None, **kwargs):
            bsz, q_len, _ = hidden_states.size()

            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

            if past_key_value is not None:
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)

            past_key_value = (key_states, value_states) if use_cache else None

            key_states = self._repeat_kv(key_states, self.num_key_value_groups)
            value_states = self._repeat_kv(value_states, self.num_key_value_groups)

            # Use hybrid attention
            Q_np = query_states.detach().cpu().numpy().astype(np.float32)
            K_np = key_states.detach().cpu().numpy().astype(np.float32)
            V_np = value_states.detach().cpu().numpy().astype(np.float32)

            output_np = self.attn.forward(Q_np, K_np, V_np, causal=True)

            attn_output = torch.from_numpy(output_np).to(hidden_states.device, dtype=hidden_states.dtype)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
            attn_output = self.o_proj(attn_output)

            return attn_output, past_key_value

    # Patch all attention layers
    patched = 0
    for name, module in model.named_modules():
        if isinstance(module, LlamaAttention):
            parent_name = '.'.join(name.split('.')[:-1])
            attr_name = name.split('.')[-1]
            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model
            setattr(parent, attr_name, HybridLlamaAttention(module, model.config))
            patched += 1

    print(f"Patched {patched} attention layers with hybrid backend")

    # Test inference
    prompts = [
        "The capital of France is",
        "def fibonacci(n):",
    ]

    print("\n--- Inference Test ---")
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")

        inputs = tokenizer(prompt, return_tensors="pt")

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        elapsed = time.perf_counter() - start

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Output: {generated}")
        print(f"Time: {elapsed*1000:.1f}ms")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

PYEOF
EOF

echo ""
echo "============================================================"
echo "TEST COMPLETE"
echo "============================================================"
