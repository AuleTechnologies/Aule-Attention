#!/bin/bash
# Deploy aule-attention to MI300X droplet and test
#
# Usage: ./deploy_mi300x.sh <droplet-ip>
# Example: ./deploy_mi300x.sh 192.168.1.100

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <droplet-ip>"
    echo "Example: $0 192.168.1.100"
    exit 1
fi

DROPLET_IP="$1"
DROPLET_USER="${2:-root}"
REMOTE="$DROPLET_USER@$DROPLET_IP"

echo "=== Deploying aule-attention to $REMOTE ==="

# Create remote directory
echo "[1/4] Creating remote directory..."
ssh "$REMOTE" "mkdir -p ~/aule-attention/python"

# Copy Python files
echo "[2/4] Copying Python files..."
cd /home/yab/Sndr
scp python/aule_hip.py python/aule_hip_mfma.py python/aule_opencl.py python/aule_unified.py python/aule_rocm.py python/aule_autograd.py python/benchmark.py "$REMOTE:~/aule-attention/python/"

# Install dependencies and test
echo "[3/4] Installing dependencies on remote..."
ssh "$REMOTE" << 'EOF'
cd ~/aule-attention

# Install pip if not available
if ! command -v pip3 &> /dev/null; then
    echo "Installing pip..."
    apt-get update -qq && apt-get install -y -qq python3-pip python3-venv
fi

# Check ROCm first
echo ""
echo "=== ROCm Status ==="
if command -v rocm-smi &> /dev/null; then
    rocm-smi --showproductname 2>/dev/null || echo "rocm-smi available"
    rocm-smi 2>/dev/null | head -20
elif command -v rocminfo &> /dev/null; then
    echo "rocminfo available:"
    rocminfo 2>/dev/null | grep -i "marketing name\|name:" | head -5
else
    echo "ROCm not installed. Checking for AMD GPU..."
    lspci | grep -i amd
    echo ""
    echo "Checking /dev/kfd (ROCm kernel driver)..."
    ls -la /dev/kfd 2>/dev/null || echo "/dev/kfd not found - ROCm kernel driver not loaded"
    echo ""
    echo "Checking /dev/dri..."
    ls -la /dev/dri/ 2>/dev/null
fi

# Install dependencies
echo ""
echo "Installing Python packages..."
pip3 install --break-system-packages numpy pyopencl 2>/dev/null || pip3 install numpy pyopencl

# Check OpenCL
echo ""
echo "Checking OpenCL devices..."
python3 -c "
try:
    import pyopencl as cl
    for p in cl.get_platforms():
        for d in p.get_devices():
            print(f'  Found: {d.name}')
except Exception as e:
    print(f'  OpenCL check failed: {e}')
" 2>/dev/null || echo "  pyopencl not working"

# Check if ROCm Python bindings are available
echo ""
echo "Checking for ROCm HIP Python support..."
if [ -d "/opt/rocm" ]; then
    echo "ROCm found at /opt/rocm"
    export PATH=$PATH:/opt/rocm/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib
    # Try to find HIP Python in ROCm
    if [ -f "/opt/rocm/lib/python*/site-packages/hip/__init__.py" ]; then
        echo "HIP Python found in ROCm"
    else
        echo "HIP Python not in ROCm, trying PyPI package..."
        pip3 install --break-system-packages hip-python 2>/dev/null
    fi
else
    echo "ROCm not found at /opt/rocm"
fi
EOF

# Run test
echo "[4/4] Running attention test..."
ssh "$REMOTE" << 'EOF'
cd ~/aule-attention
python3 << 'PYTHON'
import sys
sys.path.insert(0, 'python')

print("=== aule-attention MI300X Test ===\n")

# Check available backends
import aule_unified as aule
backends = aule.get_available_backends()
print(f"Available backends: {backends}\n")

# Test attention with various sizes
import numpy as np
import time

# Test configurations: (batch, heads, seq_len, head_dim)
configs = [
    (1, 8, 64, 64),      # Small
    (1, 8, 256, 64),     # Medium
    (1, 8, 512, 64),     # Large
    (1, 8, 1024, 64),    # XL - where GPU should shine
    (1, 8, 2048, 64),    # XXL - longer sequences
]

print("=" * 80)
print("BENCHMARK 1: Standard Attention (non-causal)")
print("=" * 80)
print(f"{'Config':<25} {'CPU':<15} {'OpenCL':<15} {'Speedup':<15}")
print("-" * 80)

for batch, heads, seq_len, head_dim in configs:
    Q = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32)
    K = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32)
    V = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32)

    results = {}
    for backend in backends:
        try:
            with aule.Attention(backend=backend) as attn:
                # Warmup
                _ = attn.forward(Q, K, V)

                # Benchmark
                n_iter = 5 if seq_len >= 512 else 10
                start = time.perf_counter()
                for _ in range(n_iter):
                    output = attn.forward(Q, K, V)
                elapsed = (time.perf_counter() - start) / n_iter * 1000
                results[backend] = elapsed
        except Exception as e:
            results[backend] = None

    config_str = f"({batch},{heads},{seq_len},{head_dim})"
    cpu_time = results.get('cpu')
    opencl_time = results.get('opencl')

    cpu_str = f"{cpu_time:.2f}" if cpu_time else "N/A"
    opencl_str = f"{opencl_time:.2f}" if opencl_time else "N/A"

    if cpu_time and opencl_time:
        speedup = cpu_time / opencl_time
        speedup_str = f"{speedup:.2f}x" if speedup > 1 else f"{1/speedup:.2f}x slower"
    else:
        speedup_str = "N/A"

    print(f"{config_str:<25} {cpu_str:<15} {opencl_str:<15} {speedup_str:<15}")

print("\n")
print("=" * 80)
print("BENCHMARK 2: Causal Attention (autoregressive)")
print("=" * 80)
print(f"{'Config':<25} {'CPU':<15} {'OpenCL':<15} {'Speedup':<15}")
print("-" * 80)

for batch, heads, seq_len, head_dim in configs:
    Q = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32)
    K = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32)
    V = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32)

    results = {}
    for backend in backends:
        try:
            with aule.Attention(backend=backend) as attn:
                # Warmup
                _ = attn.forward(Q, K, V, causal=True)

                # Benchmark
                n_iter = 5 if seq_len >= 512 else 10
                start = time.perf_counter()
                for _ in range(n_iter):
                    output = attn.forward(Q, K, V, causal=True)
                elapsed = (time.perf_counter() - start) / n_iter * 1000
                results[backend] = elapsed
        except Exception as e:
            results[backend] = None

    config_str = f"({batch},{heads},{seq_len},{head_dim})"
    cpu_time = results.get('cpu')
    opencl_time = results.get('opencl')

    cpu_str = f"{cpu_time:.2f}" if cpu_time else "N/A"
    opencl_str = f"{opencl_time:.2f}" if opencl_time else "N/A"

    if cpu_time and opencl_time:
        speedup = cpu_time / opencl_time
        speedup_str = f"{speedup:.2f}x" if speedup > 1 else f"{1/speedup:.2f}x slower"
    else:
        speedup_str = "N/A"

    print(f"{config_str:<25} {cpu_str:<15} {opencl_str:<15} {speedup_str:<15}")

print("\n")
print("=" * 80)
print("BENCHMARK 3: Batch Size Scaling (seq_len=512, causal)")
print("=" * 80)
print(f"{'Batch':<10} {'CPU':<15} {'OpenCL':<15} {'Speedup':<15}")
print("-" * 80)

for batch in [1, 2, 4, 8]:
    Q = np.random.randn(batch, 8, 512, 64).astype(np.float32)
    K = np.random.randn(batch, 8, 512, 64).astype(np.float32)
    V = np.random.randn(batch, 8, 512, 64).astype(np.float32)

    results = {}
    for backend in backends:
        try:
            with aule.Attention(backend=backend) as attn:
                # Warmup
                _ = attn.forward(Q, K, V, causal=True)

                # Benchmark
                n_iter = 5
                start = time.perf_counter()
                for _ in range(n_iter):
                    _ = attn.forward(Q, K, V, causal=True)
                elapsed = (time.perf_counter() - start) / n_iter * 1000
                results[backend] = elapsed
        except Exception as e:
            results[backend] = None

    cpu_time = results.get('cpu')
    opencl_time = results.get('opencl')

    cpu_str = f"{cpu_time:.2f}" if cpu_time else "N/A"
    opencl_str = f"{opencl_time:.2f}" if opencl_time else "N/A"

    if cpu_time and opencl_time:
        speedup = cpu_time / opencl_time
        speedup_str = f"{speedup:.2f}x" if speedup > 1 else f"{1/speedup:.2f}x slower"
    else:
        speedup_str = "N/A"

    print(f"{batch:<10} {cpu_str:<15} {opencl_str:<15} {speedup_str:<15}")

print("\n")
print("=" * 80)
print("BENCHMARK 4: FP16 vs FP32 (seq_len=1024, causal)")
print("=" * 80)

if 'opencl' in backends:
    with aule.Attention(backend='opencl') as attn:
        if attn.fp16_supported:
            print(f"FP16 supported: Yes")
            print(f"{'Dtype':<12} {'Time (ms)':<15} {'Speedup':<15}")
            print("-" * 50)

            Q = np.random.randn(1, 8, 1024, 64).astype(np.float32)
            K = np.random.randn(1, 8, 1024, 64).astype(np.float32)
            V = np.random.randn(1, 8, 1024, 64).astype(np.float32)

            # FP32 benchmark
            _ = attn.forward(Q, K, V, causal=True, dtype='float32')
            start = time.perf_counter()
            for _ in range(5):
                _ = attn.forward(Q, K, V, causal=True, dtype='float32')
            fp32_time = (time.perf_counter() - start) / 5 * 1000

            # FP16 benchmark
            _ = attn.forward(Q, K, V, causal=True, dtype='float16')
            start = time.perf_counter()
            for _ in range(5):
                _ = attn.forward(Q, K, V, causal=True, dtype='float16')
            fp16_time = (time.perf_counter() - start) / 5 * 1000

            print(f"{'float32':<12} {fp32_time:<15.2f} {'baseline':<15}")
            print(f"{'float16':<12} {fp16_time:<15.2f} {fp32_time/fp16_time:.2f}x")

            # Correctness check
            out_fp32 = attn.forward(Q, K, V, causal=True, dtype='float32')
            out_fp16 = attn.forward(Q, K, V, causal=True, dtype='float16')
            max_diff = np.abs(out_fp32.astype(np.float32) - out_fp16.astype(np.float32)).max()
            print(f"\nFP16 vs FP32 max difference: {max_diff:.6f}")
            print("PASS" if max_diff < 0.01 else "WARN - larger difference (expected for FP16)")
        else:
            print("FP16 not supported on this device (cl_khr_fp16 extension missing)")
else:
    print("OpenCL not available")

print("\n")
print("=" * 80)
print("BENCHMARK 5: FlashAttention-2 vs Simple Kernel")
print("=" * 80)

if 'opencl' in backends:
    from aule_opencl import OpenCLAttention

    print(f"{'Seq Len':<12} {'Flash (ms)':<15} {'Simple (ms)':<15} {'Flash Speedup':<15}")
    print("-" * 60)

    for seq_len in [256, 512, 1024, 2048, 4096]:
        Q = np.random.randn(1, 8, seq_len, 64).astype(np.float32)
        K = np.random.randn(1, 8, seq_len, 64).astype(np.float32)
        V = np.random.randn(1, 8, seq_len, 64).astype(np.float32)

        # FlashAttention kernel
        try:
            with OpenCLAttention(use_flash=True) as attn_flash:
                _ = attn_flash.forward(Q, K, V, causal=True)
                n_iter = 3 if seq_len >= 2048 else 5
                start = time.perf_counter()
                for _ in range(n_iter):
                    flash_out = attn_flash.forward(Q, K, V, causal=True)
                flash_time = (time.perf_counter() - start) / n_iter * 1000
        except Exception as e:
            flash_time = None
            flash_out = None

        # Simple kernel
        try:
            with OpenCLAttention(use_flash=False) as attn_simple:
                _ = attn_simple.forward(Q, K, V, causal=True)
                start = time.perf_counter()
                for _ in range(n_iter):
                    simple_out = attn_simple.forward(Q, K, V, causal=True)
                simple_time = (time.perf_counter() - start) / n_iter * 1000
        except Exception as e:
            simple_time = None
            simple_out = None

        flash_str = f"{flash_time:.2f}" if flash_time else "N/A"
        simple_str = f"{simple_time:.2f}" if simple_time else "N/A"

        if flash_time and simple_time:
            speedup = simple_time / flash_time
            speedup_str = f"{speedup:.2f}x" if speedup > 1 else f"{1/speedup:.2f}x slower"
        else:
            speedup_str = "N/A"

        print(f"{seq_len:<12} {flash_str:<15} {simple_str:<15} {speedup_str:<15}")

        # Verify correctness
        if flash_out is not None and simple_out is not None:
            max_diff = np.abs(flash_out - simple_out).max()
            if max_diff > 1e-3:
                print(f"  WARNING: Flash vs Simple diff = {max_diff:.6f}")
else:
    print("OpenCL not available")

print("\n")
print("=" * 80)
print("BENCHMARK 6: Long Sequences (8K+ tokens) - FlashAttention-2")
print("=" * 80)

if 'opencl' in backends:
    from aule_opencl import OpenCLAttention

    print(f"{'Seq Len':<12} {'Batch':<8} {'Time (ms)':<15} {'Tokens/sec':<20} {'Memory':<15}")
    print("-" * 75)

    # Test long sequences with FlashAttention-2 (O(N) memory)
    long_configs = [
        (4096, 1, 8, 64),    # 4K - baseline
        (8192, 1, 8, 64),    # 8K - LLaMA/Mistral context
        (16384, 1, 4, 64),   # 16K - long context (fewer heads to fit)
        (32768, 1, 2, 64),   # 32K - very long context (2 heads)
    ]

    for seq_len, batch, heads, head_dim in long_configs:
        try:
            # Estimate memory: O(batch * heads * seq_len * head_dim * 4 bytes) for Q,K,V,O
            mem_bytes = batch * heads * seq_len * head_dim * 4 * 4  # 4 tensors
            mem_mb = mem_bytes / (1024 * 1024)

            Q = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32)
            K = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32)
            V = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32)

            with OpenCLAttention(use_flash=True) as attn:
                # Warmup
                _ = attn.forward(Q, K, V, causal=True)

                # Benchmark
                n_iter = 3 if seq_len >= 8192 else 5
                start = time.perf_counter()
                for _ in range(n_iter):
                    _ = attn.forward(Q, K, V, causal=True)
                elapsed = (time.perf_counter() - start) / n_iter

                time_ms = elapsed * 1000
                tokens_per_sec = (batch * seq_len) / elapsed

                print(f"{seq_len:<12} {batch:<8} {time_ms:<15.2f} {tokens_per_sec:<20,.0f} {mem_mb:<15.1f}MB")
        except Exception as e:
            print(f"{seq_len:<12} {batch:<8} {'ERROR':<15} {str(e)[:30]}")

    # Test FP16 for long sequences (2x memory efficiency)
    print("\n--- FP16 Long Sequences (2x memory bandwidth) ---")
    print(f"{'Seq Len':<12} {'Batch':<8} {'FP32 (ms)':<15} {'FP16 (ms)':<15} {'Speedup':<15}")
    print("-" * 70)

    for seq_len in [4096, 8192]:
        try:
            Q = np.random.randn(1, 8, seq_len, 64).astype(np.float32)
            K = np.random.randn(1, 8, seq_len, 64).astype(np.float32)
            V = np.random.randn(1, 8, seq_len, 64).astype(np.float32)

            with OpenCLAttention(use_flash=True) as attn:
                if attn.fp16_supported:
                    # FP32
                    _ = attn.forward(Q, K, V, causal=True, dtype='float32')
                    start = time.perf_counter()
                    for _ in range(3):
                        _ = attn.forward(Q, K, V, causal=True, dtype='float32')
                    fp32_time = (time.perf_counter() - start) / 3 * 1000

                    # FP16
                    _ = attn.forward(Q, K, V, causal=True, dtype='float16')
                    start = time.perf_counter()
                    for _ in range(3):
                        _ = attn.forward(Q, K, V, causal=True, dtype='float16')
                    fp16_time = (time.perf_counter() - start) / 3 * 1000

                    speedup = fp32_time / fp16_time
                    print(f"{seq_len:<12} {'1':<8} {fp32_time:<15.2f} {fp16_time:<15.2f} {speedup:.2f}x")
                else:
                    print(f"{seq_len:<12} {'1':<8} {'N/A':<15} {'FP16 not supported':<15}")
        except Exception as e:
            print(f"{seq_len:<12} {'1':<8} {'ERROR':<15} {str(e)[:30]}")
else:
    print("OpenCL not available")

print("\n")
print("=" * 80)
print("CORRECTNESS CHECK: Causal vs Non-Causal")
print("=" * 80)

# Verify causal masking is working correctly
Q = np.random.randn(1, 1, 4, 4).astype(np.float32)
K = np.random.randn(1, 1, 4, 4).astype(np.float32)
V = np.random.randn(1, 1, 4, 4).astype(np.float32)

with aule.Attention(backend='cpu') as attn:
    out_full = attn.forward(Q, K, V, causal=False)
    out_causal = attn.forward(Q, K, V, causal=True)

print("Non-causal output[0,0,0,:4]:", out_full[0,0,0,:4].round(3))
print("Causal output[0,0,0,:4]:    ", out_causal[0,0,0,:4].round(3))
print("(First row should differ - causal only attends to first token)")

if 'opencl' in backends:
    print("\nFlashAttention GPU vs CPU correctness:")
    with aule.Attention(backend='opencl') as attn_gpu:
        with aule.Attention(backend='cpu') as attn_cpu:
            Q = np.random.randn(1, 4, 128, 64).astype(np.float32)
            K = np.random.randn(1, 4, 128, 64).astype(np.float32)
            V = np.random.randn(1, 4, 128, 64).astype(np.float32)

            gpu_out = attn_gpu.forward(Q, K, V, causal=True)
            cpu_out = attn_cpu.forward(Q, K, V, causal=True)

            max_diff = np.abs(gpu_out - cpu_out).max()
            print(f"Max GPU vs CPU difference (causal): {max_diff:.6f}")
            print("PASS" if max_diff < 1e-3 else "FAIL - numerical difference too large")

print("\n=== Test Complete ===")
PYTHON
EOF

echo ""
echo "=== Deployment Complete ==="
