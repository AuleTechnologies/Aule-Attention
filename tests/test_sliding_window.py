#!/usr/bin/env python3
"""
Sliding Window Attention - Comprehensive Tests and Benchmarks

Tests:
1. Correctness validation against PyTorch reference
2. Performance benchmarks at various window sizes
3. Memory efficiency comparison
4. Real workflow: Long document processing simulation
"""

import numpy as np
import time
import sys

try:
    from aule import Aule
    HAS_AULE = True
except ImportError:
    HAS_AULE = False
    print("WARNING: aule not available")

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: torch not available for reference comparison")


def pytorch_sliding_window_attention(q, k, v, window_size, causal=True):
    """Reference implementation using PyTorch."""
    B, H, N, D = q.shape
    scale = 1.0 / np.sqrt(D)

    # Compute full attention scores
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, N, N]

    # Create sliding window mask
    mask = torch.ones(N, N, dtype=torch.bool, device=q.device)

    for i in range(N):
        if causal:
            # Causal: can attend to [max(0, i-window_size+1), i]
            start = max(0, i - window_size + 1)
            mask[i, :start] = False
            mask[i, i+1:] = False  # causal: no future
        else:
            # Bidirectional: can attend to [i-window_size//2, i+window_size//2]
            half = window_size // 2
            start = max(0, i - half)
            end = min(N, i + half + 1)
            mask[i, :start] = False
            mask[i, end:] = False

    # Apply mask
    scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))

    # Softmax and output
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    return out


def test_correctness():
    """Test sliding window correctness against PyTorch reference."""
    print("\n" + "="*70)
    print("CORRECTNESS TEST: Sliding Window vs PyTorch Reference")
    print("="*70)

    if not HAS_AULE or not HAS_TORCH:
        print("Skipping: requires both aule and torch")
        return

    aule = Aule()
    print(f"Device: {aule.device_name}")

    # Test configurations
    configs = [
        {"B": 1, "H": 1, "N": 64, "D": 32, "window": 16},
        {"B": 1, "H": 2, "N": 128, "D": 64, "window": 32},
        {"B": 2, "H": 4, "N": 256, "D": 64, "window": 64},
    ]

    print(f"\n{'Config':<30} | {'Causal':<8} | {'Max Error':<12} | {'Mean Error':<12} | {'Status'}")
    print("-" * 85)

    all_passed = True

    for cfg in configs:
        B, H, N, D, W = cfg["B"], cfg["H"], cfg["N"], cfg["D"], cfg["window"]

        np.random.seed(42)
        q_np = np.random.randn(B, H, N, D).astype(np.float32) * 0.1
        k_np = np.random.randn(B, H, N, D).astype(np.float32) * 0.1
        v_np = np.random.randn(B, H, N, D).astype(np.float32) * 0.1

        q_torch = torch.from_numpy(q_np)
        k_torch = torch.from_numpy(k_np)
        v_torch = torch.from_numpy(v_np)

        for causal in [True, False]:
            # PyTorch reference
            ref = pytorch_sliding_window_attention(q_torch, k_torch, v_torch, W, causal=causal)
            ref_np = ref.numpy()

            # Aule sliding window
            out_aule = aule.attention(q_np, k_np, v_np, causal=causal, window_size=W)

            # Compare
            max_err = np.abs(out_aule - ref_np).max()
            mean_err = np.abs(out_aule - ref_np).mean()

            # Tolerance (fp32 accumulation differences)
            passed = max_err < 1e-2
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_passed = False

            config_str = f"B{B}H{H}N{N}D{D}W{W}"
            causal_str = "Yes" if causal else "No"
            print(f"{config_str:<30} | {causal_str:<8} | {max_err:<12.6f} | {mean_err:<12.6f} | {status}")

    aule.close()

    print("\n" + ("ALL TESTS PASSED!" if all_passed else "SOME TESTS FAILED"))
    return all_passed


def benchmark_window_sizes():
    """Benchmark different window sizes."""
    print("\n" + "="*70)
    print("PERFORMANCE BENCHMARK: Window Size vs Speed")
    print("="*70)

    if not HAS_AULE:
        print("Skipping: aule not available")
        return

    aule = Aule()
    print(f"Device: {aule.device_name}")

    # Test with longer sequences
    configs = [
        {"B": 1, "H": 8, "N": 512, "D": 64},
        {"B": 1, "H": 8, "N": 1024, "D": 64},
        {"B": 1, "H": 8, "N": 2048, "D": 64},
    ]

    window_sizes = [-1, 32, 64, 128, 256, 512]  # -1 = full attention
    n_warmup = 2
    n_iters = 10

    for cfg in configs:
        B, H, N, D = cfg["B"], cfg["H"], cfg["N"], cfg["D"]

        print(f"\n{'='*60}")
        print(f"Config: B={B}, H={H}, N={N}, D={D}")
        print(f"{'='*60}")

        np.random.seed(42)
        q = np.random.randn(B, H, N, D).astype(np.float32)
        k = np.random.randn(B, H, N, D).astype(np.float32)
        v = np.random.randn(B, H, N, D).astype(np.float32)

        # Warmup
        for _ in range(n_warmup):
            _ = aule.attention(q, k, v, causal=True)

        print(f"\n{'Window Size':<15} | {'Time (ms)':<12} | {'Speedup':<10} | {'Tokens/sec':<15}")
        print("-" * 60)

        base_time = None

        for ws in window_sizes:
            if ws > N and ws != -1:
                continue

            start = time.perf_counter()
            for _ in range(n_iters):
                out = aule.attention(q, k, v, causal=True, window_size=ws)
            elapsed = (time.perf_counter() - start) / n_iters * 1000

            if ws == -1:
                base_time = elapsed
                ws_str = "Full"
            else:
                ws_str = str(ws)

            speedup = base_time / elapsed if base_time else 1.0
            tokens_per_sec = (B * H * N) / (elapsed / 1000)

            print(f"{ws_str:<15} | {elapsed:<12.2f} | {speedup:<10.2f}x | {tokens_per_sec:<15.0f}")

    aule.close()


def benchmark_memory_efficiency():
    """Compare memory patterns between full and sliding window attention."""
    print("\n" + "="*70)
    print("MEMORY EFFICIENCY: Full vs Sliding Window")
    print("="*70)

    if not HAS_AULE:
        print("Skipping: aule not available")
        return

    aule = Aule()
    print(f"Device: {aule.device_name}")

    # Theoretical memory analysis
    print("\nTheoretical Memory Analysis:")
    print("-" * 60)

    seq_lengths = [512, 1024, 2048, 4096]
    window_size = 256
    head_dim = 64
    num_heads = 8
    batch_size = 1

    print(f"\n{'Seq Len':<10} | {'Full Attn (MB)':<15} | {'Window={window_size} (MB)':<18} | {'Savings':<10}")
    print("-" * 60)

    for N in seq_lengths:
        # Full attention: stores N x N attention matrix per head
        full_attn_mem = batch_size * num_heads * N * N * 4 / (1024 * 1024)  # fp32

        # Sliding window: only stores window_size scores per row
        window_attn_mem = batch_size * num_heads * N * min(window_size, N) * 4 / (1024 * 1024)

        savings = (1 - window_attn_mem / full_attn_mem) * 100 if full_attn_mem > 0 else 0

        print(f"{N:<10} | {full_attn_mem:<15.2f} | {window_attn_mem:<18.2f} | {savings:<10.1f}%")

    # Practical test - see how large we can go
    print("\n\nPractical Test: Maximum Sequence Length")
    print("-" * 60)

    test_lens = [512, 1024, 2048]

    for N in test_lens:
        try:
            q = np.random.randn(1, 4, N, 64).astype(np.float32)
            k = np.random.randn(1, 4, N, 64).astype(np.float32)
            v = np.random.randn(1, 4, N, 64).astype(np.float32)

            # Full attention
            start = time.perf_counter()
            out_full = aule.attention(q, k, v, causal=True, window_size=-1)
            full_time = (time.perf_counter() - start) * 1000

            # Window attention
            start = time.perf_counter()
            out_window = aule.attention(q, k, v, causal=True, window_size=256)
            window_time = (time.perf_counter() - start) * 1000

            print(f"N={N}: Full={full_time:.1f}ms, Window(256)={window_time:.1f}ms, Speedup={full_time/window_time:.2f}x")

        except Exception as e:
            print(f"N={N}: Failed - {e}")

    aule.close()


def real_workflow_long_document():
    """Simulate processing a long document with sliding window attention."""
    print("\n" + "="*70)
    print("REAL WORKFLOW: Long Document Processing")
    print("="*70)

    if not HAS_AULE:
        print("Skipping: aule not available")
        return

    aule = Aule()
    print(f"Device: {aule.device_name}")

    # Simulate a document with ~2000 tokens
    # Using sliding window mimics models like Longformer, BigBird, etc.

    doc_length = 1024  # tokens
    num_heads = 8
    head_dim = 64
    batch_size = 1

    print(f"\nDocument length: {doc_length} tokens")
    print(f"Model config: {num_heads} heads, {head_dim} head dim")

    # Create "document" embeddings (random for simulation)
    np.random.seed(42)
    q = np.random.randn(batch_size, num_heads, doc_length, head_dim).astype(np.float32) * 0.1
    k = np.random.randn(batch_size, num_heads, doc_length, head_dim).astype(np.float32) * 0.1
    v = np.random.randn(batch_size, num_heads, doc_length, head_dim).astype(np.float32) * 0.1

    print("\n" + "-"*60)
    print("Comparing attention strategies:")
    print("-"*60)

    strategies = [
        ("Full O(N^2)", -1),
        ("Window 64 (local)", 64),
        ("Window 128", 128),
        ("Window 256", 256),
        ("Window 512", 512),
    ]

    n_iters = 5
    results = {}

    # Warmup
    _ = aule.attention(q, k, v, causal=True)

    # Get reference output
    ref_output = aule.attention(q, k, v, causal=True, window_size=-1)

    print(f"\n{'Strategy':<25} | {'Time (ms)':<12} | {'Throughput':<15} | {'Quality':<12}")
    print("-" * 70)

    for name, window in strategies:
        start = time.perf_counter()
        for _ in range(n_iters):
            out = aule.attention(q, k, v, causal=True, window_size=window)
        elapsed = (time.perf_counter() - start) / n_iters * 1000

        # Measure quality (cosine similarity to full attention)
        if window == -1:
            quality = 1.0
        else:
            # Compute cosine similarity
            out_flat = out.flatten()
            ref_flat = ref_output.flatten()
            cos_sim = np.dot(out_flat, ref_flat) / (np.linalg.norm(out_flat) * np.linalg.norm(ref_flat) + 1e-8)
            quality = cos_sim

        throughput = doc_length / (elapsed / 1000)  # tokens/sec

        results[name] = {
            "time_ms": elapsed,
            "throughput": throughput,
            "quality": quality
        }

        print(f"{name:<25} | {elapsed:<12.2f} | {throughput:<15.0f} | {quality:<12.4f}")

    # Analysis
    print("\n" + "-"*60)
    print("Analysis:")
    print("-"*60)

    full_time = results["Full O(N^2)"]["time_ms"]

    for name, data in results.items():
        if name != "Full O(N^2)":
            speedup = full_time / data["time_ms"]
            quality_loss = (1 - data["quality"]) * 100
            print(f"{name}: {speedup:.2f}x faster, {quality_loss:.2f}% quality loss")

    # Recommendation
    print("\n" + "-"*60)
    print("Recommendations:")
    print("-"*60)

    # Find best speed/quality tradeoff
    best_tradeoff = None
    best_score = 0

    for name, data in results.items():
        if name == "Full O(N^2)":
            continue
        speedup = full_time / data["time_ms"]
        quality = data["quality"]
        # Score: balance speed and quality (adjust weights as needed)
        score = speedup * (quality ** 2)  # Penalize quality loss
        if score > best_score:
            best_score = score
            best_tradeoff = name

    print(f"Best speed/quality tradeoff: {best_tradeoff}")
    print(f"  - Use Window 64-128 for maximum speed with local context")
    print(f"  - Use Window 256-512 for balanced speed/quality")
    print(f"  - Use Full attention when global context is critical")

    aule.close()


def test_causal_vs_bidirectional():
    """Test both causal and bidirectional sliding window modes."""
    print("\n" + "="*70)
    print("CAUSAL vs BIDIRECTIONAL Sliding Window")
    print("="*70)

    if not HAS_AULE:
        print("Skipping: aule not available")
        return

    aule = Aule()

    B, H, N, D = 1, 4, 256, 64
    window_size = 64

    np.random.seed(42)
    q = np.random.randn(B, H, N, D).astype(np.float32) * 0.1
    k = np.random.randn(B, H, N, D).astype(np.float32) * 0.1
    v = np.random.randn(B, H, N, D).astype(np.float32) * 0.1

    # Causal (autoregressive - for generation)
    out_causal = aule.attention(q, k, v, causal=True, window_size=window_size)

    # Bidirectional (encoder - for understanding)
    out_bidir = aule.attention(q, k, v, causal=False, window_size=window_size)

    # Compare
    diff = np.abs(out_causal - out_bidir).mean()

    print(f"\nConfig: B={B}, H={H}, N={N}, D={D}, Window={window_size}")
    print(f"\nCausal output mean: {out_causal.mean():.6f}")
    print(f"Bidirectional output mean: {out_bidir.mean():.6f}")
    print(f"Mean absolute difference: {diff:.6f}")

    # Verify they're actually different (bidirectional sees more context)
    print(f"\nExpected: Outputs should differ (bidirectional sees future tokens)")
    print(f"Result: {'PASS' if diff > 0.01 else 'CHECK - outputs very similar'}")

    aule.close()


def main():
    print("="*70)
    print("SLIDING WINDOW ATTENTION - Comprehensive Test Suite")
    print("="*70)

    if not HAS_AULE:
        print("\nERROR: aule not available. Build with 'zig build' first.")
        sys.exit(1)

    # Run all tests
    test_correctness()
    benchmark_window_sizes()
    benchmark_memory_efficiency()
    real_workflow_long_document()
    test_causal_vs_bidirectional()

    print("\n" + "="*70)
    print("ALL TESTS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
