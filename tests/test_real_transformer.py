#!/usr/bin/env python3
"""
Test PagedAttention with real transformer-like workload
Simulates autoregressive generation with growing KV cache
"""

import sys
import os
import ctypes
import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

def load_library():
    lib_paths = [
        '../zig-out/lib/libaule.so',
        './zig-out/lib/libaule.so',
        '/home/yab/Sndr/zig-out/lib/libaule.so',
    ]
    for path in lib_paths:
        if os.path.exists(path):
            return ctypes.CDLL(path)
    raise FileNotFoundError("Could not find libaule.so")

def setup_library(lib):
    """Set up ctypes function signatures"""
    lib.aule_init.restype = ctypes.c_int32
    lib.aule_shutdown.restype = None
    lib.aule_get_error.restype = ctypes.c_char_p
    lib.aule_get_vendor.restype = ctypes.c_int32
    lib.aule_get_device_name.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_uint32]
    lib.aule_get_device_name.restype = ctypes.c_int32

    lib.aule_tensor_create.argtypes = [ctypes.c_uint32] * 4
    lib.aule_tensor_create.restype = ctypes.c_uint64
    lib.aule_tensor_destroy.argtypes = [ctypes.c_uint64]
    lib.aule_tensor_upload.argtypes = [ctypes.c_uint64, ctypes.POINTER(ctypes.c_float), ctypes.c_uint32]
    lib.aule_tensor_upload.restype = ctypes.c_int32
    lib.aule_tensor_download.argtypes = [ctypes.c_uint64, ctypes.POINTER(ctypes.c_float), ctypes.c_uint32]
    lib.aule_tensor_download.restype = ctypes.c_int32

    lib.aule_attention_forward_gpu.argtypes = [ctypes.c_uint64] * 4 + [ctypes.c_uint64] * 2 + [ctypes.c_int32, ctypes.c_int32]
    lib.aule_attention_forward_gpu.restype = ctypes.c_int32
    lib.aule_attention_forward_paged.argtypes = [ctypes.c_uint64] * 4 + [ctypes.c_uint64] * 2 + [ctypes.c_int32, ctypes.c_int32]
    lib.aule_attention_forward_paged.restype = ctypes.c_int32

def test_autoregressive_generation(lib):
    """
    Simulate autoregressive transformer generation
    Start with prompt, then generate tokens one by one
    KV cache grows with each step
    """
    print("\n" + "=" * 70)
    print("  Test 1: Autoregressive Generation (Growing KV Cache)")
    print("=" * 70)

    # Model config (GPT-2 small-like)
    num_heads = 12
    head_dim = 64
    num_layers = 1  # Test single layer for now

    # Generation config
    prompt_len = 64  # Input prompt length
    max_new_tokens = 64  # Generate 64 new tokens
    batch_size = 1

    print(f"\nConfig:")
    print(f"  Heads: {num_heads}")
    print(f"  Head dim: {head_dim}")
    print(f"  Prompt length: {prompt_len}")
    print(f"  Generating: {max_new_tokens} tokens")

    # Initialize
    ret = lib.aule_init()
    if ret != 0:
        error = lib.aule_get_error()
        print(f"✗ Init failed: {ctypes.string_at(error).decode()}")
        return False

    # Get device info
    vendor = lib.aule_get_vendor()
    vendor_names = {0: "Other", 1: "AMD", 2: "NVIDIA", 3: "Intel", 4: "Apple"}
    device_name = ctypes.create_string_buffer(256)
    lib.aule_get_device_name(device_name, 256)
    print(f"\n✓ Running on: {device_name.value.decode()} ({vendor_names.get(vendor, 'Unknown')})")

    # Phase 1: Prefill (process entire prompt at once)
    print(f"\n--- Phase 1: Prefill (seq_len={prompt_len}) ---")

    prefill_count = batch_size * num_heads * prompt_len * head_dim

    Q_prefill = lib.aule_tensor_create(batch_size, num_heads, prompt_len, head_dim)
    K_prefill = lib.aule_tensor_create(batch_size, num_heads, prompt_len, head_dim)
    V_prefill = lib.aule_tensor_create(batch_size, num_heads, prompt_len, head_dim)
    O_prefill = lib.aule_tensor_create(batch_size, num_heads, prompt_len, head_dim)

    # Create realistic input (small random values)
    np.random.seed(42)
    Q_data = np.random.randn(prefill_count).astype(np.float32) * 0.02
    K_data = np.random.randn(prefill_count).astype(np.float32) * 0.02
    V_data = np.random.randn(prefill_count).astype(np.float32) * 0.02

    lib.aule_tensor_upload(Q_prefill, Q_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), prefill_count)
    lib.aule_tensor_upload(K_prefill, K_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), prefill_count)
    lib.aule_tensor_upload(V_prefill, V_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), prefill_count)

    # Run prefill with PagedAttention
    start = time.time()
    ret = lib.aule_attention_forward_paged(
        Q_prefill, K_prefill, V_prefill, O_prefill,
        0, 0,  # no RoPE for simplicity
        1,     # causal masking
        -1     # no sliding window
    )
    prefill_time = (time.time() - start) * 1000

    if ret != 0:
        error = lib.aule_get_error()
        print(f"✗ Prefill failed: {ctypes.string_at(error).decode()}")
        return False

    # Verify output
    output_prefill = np.zeros(prefill_count, dtype=np.float32)
    lib.aule_tensor_download(O_prefill, output_prefill.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), prefill_count)

    non_zero = np.count_nonzero(output_prefill)
    print(f"✓ Prefill completed in {prefill_time:.2f}ms")
    print(f"  Output: {non_zero}/{prefill_count} non-zero")
    print(f"  Mean: {np.mean(np.abs(output_prefill)):.6f}")
    print(f"  Max: {np.max(np.abs(output_prefill)):.6f}")

    if non_zero == 0:
        print("✗ ERROR: Prefill produced all zeros!")
        return False

    lib.aule_tensor_destroy(Q_prefill)
    lib.aule_tensor_destroy(K_prefill)
    lib.aule_tensor_destroy(V_prefill)
    lib.aule_tensor_destroy(O_prefill)

    # Phase 2: Decode (generate tokens one by one with growing KV cache)
    print(f"\n--- Phase 2: Decode (generating {max_new_tokens} tokens) ---")

    decode_times = []

    for step in range(max_new_tokens):
        current_kv_len = prompt_len + step  # KV cache grows
        query_len = 1  # Only query for new token

        # For decode, Q is 1 token, K/V are full cache
        q_count = batch_size * num_heads * query_len * head_dim
        kv_count = batch_size * num_heads * current_kv_len * head_dim

        Q_decode = lib.aule_tensor_create(batch_size, num_heads, query_len, head_dim)
        K_decode = lib.aule_tensor_create(batch_size, num_heads, current_kv_len, head_dim)
        V_decode = lib.aule_tensor_create(batch_size, num_heads, current_kv_len, head_dim)
        O_decode = lib.aule_tensor_create(batch_size, num_heads, query_len, head_dim)

        # Create new query and extended KV cache
        Q_new = np.random.randn(q_count).astype(np.float32) * 0.02
        K_new = np.random.randn(kv_count).astype(np.float32) * 0.02
        V_new = np.random.randn(kv_count).astype(np.float32) * 0.02

        lib.aule_tensor_upload(Q_decode, Q_new.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), q_count)
        lib.aule_tensor_upload(K_decode, K_new.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), kv_count)
        lib.aule_tensor_upload(V_decode, V_new.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), kv_count)

        # Run decode step with PagedAttention
        start = time.time()
        ret = lib.aule_attention_forward_paged(
            Q_decode, K_decode, V_decode, O_decode,
            0, 0,  # no RoPE
            1,     # causal
            -1     # no window
        )
        step_time = (time.time() - start) * 1000
        decode_times.append(step_time)

        if ret != 0:
            error = lib.aule_get_error()
            print(f"✗ Decode step {step} failed: {ctypes.string_at(error).decode()}")
            return False

        # Verify output
        output_decode = np.zeros(q_count, dtype=np.float32)
        lib.aule_tensor_download(O_decode, output_decode.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), q_count)

        if step % 16 == 0 or step < 3:
            print(f"  Step {step:3d}: KV_len={current_kv_len:3d}, time={step_time:.2f}ms, output_mean={np.mean(np.abs(output_decode)):.6f}")

        if np.count_nonzero(output_decode) == 0:
            print(f"✗ ERROR: Decode step {step} produced all zeros!")
            return False

        lib.aule_tensor_destroy(Q_decode)
        lib.aule_tensor_destroy(K_decode)
        lib.aule_tensor_destroy(V_decode)
        lib.aule_tensor_destroy(O_decode)

    # Statistics
    print(f"\n✓ Decode completed!")
    print(f"  Total decode time: {sum(decode_times):.2f}ms")
    print(f"  Avg per token: {np.mean(decode_times):.2f}ms")
    print(f"  Median: {np.median(decode_times):.2f}ms")
    print(f"  Min: {np.min(decode_times):.2f}ms")
    print(f"  Max: {np.max(decode_times):.2f}ms")
    print(f"  Total latency: {prefill_time + sum(decode_times):.2f}ms")

    lib.aule_shutdown()
    return True

def test_batched_inference(lib):
    """
    Test batched inference (multiple sequences in parallel)
    This is common in serving scenarios
    """
    print("\n" + "=" * 70)
    print("  Test 2: Batched Inference (Multiple Sequences)")
    print("=" * 70)

    batch_size = 4
    num_heads = 8
    seq_len = 128
    head_dim = 64

    print(f"\nConfig:")
    print(f"  Batch size: {batch_size}")
    print(f"  Heads: {num_heads}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Head dim: {head_dim}")

    ret = lib.aule_init()
    if ret != 0:
        error = lib.aule_get_error()
        print(f"✗ Init failed: {ctypes.string_at(error).decode()}")
        return False

    count = batch_size * num_heads * seq_len * head_dim

    Q = lib.aule_tensor_create(batch_size, num_heads, seq_len, head_dim)
    K = lib.aule_tensor_create(batch_size, num_heads, seq_len, head_dim)
    V = lib.aule_tensor_create(batch_size, num_heads, seq_len, head_dim)
    O = lib.aule_tensor_create(batch_size, num_heads, seq_len, head_dim)

    # Create different data for each batch item
    np.random.seed(123)
    Q_data = np.random.randn(count).astype(np.float32) * 0.02
    K_data = np.random.randn(count).astype(np.float32) * 0.02
    V_data = np.random.randn(count).astype(np.float32) * 0.02

    lib.aule_tensor_upload(Q, Q_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), count)
    lib.aule_tensor_upload(K, K_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), count)
    lib.aule_tensor_upload(V, V_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), count)

    print("\n--- Running PagedAttention ---")
    start = time.time()
    ret = lib.aule_attention_forward_paged(Q, K, V, O, 0, 0, 1, -1)
    elapsed = (time.time() - start) * 1000

    if ret != 0:
        error = lib.aule_get_error()
        print(f"✗ Batched inference failed: {ctypes.string_at(error).decode()}")
        return False

    output = np.zeros(count, dtype=np.float32)
    lib.aule_tensor_download(O, output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), count)

    # Reshape to analyze per-batch
    output_batched = output.reshape(batch_size, num_heads, seq_len, head_dim)

    print(f"✓ Batched inference completed in {elapsed:.2f}ms")
    print(f"  Throughput: {batch_size * seq_len / (elapsed / 1000):.0f} tokens/sec")
    print(f"\nPer-batch statistics:")
    for b in range(batch_size):
        batch_output = output_batched[b]
        non_zero = np.count_nonzero(batch_output)
        total = num_heads * seq_len * head_dim
        print(f"  Batch {b}: {non_zero}/{total} non-zero, mean={np.mean(np.abs(batch_output)):.6f}")

    lib.aule_tensor_destroy(Q)
    lib.aule_tensor_destroy(K)
    lib.aule_tensor_destroy(V)
    lib.aule_tensor_destroy(O)
    lib.aule_shutdown()

    return True

def test_long_context(lib):
    """
    Test with long context (4096 tokens)
    This stresses memory management and block allocation
    """
    print("\n" + "=" * 70)
    print("  Test 3: Long Context (4096 tokens)")
    print("=" * 70)

    batch_size = 1
    num_heads = 8
    seq_len = 4096  # Long context
    head_dim = 64

    print(f"\nConfig:")
    print(f"  Batch size: {batch_size}")
    print(f"  Heads: {num_heads}")
    print(f"  Sequence length: {seq_len} (requires {seq_len // 32} blocks)")
    print(f"  Head dim: {head_dim}")
    print(f"  Total memory: {batch_size * num_heads * seq_len * head_dim * 4 / 1024 / 1024:.2f} MB per tensor")

    ret = lib.aule_init()
    if ret != 0:
        error = lib.aule_get_error()
        print(f"✗ Init failed: {ctypes.string_at(error).decode()}")
        return False

    count = batch_size * num_heads * seq_len * head_dim

    print("\n--- Creating tensors ---")
    Q = lib.aule_tensor_create(batch_size, num_heads, seq_len, head_dim)
    K = lib.aule_tensor_create(batch_size, num_heads, seq_len, head_dim)
    V = lib.aule_tensor_create(batch_size, num_heads, seq_len, head_dim)
    O = lib.aule_tensor_create(batch_size, num_heads, seq_len, head_dim)

    print("✓ Tensors created")

    # Use zeros for speed (we just want to test infrastructure)
    print("--- Uploading data ---")
    data = np.random.randn(count).astype(np.float32) * 0.01

    lib.aule_tensor_upload(Q, data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), count)
    lib.aule_tensor_upload(K, data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), count)
    lib.aule_tensor_upload(V, data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), count)

    print("✓ Data uploaded")

    print("\n--- Running PagedAttention ---")
    start = time.time()
    ret = lib.aule_attention_forward_paged(Q, K, V, O, 0, 0, 1, -1)
    elapsed = (time.time() - start) * 1000

    if ret != 0:
        error = lib.aule_get_error()
        print(f"✗ Long context failed: {ctypes.string_at(error).decode()}")
        return False

    print(f"✓ Completed in {elapsed:.2f}ms")
    print(f"  Throughput: {seq_len / (elapsed / 1000):.0f} tokens/sec")

    output = np.zeros(count, dtype=np.float32)
    lib.aule_tensor_download(O, output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), count)

    non_zero = np.count_nonzero(output)
    print(f"  Output: {non_zero}/{count} non-zero ({100 * non_zero / count:.1f}%)")
    print(f"  Mean: {np.mean(np.abs(output)):.6f}")

    if non_zero == 0:
        print("✗ ERROR: Long context produced all zeros!")
        return False

    lib.aule_tensor_destroy(Q)
    lib.aule_tensor_destroy(K)
    lib.aule_tensor_destroy(V)
    lib.aule_tensor_destroy(O)
    lib.aule_shutdown()

    return True

def main():
    print("=" * 70)
    print("  PagedAttention: Real ML/AI Workflow Tests")
    print("=" * 70)

    try:
        lib = load_library()
        setup_library(lib)

        tests = [
            ("Autoregressive Generation", test_autoregressive_generation),
            ("Batched Inference", test_batched_inference),
            ("Long Context", test_long_context),
        ]

        passed = 0
        failed = 0

        for name, test_fn in tests:
            try:
                if test_fn(lib):
                    passed += 1
                    print(f"\n✅ {name} PASSED\n")
                else:
                    failed += 1
                    print(f"\n❌ {name} FAILED\n")
            except Exception as e:
                failed += 1
                print(f"\n❌ {name} CRASHED: {e}\n")
                import traceback
                traceback.print_exc()

        print("=" * 70)
        print(f"Final Results: {passed}/{len(tests)} passed")
        print("=" * 70)

        return failed == 0

    except Exception as e:
        print(f"✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
