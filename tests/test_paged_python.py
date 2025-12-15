#!/usr/bin/env python3
"""
Test PagedAttention implementation from Python
Tests on both Intel iGPU and AMD MI300X
"""

import sys
import os
import ctypes
import numpy as np
import time

# Add python module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

def load_library():
    """Load the aule shared library"""
    lib_paths = [
        '../zig-out/lib/libaule.so',
        './zig-out/lib/libaule.so',
        '/home/yab/Sndr/zig-out/lib/libaule.so',
    ]

    for path in lib_paths:
        if os.path.exists(path):
            print(f"✓ Loading library from: {path}")
            return ctypes.CDLL(path)

    raise FileNotFoundError("Could not find libaule.so - run 'zig build' first")

def test_basic_attention(lib):
    """Test basic attention (non-paged) as baseline"""
    print("\n=== Test 1: Basic Attention (Baseline) ===")

    # Initialize library
    ret = lib.aule_init()
    if ret != 0:
        error = lib.aule_get_error()
        print(f"✗ Init failed: {ctypes.string_at(error).decode()}")
        return False

    # Get device info
    vendor = lib.aule_get_vendor()
    vendor_names = {0: "Other", 1: "AMD", 2: "NVIDIA", 3: "Intel", 4: "Apple"}
    print(f"✓ GPU Vendor: {vendor_names.get(vendor, 'Unknown')} (ID: {vendor})")

    # Get device name
    device_name = ctypes.create_string_buffer(256)
    name_len = lib.aule_get_device_name(device_name, 256)
    if name_len > 0:
        print(f"✓ Device: {device_name.value.decode()}")

    # Small test case
    batch, heads, seq_len, head_dim = 1, 2, 64, 32
    count = batch * heads * seq_len * head_dim

    print(f"✓ Shape: [{batch}, {heads}, {seq_len}, {head_dim}]")

    # Create tensors
    Q_handle = lib.aule_tensor_create(batch, heads, seq_len, head_dim)
    K_handle = lib.aule_tensor_create(batch, heads, seq_len, head_dim)
    V_handle = lib.aule_tensor_create(batch, heads, seq_len, head_dim)
    O_handle = lib.aule_tensor_create(batch, heads, seq_len, head_dim)

    if not all([Q_handle, K_handle, V_handle, O_handle]):
        print("✗ Failed to create tensors")
        return False

    print("✓ Created 4 GPU tensors")

    # Initialize with simple data
    Q_data = np.random.randn(count).astype(np.float32) * 0.1
    K_data = np.random.randn(count).astype(np.float32) * 0.1
    V_data = np.random.randn(count).astype(np.float32) * 0.1

    # Upload data
    lib.aule_tensor_upload(Q_handle, Q_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), count)
    lib.aule_tensor_upload(K_handle, K_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), count)
    lib.aule_tensor_upload(V_handle, V_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), count)

    print("✓ Uploaded input data")

    # Run attention
    start = time.time()
    ret = lib.aule_attention_forward_gpu(
        Q_handle, K_handle, V_handle, O_handle,
        0, 0,  # no RoPE
        0,     # not causal
        -1     # no window
    )
    elapsed = (time.time() - start) * 1000

    if ret != 0:
        error = lib.aule_get_error()
        print(f"✗ Forward failed: {ctypes.string_at(error).decode()}")
        return False

    print(f"✓ Forward pass completed in {elapsed:.2f}ms")

    # Download output
    output = np.zeros(count, dtype=np.float32)
    lib.aule_tensor_download(O_handle, output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), count)

    # Verify output
    non_zero = np.count_nonzero(output)
    mean_val = np.mean(np.abs(output))
    max_val = np.max(np.abs(output))

    print(f"✓ Output: {non_zero}/{count} non-zero, mean={mean_val:.4f}, max={max_val:.4f}")

    # Cleanup
    lib.aule_tensor_destroy(Q_handle)
    lib.aule_tensor_destroy(K_handle)
    lib.aule_tensor_destroy(V_handle)
    lib.aule_tensor_destroy(O_handle)
    lib.aule_shutdown()

    print("✓ Test passed!\n")
    return True

def test_large_sequence(lib):
    """Test with longer sequence (should benefit from paging)"""
    print("=== Test 2: Large Sequence (2048 tokens) ===")

    ret = lib.aule_init()
    if ret != 0:
        print("✗ Init failed")
        return False

    batch, heads, seq_len, head_dim = 2, 4, 2048, 64
    count = batch * heads * seq_len * head_dim

    print(f"✓ Shape: [{batch}, {heads}, {seq_len}, {head_dim}]")
    print(f"✓ Total elements: {count:,}")

    Q_handle = lib.aule_tensor_create(batch, heads, seq_len, head_dim)
    K_handle = lib.aule_tensor_create(batch, heads, seq_len, head_dim)
    V_handle = lib.aule_tensor_create(batch, heads, seq_len, head_dim)
    O_handle = lib.aule_tensor_create(batch, heads, seq_len, head_dim)

    if not all([Q_handle, K_handle, V_handle, O_handle]):
        print("✗ Failed to create tensors")
        return False

    # Initialize with zeros for speed
    data = np.zeros(count, dtype=np.float32)
    data[:100] = 0.1  # Just a few non-zero values

    lib.aule_tensor_upload(Q_handle, data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), count)
    lib.aule_tensor_upload(K_handle, data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), count)
    lib.aule_tensor_upload(V_handle, data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), count)

    print("✓ Uploaded data")

    # Run attention
    start = time.time()
    ret = lib.aule_attention_forward_gpu(Q_handle, K_handle, V_handle, O_handle, 0, 0, 0, -1)
    elapsed = (time.time() - start) * 1000

    if ret != 0:
        error = lib.aule_get_error()
        print(f"✗ Forward failed: {ctypes.string_at(error).decode()}")
        lib.aule_tensor_destroy(Q_handle)
        lib.aule_tensor_destroy(K_handle)
        lib.aule_tensor_destroy(V_handle)
        lib.aule_tensor_destroy(O_handle)
        lib.aule_shutdown()
        return False

    print(f"✓ Forward pass completed in {elapsed:.2f}ms")

    # Cleanup
    lib.aule_tensor_destroy(Q_handle)
    lib.aule_tensor_destroy(K_handle)
    lib.aule_tensor_destroy(V_handle)
    lib.aule_tensor_destroy(O_handle)
    lib.aule_shutdown()

    print("✓ Test passed!\n")
    return True

def test_paged_attention(lib):
    """Test PagedAttention path specifically"""
    print("\n=== Test 3: PagedAttention (Block-based KV Cache) ===")

    ret = lib.aule_init()
    if ret != 0:
        print("✗ Init failed")
        return False

    batch, heads, seq_len, head_dim = 1, 2, 128, 64
    count = batch * heads * seq_len * head_dim

    print(f"✓ Shape: [{batch}, {heads}, {seq_len}, {head_dim}]")

    Q_handle = lib.aule_tensor_create(batch, heads, seq_len, head_dim)
    K_handle = lib.aule_tensor_create(batch, heads, seq_len, head_dim)
    V_handle = lib.aule_tensor_create(batch, heads, seq_len, head_dim)
    O_handle = lib.aule_tensor_create(batch, heads, seq_len, head_dim)

    if not all([Q_handle, K_handle, V_handle, O_handle]):
        print("✗ Failed to create tensors")
        return False

    # Initialize with random data
    data = np.random.randn(count).astype(np.float32) * 0.1
    lib.aule_tensor_upload(Q_handle, data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), count)
    lib.aule_tensor_upload(K_handle, data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), count)
    lib.aule_tensor_upload(V_handle, data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), count)

    print("✓ Uploaded input data")

    # Run PagedAttention
    start = time.time()
    ret = lib.aule_attention_forward_paged(
        Q_handle, K_handle, V_handle, O_handle,
        0, 0,  # no RoPE
        0,     # not causal
        -1     # no window
    )
    elapsed = (time.time() - start) * 1000

    if ret != 0:
        error = lib.aule_get_error()
        print(f"✗ PagedAttention failed: {ctypes.string_at(error).decode()}")
        lib.aule_tensor_destroy(Q_handle)
        lib.aule_tensor_destroy(K_handle)
        lib.aule_tensor_destroy(V_handle)
        lib.aule_tensor_destroy(O_handle)
        lib.aule_shutdown()
        return False

    print(f"✓ PagedAttention completed in {elapsed:.2f}ms")

    # Download output
    output = np.zeros(count, dtype=np.float32)
    lib.aule_tensor_download(O_handle, output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), count)

    # Verify output
    non_zero = np.count_nonzero(output)
    mean_val = np.mean(np.abs(output))
    max_val = np.max(np.abs(output))

    print(f"✓ Output: {non_zero}/{count} non-zero, mean={mean_val:.4f}, max={max_val:.4f}")

    # Cleanup
    lib.aule_tensor_destroy(Q_handle)
    lib.aule_tensor_destroy(K_handle)
    lib.aule_tensor_destroy(V_handle)
    lib.aule_tensor_destroy(O_handle)
    lib.aule_shutdown()

    print("✓ PagedAttention test passed!\n")
    return True

def test_benchmark(lib):
    """Benchmark different sequence lengths"""
    print("=== Test 4: Benchmark Sequence Lengths ===")

    ret = lib.aule_init()
    if ret != 0:
        print("✗ Init failed")
        return False

    batch, heads, head_dim = 1, 8, 64
    seq_lengths = [128, 256, 512, 1024, 2048]

    results = []

    for seq_len in seq_lengths:
        count = batch * heads * seq_len * head_dim

        Q = lib.aule_tensor_create(batch, heads, seq_len, head_dim)
        K = lib.aule_tensor_create(batch, heads, seq_len, head_dim)
        V = lib.aule_tensor_create(batch, heads, seq_len, head_dim)
        O = lib.aule_tensor_create(batch, heads, seq_len, head_dim)

        # Upload zeros
        data = np.zeros(count, dtype=np.float32)
        lib.aule_tensor_upload(Q, data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), count)
        lib.aule_tensor_upload(K, data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), count)
        lib.aule_tensor_upload(V, data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), count)

        # Warmup
        lib.aule_attention_forward_gpu(Q, K, V, O, 0, 0, 0, -1)

        # Benchmark
        times = []
        for _ in range(5):
            start = time.time()
            lib.aule_attention_forward_gpu(Q, K, V, O, 0, 0, 0, -1)
            times.append((time.time() - start) * 1000)

        avg_time = np.mean(times)
        results.append((seq_len, avg_time))

        print(f"  seq_len={seq_len:4d}: {avg_time:.2f}ms (±{np.std(times):.2f}ms)")

        lib.aule_tensor_destroy(Q)
        lib.aule_tensor_destroy(K)
        lib.aule_tensor_destroy(V)
        lib.aule_tensor_destroy(O)

    lib.aule_shutdown()

    print("✓ Benchmark completed!\n")
    return True

def main():
    """Run all tests"""
    print("=" * 60)
    print("  PagedAttention Test Suite")
    print("=" * 60)

    try:
        lib = load_library()

        # Set up function signatures
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

        # Run tests
        tests = [
            ("Basic Attention", test_basic_attention),
            ("Large Sequence", test_large_sequence),
            ("PagedAttention", test_paged_attention),
            ("Benchmark", test_benchmark),
        ]

        passed = 0
        failed = 0

        for name, test_fn in tests:
            try:
                if test_fn(lib):
                    passed += 1
                else:
                    failed += 1
                    print(f"✗ {name} FAILED\n")
            except Exception as e:
                failed += 1
                print(f"✗ {name} CRASHED: {e}\n")

        print("=" * 60)
        print(f"Results: {passed} passed, {failed} failed")
        print("=" * 60)

        return failed == 0

    except Exception as e:
        print(f"✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
