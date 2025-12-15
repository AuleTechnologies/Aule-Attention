#!/usr/bin/env python3
"""
Comprehensive benchmark for AMD MI300X
Measures raw performance and compares against theoretical peaks
"""

import sys
import os
import ctypes
import numpy as np
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

def load_library():
    lib_paths = [
        '../zig-out/lib/libaule.so',
        './zig-out/lib/libaule.so',
        '/root/Sndr/zig-out/lib/libaule.so',
    ]
    for path in lib_paths:
        if os.path.exists(path):
            print(f"✓ Loading library from: {path}")
            return ctypes.CDLL(path)
    raise FileNotFoundError("Could not find libaule.so - run 'zig build' first")

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

def get_device_info(lib):
    """Get GPU device information"""
    vendor = lib.aule_get_vendor()
    vendor_names = {0: "Other", 1: "AMD", 2: "NVIDIA", 3: "Intel", 4: "Apple"}

    device_name = ctypes.create_string_buffer(256)
    lib.aule_get_device_name(device_name, 256)

    return {
        'vendor': vendor_names.get(vendor, 'Unknown'),
        'vendor_id': vendor,
        'device': device_name.value.decode(),
        'is_mi300x': 'MI300X' in device_name.value.decode().upper()
    }

def calculate_flops(batch, heads, seq_len, head_dim, time_ms):
    """
    Calculate FLOPS for attention operation
    Attention FLOPS = 4 * batch * heads * seq_len^2 * head_dim
    (QK matmul, softmax, PV matmul, plus overhead)
    """
    flops = 4 * batch * heads * seq_len * seq_len * head_dim
    tflops = (flops / (time_ms / 1000)) / 1e12
    return tflops

def benchmark_config(lib, name, batch, heads, seq_len, head_dim, use_paged=False, warmup=3, iterations=10):
    """Benchmark a specific configuration"""
    print(f"\n--- {name} ---")
    print(f"Config: batch={batch}, heads={heads}, seq_len={seq_len}, head_dim={head_dim}")

    count = batch * heads * seq_len * head_dim

    # Create tensors
    Q = lib.aule_tensor_create(batch, heads, seq_len, head_dim)
    K = lib.aule_tensor_create(batch, heads, seq_len, head_dim)
    V = lib.aule_tensor_create(batch, heads, seq_len, head_dim)
    O = lib.aule_tensor_create(batch, heads, seq_len, head_dim)

    # Upload data
    data = np.random.randn(count).astype(np.float32) * 0.02
    lib.aule_tensor_upload(Q, data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), count)
    lib.aule_tensor_upload(K, data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), count)
    lib.aule_tensor_upload(V, data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), count)

    # Choose forward function
    forward_fn = lib.aule_attention_forward_paged if use_paged else lib.aule_attention_forward_gpu

    # Warmup
    for _ in range(warmup):
        forward_fn(Q, K, V, O, 0, 0, 1, -1)

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.time()
        ret = forward_fn(Q, K, V, O, 0, 0, 1, -1)
        elapsed = (time.time() - start) * 1000

        if ret != 0:
            error = lib.aule_get_error()
            print(f"✗ Error: {ctypes.string_at(error).decode()}")
            break

        times.append(elapsed)

    # Verify output
    output = np.zeros(count, dtype=np.float32)
    lib.aule_tensor_download(O, output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), count)
    non_zero = np.count_nonzero(output)

    # Cleanup
    lib.aule_tensor_destroy(Q)
    lib.aule_tensor_destroy(K)
    lib.aule_tensor_destroy(V)
    lib.aule_tensor_destroy(O)

    # Statistics
    if times:
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        tflops = calculate_flops(batch, heads, seq_len, head_dim, avg_time)
        throughput = (batch * seq_len) / (avg_time / 1000)  # tokens/sec

        return {
            'name': name,
            'batch': batch,
            'heads': heads,
            'seq_len': seq_len,
            'head_dim': head_dim,
            'use_paged': use_paged,
            'avg_time_ms': avg_time,
            'std_time_ms': std_time,
            'min_time_ms': min_time,
            'max_time_ms': max_time,
            'tflops': tflops,
            'throughput': throughput,
            'non_zero': non_zero,
            'total': count,
            'success': non_zero > 0
        }

    return None

def print_results_table(results):
    """Print results in a formatted table"""
    print("\n" + "=" * 120)
    print("BENCHMARK RESULTS")
    print("=" * 120)
    print(f"{'Config':<35} {'Batch':<6} {'Heads':<6} {'SeqLen':<7} {'Time(ms)':<12} {'TFLOPS':<10} {'Throughput':<15} {'Status':<8}")
    print("-" * 120)

    for r in results:
        status = "✓ PASS" if r['success'] else "✗ FAIL"
        throughput_str = f"{r['throughput']:.0f} tok/s"
        print(f"{r['name']:<35} {r['batch']:<6} {r['heads']:<6} {r['seq_len']:<7} "
              f"{r['avg_time_ms']:>8.2f}±{r['std_time_ms']:.2f} {r['tflops']:>8.2f} "
              f"{throughput_str:<15} {status:<8}")

    print("=" * 120)

def main():
    print("=" * 80)
    print("  AMD MI300X PagedAttention Benchmark Suite")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 80)

    try:
        lib = load_library()
        setup_library(lib)

        # Initialize
        ret = lib.aule_init()
        if ret != 0:
            error = lib.aule_get_error()
            print(f"✗ Init failed: {ctypes.string_at(error).decode()}")
            return False

        # Get device info
        device_info = get_device_info(lib)
        print(f"\n✓ GPU: {device_info['device']}")
        print(f"  Vendor: {device_info['vendor']}")

        if device_info['is_mi300x']:
            print(f"  🚀 AMD MI300X detected - running production benchmarks")
            print(f"  Expected peak: ~650 TFLOPS FP32, 5.3 TB/s memory bandwidth")
        else:
            print(f"  ℹ  Not MI300X - running standard benchmarks")

        print("\n" + "=" * 80)
        print("  Running benchmarks (this may take several minutes)...")
        print("=" * 80)

        results = []

        # Benchmark configurations
        configs = [
            # Small: Typical decode step
            ("Small Decode (1 token)", 1, 32, 1, 128, True),
            ("Small Decode (std)", 1, 32, 1, 128, False),

            # Medium: Short sequence
            ("Medium Prefill (256)", 1, 32, 256, 128, True),
            ("Medium Prefill (std)", 1, 32, 256, 128, False),

            # Large: Standard context
            ("Large Context (2K)", 1, 32, 2048, 128, True),
            ("Large Context (std)", 1, 32, 2048, 128, False),

            # Very Large: Long context
            ("Very Large (4K)", 1, 32, 4096, 128, True),
            ("Very Large (std)", 1, 32, 4096, 128, False),

            # Batched: Multi-sequence
            ("Batched Small (B=8)", 8, 32, 128, 128, True),
            ("Batched Small (std)", 8, 32, 128, 128, False),

            ("Batched Medium (B=16)", 16, 32, 256, 128, True),
            ("Batched Medium (std)", 16, 32, 256, 128, False),

            # Extreme: Stress test
            ("Extreme (8K)", 1, 32, 8192, 128, True),
        ]

        for name, batch, heads, seq_len, head_dim, use_paged in configs:
            result = benchmark_config(lib, name, batch, heads, seq_len, head_dim,
                                    use_paged=use_paged, warmup=3, iterations=10)
            if result:
                results.append(result)
                print(f"  ✓ {name}: {result['avg_time_ms']:.2f}ms, {result['tflops']:.2f} TFLOPS")

        # Print summary
        print_results_table(results)

        # Analysis
        print("\n" + "=" * 80)
        print("  ANALYSIS")
        print("=" * 80)

        # Compare paged vs standard
        paged_results = [r for r in results if r['use_paged']]
        std_results = [r for r in results if not r['use_paged']]

        if paged_results:
            avg_paged_tflops = np.mean([r['tflops'] for r in paged_results])
            print(f"\n  PagedAttention Average: {avg_paged_tflops:.2f} TFLOPS")

        if std_results:
            avg_std_tflops = np.mean([r['tflops'] for r in std_results])
            print(f"  Standard Attention Average: {avg_std_tflops:.2f} TFLOPS")

        if paged_results and std_results:
            speedup = avg_paged_tflops / avg_std_tflops
            print(f"  PagedAttention Speedup: {speedup:.2f}x")

        # Peak performance
        best_result = max(results, key=lambda r: r['tflops'])
        print(f"\n  Peak Performance: {best_result['tflops']:.2f} TFLOPS")
        print(f"    Config: {best_result['name']}")
        print(f"    Latency: {best_result['avg_time_ms']:.2f}ms")
        print(f"    Throughput: {best_result['throughput']:.0f} tokens/sec")

        # MI300X specific analysis
        if device_info['is_mi300x']:
            theoretical_peak = 650  # TFLOPS for MI300X FP32
            efficiency = (best_result['tflops'] / theoretical_peak) * 100
            print(f"\n  MI300X Efficiency: {efficiency:.1f}% of theoretical peak")
            print(f"  Theoretical Peak: {theoretical_peak} TFLOPS (FP32)")

            if efficiency > 50:
                print(f"  ✓ Excellent performance!")
            elif efficiency > 30:
                print(f"  ℹ  Good performance, room for optimization")
            else:
                print(f"  ⚠  Performance below expected - check GPU clocks and drivers")

        lib.aule_shutdown()

        # Save results to file
        results_file = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(results_file, 'w') as f:
            f.write(f"AMD MI300X PagedAttention Benchmark Results\n")
            f.write(f"Date: {datetime.now()}\n")
            f.write(f"Device: {device_info['device']}\n")
            f.write(f"Vendor: {device_info['vendor']}\n\n")

            for r in results:
                f.write(f"{r['name']}: {r['avg_time_ms']:.2f}ms, {r['tflops']:.2f} TFLOPS\n")

        print(f"\n  Results saved to: {results_file}")
        print("\n" + "=" * 80)

        return True

    except Exception as e:
        print(f"✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
