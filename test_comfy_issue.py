#!/usr/bin/env python3
"""
Test script to reproduce and diagnose the ComfyUI WAN 2.2 blocky noise issue.

This simulates what happens when ComfyUI calls aule.install() and then runs
a diffusion model (like Stable Diffusion) through KSampler.
"""

import torch
import numpy as np
import sys

print("=" * 80)
print("AULE COMFYUI INTEGRATION TEST")
print("=" * 80)

# Import aule
try:
    import aule
    print(f"\n✓ aule-attention v{aule.__version__} imported")
    print(f"  Location: {aule.__file__}")
except ImportError as e:
    print(f"\n✗ Failed to import aule: {e}")
    sys.exit(1)

# Print backend info
print("\n" + "=" * 80)
print("BACKEND INFORMATION")
print("=" * 80)
aule.print_backend_info()

# Test 1: Non-causal attention (Stable Diffusion style)
print("\n" + "=" * 80)
print("TEST 1: Non-Causal Attention (Diffusion Models)")
print("=" * 80)

def test_diffusion_attention():
    """
    Simulate Stable Diffusion / FLUX / SD3 attention.
    These models use NON-CAUSAL attention.
    """
    print("\nSimulating SD/FLUX attention pattern...")

    # Typical SD UNet attention shape
    batch = 2
    heads = 8
    seq = 64  # 8x8 spatial resolution flattened
    dim = 64  # head_dim

    print(f"  Shape: batch={batch}, heads={heads}, seq={seq}, head_dim={dim}")
    print(f"  Causal: FALSE (bidirectional)")

    # Create random inputs
    q = torch.randn(batch, heads, seq, dim, dtype=torch.float32)
    k = torch.randn(batch, heads, seq, dim, dtype=torch.float32)
    v = torch.randn(batch, heads, seq, dim, dtype=torch.float32)

    # Reference: PyTorch SDPA
    print("\n  Computing reference (PyTorch SDPA)...")
    with torch.no_grad():
        ref_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=False
        )

    print(f"    ✓ Reference output: shape={ref_output.shape}")
    print(f"      mean={ref_output.mean():.6f}, std={ref_output.std():.6f}")
    print(f"      min={ref_output.min():.6f}, max={ref_output.max():.6f}")

    # Test with aule (using install() like ComfyUI does)
    print("\n  Installing aule-attention globally...")
    aule.install(verbose=True)

    print("\n  Computing with aule-attention...")
    with torch.no_grad():
        aule_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=False
        )

    print(f"    ✓ Aule output: shape={aule_output.shape}")
    print(f"      mean={aule_output.mean():.6f}, std={aule_output.std():.6f}")
    print(f"      min={aule_output.min():.6f}, max={aule_output.max():.6f}")

    # Compare
    print("\n  Comparing outputs...")
    diff = (aule_output - ref_output).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"    Max absolute error:  {max_diff:.6f}")
    print(f"    Mean absolute error: {mean_diff:.6f}")

    # Check for "blocky" patterns (high variance in patches)
    patch_size = 8
    for i in range(0, seq, patch_size):
        patch_diff = diff[:, :, i:i+patch_size, :].mean().item()
        if patch_diff > 0.1:
            print(f"    ⚠ High error in patch {i//patch_size}: {patch_diff:.6f}")

    # Verdict
    threshold = 0.01  # 1% tolerance for fp32
    if max_diff < threshold:
        print(f"\n  ✓ PASS: Outputs match (max_diff={max_diff:.6f} < {threshold})")
        return True
    else:
        print(f"\n  ✗ FAIL: Outputs differ (max_diff={max_diff:.6f} >= {threshold})")

        # Diagnose failure
        print("\n  Diagnosing failure...")

        # Check if output is garbage (NaN, Inf, or way out of range)
        has_nan = torch.isnan(aule_output).any()
        has_inf = torch.isinf(aule_output).any()

        if has_nan:
            print("    ✗ Output contains NaN values!")
        if has_inf:
            print("    ✗ Output contains Inf values!")

        # Check if output is all zeros or constant
        if aule_output.std() < 1e-6:
            print("    ✗ Output is nearly constant (std < 1e-6)!")

        # Check if output looks like noise
        ref_std = ref_output.std().item()
        aule_std = aule_output.std().item()
        if abs(aule_std - ref_std) / ref_std > 2.0:
            print(f"    ✗ Output std differs by >200%: ref={ref_std:.6f}, aule={aule_std:.6f}")
            print("      This suggests WRONG ATTENTION was computed!")

        return False

try:
    result = test_diffusion_attention()
except Exception as e:
    print(f"\n✗ TEST FAILED WITH EXCEPTION:")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    result = False

# Test 2: Check backend selection logic
print("\n" + "=" * 80)
print("TEST 2: Backend Selection with Different Configurations")
print("=" * 80)

def test_backend_selection():
    """Test which backend gets selected for different input types."""

    print("\nTest 2a: PyTorch tensor on CPU")
    q_cpu = torch.randn(1, 4, 32, 64)
    k_cpu = torch.randn(1, 4, 32, 64)
    v_cpu = torch.randn(1, 4, 32, 64)

    aule.uninstall()
    aule.install(verbose=True)
    print("  Calling SDPA with CPU tensors...")
    try:
        out = torch.nn.functional.scaled_dot_product_attention(q_cpu, k_cpu, v_cpu, is_causal=False)
        print(f"    ✓ Success: {out.shape}")
    except Exception as e:
        print(f"    ✗ Failed: {e}")

    print("\nTest 2b: NumPy arrays (direct call)")
    q_np = np.random.randn(1, 4, 32, 64).astype(np.float32)
    k_np = np.random.randn(1, 4, 32, 64).astype(np.float32)
    v_np = np.random.randn(1, 4, 32, 64).astype(np.float32)

    print("  Calling flash_attention directly with NumPy...")
    try:
        out = aule.flash_attention(q_np, k_np, v_np, causal=False)
        print(f"    ✓ Success: {out.shape}, type={type(out)}")
    except Exception as e:
        print(f"    ✗ Failed: {e}")

    print("\nTest 2c: Force Vulkan backend")
    aule.uninstall()
    aule.install(backend='vulkan', verbose=True)
    q = torch.randn(1, 4, 32, 64)
    k = torch.randn(1, 4, 32, 64)
    v = torch.randn(1, 4, 32, 64)

    print("  Calling SDPA with forced Vulkan backend...")
    try:
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
        print(f"    ✓ Success: {out.shape}")
    except Exception as e:
        print(f"    ✗ Failed: {e}")

try:
    test_backend_selection()
except Exception as e:
    print(f"\n✗ Backend selection tests failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Check head_dim limits
print("\n" + "=" * 80)
print("TEST 3: Head Dimension Limits")
print("=" * 80)

def test_head_dim_limits():
    """Test various head dimensions to find breaking points."""

    test_dims = [32, 64, 80, 96, 128]

    aule.uninstall()
    aule.install(backend='vulkan', verbose=False)

    for dim in test_dims:
        print(f"\nTesting head_dim={dim}...")
        q = torch.randn(1, 4, 32, dim)
        k = torch.randn(1, 4, 32, dim)
        v = torch.randn(1, 4, 32, dim)

        try:
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
            print(f"  ✓ head_dim={dim} works")
        except Exception as e:
            print(f"  ✗ head_dim={dim} failed: {e}")

try:
    test_head_dim_limits()
except Exception as e:
    print(f"\n✗ Head dim tests failed: {e}")

# Cleanup
aule.uninstall()

# Summary
print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)
if result:
    print("✓ All critical tests passed")
    print("\nThe library appears to be working correctly.")
    print("The ComfyUI issue may be configuration-related (causal=True vs False)")
else:
    print("✗ Critical tests FAILED")
    print("\nThe library has bugs that need to be fixed.")
    print("This is likely causing the 'blocky noise' in ComfyUI.")

print("\n" + "=" * 80)
