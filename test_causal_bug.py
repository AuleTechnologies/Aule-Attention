#!/usr/bin/env python3
"""
Test to reproduce the "blocky noise" by using WRONG causal setting.

Hypothesis: The blocky noise happens because aule.install() or AulePatchModel
was using causal=True (default for LLMs) on a diffusion model that needs causal=False.
"""

import torch
import numpy as np
import aule

print("=" * 80)
print("TESTING: Wrong Causal Masking = Blocky Noise")
print("=" * 80)

# Simulate a 2D spatial attention (like SD UNet)
# In practice, this is a flattened 8x8 or 16x16 spatial grid
batch = 1
heads = 8
spatial_h = 8
spatial_w = 8
seq = spatial_h * spatial_w  # 64 positions
dim = 64

print(f"\nSimulating 2D spatial attention (SD UNet style)")
print(f"  Spatial: {spatial_h}x{spatial_w} = {seq} tokens")
print(f"  Heads: {heads}, Head dim: {dim}")

# Create inputs
q = torch.randn(batch, heads, seq, dim)
k = torch.randn(batch, heads, seq, dim)
v = torch.randn(batch, heads, seq, dim)

# CORRECT: causal=False (bidirectional attention for images)
print("\n" + "=" * 80)
print("TEST 1: CORRECT - causal=False (bidirectional)")
print("=" * 80)

aule.install()
out_correct = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)

print(f"✓ Output shape: {out_correct.shape}")
print(f"  mean={out_correct.mean():.6f}, std={out_correct.std():.6f}")

# Reshape to 2D to visualize
out_2d = out_correct[0, 0].view(spatial_h, spatial_w, dim)
print(f"\nSpatial statistics (first head, first dim):")
for i in range(spatial_h):
    row_mean = out_2d[i, :, 0].mean().item()
    print(f"  Row {i}: mean={row_mean:+.4f}")

# WRONG: causal=True (autoregressive masking on 2D image!)
print("\n" + "=" * 80)
print("TEST 2: WRONG - causal=True (this causes blocky noise!)")
print("=" * 80)

out_wrong = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

print(f"✓ Output shape: {out_wrong.shape}")
print(f"  mean={out_wrong.mean():.6f}, std={out_wrong.std():.6f}")

# Reshape to 2D
out_2d_wrong = out_wrong[0, 0].view(spatial_h, spatial_w, dim)
print(f"\nSpatial statistics (first head, first dim):")
for i in range(spatial_h):
    row_mean = out_2d_wrong[i, :, 0].mean().item()
    print(f"  Row {i}: mean={row_mean:+.4f}")

# Compare
print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

diff = (out_wrong - out_correct).abs()
print(f"\nAbsolute difference:")
print(f"  Max: {diff.max():.6f}")
print(f"  Mean: {diff.mean():.6f}")

# Check for "blocky" patterns
print(f"\nChecking for blocky artifacts...")
print(f"  (Causal masking creates triangular patterns in 1D,")
print(f"   which become diagonal block patterns when reshaped to 2D)")

# Visualize the difference pattern
diff_2d = diff[0, 0].view(spatial_h, spatial_w, dim).mean(dim=2)  # Average across dims
print(f"\nDifference heatmap (row x col):")
print("   ", "  ".join([f"{c:2d}" for c in range(spatial_w)]))
for i in range(spatial_h):
    row_str = f"{i}: "
    for j in range(spatial_w):
        val = diff_2d[i, j].item()
        if val < 0.01:
            row_str += " . "
        elif val < 0.1:
            row_str += " o "
        else:
            row_str += " X "
    print(row_str)

print(f"\nLegend: . = small diff, o = medium diff, X = large diff")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

if diff.max() > 0.1:
    print("✗ LARGE DIFFERENCE detected!")
    print("\n  This confirms that using causal=True on a diffusion model")
    print("  produces WRONG outputs with diagonal block patterns.")
    print("\n  The 'blocky noise' in ComfyUI is likely caused by:")
    print("    1. aule.install() defaulting to causal=True (for LLMs)")
    print("    2. Or AuleInstall node not configuring causal=False")
    print("\n  FIX: Diffusion models MUST use causal=False!")
else:
    print("✓ Outputs are similar")
    print("  (Unexpected - investigate further)")

aule.uninstall()

print("\n" + "=" * 80)
