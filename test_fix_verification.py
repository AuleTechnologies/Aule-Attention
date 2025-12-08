#!/usr/bin/env python3
"""
Verify that the fix resolves the blocky noise issue.

Tests both code paths:
1. aule.install() - should work (already correct)
2. aule.patch_model() - should now work after fix
"""

import torch
import aule
from transformers import GPT2Model, GPT2Config

print("=" * 80)
print("VERIFICATION: Fix for Blocky Noise Issue")
print("=" * 80)

def test_simulated_diffusion():
    """Simulate diffusion model attention (non-causal)."""
    print("\nTest 1: Simulated Diffusion Model")
    print("-" * 80)

    batch, heads, seq, dim = 1, 8, 64, 64
    q = torch.randn(batch, heads, seq, dim)
    k = torch.randn(batch, heads, seq, dim)
    v = torch.randn(batch, heads, seq, dim)

    # Reference
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)

    # Test with aule.install()
    aule.install(verbose=False)
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
    aule.uninstall()

    diff = (out - ref).abs().max().item()
    print(f"  aule.install() path: diff={diff:.6f}")

    if diff < 0.01:
        print(f"    ✓ PASS")
        return True
    else:
        print(f"    ✗ FAIL - outputs differ!")
        return False


def test_gpt2_patch():
    """Test patching GPT-2 (causal model) - should still work."""
    print("\nTest 2: GPT-2 Model (causal=True)")
    print("-" * 80)

    config = GPT2Config(n_embd=256, n_head=4, n_layer=1)
    model = GPT2Model(config)
    model.eval()

    input_ids = torch.randint(0, 1000, (1, 32))

    # Reference (unpatched)
    with torch.no_grad():
        ref_out = model(input_ids).last_hidden_state

    # Patched with explicit causal=True
    aule.patch_model(model, config={"causal": True})
    with torch.no_grad():
        patched_out = model(input_ids).last_hidden_state

    diff = (patched_out - ref_out).abs().max().item()
    print(f"  aule.patch_model(causal=True): diff={diff:.6f}")

    if diff < 0.01:
        print(f"    ✓ PASS")
        return True
    else:
        print(f"    ✗ FAIL - GPT-2 broken after patch!")
        return False


def test_default_config():
    """Test that default PATCH_CONFIG is now causal=False."""
    print("\nTest 3: Default Patch Config")
    print("-" * 80)

    # Force reimport to avoid caching issues
    import importlib
    import aule.patching
    importlib.reload(aule.patching)
    from aule.patching import PATCH_CONFIG

    print(f"  PATCH_CONFIG['causal'] = {PATCH_CONFIG['causal']}")

    if PATCH_CONFIG['causal'] == False:
        print(f"    ✓ PASS - defaults to False (diffusion-friendly)")
        return True
    else:
        print(f"    ✗ FAIL - defaults to True (causes blocky noise!)")
        return False


def test_comfy_auleinstall():
    """
    Simulate what happens when user uses AuleInstall node in ComfyUI.
    This just calls aule.install() with no args.
    """
    print("\nTest 4: ComfyUI AuleInstall Node")
    print("-" * 80)

    # Simulate diffusion model
    batch, heads, seq, dim = 1, 8, 64, 64
    q = torch.randn(batch, heads, seq, dim)
    k = torch.randn(batch, heads, seq, dim)
    v = torch.randn(batch, heads, seq, dim)

    # Reference
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)

    # Simulate AuleInstall node
    aule.install()  # <-- This is what AuleInstall.install() does

    # Model calls SDPA with is_causal=False (standard for SD)
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)

    aule.uninstall()

    diff = (out - ref).abs().max().item()
    print(f"  After aule.install(): diff={diff:.6f}")

    if diff < 0.01:
        print(f"    ✓ PASS - AuleInstall works correctly")
        return True
    else:
        print(f"    ✗ FAIL - AuleInstall breaks diffusion models!")
        return False


# Run all tests
print("\n" + "=" * 80)
print("RUNNING TESTS")
print("=" * 80)

results = []
results.append(("Simulated Diffusion", test_simulated_diffusion()))
results.append(("GPT-2 Patch", test_gpt2_patch()))
results.append(("Default Config", test_default_config()))
results.append(("ComfyUI AuleInstall", test_comfy_auleinstall()))

# Summary
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

for name, passed in results:
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}: {name}")

all_passed = all(r[1] for r in results)

print("\n" + "=" * 80)
if all_passed:
    print("✓ ALL TESTS PASSED")
    print("\nThe fix is successful! Blocky noise issue resolved.")
else:
    print("✗ SOME TESTS FAILED")
    print("\nFurther investigation needed.")
print("=" * 80 + "\n")
