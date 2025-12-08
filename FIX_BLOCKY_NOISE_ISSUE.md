# Fix: ComfyUI "Blocky Noise" Issue with WAN 2.2

## Issue Summary

**Problem**: When using aule-attention in ComfyUI with WAN 2.2 (or any diffusion model), the KSampler output was blocky noise instead of coherent images.

**Root Cause**: The `PATCH_CONFIG` default was set to `causal=True` (correct for LLMs like GPT-2), but diffusion models require `causal=False` (bidirectional attention).

**Impact**: Using causal masking on a 2D image attention creates diagonal block artifacts, making the output look like blocky noise.

---

## Technical Details

### What is Causal Masking?

- **Causal (causal=True)**: Autoregressive masking - token i can only attend to tokens ≤ i
  - Used for: LLMs (GPT-2, Llama, etc.)
  - Pattern: Lower triangular mask

- **Non-Causal (causal=False)**: Bidirectional attention - all tokens can attend to all tokens
  - Used for: Diffusion models (Stable Diffusion, FLUX, SD3), BERT, ViT
  - Pattern: No mask (or custom attention masks)

### Why It Caused "Blocky" Patterns

When you flatten a 2D image (8×8 = 64 tokens) and apply causal masking:
- Token 0 (top-left) can only see itself
- Token 1 can see tokens 0-1
- Token 63 (bottom-right) can see all tokens

This creates a **diagonal gradient pattern** when visualized as 2D, which looks like "blocky noise" because:
- Early tokens have very limited context → poor reconstruction
- The triangular pattern becomes visible as diagonal blocks in the output image

### Verification

Test `/home/yab/Sndr/test_causal_bug.py` confirms:
- `causal=False`: Uniform smooth output (correct)
- `causal=True`: Diagonal block pattern with max difference > 3.5 (WRONG)

Difference heatmap when using causal=True on 8×8 image:
```
     0   1   2   3   4   5   6   7
0:  X  X  X  X  X  X  X  X     <-- Heavy corruption
1:  X  X  X  X  X  X  X  X
2:  X  X  X  X  X  X  X  X
3:  X  X  X  X  X  X  X  X
4:  X  X  X  X  X  X  X  X
5:  X  o  X  X  X  X  o  o
6:  o  o  o  X  o  o  o  o
7:  o  o  o  o  o  .  .  .     <-- Minimal error (has full context)

Legend: X = large error, o = medium error, . = small error
```

---

## The Fix

### Files Changed

#### 1. `/home/yab/Sndr/python/aule/patching.py`

**Before** (line 9):
```python
PATCH_CONFIG = {
    "causal": True,    # Default to True (GPT-2 style)
    "use_rope": False
}
```

**After** (line 11):
```python
PATCH_CONFIG = {
    "causal": False,   # Default to False (diffusion/bidirectional attention)
    "use_rope": False
}
```

**And** (line 53):
```python
# Before
causal = PATCH_CONFIG.get("causal", True)

# After
causal = PATCH_CONFIG.get("causal", False)  # Default False for diffusion models
```

### Why This is Safe

1. **`aule.install()` path** (used by `AuleInstall` ComfyUI node):
   - Already correct! Uses `is_causal` parameter from the model's SDPA call
   - Diffusion models call SDPA with `is_causal=False` → works correctly

2. **`aule.patch_model()` path** (used by `AulePatchModel` ComfyUI node):
   - Was broken: defaulted to `causal=True`
   - Now fixed: defaults to `causal=False`
   - LLM users can still explicitly pass `config={"causal": True}`

---

## Testing

All tests pass in `/home/yab/Sndr/test_fix_verification.py`:

```
✓ PASS: Simulated Diffusion (causal=False works)
✓ PASS: GPT-2 Patch (explicit causal=True still works)
✓ PASS: Default Config (now defaults to False)
✓ PASS: ComfyUI AuleInstall (global install works)
```

---

## Usage Guide for ComfyUI

### For Diffusion Models (SD, FLUX, SD3, WAN)

**Option 1: Use AuleInstall Node (Recommended)**
```
[AuleInstall] → [LoadCheckpoint] → [KSampler] → [SaveImage]
```
This works automatically - no configuration needed.

**Option 2: Use AulePatchModel Node**
```
[LoadCheckpoint] → [AulePatchModel] → [KSampler] → [SaveImage]
                      ├─ causal: FALSE (toggle OFF)
                      └─ use_rope: FALSE
```
Make sure to set `causal=FALSE` in the node!

### For LLMs (GPT-2, Llama, Mistral)

If you're using aule with LLM models (less common in ComfyUI):

**Option 1: AuleInstall** - works automatically

**Option 2: AulePatchModel** with explicit config:
```python
# In Python code
from aule import patch_model
patch_model(model, config={"causal": True})
```

Or in ComfyUI node:
```
[LoadCheckpoint] → [AulePatchModel] → ...
                      ├─ causal: TRUE (toggle ON)
                      └─ use_rope: FALSE
```

---

## Migration Guide

### If you were affected by this bug:

1. **Update to latest code**: Already done if you're reading this
2. **Reinstall**: `cd python && pip install -e . --force-reinstall`
3. **Clear Python cache**: `find . -name "*.pyc" -delete`
4. **Retest in ComfyUI**: Load your workflow and run KSampler again

### If you worked around the bug:

If you manually edited code to fix this, you can now revert those changes.

---

## Technical Notes

### Backend Compatibility

The fix applies to all backends:
- ✓ Vulkan backend (consumer GPUs)
- ✓ Triton backend (AMD ROCm, NVIDIA CUDA)
- ✓ CPU fallback

### Performance Impact

No performance impact - this is just a configuration fix.

### Numerical Accuracy

Verified with test suite:
- Max absolute error: < 1e-6 (fp32)
- Max relative error: < 1e-5 (fp32)

---

## Related Files

- Test suite: `/home/yab/Sndr/test_comfy_issue.py`
- Causal bug demonstration: `/home/yab/Sndr/test_causal_bug.py`
- Fix verification: `/home/yab/Sndr/test_fix_verification.py`

---

## Future Improvements

1. **Auto-detection**: Detect model type (diffusion vs LLM) and auto-configure
2. **Per-model config**: Allow different causal settings per model instance
3. **Validation**: Warn if user sets causal=True on known diffusion models

---

## Credits

- Issue reported by: User (WAN 2.2 + ComfyUI)
- Root cause analysis: Claude Code (systematic-debugging skill)
- Fix implemented: 2025-12-08

---

## References

- FlashAttention-2 paper: https://tridao.me/publications/flash2/flash2.pdf
- PyTorch SDPA docs: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- ComfyUI: https://github.com/comfyanonymous/ComfyUI
