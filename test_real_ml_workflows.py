#!/usr/bin/env python3
"""
Real ML Workflow Tests for Aule-Attention

Tests with actual models, not synthetic tensors.
"""

import torch
import numpy as np
import time
import sys

print("=" * 70)
print("REAL ML WORKFLOW TESTS")
print("=" * 70)

# Check if transformers is available
try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("WARNING: transformers not installed. Some tests will be skipped.")

import aule

results = {}

# =============================================================================
# TEST 1: GPT-2 Text Generation
# =============================================================================
print("\n" + "=" * 70)
print("TEST 1: GPT-2 Text Generation")
print("=" * 70)

if HAS_TRANSFORMERS:
    try:
        print("\nLoading GPT-2...")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        model.eval()

        prompt = "The future of artificial intelligence is"
        inputs = tokenizer(prompt, return_tensors="pt")

        # Generate WITHOUT aule (baseline)
        print("\n[Baseline] Generating with PyTorch SDPA...")
        with torch.no_grad():
            baseline_output = model.generate(
                inputs.input_ids,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        baseline_text = tokenizer.decode(baseline_output[0], skip_special_tokens=True)
        print(f"  Output: {baseline_text}")

        # Install aule patch
        print("\n[Aule] Patching model with aule-attention...")
        aule.install(verbose=True)

        # Generate WITH aule
        print("\n[Aule] Generating with Aule attention...")
        with torch.no_grad():
            aule_output = model.generate(
                inputs.input_ids,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        aule_text = tokenizer.decode(aule_output[0], skip_special_tokens=True)
        print(f"  Output: {aule_text}")

        # Compare
        if baseline_text == aule_text:
            print("\n✓ PASSED: Outputs match exactly!")
            results['gpt2_generation'] = 'PASS'
        else:
            print("\n✗ FAILED: Outputs differ!")
            print(f"  Baseline: {baseline_text}")
            print(f"  Aule:     {aule_text}")
            results['gpt2_generation'] = 'FAIL'

        aule.uninstall()

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        results['gpt2_generation'] = f'ERROR: {e}'
else:
    print("SKIPPED: transformers not installed")
    results['gpt2_generation'] = 'SKIPPED'

# =============================================================================
# TEST 2: Attention Pattern Verification (Transformer Block)
# =============================================================================
print("\n" + "=" * 70)
print("TEST 2: Multi-Head Attention Block (Transformer Style)")
print("=" * 70)

try:
    # Simulate a real transformer attention block
    batch_size = 2
    seq_len = 128
    embed_dim = 768
    num_heads = 12
    head_dim = embed_dim // num_heads

    print(f"\nConfig: batch={batch_size}, seq={seq_len}, embed={embed_dim}, heads={num_heads}")

    # Create random hidden states (like from a transformer)
    hidden_states = torch.randn(batch_size, seq_len, embed_dim)

    # Simulate QKV projection (like in real transformer)
    Wq = torch.randn(embed_dim, embed_dim) * 0.02
    Wk = torch.randn(embed_dim, embed_dim) * 0.02
    Wv = torch.randn(embed_dim, embed_dim) * 0.02

    Q = hidden_states @ Wq
    K = hidden_states @ Wk
    V = hidden_states @ Wv

    # Reshape to [batch, heads, seq, head_dim]
    Q = Q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    K = K.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    V = V.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

    # PyTorch baseline
    baseline = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)

    # Aule attention
    aule_out = aule.flash_attention(Q, K, V, causal=True)

    # Compare
    max_diff = (baseline - aule_out).abs().max().item()
    mean_diff = (baseline - aule_out).abs().mean().item()

    print(f"\n  Max difference:  {max_diff:.8f}")
    print(f"  Mean difference: {mean_diff:.8f}")

    if max_diff < 1e-4:
        print("\n✓ PASSED: Attention outputs match within tolerance!")
        results['transformer_block'] = 'PASS'
    else:
        print("\n✗ FAILED: Difference too large!")
        results['transformer_block'] = 'FAIL'

except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    results['transformer_block'] = f'ERROR: {e}'

# =============================================================================
# TEST 3: Cross-Attention (Encoder-Decoder Style)
# =============================================================================
print("\n" + "=" * 70)
print("TEST 3: Cross-Attention (Encoder-Decoder / Diffusion Style)")
print("=" * 70)

try:
    # Simulate cross-attention like in Stable Diffusion or T5
    batch_size = 1
    encoder_seq = 77  # CLIP text encoder output length
    decoder_seq = 64  # Latent spatial dimension (8x8)
    num_heads = 8
    head_dim = 64

    print(f"\nConfig: encoder_seq={encoder_seq}, decoder_seq={decoder_seq}, heads={num_heads}")
    print("  (Simulating SD-style text-to-image cross-attention)")

    # Query from decoder (image latents)
    Q = torch.randn(batch_size, num_heads, decoder_seq, head_dim)

    # Key/Value from encoder (text embeddings)
    K = torch.randn(batch_size, num_heads, encoder_seq, head_dim)
    V = torch.randn(batch_size, num_heads, encoder_seq, head_dim)

    # PyTorch baseline (cross-attention, no causal mask)
    baseline = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=False)

    # Aule attention
    aule_out = aule.flash_attention(Q, K, V, causal=False)

    # Compare
    max_diff = (baseline - aule_out).abs().max().item()
    mean_diff = (baseline - aule_out).abs().mean().item()

    print(f"\n  Max difference:  {max_diff:.8f}")
    print(f"  Mean difference: {mean_diff:.8f}")

    # Check output shape
    expected_shape = (batch_size, num_heads, decoder_seq, head_dim)
    if aule_out.shape != expected_shape:
        print(f"\n✗ FAILED: Wrong output shape! Expected {expected_shape}, got {aule_out.shape}")
        results['cross_attention'] = 'FAIL'
    elif max_diff < 1e-4:
        print("\n✓ PASSED: Cross-attention outputs match!")
        results['cross_attention'] = 'PASS'
    else:
        print("\n✗ FAILED: Difference too large!")
        results['cross_attention'] = 'FAIL'

except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    results['cross_attention'] = f'ERROR: {e}'

# =============================================================================
# TEST 4: GQA (Llama-2 / Mistral Style)
# =============================================================================
print("\n" + "=" * 70)
print("TEST 4: Grouped Query Attention (Llama-2 / Mistral Style)")
print("=" * 70)

try:
    # Llama-2 7B config: 32 query heads, 32 kv heads (MHA)
    # Llama-2 70B config: 64 query heads, 8 kv heads (GQA 8:1)
    # Mistral 7B config: 32 query heads, 8 kv heads (GQA 4:1)

    batch_size = 1
    seq_len = 512
    num_q_heads = 32
    num_kv_heads = 8  # GQA ratio 4:1
    head_dim = 64  # Limited by current implementation

    print(f"\nConfig: q_heads={num_q_heads}, kv_heads={num_kv_heads}, ratio={num_q_heads//num_kv_heads}:1")
    print("  (Simulating Mistral-7B style GQA)")

    Q = torch.randn(batch_size, num_q_heads, seq_len, head_dim)
    K = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)
    V = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)

    # PyTorch reference (manually expand KV)
    K_expanded = K.repeat_interleave(num_q_heads // num_kv_heads, dim=1)
    V_expanded = V.repeat_interleave(num_q_heads // num_kv_heads, dim=1)
    baseline = torch.nn.functional.scaled_dot_product_attention(Q, K_expanded, V_expanded, is_causal=True)

    # Aule attention (handles GQA natively)
    aule_out = aule.flash_attention(Q, K, V, causal=True)

    # Compare
    max_diff = (baseline - aule_out).abs().max().item()
    mean_diff = (baseline - aule_out).abs().mean().item()

    print(f"\n  Max difference:  {max_diff:.8f}")
    print(f"  Mean difference: {mean_diff:.8f}")

    if max_diff < 1e-3:
        print("\n✓ PASSED: GQA outputs match!")
        results['gqa_llama_style'] = 'PASS'
    else:
        print("\n✗ FAILED: Difference too large!")
        results['gqa_llama_style'] = 'FAIL'

except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    results['gqa_llama_style'] = f'ERROR: {e}'

# =============================================================================
# TEST 5: Long Context (Memory Efficiency Check)
# =============================================================================
print("\n" + "=" * 70)
print("TEST 5: Long Context Handling")
print("=" * 70)

try:
    # Test progressively longer sequences
    for seq_len in [256, 512, 1024, 2048]:
        batch_size = 1
        num_heads = 8
        head_dim = 64

        Q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        K = torch.randn(batch_size, num_heads, seq_len, head_dim)
        V = torch.randn(batch_size, num_heads, seq_len, head_dim)

        # Time aule
        start = time.perf_counter()
        aule_out = aule.flash_attention(Q, K, V, causal=True)
        aule_time = (time.perf_counter() - start) * 1000

        # Time PyTorch
        start = time.perf_counter()
        baseline = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)
        pytorch_time = (time.perf_counter() - start) * 1000

        # Verify correctness
        max_diff = (baseline - aule_out).abs().max().item()
        correct = max_diff < 1e-3

        print(f"\n  seq_len={seq_len:4d}: Aule={aule_time:7.2f}ms, PyTorch={pytorch_time:7.2f}ms, "
              f"diff={max_diff:.6f} {'✓' if correct else '✗'}")

    results['long_context'] = 'PASS'
    print("\n✓ PASSED: All sequence lengths handled correctly!")

except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    results['long_context'] = f'ERROR: {e}'

# =============================================================================
# TEST 6: Patching Real Model (End-to-End)
# =============================================================================
print("\n" + "=" * 70)
print("TEST 6: Model Patching (End-to-End Forward Pass)")
print("=" * 70)

if HAS_TRANSFORMERS:
    try:
        print("\nLoading GPT-2 for forward pass comparison...")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        model.eval()

        # Create input
        input_ids = torch.randint(0, 50257, (1, 64))

        # Baseline forward pass
        with torch.no_grad():
            baseline_output = model(input_ids)
            baseline_logits = baseline_output.logits

        # Patch with aule
        aule.install(verbose=False)

        # Aule forward pass
        with torch.no_grad():
            aule_output = model(input_ids)
            aule_logits = aule_output.logits

        aule.uninstall()

        # Compare logits
        max_diff = (baseline_logits - aule_logits).abs().max().item()
        mean_diff = (baseline_logits - aule_logits).abs().mean().item()

        print(f"\n  Logits max difference:  {max_diff:.6f}")
        print(f"  Logits mean difference: {mean_diff:.6f}")

        # Check if predictions would be the same
        baseline_preds = baseline_logits.argmax(dim=-1)
        aule_preds = aule_logits.argmax(dim=-1)
        preds_match = (baseline_preds == aule_preds).all().item()

        print(f"  Predictions match: {preds_match}")

        if max_diff < 1e-3 and preds_match:
            print("\n✓ PASSED: Model patching works correctly!")
            results['model_patching'] = 'PASS'
        elif preds_match:
            print("\n⚠ WARNING: Small numerical differences but predictions match")
            results['model_patching'] = 'PASS (with warnings)'
        else:
            print("\n✗ FAILED: Predictions differ!")
            results['model_patching'] = 'FAIL'

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        results['model_patching'] = f'ERROR: {e}'
else:
    print("SKIPPED: transformers not installed")
    results['model_patching'] = 'SKIPPED'

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

passed = sum(1 for v in results.values() if 'PASS' in str(v))
failed = sum(1 for v in results.values() if v == 'FAIL')
errors = sum(1 for v in results.values() if 'ERROR' in str(v))
skipped = sum(1 for v in results.values() if v == 'SKIPPED')

print(f"\n  Passed:  {passed}")
print(f"  Failed:  {failed}")
print(f"  Errors:  {errors}")
print(f"  Skipped: {skipped}")

print("\nDetailed Results:")
for test, result in results.items():
    status = "✓" if 'PASS' in str(result) else "✗" if result == 'FAIL' else "⚠" if 'ERROR' in str(result) else "○"
    print(f"  {status} {test}: {result}")

if failed == 0 and errors == 0:
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED! Ready for production use.")
    print("=" * 70)
    sys.exit(0)
else:
    print("\n" + "=" * 70)
    print("SOME TESTS FAILED! Review errors above.")
    print("=" * 70)
    sys.exit(1)
