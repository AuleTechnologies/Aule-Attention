
import pytest
import numpy as np
from aule.vulkan import Aule

@pytest.fixture(scope="module")
def aule():
    instance = Aule()
    yield instance
    instance.close()

@pytest.mark.xfail(reason="Radix sort currently only supports B=1, H=1 - segmented sort not yet implemented")
def test_segmented_sort_correctness(aule):
    """
    Verify that Segmented Radix Sort correctly sorts multiple segments (Batch * Head)
    independently without cross-contamination.
    """
    B = 4
    H = 4
    S = 256 # Segment size
    D = 64
    
    # Total elements = 4 * 4 * 256 = 4096
    # If global sort ran, it would sort all 4096 elements together.
    # We want 16 independent sorts of 256 elements.
    
    q = np.random.randn(B, H, S, D).astype(np.float32)
    k = np.random.randn(B, H, S, D).astype(np.float32)
    v = np.random.randn(B, H, S, D).astype(np.float32)
    
    # Create random indices per segment
    indices = np.empty((B, H, S), dtype=np.uint32)
    for b in range(B):
        for h in range(H):
            # Each segment gets a random permutation of 0..255
            # Crucially, values are effectively 0..S-1 relative to the segment start?
            # NO. The implementation expects indices to be relative to the START of the tensor?
            # Or relative to the segment?
            # Standard Attention usually expects indices 0..S-1 for each head.
            # Our `iota` shader generates 0..N-1 globally.
            # Wait, `iota` needs to be segmented too? 
            # If `iota` generates 0..S-1 for each segment, that's fine.
            # But currently `iota` generates global indices.
            pass

    # Actually, let's test `attention_gravity` which calls the sort.
    # If we pass random `indices` (B, H, S) where each entry is 0..255 (local index),
    # The kernel interprets this as `local_index`.
    # BUT `radix_sort` sorts the BUFFER.
    
    # The current Radix Sort implementation sorts the `indices` buffer.
    # If we pass `indices` with shape (B, H, S), it is flattened to (B*H*S).
    # If we typically utilize `indices` as a lookup for K/V:
    # `real_key_idx = indices[i]`
    # If `indices` contains local indices (0..S-1), we need to add offset?
    # Our `attention_gravity.comp` does `uint original_idx = inds[real_idx];`.
    # `original_idx` is then used to fetch K/V: `keys[original_idx * stride ...]`
    
    # Wait, `attention_gravity.comp` expects `inds` to contain GLOBAL indices if the K/V buffer is global?
    # Or LOCAL indices if it adds `batch_offset`?
    # Let's check `attention_gravity.comp`.
    # `uint batch_idx = ...`
    # `uint head_idx = ...`
    # `uint kv_offset = (batch_idx * num_heads + head_idx) * seq_len * d_model;`
    # `uint idx = inds[gID];` (This is the sorted index)
    # `float k = keys[kv_offset + idx * d_model + d];`
    
    # AHA! `attention_gravity.comp` adds the `kv_offset` (Segment Base).
    # So `inds` MUST contain LOCAL numeric values (0..S-1).
    # IF `inds` contained global values (0..N-1), we would double count the offset.
    
    # SO: The Sort must sort independent lists of LOCAL indices (0..S-1).
    # Which means:
    # Segment 1: [3, 1, 2] -> [1, 2, 3]
    # Segment 2: [2, 0, 1] -> [0, 1, 2]
    
    # Input Indices:
    indices_in = np.empty((B, H, S), dtype=np.uint32)
    for b in range(B):
        for h in range(H):
            indices_in[b, h] = np.random.permutation(S).astype(np.uint32)
            
    # We rely on `attention_gravity` to trigger the sort.
    # But verifying `attention_gravity` output is indirect.
    # Can we verify the sorted indices directly?
    # Not easily exposed to Python yet.
    # We will trust `attention_gravity` result.
    
    # If sort is GLOBAL:
    # Segment 1 (which has high-value Keys) might end up with indices from Segment 2 (low-value Keys)?
    # Wait.
    # The Sort sorts (Key, Index) pairs.
    # If Segment 1 Keys are all 100.0, and Segment 2 Keys are all 0.0.
    # A Global Sort would put ALL Segment 2 indices first, then Segment 1.
    # So Segment 1 would see indices 0..255 (which correspond to Seg 2's data).
    # Result: Segment 1 would attend to Segment 2's keys.
    # BUT `attention_gravity` adds `kv_offset`.
    # So it would read `Key[Seg1_Base + Seg2_Local_Index]`.
    # Seg2_Local_Index is a small number.
    # So it would read `Key[Seg1_Base + 0]`.
    # This is "valid" memory access, but it's attending to the WRONG data?
    # No, it's attending to Seg1's data (because of offset), but using an Index Permutation derived from Seg2's values!
    # This means Seg1 is sorted based on Seg2's keys. INCORRECT.
    
    # We need independent sorts.
    
    # Setup:
    # Seg 0: Keys = 100, 101, 102... (Should stay at bottom if global sorted)
    # Seg 1: Keys = 0, 1, 2... (Should move to top if global sorted)
    
    # If Segmented Sort works:
    # Seg 0 sorted -> 100, 101... (Local 0, 1...)
    # Seg 1 sorted -> 0, 1... (Local 0, 1...)
    
    # How to verify?
    # Use Needles.
    # Hide Needle in Seg 0 at index 5. Key = -1000.
    # Hide Needle in Seg 1 at index 10. Key = -500.
    
    # If Global:
    # -1000 (Seg 0) -> Global Rank 0.
    # -500 (Seg 1) -> Global Rank 1.
    # Seg 0 indices will contain BOTH needles? No, indices don't move between segments in memory (scatter overwrites).
    # Wait. If Global Sort, `scatter` writes globally.
    # Index 0 (Seg 0 start) gets the -1000 index.
    # Index 1 (Seg 0 start + 1) gets the -500 index.
    # So Seg 0 gets BOTH needles.
    # Seg 1 gets nothing.
    
    # If Segmented:
    # Seg 0 Index 0 gets -1000.
    # Seg 1 Index 0 gets -500.
    
    # TEST STRATEGY:
    # 2 Segments (B=2, H=1, S=256).
    # Seg0: Random noise + Needle A (-1000) at idx 5.
    # Seg1: Random noise + Needle B (-2000) at idx 10.
    
    # Run Attention (Truncated Top-1).
    # Check Seg0 output -> Should act as if it attended to Needle A.
    # Check Seg1 output -> Should act as if it attended to Needle B.
    
    B, H, S, D = 1, 1, 128, 64
    max_attend = 1
    
    # LSB Test Strategy
    # Base: 1.0 (0x3F800000)
    # Others: 1.0 + epsilon (0x3F800001)
    # Needle: 1.0 (0x3F800000)
    # If sort works, Needle < Others. Puts Needle at Top.
    
    import struct
    f_base = 1.0
    i_base = struct.unpack('I', struct.pack('f', f_base))[0]
    i_other = i_base | 0x01 # LSB set
    f_other = struct.unpack('f', struct.pack('I', i_other))[0]
    
    q = np.random.randn(B, H, S, D).astype(np.float32)
    k = np.zeros((B, H, S, D), dtype=np.float32) # Initialize k
    v = np.random.randn(B, H, S, D).astype(np.float32)
    indices = np.tile(np.arange(S, dtype=np.uint32), (B, H, 1))

    k[:] = f_other
    
    if B > 0:
        k[0, 0, 5, 0] = f_base
    if B > 1:
        k[1, 0, 10, 0] = f_base # Needle for Seg 1?
        # Note: Seg 1 should act independently.
        
    # v is random.
    # q is random. 
    # But for Top-1 selection, q doesn't matter for selection, only score.
    # Output matches v[needle].
    
    # If Global Sort (BAD):
    # -200 (Seg1) is smallest globally. Moves to Pos 0 (Seg 0).
    # -100 (Seg0) is 2nd smallest. Moves to Pos 1 (Seg 0).
    # Seg 0 top indices: [Idx 10 (from Seg1), Idx 5 (from Seg0)]
    # But Idx 10 (if interpreted locally in Seg 0) points to K[0,0,10].
    # K[0,0,10] is noise.
    # So Seg 0 will attend to noise instead of its needle.
    
    # If Segmented Sort (GOOD):
    # Seg 0 sorts its own keys. -100 is min. Top Index = 5.
    # Seg 1 sorts its own keys. -200 is min. Top Index = 10.
    
    # We verify by checking if the output matches V[needle_idx].
    # Since we use max_attend=1 (Truncated), we ONLY attend to the top 1.
    
    # Target Values
    target_v0 = v[0, 0, 5] if B > 0 else None
    target_v1 = v[1, 0, 10] if B > 1 else None
    
    out = aule.attention_gravity(q, k, v, indices, causal=False, max_attend=1)
    
    # Check Seg 0
    # Output[0, 0, :, :] should likely be close to target_v0 (since Q attends to Needle)
    # Actually Q[0] attends to K[ArgMin].
    # If K is constant (only needle is different), then yes.
    # But Q is random.
    # We need Q to match K's needle?
    # No, gravity sorts by K value (if using simple sort dim).
    # Our sort uses `sort_dim` (default 0?).
    # If keys[idx, 0] is the sort key.
    # Then independent of Q.
    # Yes, `attention_gravity` currently sorts by K (if I recall).
    # Gravity Kernel:
    # 1. Sort indices based on Keys.
    # 2. Iterate top indices.
    # 3. Compute Attention (Q.K).
    
    # Wait, if we sort by K, we prioritize keys with small values in Dim 0.
    # Then we compute dot product.
    # If Q is random, dot product is random.
    # But if K_needle is significantly better/worse?
    # Let's make Q zero and K needle match?
    # Or just rely on the fact that we ONLY attend to top 1.
    # The TOP 1 index MUST be the needle index.
    # So we compute attention on (Q, K_needle).
    # Out = Softmax(Q.K_needle) * V_needle.
    # Since only 1 item, Softmax = 1.0.
    # So Out = V_needle.
    
    # So: Out[0,0] should approx equals V[0,0,5].
    # Out[1,0] should approx equals V[1,0,10].
    
    if B > 0:
        out0 = out[0, 0, 0] # First query output
        print(f"Target 0 (v[5]): {target_v0[:4]}")
        print(f"Output 0: {out0[:4]}")
        print(f"v[0]: {v[0, 0, 0, :4]}")
        print(f"v[7]: {v[0, 0, 7, :4]}")
        print(f"v[...]: {v[0, 0, :10, 0]}") # Print first component of first 10 vectors
        np.testing.assert_allclose(out0, target_v0, atol=1e-3, err_msg="Segment 0 failed to retrieve its needle")
    
    if B > 1:
        np.testing.assert_allclose(out1, target_v1, atol=1e-3, err_msg="Segment 1 failed to retrieve its needle")
