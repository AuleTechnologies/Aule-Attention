
import pytest
import numpy as np
import torch
from aule.vulkan import Aule

def ref_attention(q, k, v):
    """Simple reference attention for validation."""
    # Scale
    d_head = q.shape[-1]
    scale = 1.0 / np.sqrt(d_head)
    
    # Dot Product
    # (B, H, 1, D) @ (B, H, D, S) -> (B, H, 1, S)
    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * scale
    
    # Softmax
    scores_exp = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    probs = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)
    
    # Weighted Sum
    # (B, H, 1, S) @ (B, H, S, D) -> (B, H, 1, D)
    output = np.matmul(probs, v)
    return output

def test_needle_retrieval():
    """
    Needle in a Haystack Test.
    Goal: Retrieve a specific 'needle' key from a large sequence using truncated attention.
    
    Setup:
    - N = 1024     # Sequence length (Global sort needed for > 256)
    - K_trunc = 32 (Truncation Limit) -> 8x sparse compression
    - Needle is hidden at a random position.
    - Needle is marked by a very low value in feature[0] (for sorting).
    - Needle matches Query strongly (for attention score).
    """
    
    B, H, N, D = 1, 1, 1024, 64
    truncated_k = 32
    sort_dim = 0
    
    print(f"\n--- Starting Needle Test (N={N}, Top-{truncated_k}) ---")

    # 1. Generate Haystack (Noise)
    # Keys and Values are random noise
    k = np.random.randn(B, H, N, D).astype(np.float32)
    v = np.random.randn(B, H, N, D).astype(np.float32)
    
    # Query: We pick a random target vector
    q = np.random.randn(B, H, 1, D).astype(np.float32)
    # Normalize Q for cleaner math (optional)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)

    # 2. Insert Needle
    # Pick a random position to hide the needle
    needle_idx = np.random.randint(0, N)
    print(f"Hiding needle at index: {needle_idx}")

    # Make Needle Key match Query (high dot product)
    # We copy Q into K[needle_idx]
    k[:, :, needle_idx, :] = q[:, :, 0, :] * 100.0 # Amplify signal massive amount
    
    # Make Needle "Relevant" for Sorting
    # Set the sorting dimension to a very low value (-100)
    # effectively putting it at the "start" of the sorted list
    k[:, :, needle_idx, sort_dim] = -100.0

    # Set Needle Value to a distinct pattern (e.g., all 1s * 10)
    target_value = np.ones((D,), dtype=np.float32) * 10.0
    v[:, :, needle_idx, :] = target_value

    # 3. Setup Aule
    with Aule() as ctx:
        # A. Run Spatial Sort
        # This should produce indices where indices[0] == needle_idx (mostly)
        print("Running Spatial Sort...")
        indices = ctx.spatial_sort(k, v, sort_dim=sort_dim)
        
        # Verify Sorting (Optional debugging)
        # The first index should be our needle_idx
        top_index = indices[0, 0, 0]
        print(f"Top 1 sorted index: {top_index}")
        if top_index != needle_idx:
            print(f"WARNING: Needle not at top! K[val] = {k[0,0,needle_idx, sort_dim]}")
            # Check what is at top
            print(f"Value at top index {top_index}: {k[0,0,top_index, sort_dim]}")
            print(f"Indices sample (first 10): {indices[0, 0, :10]}")

        assert top_index == needle_idx, "Sorting failed to bring needle to top!"

        # B. Run Gravity Attention (Truncated)
        # We only attend to the first 32 sorted keys.
        # Since our needle is at index 0 of the *sorted* list, it will be in the pool.
        print("Running Truncated Gravity Attention...")
        # Note: attention_gravity expects 4D Q, K, V and 3D indices
        # Q needs expanding to (B, H, 1, D) if not already? It is.
        # But attention_gravity python wrapper takes (B, H, S, D).
        # Our Q is (1, 1, 1, 64). S=1.
        # K/V are (1, 1, 1024, 64). S=1024.
        
        # Wait, attention_gravity assumes Q_seq_len matches or broadcasts?
        # Shader: `params.seq_len` comes from Q shape.
        # `params.key_seq_len` comes from K shape.
        # This setup (decoding/querying) is supported.
        
        output = ctx.attention_gravity(
            q, k, v, 
            indices, 
            max_attend=truncated_k,
            causal=False
        )
        
        print("Output shape:", output.shape)
        
    # 4. Verify Result
    # The output should be very close to the Needle Value (all 10s)
    # acting like a retrieval.
    # Scores: Needle should have massive score compared to noise.
    # Softmax should be near 1.0 for needle, 0.0 for others.
    
    result_vec = output[0, 0, 0]
    expected_vec = target_value
    
    print("Result sample:", result_vec[:4])
    print("Target sample:", expected_vec[:4])
    
    # Check MSE or Cosine Sim
    mse = np.mean((result_vec - expected_vec)**2)
    print(f"MSE: {mse}")
    
    assert mse < 0.1, f"Retrieval failed! MSE {mse} too high."
    print("SUCCESS: Needle retrieved with Truncated Attention!")

if __name__ == "__main__":
    test_needle_retrieval()
