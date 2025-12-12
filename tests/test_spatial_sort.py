
import numpy as np
import pytest
from aule.vulkan import Aule

@pytest.fixture(scope="module")
def aule_ctx():
    """Shared Aule context for all tests."""
    with Aule() as ctx:
        yield ctx

@pytest.mark.xfail(reason="Radix sort currently uses magnitude-based keys which may not match simple argsort")
def test_spatial_sort_basic(aule_ctx):
    """Test standard spatial sorting against numpy argsort."""
    # Shader currently implements local bitonic sort (wg=256).
    # Testing with S=256 to verify correctness of the kernel.
    B, H, S, D = 1, 1, 256, 64
    sort_dim = 0
    
    # Random keys
    keys = np.random.randn(B, H, S, D).astype(np.float32)
    values = np.random.randn(B, H, S, D).astype(np.float32)
    
    # Get Aule result
    indices = aule_ctx.spatial_sort(keys, values, sort_dim=sort_dim)
    
    assert indices.shape == (B, H, S)
    assert indices.dtype == np.uint32
    
    # Verify with Numpy
    # Projection onto sort_dim is just keys[..., sort_dim]
    projections = keys[..., sort_dim] # [B, H, S]
    
    # Numpy argsort
    ref_indices = np.argsort(projections, axis=-1)
    
    # Check if indices are valid (should contain all 0..S-1)
    for b in range(B):
        for h in range(H):
            idx_set = set(indices[b,h])
            assert len(idx_set) == S, "Indices must be a permutation!"
            
    
    # Verify using flattened arrays (since indices are global)
    keys_flat = keys.reshape(-1, D)
    indices_flat = indices.reshape(-1)
    
    # Check that indices are a permutation of 0..S-1 within each 256-block?
    # No, indices are global pointers.
    # Just check if keys_flat[indices_flat] is locally sorted (per 256 block).
    
    sorted_flat = keys_flat[indices_flat]
    projections_flat = sorted_flat[:, sort_dim]
    
    # Verify local sortedness (chunks of 256)
    for i in range(0, len(projections_flat), 256):
        chunk = projections_flat[i : i+256]
        is_sorted_chunk = np.all(chunk[1:] >= chunk[:-1])
        if not is_sorted_chunk:
             print(f"Chunk {i//256} not sorted!")
             print(chunk[:10])
             assert False, "Chunk not sorted"

@pytest.mark.xfail(reason="Radix sort currently only supports B=1, H=1 - segmented sort not yet implemented")
def test_spatial_sort_multidim(aule_ctx):
    """Test sorting on a different dimension."""
    # Use S=256 again for valid checking with current shader
    B, H, S, D = 2, 4, 256, 64
    sort_dim = 32
    
    keys = np.random.randn(B, H, S, D).astype(np.float32)
    values = np.random.randn(B, H, S, D).astype(np.float32)
    
    indices = aule_ctx.spatial_sort(keys, values, sort_dim=sort_dim)
    
    keys_flat = keys.reshape(-1, D)
    indices_flat = indices.reshape(-1)
    
    sorted_flat = keys_flat[indices_flat]
    projections = sorted_flat[:, sort_dim]
    
    for i in range(0, len(projections), 256):
        chunk = projections[i : i+256]
        assert np.all(chunk[1:] >= chunk[:-1])

if __name__ == "__main__":
    # Manually run if executed as script
    with Aule() as ctx:
        print("Running manual test...")
        test_spatial_sort_basic(ctx)
        test_spatial_sort_multidim(ctx)
        print("Tests passed!")
