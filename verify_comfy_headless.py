
import sys
import os
import torch
import importlib.util

# 1. Setup paths
current_dir = os.getcwd()
aule_path = os.path.join(current_dir, "python")
comfy_path = os.path.join(current_dir, "ComfyUI")
custom_nodes_path = os.path.join(comfy_path, "custom_nodes")

sys.path.append(aule_path)
sys.path.append(comfy_path)

print(f"Added paths: {aule_path}, {comfy_path}")

# 2. Import Aule (to check pre-patch state)
import aule
import aule.vulkan

print("Initial Torch Attention:", torch.nn.functional.scaled_dot_product_attention)

# 3. Import ComfyUI Custom Node
# We simulate how ComfyUI imports nodes
spec = importlib.util.spec_from_file_location(
    "aule_custom_node", 
    os.path.join(custom_nodes_path, "aule-attention", "comfy_node.py")
)
aule_node_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aule_node_module)

print("Imported Aule Custom Node module")

# 4. Instantiate and Run AuleInstall Node
AuleInstall = aule_node_module.AuleInstall
install_node = AuleInstall()

print("Running AuleInstall.install()...")
install_node.install()

# 5. Verify Patch
print("Post-Install Torch Attention:", torch.nn.functional.scaled_dot_product_attention)

# Check if it's the custom wrapper
if aule._original_sdpa == torch.nn.functional.scaled_dot_product_attention:
    print("FAILURE: torch.nn.functional.scaled_dot_product_attention was not patched!")
    sys.exit(1)

# 6. Run Actual Attention (Integration Test)
print("\nRunning Test Attention Operation...")
try:
    # Test Cross-Attention case (Q_len != K_len)
    B, H, D = 1, 4, 64
    Q_N, K_N = 16, 32 # Different lengths
    
    q = torch.randn(B, H, Q_N, D)
    k = torch.randn(B, H, K_N, D)
    v = torch.randn(B, H, K_N, D)
    
    print(f"Testing Cross-Attention: Q={q.shape}, K={k.shape}")
    
    # This should now go through aule's vulkan backend
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    print("Attention Output Shape:", out.shape)
    
    if out.shape != (B, H, Q_N, D):
         raise ValueError(f"Output shape mismatch! Expected {(B, H, Q_N, D)}, got {out.shape}")
         
    print("SUCCESS: Cross-Attention ran without error via patched function.")
except Exception as e:
    print(f"FAILURE: Attention execution failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 7. Uninstall
uninstall_node = aule_node_module.AuleUninstall()
uninstall_node.uninstall()

if aule._original_sdpa != torch.nn.functional.scaled_dot_product_attention:
    print("FAILURE: Uninstall did not restore original SDPA!")
    sys.exit(1)

print("\nINTEGRATION VERIFIED: ComfyUI node successfully patches and runs attention.")
sys.exit(0)
