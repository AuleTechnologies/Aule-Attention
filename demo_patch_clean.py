import torch
import aule
from transformers import GPT2Model, GPT2Config

print(f"DEBUG: aule file: {aule.__file__}")

def demo_clean():
    print("--- Aule Clean Patch Demo ---")
    
    # 1. Initialize Model (Using small config for speed/offline safety)
    print("Initializing GPT-2...")
    config = GPT2Config(n_embd=256, n_head=4, n_layer=2)
    model = GPT2Model(config)
    model.eval()

    # 2. THE MAGIC LINE
    print("Applying Aule patch...")
    aule.patch_model(model)
    
    # 3. Run Forward Pass
    print("Running forward pass...")
    input_ids = torch.randint(0, 1000, (1, 32)) 
    
    try:
        with torch.no_grad():
            outputs = model(input_ids)
        
        print(f"Success! Output shape: {outputs.last_hidden_state.shape}")
        
    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_clean()
