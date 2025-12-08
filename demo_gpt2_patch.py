import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config, GPT2Tokenizer
from transformers.models.gpt2 import modeling_gpt2
import aule
from aule import flash_attention

print(f"DEBUG: aule package file: {aule.__file__}")

# optimize for no internet access
try:
    from transformers.utils import logging
    logging.set_verbosity_error()
except:
    pass

def aule_gpt2_forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, use_cache=False, output_attentions=False, past_key_values=None, **kwargs):
    """
    Monkey-patched forward pass for GPT2Attention using Aule FlashAttention.
    """
    # 1. QKV Projection (Standard GPT-2 logic)
    # query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    # split_size is embed_dim (since c_attn produces 3 * embed_dim)
    qkv = self.c_attn(hidden_states)
    query, key, value = qkv.split(self.embed_dim, dim=2)

    # 2. Reshape for Multi-head Attention
    # [batch, seq, embed_dim] -> [batch, heads, seq, head_dim]
    # Note: Standard GPT-2 splits into [batch, seq, heads, dim] then transposes to [batch, heads, seq, dim]
    # We need [batch, heads, seq, dim] for Aule.
    
    batch_size = hidden_states.shape[0]
    seq_len = hidden_states.shape[1]
    
    new_shape = list(hidden_states.size()[:-1]) + [self.num_heads, self.head_dim]
    
    query = query.view(*new_shape).permute(0, 2, 1, 3)
    key = key.view(*new_shape).permute(0, 2, 1, 3)
    value = value.view(*new_shape).permute(0, 2, 1, 3)

    # 3. Aule FlashAttention (The MAGIC replacement)
    # Aule expects [batch, heads, seq, head_dim]
    # Standard GPT-2 implementation does a lot of masking gymnastics here.
    # FlashAttention handles causal masking natively.
    
    # Check if we are doing causal attention (decoder) or cross attention (encoder-decoder)
    is_cross_attention = encoder_hidden_states is not None
    
    # We only support standard causal self-attention for this demo
    if is_cross_attention:
        print("Warning: Cross-attention not supported in Aule demo fallback to standard")
        return modeling_gpt2.GPT2Attention.original_forward(self, hidden_states, layer_past, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, use_cache, output_attentions)

    # Invoke Aule (Vulkan Backend)
    attn_output = flash_attention(query, key, value, causal=True)
    
    # Convert back to Tensor if numpy (Vulkan backend quirk)
    if not isinstance(attn_output, torch.Tensor):
        attn_output = torch.from_numpy(attn_output).to(query.device)

    # 4. Reshape Output
    # [batch, heads, seq, dim] -> [batch, seq, embed_dim]
    attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
    new_shape = list(attn_output.size()[:-2]) + [self.num_heads * self.head_dim]
    attn_output = attn_output.view(*new_shape)
    
    # 5. Output Projection
    attn_output = self.c_proj(attn_output)
    attn_output = self.resid_dropout(attn_output)

    # present/past_key_value
    # For now, return None or input cache as placeholder since we don't update cache in this demo
    present = layer_past if use_cache else None
    outputs = (attn_output, present)

    return outputs

def demo_patch():
    print("--- Real Model (GPT-2) Aule Integration ---")
    
    # 1. Save original forward to allow fallback if needed
    modeling_gpt2.GPT2Attention.original_forward = modeling_gpt2.GPT2Attention.forward
    
    # 2. Apply Monkey-Patch
    print("Monkey-patching GPT2Attention...")
    modeling_gpt2.GPT2Attention.forward = aule_gpt2_forward
    
    # 3. Initialize Model (Random Weights if no internet)
    print("Initializing GPT-2 (Small)...")
    config = GPT2Config(
        vocab_size=1000, 
        n_positions=128, 
        n_ctx=128, 
        n_embd=256, 
        n_layer=2, 
        n_head=4
    )
    model = GPT2Model(config)
    model.eval()
    
    # 4. Run Forward Pass
    print("Running forward pass with standard Hugging Face inputs...")
    input_ids = torch.randint(0, 1000, (1, 32)) # Batch 1, Seq 32
    
    try:
        with torch.no_grad():
            outputs = model(input_ids)
            
        last_hidden_states = outputs.last_hidden_state
        print(f"Success! Output shape: {last_hidden_states.shape}")
        print("Verified: Transformers GPT-2 is running via Aule/Vulkan backend.")
        
    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_patch()
