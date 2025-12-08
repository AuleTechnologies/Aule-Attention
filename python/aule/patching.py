
import torch
import warnings
# from . import flash_attention # Circular import fix: imported inside function

# Global configuration applied to all patches
# Ideally this would be per-model, but for ComfyUI single-model usage this is safe
# DEFAULT: causal=False for diffusion models (SD, FLUX, etc.)
# For LLMs (GPT-2, Llama), explicitly set causal=True when patching
PATCH_CONFIG = {
    "causal": False,   # Default to False (diffusion/bidirectional attention)
    "use_rope": False  # Default to False
}

def _aule_gpt2_forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, use_cache=False, output_attentions=False, past_key_values=None, **kwargs):
    """
    Monkey-patched forward pass for GPT2Attention using Aule FlashAttention.
    """
    # 1. QKV Projection (Standard GPT-2 logic)
    # [batch, seq, 3*embed_dim]
    qkv = self.c_attn(hidden_states)
    
    # Split Q, K, V
    query, key, value = qkv.split(self.embed_dim, dim=2)

    # 2. Reshape for Multi-head Attention
    # [batch, seq, embed_dim] -> [batch, heads, seq, head_dim]
    # We implement this manually to avoid relying on private _split_heads method
    
    batch_size = hidden_states.shape[0]
    seq_len = hidden_states.shape[1]
    
    new_shape = list(hidden_states.size()[:-1]) + [self.num_heads, self.head_dim]
    
    query = query.view(*new_shape).permute(0, 2, 1, 3)
    key = key.view(*new_shape).permute(0, 2, 1, 3)
    value = value.view(*new_shape).permute(0, 2, 1, 3)

    # 3. Aule FlashAttention
    # Check for cross-attention
    is_cross_attention = encoder_hidden_states is not None
    
    if is_cross_attention:
        warnings.warn("Aule: Cross-attention not supported yet, falling back to CPU/Standard path.")
        # We need the original class to call the original method.
        # This assumes the patcher stored 'original_forward' on the class.
        return self.__class__.original_forward(self, hidden_states, layer_past, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, use_cache, output_attentions, past_key_values, **kwargs)

    # Invoke Aule (Vulkan Backend)
    from . import flash_attention
    
    # Read config
    causal = PATCH_CONFIG.get("causal", False)  # Default False for diffusion models
    # TODO: Pass use_rope when shader supports it
    
    attn_output = flash_attention(query, key, value, causal=causal)
    
    # Ensure tensor output (Vulkan backend fix included in __init__.py but doubling safety here)
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

    # Handle return tuple
    present = layer_past if use_cache else None
    outputs = (attn_output, present)
    
    if output_attentions:
        # We don't calculate weights in FlashAttention, so return None
        outputs = outputs + (None,)

    return outputs

def _patch_gpt2(model):
    """Patch a GPT-2 model or model class."""
    import transformers.models.gpt2.modeling_gpt2 as modeling_gpt2
    
    target_class = modeling_gpt2.GPT2Attention
    
    # Check if already patched
    if getattr(target_class, "_aule_patched", False):
        print("Aule: GPT2Attention already patched.")
        return

    print("Aule: Patching GPT2Attention...")
    
    # Save original forward
    target_class.original_forward = target_class.forward
    target_class._aule_patched = True
    
    # Apply patch
    target_class.forward = _aule_gpt2_forward


def patch_model(model, config=None):
    """
    Automatically patch a Hugging Face model to use Aule FlashAttention.
    
    Args:
        model: A transformers.PreTrainedModel instance or class.
        config: Optional dict overriding defaults {"causal": bool, "use_rope": bool}
    
    Supported Models:
        - GPT-2
    """
    model_type = getattr(model.config, "model_type", None) if hasattr(model, "config") else None
    
    if config:
        print(f"Aule: Applying patch config: {config}")
        PATCH_CONFIG.update(config)
    
    if model_type == "gpt2":
        _patch_gpt2(model)
    else:
        # Fallback: Try to detect class name
        class_name = model.__class__.__name__.lower()
        if "gpt2" in class_name:
            _patch_gpt2(model)
        else:
            warnings.warn(f"Aule: Model type '{model_type}' (class {class_name}) not currently supported for automatic patching.")
