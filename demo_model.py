import torch
import torch.nn as nn
from aule import flash_attention, print_backend_info

class AuleAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x: [batch, seq_len, embed_dim]
        batch, seq_len, _ = x.shape
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention: [batch, heads, seq, head_dim]
        # aule-attention expects [batch, heads, seq, head_dim] layout
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Use Aule FlashAttention
        # Automatically selects Vulkan backend on Intel GPU
        attn_out = flash_attention(q, k, v, causal=True)

        # Reshape back: [batch, seq, embed_dim]
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        
        return self.out_proj(attn_out)

def demo():
    print("--- Aule Model Integration Demo ---")
    print_backend_info()

    # Configuration
    batch_size = 2
    seq_len = 128
    embed_dim = 256 # 4 heads * 64 dim
    num_heads = 4

    print(f"\nConfiguration:\n  Batch: {batch_size}\n  Seq Len: {seq_len}\n  Embed Dim: {embed_dim}\n  Heads: {num_heads}")

    # Initialize model
    model = AuleAttention(embed_dim, num_heads)
    model.eval() # Inference mode

    # Create dummy input
    x = torch.randn(batch_size, seq_len, embed_dim)

    print("\nRunning forward pass...")
    try:
        with torch.no_grad():
            output = model(x)
        
        print(f"\nSuccess! Output shape: {output.shape}")
        print("Aule-Attention successfully integrated into PyTorch model.")
        
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo()
