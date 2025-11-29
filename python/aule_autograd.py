"""
PyTorch Autograd Integration for aule-attention

This module provides a torch.autograd.Function wrapper that enables
gradient computation through the aule-attention backends.

Usage:
    import torch
    from aule_autograd import AuleAttentionFunction, aule_attention

    # Functional API
    Q = torch.randn(1, 8, 64, 64, requires_grad=True)
    K = torch.randn(1, 8, 64, 64, requires_grad=True)
    V = torch.randn(1, 8, 64, 64, requires_grad=True)

    output = aule_attention(Q, K, V, causal=True)
    loss = output.sum()
    loss.backward()  # Computes dQ, dK, dV
"""

import numpy as np
from typing import Optional, Tuple, Any

# Try to import PyTorch
try:
    import torch
    from torch.autograd import Function
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    Function = object  # Placeholder for type hints

# Import aule backends
try:
    import aule_unified as aule
    AULE_AVAILABLE = True
except ImportError:
    AULE_AVAILABLE = False


if PYTORCH_AVAILABLE and AULE_AVAILABLE:

    class AuleAttentionFunction(Function):
        """
        PyTorch autograd function for aule-attention.

        Implements custom forward and backward passes using OpenCL/Vulkan kernels.
        Falls back to CPU implementation if GPU backward is not available.
        """

        @staticmethod
        def forward(
            ctx: Any,
            Q: torch.Tensor,
            K: torch.Tensor,
            V: torch.Tensor,
            causal: bool = False,
            scale: Optional[float] = None,
            backend: str = 'auto',
        ) -> torch.Tensor:
            """
            Forward pass of scaled dot-product attention.

            Args:
                ctx: Autograd context for saving tensors
                Q: Query tensor [batch, heads, seq, dim]
                K: Key tensor [batch, heads, seq, dim]
                V: Value tensor [batch, heads, seq, dim]
                causal: Apply causal masking
                scale: Attention scale (default: 1/sqrt(head_dim))
                backend: Backend to use ('auto', 'opencl', 'cpu', etc.)

            Returns:
                Output tensor [batch, heads, seq, dim]
            """
            # Store input tensors for backward
            ctx.save_for_backward(Q, K, V)
            ctx.causal = causal
            ctx.scale = scale if scale is not None else 1.0 / np.sqrt(Q.shape[-1])
            ctx.backend = backend

            # Convert to numpy
            Q_np = Q.detach().cpu().numpy()
            K_np = K.detach().cpu().numpy()
            V_np = V.detach().cpu().numpy()

            # Use aule unified API for forward
            with aule.Attention(backend=backend) as attn:
                output_np = attn.forward(Q_np, K_np, V_np, causal=causal)

                # Check if backend supports backward via _impl (OpenCL) or CPU
                ctx.has_native_backward = getattr(attn, 'backward_available', False)
                if ctx.has_native_backward and attn._backend == 'opencl' and attn._impl:
                    # Save the OpenCL backend's saved tensors
                    ctx.saved_attn_tensors = attn._impl._saved_tensors.copy()
                elif ctx.has_native_backward and attn._backend == 'cpu':
                    # Save CPU backend's saved tensors
                    ctx.saved_attn_tensors = getattr(attn, '_saved_for_backward', None)

            # Convert back to torch
            output = torch.from_numpy(output_np).to(Q.device, dtype=Q.dtype)

            return output

        @staticmethod
        def backward(
            ctx: Any,
            grad_output: torch.Tensor,
        ) -> Tuple[Optional[torch.Tensor], ...]:
            """
            Backward pass computing gradients dQ, dK, dV.

            Args:
                ctx: Autograd context with saved tensors
                grad_output: Gradient of output [batch, heads, seq, dim]

            Returns:
                Tuple of (dQ, dK, dV, None, None, None) - None for non-tensor args
            """
            Q, K, V = ctx.saved_tensors
            causal = ctx.causal
            scale = ctx.scale

            # Convert grad_output to numpy
            dO_np = grad_output.detach().cpu().numpy()

            if ctx.has_native_backward and hasattr(ctx, 'saved_attn_tensors') and ctx.saved_attn_tensors is not None:
                # Check if we have OpenCL or CPU saved tensors
                saved = ctx.saved_attn_tensors

                if isinstance(saved, dict):
                    # OpenCL backend saved tensors
                    try:
                        from aule_opencl import OpenCLAttention

                        with OpenCLAttention() as attn:
                            # Restore saved tensors from forward
                            attn._saved_tensors = saved

                            # Run backward
                            dQ_np, dK_np, dV_np = attn.backward(dO_np)

                    except Exception as e:
                        # Fall back to CPU
                        print(f"Warning: Native backward failed ({e}), falling back to CPU")
                        dQ_np, dK_np, dV_np = _cpu_backward(
                            Q.detach().cpu().numpy(),
                            K.detach().cpu().numpy(),
                            V.detach().cpu().numpy(),
                            dO_np,
                            scale,
                            causal,
                        )
                elif isinstance(saved, tuple):
                    # CPU backend saved tensors
                    Q_saved, K_saved, V_saved, scale_saved, causal_saved = saved
                    dQ_np, dK_np, dV_np = _cpu_backward(
                        Q_saved, K_saved, V_saved,
                        dO_np,
                        scale_saved,
                        causal_saved,
                    )
                else:
                    # Unknown format, fall back to CPU
                    dQ_np, dK_np, dV_np = _cpu_backward(
                        Q.detach().cpu().numpy(),
                        K.detach().cpu().numpy(),
                        V.detach().cpu().numpy(),
                        dO_np,
                        scale,
                        causal,
                    )
            else:
                # Use CPU backward implementation
                dQ_np, dK_np, dV_np = _cpu_backward(
                    Q.detach().cpu().numpy(),
                    K.detach().cpu().numpy(),
                    V.detach().cpu().numpy(),
                    dO_np,
                    scale,
                    causal,
                )

            # Convert back to torch
            dQ = torch.from_numpy(dQ_np).to(Q.device, dtype=Q.dtype)
            dK = torch.from_numpy(dK_np).to(K.device, dtype=K.dtype)
            dV = torch.from_numpy(dV_np).to(V.device, dtype=V.dtype)

            # Return gradients for all inputs (None for non-tensor args)
            return dQ, dK, dV, None, None, None


    def _cpu_backward(
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        dO: np.ndarray,
        scale: float,
        causal: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        CPU reference implementation of attention backward pass.

        Algorithm from FlashAttention-2:
        1. Recompute attention weights P = softmax(Q @ K^T * scale)
        2. dV = P^T @ dO
        3. dP = dO @ V^T
        4. D = rowsum(dO * O) = rowsum(P * dP)
        5. dS = P * (dP - D)
        6. dQ = scale * dS @ K
        7. dK = scale * dS^T @ Q
        """
        batch_size, num_heads, seq_len, head_dim = Q.shape

        dQ = np.zeros_like(Q)
        dK = np.zeros_like(K)
        dV = np.zeros_like(V)

        for b in range(batch_size):
            for h in range(num_heads):
                q = Q[b, h]  # [seq, dim]
                k = K[b, h]
                v = V[b, h]
                do = dO[b, h]

                # Recompute attention scores
                scores = np.matmul(q, k.T) * scale  # [seq, seq]

                # Apply causal mask
                if causal:
                    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
                    scores = scores - mask * 1e9

                # Softmax
                scores_max = np.max(scores, axis=-1, keepdims=True)
                exp_scores = np.exp(scores - scores_max)
                P = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

                # dV = P^T @ dO
                dV[b, h] = np.matmul(P.T, do)

                # dP = dO @ V^T
                dP = np.matmul(do, v.T)

                # D = rowsum(P * dP)
                D = np.sum(P * dP, axis=-1, keepdims=True)

                # dS = P * (dP - D)
                dS = P * (dP - D)

                # Apply causal mask to dS
                if causal:
                    dS = dS * (1 - mask)

                # dQ = scale * dS @ K
                dQ[b, h] = scale * np.matmul(dS, k)

                # dK = scale * dS^T @ Q
                dK[b, h] = scale * np.matmul(dS.T, q)

        return dQ, dK, dV


    def aule_attention(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        causal: bool = False,
        scale: Optional[float] = None,
        backend: str = 'auto',
    ) -> torch.Tensor:
        """
        Compute scaled dot-product attention with autograd support.

        This is the main entry point for using aule-attention with PyTorch.
        Supports automatic differentiation for training.

        Args:
            Q: Query tensor [batch, heads, seq, dim]
            K: Key tensor [batch, heads, seq, dim]
            V: Value tensor [batch, heads, seq, dim]
            causal: Apply causal masking for autoregressive models
            scale: Attention scale (default: 1/sqrt(head_dim))
            backend: Backend to use ('auto', 'opencl', 'cpu', etc.)

        Returns:
            Output tensor [batch, heads, seq, dim]

        Example:
            >>> Q = torch.randn(1, 8, 64, 64, requires_grad=True)
            >>> K = torch.randn(1, 8, 64, 64, requires_grad=True)
            >>> V = torch.randn(1, 8, 64, 64, requires_grad=True)
            >>> out = aule_attention(Q, K, V, causal=True)
            >>> loss = out.sum()
            >>> loss.backward()
            >>> print(Q.grad.shape)  # [1, 8, 64, 64]
        """
        return AuleAttentionFunction.apply(Q, K, V, causal, scale, backend)


    class AuleAttention(torch.nn.Module):
        """
        PyTorch nn.Module wrapper for aule-attention.

        Can be used as a drop-in replacement for attention layers.

        Example:
            >>> attn = AuleAttention(causal=True)
            >>> Q = torch.randn(1, 8, 64, 64)
            >>> K = torch.randn(1, 8, 64, 64)
            >>> V = torch.randn(1, 8, 64, 64)
            >>> out = attn(Q, K, V)
        """

        def __init__(
            self,
            causal: bool = False,
            scale: Optional[float] = None,
            backend: str = 'auto',
        ):
            super().__init__()
            self.causal = causal
            self.scale = scale
            self.backend = backend

        def forward(
            self,
            Q: torch.Tensor,
            K: torch.Tensor,
            V: torch.Tensor,
        ) -> torch.Tensor:
            return aule_attention(Q, K, V, causal=self.causal, scale=self.scale, backend=self.backend)


else:
    # Stubs when PyTorch or aule is not available
    def aule_attention(*args, **kwargs):
        raise ImportError(
            "aule_attention requires PyTorch and aule_unified. "
            "Install with: pip install torch"
        )

    class AuleAttention:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "AuleAttention requires PyTorch and aule_unified. "
                "Install with: pip install torch"
            )


def test_backward():
    """Test backward pass correctness against PyTorch autograd."""
    if not PYTORCH_AVAILABLE:
        print("PyTorch not available, skipping test")
        return

    print("Testing aule-attention backward pass...")

    # Small test case
    batch, heads, seq, dim = 1, 2, 16, 32
    Q = torch.randn(batch, heads, seq, dim, requires_grad=True, dtype=torch.float32)
    K = torch.randn(batch, heads, seq, dim, requires_grad=True, dtype=torch.float32)
    V = torch.randn(batch, heads, seq, dim, requires_grad=True, dtype=torch.float32)

    # Forward pass
    output = aule_attention(Q, K, V, causal=False, backend='cpu')

    # Backward pass
    loss = output.sum()
    loss.backward()

    print(f"  Q.grad shape: {Q.grad.shape}")
    print(f"  K.grad shape: {K.grad.shape}")
    print(f"  V.grad shape: {V.grad.shape}")

    # Verify against PyTorch SDPA
    Q2 = Q.detach().clone().requires_grad_(True)
    K2 = K.detach().clone().requires_grad_(True)
    V2 = V.detach().clone().requires_grad_(True)

    output_ref = torch.nn.functional.scaled_dot_product_attention(Q2, K2, V2, is_causal=False)
    loss_ref = output_ref.sum()
    loss_ref.backward()

    # Compare gradients
    dQ_diff = (Q.grad - Q2.grad).abs().max().item()
    dK_diff = (K.grad - K2.grad).abs().max().item()
    dV_diff = (V.grad - V2.grad).abs().max().item()

    print(f"\nGradient differences from PyTorch SDPA:")
    print(f"  dQ max diff: {dQ_diff:.6f}")
    print(f"  dK max diff: {dK_diff:.6f}")
    print(f"  dV max diff: {dV_diff:.6f}")

    tol = 1e-4
    if dQ_diff < tol and dK_diff < tol and dV_diff < tol:
        print("\n✓ Backward pass matches PyTorch SDPA!")
    else:
        print(f"\n✗ Gradients differ (tolerance: {tol})")

    # Test causal
    print("\nTesting causal attention backward...")
    Q3 = torch.randn(batch, heads, seq, dim, requires_grad=True, dtype=torch.float32)
    K3 = torch.randn(batch, heads, seq, dim, requires_grad=True, dtype=torch.float32)
    V3 = torch.randn(batch, heads, seq, dim, requires_grad=True, dtype=torch.float32)

    output_causal = aule_attention(Q3, K3, V3, causal=True, backend='cpu')
    loss_causal = output_causal.sum()
    loss_causal.backward()

    Q4 = Q3.detach().clone().requires_grad_(True)
    K4 = K3.detach().clone().requires_grad_(True)
    V4 = V3.detach().clone().requires_grad_(True)

    output_ref_causal = torch.nn.functional.scaled_dot_product_attention(Q4, K4, V4, is_causal=True)
    loss_ref_causal = output_ref_causal.sum()
    loss_ref_causal.backward()

    dQ_causal_diff = (Q3.grad - Q4.grad).abs().max().item()
    dK_causal_diff = (K3.grad - K4.grad).abs().max().item()
    dV_causal_diff = (V3.grad - V4.grad).abs().max().item()

    print(f"  dQ max diff: {dQ_causal_diff:.6f}")
    print(f"  dK max diff: {dK_causal_diff:.6f}")
    print(f"  dV max diff: {dV_causal_diff:.6f}")

    if dQ_causal_diff < tol and dK_causal_diff < tol and dV_causal_diff < tol:
        print("\n✓ Causal backward pass matches PyTorch SDPA!")
    else:
        print(f"\n✗ Causal gradients differ (tolerance: {tol})")


if __name__ == "__main__":
    test_backward()
