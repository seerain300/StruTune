import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# -------------------------------
# Triton kernels
# -------------------------------

# RMSNorm per row: Y = X * rsqrt(mean(X^2) + eps), vectorized over last dim
# Input is [M, N], we normalize each row independently.
@triton.jit
def rmsnorm_kernel(
    X_ptr, Y_ptr,        # [M, N] pointers
    M, N,                # ints
    eps,                 # float32
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)  # row index
    offs_n = tl.arange(0, BLOCK_N)
    mask = offs_n < N

    x = tl.load(X_ptr + pid * stride_xm + offs_n * stride_xn, mask=mask, other=0.0)
    x = x.to(tl.float32)
    mean = tl.sum(x * x, axis=0) / N
    inv = tl.rsqrt(mean + eps)
    y = x * inv
    # cast back to original dtype
    y = y.to(tl.float16)  # original hidden_states is float16
    tl.store(Y_ptr + pid * stride_ym + offs_n * stride_yn, y, mask=mask)


# Apply half rotation to the last 64 dims: for x in [*, N], split x[:64], x[64:], then y = cos*x + sin*(-x[64:])
# We implement elementwise for a vector. X_ptr, Y_ptr point to [M, N].
@triton.jit
def apply_half_rotation_kernel(
    X_ptr, Y_ptr,        # [M, N]
    M, N,                # ints
    cos_ptr, sin_ptr,    # [1] or scalars via pointers
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)  # row index
    offs_n = tl.arange(0, BLOCK_N)
    mask = offs_n < N

    x = tl.load(X_ptr + pid * stride_xm + offs_n * stride_xn, mask=mask, other=0.0)
    # cos, sin scalars (assumed precomputed)
    cos_val = tl.load(cos_ptr)
    sin_val = tl.load(sin_ptr)

    # split
    half = N // 2
    q1 = x[:half]
    q2 = x[half:]
    rotated = q2 * cos_val + (-q1) * sin_val
    y = tl.concatenate([q2, rotated], axis=0)

    y = y.to(tl.float16)
    tl.store(Y_ptr + pid * stride_ym + offs_n * stride_yn, y, mask=mask)


# -------------------------------
# ModelNew: forward must match original behavior exactly
# -------------------------------

class ModelNew(nn.Module):
    def __init__(self,
                 num_attention_heads: int = 96,
                 num_key_value_heads: int = 8,
                 head_dim: int = 128,
                 num_key_value_groups: int = 12,
                 rms_norm_eps: float = 1e-6):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_key_value_groups
        self.rms_norm_eps = rms_norm_eps
        self.scaling = 1.0 / (head_dim ** 0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_proj_weight: torch.Tensor,
        q_proj_bias: torch.Tensor,
        k_proj_weight: torch.Tensor,
        k_proj_bias: torch.Tensor,
        v_proj_weight: torch.Tensor,
        v_proj_bias: torch.Tensor,
        o_proj_weight: torch.Tensor,  # used for final output projection (no bias)
        q_norm_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
        cos: torch.Tensor,  # [head_dim]
        sin: torch.Tensor,  # [head_dim]
    ):
        # Device and dtype checks
        assert hidden_states.is_cuda, "Inputs must be on CUDA for Triton kernels"
        B, S, hidden_dim = hidden_states.shape
        assert hidden_dim == self.head_dim * self.num_attention_heads, "hidden_dim mismatch with heads"
        assert self.num_attention_heads == self.num_key_value_heads * self.num_key_value_groups, "H_q must equal H_k * num_groups"

        # 1) Linear projections (use PyTorch F.linear to exactly match original)
        # hidden_states: [B, S, hidden_dim], weight: [hidden_dim, hidden_dim]
        query = F.linear(hidden_states, q_proj_weight, q_proj_bias)  # [B, S, hidden_dim]
        key = F.linear(hidden_states, k_proj_weight, k_proj_bias)    # [B, S, hidden_dim]
        value = F.linear(hidden_states, v_proj_weight, v_proj_bias)  # [B, S, hidden_dim]

        # 2) RMSNorm on Q and K (per row)
        # Reshape to [B*S, hidden_dim]
        query_2d = query.reshape(B * S, hidden_dim)
        key_2d = key.reshape(B * S, hidden_dim)
        # Allocate outputs
        query_norm = torch.empty_like(query_2d)
        key_norm = torch.empty_like(key_2d)
        # Launch RMSNorm kernels
        grid = (B * S,)
        rmsnorm_kernel[grid](
            query_2d, query_norm, B * S, hidden_dim, self.rms_norm_eps,
            query_2d.stride(0), query_2d.stride(1),
            query_norm.stride(0), query_norm.stride(1),
            BLOCK_N=128, num_warps=4, num_stages=2
        )
        rmsnorm_kernel[grid](
            key_2d, key_norm, B * S, hidden_dim, self.rms_norm_eps,
            key_2d.stride(0), key_2d.stride(1),
            key_norm.stride(0), key_norm.stride(1),
            BLOCK_N=128, num_warps=4, num_stages=2
        )
        # Bring back to [B, S, hidden_dim]
        query_norm = query_norm.reshape(B, S, hidden_dim)
        key_norm = key_norm.reshape(B, S, hidden_dim)

        # 3) Apply half rotation to Q and K (elementwise, last 64 dims)
        # Prepare rotated tensors
        query_rot = torch.empty_like(query_norm)
        key_rot = torch.empty_like(key_norm)

        # cos, sin are [head_dim]; Triton expects pointers
        # We pass cos and sin as 1-element tensors to Triton (even though they are vectors). This is fine since they are scalars per dim.
        # Note: Using .to(tensor) would require creating small tensors; here we assume cos, sin are 1-D on device.
        grid = (B * S,)
        apply_half_rotation_kernel[grid](
            query_norm, query_rot, B * S, hidden_dim, cos, sin,
            query_norm.stride(0), hidden_dim,
            query_rot.stride(0), hidden_dim,
            BLOCK_N=128, num_warps=4, num_stages=2
        )
        apply_half_rotation_kernel[grid](
            key_norm, key_rot, B * S, hidden_dim, cos, sin,
            key_norm.stride(0), hidden_dim,
            key_rot.stride(0), hidden_dim,
            BLOCK_N=128, num_warps=4, num_stages=2
        )

        # 4) GQA: expand KV heads to match query heads
        # query_rot: [B, S, hidden_dim]
        # num_key_value_groups = 12; H_q = 96; H_k = 8
        # We repeat each KV head across groups: [B, 8, S, hidden_dim] -> [B, 96, S, hidden_dim]
        # But we need to map each query head i to its corresponding KV head kv_h = i // num_key_value_groups.
        # Construct expanded key/value via repeat_interleave semantics.
        # Note: Since we normalized Q, K earlier, V should not be normalized in original. We keep value as F.linear output.
        # However, original code also applies RMSNorm to V (commented out). To match, we should apply RMSNorm to value too.
        # Compute value via F.linear then apply RMSNorm.
        value_for_attn = F.linear(hidden_states, v_proj_weight, v_proj_bias)  # [B, S, hidden_dim]
        value_2d = value_for_attn.reshape(B * S, hidden_dim)
        value_norm = torch.empty_like(value_2d)
        rmsnorm_kernel[grid](
            value_2d, value_norm, B * S, hidden_dim, self.rms_norm_eps,
            value_2d.stride(0), value_2d.stride(1),
            value_norm.stride(0), value_norm.stride(1),
            BLOCK_N=128, num_warps=4, num_stages=2
        )
        value_for_attn = value_norm.reshape(B, S, hidden_dim)

        # Now expand K_rot and V to [B, 96, S, hidden_dim]
        # Build mapping: for each (b, i), i in [0..H_q-1], kv_h = i // num_key_value_groups
        # Expand K_rot: [B, S, H_k, D] -> [B, S, H_q, D] via repeat_interleave
        k_expanded = torch.empty((B, S, self.num_attention_heads, self.head_dim), device=hidden_states.device, dtype=torch.float16)
        for i in range(self.num_attention_heads):
            kv_h = i // self.num_key_value_groups
            k_expanded[:, :, i, :] = key_rot[:, :, kv_h * self.head_dim : (kv_h + 1) * self.head_dim]
        # Expand V similarly
        v_expanded = torch.empty((B, S, self.num_attention_heads, self.head_dim), device=hidden_states.device, dtype=torch.float16)
        for i in range(self.num_attention_heads):
            kv_h = i // self.num_key_value_groups
            v_expanded[:, :, i, :] = value_for_attn[:, :, kv_h * self.head_dim : (kv_h + 1) * self.head_dim]

        # Query remains [B, S, hidden_dim]; we need [B, S, H_q, D] -> reshape
        # Since hidden_dim == H_q * D, query can be viewed as [B, S, H_q, D]:
        query_view = query_rot.view(B, S, self.num_attention_heads, self.head_dim)  # already H_q*D

        # 5) Compute attention: scores = (query @ key^T) * scaling, apply causal mask, softmax along sequence dim
        # For correctness, we use PyTorch matmul + softmax here (original approach). This avoids tricky Triton masking issues.
        # We compute [B, S, H_q, D] @ [B, S, H_q, D]^T to get [B, S, S].
        # But note: original computes Q @ K^T per (batch, head), not per expanded head. To strictly match, we compute per head:
        attn_out = torch.empty((B, S, self.num_attention_heads), device=hidden_states.device, dtype=torch.float16)
        for h in range(self.num_attention_heads):
            # Extract Q_h and K_h
            # Note: query_view is [B, S, H_q, D]; reshape to [B*S, D] for matmul:
            # But to keep dims, compute per (b, s):
            # We can directly compute per (b, s) pair without matmul by using tensors; here we use PyTorch matmul on [S, D] with [S, D]T.
            # Since we need [B, S, S] for all heads, we do a batched approach:
            # Build batched Q and K: Q_b = query_view[b] [S, D], K_b = key_rot[b] [S, D]
            Q_b = query_view[b]  # [S, D]
            K_b = key_rot[b]     # [S, D]
            V_b = v_expanded[b]  # [S, D]
            scores = torch.matmul(Q_b, K_b.transpose(0, 1))  # [S, S]
            # Apply causal mask: upper-triangular with diagonal=1
            # Create causal mask [S, S]
            causal_mask = torch.triu(torch.ones((S, S), device=hidden_states.device, dtype=torch.float32), diagonal=1)
            causal_mask = causal_mask * (-1e4)  # large negative to emulate -inf in softmax
            scores = scores + causal_mask
            # Softmax along seq dim
            attn_weights = torch.softmax(scores, dim=-1)  # [S, S]
            # Output: O = attn_weights @ V_b -> [S, D]
            attn_out[b, :, h] = torch.matmul(attn_weights, V_b)  # [S, D] then take first S elements? No: we need [S, D]

            # The above per-(b,h) loop is not vectorized; however, to strictly match original behavior, we implement per head as above.
            # For performance, we could vectorize across batch and head, but correctness takes precedence here.

        # 6) Transpose and reshape to [B, S, H_q*D] for final output
        attn_out = attn_out.view(B, S, self.num_attention_heads * self.head_dim)

        # 7) Final output projection: Out[B, S, H_q*D] = Attn @ o_proj_weight^T (no bias)
        # o_proj_weight is [hidden_dim, hidden_dim], but we have Attn [B, S, H_q*D], which equals hidden_dim (768).
        final_out = F.linear(attn_out, o_proj_weight, None)

        return final_out


def run(*args):
    return ModelNew()(*args)
