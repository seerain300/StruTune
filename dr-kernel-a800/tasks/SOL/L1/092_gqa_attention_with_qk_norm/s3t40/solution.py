import torch
import torch.nn as nn
import torch.nn.functional as F

# Fixed constants from the original code
num_attention_heads = 96
num_key_value_heads = 8
num_key_value_groups = 12
head_dim = 128
scaling = head_dim ** -0.5

# Try to import Triton
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False

# Triton kernel: linear Y = X @ W^T + B
# X: [M, D_in], W: [D_out, D_in], B: [D_out], Y: [M, D_out]
# Grid: (M, D_out)
@triton.jit
def linear_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, D_in: tl.constexpr, D_out: tl.constexpr,
    stride_xm, stride_xk,
    stride_w0, stride_w1,
    stride_ym, stride_yn,
):
    m = tl.program_id(axis=0)   # row index
    n = tl.program_id(axis=1)   # output feature index
    acc = tl.zeros((), dtype=tl.float32)
    # Loop over input dimension in chunks of 64
    for k0 in range(0, D_in, 64):
        offs = k0 + tl.arange(0, 64)
        mask = offs < D_in
        x = tl.load(X_ptr + m * stride_xm + offs * stride_xk, mask=mask, other=0.0)  # [64]
        w = tl.load(W_ptr + n * stride_w0 + offs * stride_w1, mask=mask, other=0.0)  # [64]
        acc += tl.sum(x * w, axis=0)
    # Add bias
    b = tl.load(B_ptr + n)
    acc += b
    tl.store(Y_ptr + m * stride_ym + n * stride_yn, acc)

# Triton kernel: output linear Y = X @ W^T (no bias)
@triton.jit
def linear_kernel_no_bias(
    X_ptr, W_ptr, Y_ptr,
    M, D_in: tl.constexpr, D_out: tl.constexpr,
    stride_xm, stride_xk,
    stride_w0, stride_w1,
    stride_ym, stride_yn,
):
    m = tl.program_id(axis=0)
    n = tl.program_id(axis=1)
    acc = tl.zeros((), dtype=tl.float32)
    for k0 in range(0, D_in, 64):
        offs = k0 + tl.arange(0, 64)
        mask = offs < D_in
        x = tl.load(X_ptr + m * stride_xm + offs * stride_xk, mask=mask, other=0.0)
        w = tl.load(W_ptr + n * stride_w0 + offs * stride_w1, mask=mask, other=0.0)
        acc += tl.sum(x * w, axis=0)
    tl.store(Y_ptr + m * stride_ym + n * stride_yn, acc)

# Triton kernel: RMSNorm per (b, h, s, d)
# X: [B, H, S, D], W: [D], Y: [B, H, S, D]
# For each element: inv_rms = 1/sqrt(mean(x^2) + eps), y = x * inv_rms * w
@triton.jit
def rmsnorm_kernel(
    X_ptr, W_ptr, Y_ptr,
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    stride_xb, stride_xh, stride_xs, stride_xd,
    stride_yb, stride_yh, stride_ys, stride_yd,
    eps: tl.float32,
):
    b = tl.program_id(axis=0)
    h = tl.program_id(axis=1)
    s = tl.program_id(axis=2)
    d = tl.program_id(axis=3)
    x = tl.load(X_ptr + b * stride_xb + h * stride_xh + s * stride_xs + d * stride_xd).to(tl.float32)
    sum_sq = x * x
    mean_sq = tl.sum(sum_sq, axis=0) / D
    inv_rms = tl.rsqrt(mean_sq + eps)
    w = tl.load(W_ptr + d).to(tl.float32)
    y = (x * inv_rms) * w
    tl.store(Y_ptr + b * stride_yb + h * stride_yh + s * stride_ys + d * stride_yd, y)

class ModelNew(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        q_proj_weight: torch.Tensor,
        q_proj_bias: torch.Tensor,
        k_proj_weight: torch.Tensor,
        k_proj_bias: torch.Tensor,
        v_proj_weight: torch.Tensor,
        v_proj_bias: torch.Tensor,
        o_proj_weight: torch.Tensor,
        q_norm_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        rms_norm_eps: float,
    ):
        # Ensure tensors are on the same device and contiguous
        device = hidden_states.device
        B, S, D_in = hidden_states.shape
        assert D_in == head_dim, "hidden_states last dim must equal head_dim (128)."

        # Flatten [B, S] -> M
        M = B * S
        hidden_flat = hidden_states.reshape(M, D_in).contiguous()

        # 1) Q, K, V via Triton linear
        # Q
        q_w = q_proj_weight.contiguous()        # [128, 128]
        q_y = torch.empty((M, q_w.shape[0]), dtype=torch.float32, device=device)
        grid_q = (M, q_w.shape[0])
        linear_kernel[grid_q](
            hidden_flat, q_w, q_proj_bias.contiguous(), q_y,
            M, q_w.shape[1], q_w.shape[0],
            hidden_flat.stride(0), hidden_flat.stride(1),
            q_w.stride(0), q_w.stride(1),
            q_y.stride(0), q_y.stride(1),
            num_warps=4, num_stages=2,
        )
        query_states = q_y.view(B, S, q_w.shape[0]).contiguous()  # [B, S, 128]

        # K
        k_w = k_proj_weight.contiguous()        # [128, 128]
        k_y = torch.empty((M, k_w.shape[0]), dtype=torch.float32, device=device)
        grid_k = (M, k_w.shape[0])
        linear_kernel[grid_k](
            hidden_flat, k_w, k_proj_bias.contiguous(), k_y,
            M, k_w.shape[1], k_w.shape[0],
            hidden_flat.stride(0), hidden_flat.stride(1),
            k_w.stride(0), k_w.stride(1),
            k_y.stride(0), k_y.stride(1),
            num_warps=4, num_stages=2,
        )
        key_states = k_y.view(B, S, k_w.shape[0]).contiguous()  # [B, S, 128]

        # V
        v_w = v_proj_weight.contiguous()        # [128, 128]
        v_y = torch.empty((M, v_w.shape[0]), dtype=torch.float32, device=device)
        grid_v = (M, v_w.shape[0])
        linear_kernel[grid_v](
            hidden_flat, v_w, v_proj_bias.contiguous(), v_y,
            M, v_w.shape[1], v_w.shape[0],
            hidden_flat.stride(0), hidden_flat.stride(1),
            v_w.stride(0), v_w.stride(1),
            v_y.stride(0), v_y.stride(1),
            num_warps=4, num_stages=2,
        )
        value_states = v_y.view(B, S, v_w.shape[0]).contiguous()  # [B, S, 128]

        # 2) RMSNorm for Q and K (Triton)
        # Query
        q_norm_w = q_norm_weight.contiguous()   # [128]
        query_norm = torch.empty_like(query_states, dtype=torch.float32, device=device)
        grid_qn = (B, num_attention_heads, S, head_dim)
        rmsnorm_kernel[grid_qn](
            query_states.reshape(B, num_attention_heads, S, head_dim),
            q_norm_w, query_norm.reshape(B, num_attention_heads, S, head_dim),
            B, num_attention_heads, S, head_dim,
            query_states.reshape(B, num_attention_heads, S, head_dim).stride(0),
            query_states.reshape(B, num_attention_heads, S, head_dim).stride(1),
            query_states.reshape(B, num_attention_heads, S, head_dim).stride(2),
            query_states.reshape(B, num_attention_heads, S, head_dim).stride(3),
            query_norm.reshape(B, num_attention_heads, S, head_dim).stride(0),
            query_norm.reshape(B, num_attention_heads, S, head_dim).stride(1),
            query_norm.reshape(B, num_attention_heads, S, head_dim).stride(2),
            query_norm.reshape(B, num_attention_heads, S, head_dim).stride(3),
            rms_norm_eps,
            num_warps=4, num_stages=2,
        )
        query_states = query_norm

        # Key
        k_norm_w = k_norm_weight.contiguous()   # [128]
        key_norm = torch.empty_like(key_states, dtype=torch.float32, device=device)
        grid_kn = (B, num_key_value_heads, S, head_dim)
        rmsnorm_kernel[grid_kn](
            key_states.reshape(B, num_key_value_heads, S, head_dim),
            k_norm_w, key_norm.reshape(B, num_key_value_heads, S, head_dim),
            B, num_key_value_heads, S, head_dim,
            key_states.reshape(B, num_key_value_heads, S, head_dim).stride(0),
            key_states.reshape(B, num_key_value_heads, S, head_dim).stride(1),
            key_states.reshape(B, num_key_value_heads, S, head_dim).stride(2),
            key_states.reshape(B, num_key_value_heads, S, head_dim).stride(3),
            key_norm.reshape(B, num_key_value_heads, S, head_dim).stride(0),
            key_norm.reshape(B, num_key_value_heads, S, head_dim).stride(1),
            key_norm.reshape(B, num_key_value_heads, S, head_dim).stride(2),
            key_norm.reshape(B, num_key_value_heads, S, head_dim).stride(3),
            rms_norm_eps,
            num_warps=4, num_stages=2,
        )
        key_states = key_norm

        # 3) Rotation (RoPE) - apply to query and key. We keep it in PyTorch to avoid Triton kernel complexity.
        # Split into halves: first 64 and last 64
        q1 = query_states[..., :64]
        q2 = query_states[..., 64:]
        # rotated = cat((-q2, q1), -1)
        rotated_q_half = torch.cat((-q2, q1), dim=-1)
        # Apply rotation: Y = X * cos + rotated_half * sin
        # cos/sin are scalars per position; hidden sin/cos are length S and applied to last 64 dims
        # Build cos/sin of shape [B, S, 64]
        cos = cos.unsqueeze(0)  # [1, S, 64]
        sin = sin.unsqueeze(0)  # [1, S, 64]
        cos_exp = cos.expand(B, S, 64)
        sin_exp = sin.expand(B, S, 64)
        query_rotated = query_states * cos_exp + rotated_q_half * sin_exp

        k1 = key_states[..., :64]
        k2 = key_states[..., 64:]
        rotated_k_half = torch.cat((-k2, k1), dim=-1)
        key_rotated = key_states * cos_exp + rotated_k_half * sin_exp

        # 4) GQA: expand key/value from 8 heads to 96 using groups of 12
        key_expanded = key_rotated.view(B, S, num_key_value_heads, head_dim)[:, :, None, :, :].expand(B, num_key_value_heads, num_key_value_groups, S, head_dim).reshape(B, num_attention_heads, S, head_dim)
        value_expanded = value_states.view(B, S, num_key_value_heads, head_dim)[:, :, None, :, :].expand(B, num_key_value_heads, num_key_value_groups, S, head_dim).reshape(B, num_attention_heads, S, head_dim)

        # 5) Compute attention in PyTorch for correctness
        query_attn = query_rotated.view(B, S, num_attention_heads, head_dim)
        attn_weights = torch.matmul(query_attn, key_expanded.transpose(2, 3)) * scaling  # [B, S, S]
        # Causal mask: j > i -> -inf
        causal_mask = torch.triu(
            torch.full((S, S), float("-inf"), device=device, dtype=torch.float32),
            diagonal=1
        ).unsqueeze(0).unsqueeze(1)  # [1, 1, S, S]
        attn_weights = attn_weights + causal_mask
        attn_weights = F.softmax(attn_weights, dim=-1)  # [B, S, S]

        attn_output = torch.matmul(attn_weights, value_expanded)  # [B, S, 128]
        attn_output = attn_output.view(B, S, num_attention_heads * head_dim)

        # 6) Final output projection via Triton (no bias)
        out_x = attn_output.reshape(M, head_dim).contiguous()
        out_w = o_proj_weight.contiguous()            # [128, 128]
        out_y = torch.empty((M, out_w.shape[0]), dtype=torch.float32, device=device)
        grid_out = (M, out_w.shape[0])
        linear_kernel_no_bias[grid_out](
            out_x, out_w, out_y,
            M, out_w.shape[1], out_w.shape[0],
            out_x.stride(0), out_x.stride(1),
            out_w.stride(0), out_w.stride(1),
            out_y.stride(0), out_y.stride(1),
            num_warps=4, num_stages=2,
        )
        output = out_y.view(B, S, out_w.shape[0]).contiguous()  # [B, S, 128], float32

        return output


def run(*args):
    return ModelNew()(*args)
