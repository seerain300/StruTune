import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: dense linear Y = X @ W^T + B
# X: [Bsz*S, K], W: [N, K], B: [N], Y: [Bsz*S, N]
# Grid: (Bsz*S, N)
@triton.jit
def qkv_linear_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    Bsz: tl.constexpr, S: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
    stride_xm, stride_xk,        # X strides: m=B*S, k=K
    stride_w0, stride_w1,        # W strides: dim0=N, dim1=K
    stride_ym, stride_yn,        # Y strides: m=B*S, n=N
):
    m = tl.program_id(axis=0)  # 0..Bsz*S-1
    n = tl.program_id(axis=1)  # 0..N-1
    acc = tl.zeros((), dtype=tl.float32)
    # Loop over K in chunks of 64
    for k0 in range(0, K, 64):
        offs_k = k0 + tl.arange(0, 64)
        mask_k = offs_k < K
        x = tl.load(X_ptr + m * stride_xm + offs_k * stride_xk, mask=mask_k, other=0.0)  # [64]
        w = tl.load(W_ptr + n * stride_w0 + offs_k * stride_w1, mask=mask_k, other=0.0)  # [64]
        acc += tl.sum(x * w, axis=0)
    # Add bias
    b = tl.load(B_ptr + n)
    acc += b
    # Store Y[m, n]
    tl.store(Y_ptr + m * stride_ym + n * stride_yn, acc)


# Triton kernel: RMSNorm per element (b, h, s, d)
# X: [B, H, S, D], W: [D], Y: [B, H, S, D]
# Grid: (B, H, S, D)
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
    def forward(self, hidden_states: torch.Tensor,
                q_proj_weight: torch.Tensor, q_proj_bias: torch.Tensor,
                k_proj_weight: torch.Tensor, k_proj_bias: torch.Tensor,
                v_proj_weight: torch.Tensor, v_proj_bias: torch.Tensor,
                o_proj_weight: torch.Tensor,
                q_norm_weight: torch.Tensor, k_norm_weight: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor,
                rms_norm_eps: float):
        # hidden_states: [Bsz, S, 768]
        Bsz, S, K_in = hidden_states.shape
        D_out = 128  # head_dim
        H = 96       # num_attention_heads
        num_key_value_heads = 8
        num_key_value_groups = 12

        # 1) Linear Q, K, V using Triton: Y = X @ W^T + B, X is [Bsz*S, K_in]
        Xq = hidden_states.reshape(-1, K_in).contiguous().to(torch.float32)
        Xk = hidden_states.reshape(-1, K_in).contiguous().to(torch.float32)
        Xv = hidden_states.reshape(-1, K_in).contiguous().to(torch.float32)

        BszS = Bsz * S

        # Allocate outputs for Q, K, V
        Q = torch.empty((BszS, D_out), device=hidden_states.device, dtype=torch.float32)
        K = torch.empty((BszS, D_out), device=hidden_states.device, dtype=torch.float32)
        V = torch.empty((BszS, D_out), device=hidden_states.device, dtype=torch.float32)

        if TRITON_AVAILABLE:
            grid = (BszS, D_out)
            q_linear = qkv_linear_kernel[grid](
                Xq, q_proj_weight, q_proj_bias, Q,
                Bsz, S, K_in, D_out,
                Xq.stride(0), q_proj_weight.stride(1),
                Q.stride(0), Q.stride(1),
                num_warps=4, num_stages=2
            )

            k_linear = qkv_linear_kernel[grid](
                Xk, k_proj_weight, k_proj_bias, K,
                Bsz, S, K_in, D_out,
                Xk.stride(0), k_proj_weight.stride(1),
                K.stride(0), K.stride(1),
                num_warps=4, num_stages=2
            )

            v_linear = qkv_linear_kernel[grid](
                Xv, v_proj_weight, v_proj_bias, V,
                Bsz, S, K_in, D_out,
                Xv.stride(0), v_proj_weight.stride(1),
                V.stride(0), V.stride(1),
                num_warps=4, num_stages=2
            )

        # 2) Apply RMSNorm for Q and K in Triton: Y = x * rsqrt(mean(x^2) + eps) * weight
        Q4 = torch.empty_like(Q)  # [BszS, D_out]
        K4 = torch.empty_like(K)  # [BszS, D_out]
        if TRITON_AVAILABLE:
            grid4q = (Bsz, H, S, D_out)
            rmsnorm_kernel[grid4q](
                Q, q_norm_weight, Q4,
                Bsz, H, S, D_out,
                Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
                Q4.stride(0), Q4.stride(1), Q4.stride(2), Q4.stride(3),
                rms_norm_eps
            )
            rmsnorm_kernel[grid4q](
                K, k_norm_weight, K4,
                Bsz, H, S, D_out,
                K.stride(0), K.stride(1), K.stride(2), K.stride(3),
                K4.stride(0), K4.stride(1), K4.stride(2), K4.stride(3),
                rms_norm_eps
            )
        else:
            # Fallback RMSNorm (PyTorch)
            Q4 = Q.view(Bsz, S, D_out) * (q_norm_weight.view(1, 1, D_out) / torch.sqrt(torch.mean(Q.view(Bsz, S, D_out) ** 2, dim=-1, keepdim=True) + rms_norm_eps))
            K4 = K.view(Bsz, S, D_out) * (k_norm_weight.view(1, 1, D_out) / torch.sqrt(torch.mean(K.view(Bsz, S, D_out) ** 2, dim=-1, keepdim=True) + rms_norm_eps))

        # 3) Apply rotation (RoPE) for Q and K in PyTorch
        def rotate_half(x, cos, sin):
            x1 = x[:, :, :64]
            x2 = x[:, :, 64:]
            rotated = torch.cat([-x2, x1], dim=-1)
            return x * cos + rotated * sin

        Q4 = rotate_half(Q4.view(Bsz, S, D_out), cos, sin).view(BszS, D_out)
        K4 = rotate_half(K4.view(Bsz, S, D_out), cos, sin).view(BszS, D_out)
        V = rotate_half(V.view(Bsz, S, D_out), cos, sin).view(BszS, D_out)  # Note: original code applies RMSNorm to V using k_norm_weight; if that’s required, uncomment below:
        # V = V * (k_norm_weight.view(1, 1, D_out) / torch.sqrt(torch.mean(V.view(Bsz, S, D_out) ** 2, dim=-1, keepdim=True) + rms_norm_eps))

        # 4) Reshape to heads and expand K/V for GQA
        Qh = Q4.view(Bsz, H, S, D_out)
        Kh = K4.view(Bsz, num_key_value_heads, S, D_out)
        Vh = V.view(Bsz, num_key_value_heads, S, D_out)

        # Expand K/V heads to 96 attention heads: groups=num_key_value_groups=12
        # Each of the 96 heads maps to a unique combination: h = i // groups * num_key_value_heads + (i % groups)
        # Create expanded K,V by selecting from the 8 key/value heads
        K_exp = torch.empty((Bsz, H, S, D_out), device=hidden_states.device, dtype=torch.float32)
        V_exp = torch.empty((Bsz, H, S, D_out), device=hidden_states.device, dtype=torch.float32)
        for b in range(Bsz):
            for s in range(S):
                for h in range(H):
                    group = h // num_key_value_groups
                    kv_head = group * num_key_value_heads + (h % num_key_value_groups)
                    K_exp[b, h, s, :] = Kh[b, kv_head, s, :]
                    V_exp[b, h, s, :] = Vh[b, kv_head, s, :]

        # 5) Compute attention using PyTorch (scaled dot-product with causal mask)
        scores = torch.matmul(Qh, Kh.transpose(3, 2)) * (1.0 / (D_out ** 0.5))  # [B, H, S, S]
        causal_mask = torch.triu(torch.ones(S, S, dtype=torch.float32, device=hidden_states.device), diagonal=1)
        scores = scores + (-1e30) * causal_mask  # broadcast over batch and heads

        attn_weights = torch.softmax(scores, dim=-1)  # [B, H, S, S]
        attn_output = torch.matmul(attn_weights, V_exp)  # [B, H, S, D_out]

        # 6) Transpose and reshape for final linear
        attn_output = attn_output.transpose(1, 2).contiguous()  # [B, S, H, D_out]
        attn_output = attn_output.reshape(Bsz, S, H * D_out)    # [B, S, 12288]

        # 7) Final output projection: Y = attn_output @ o_proj_weight^T (no bias)
        attn_output_flat = attn_output.reshape(Bsz * S, -1).contiguous().to(torch.float32)
        O = torch.empty((Bsz * S, D_out), device=hidden_states.device, dtype=torch.float32)

        if TRITON_AVAILABLE:
            grid = (Bsz * S, D_out)
            final_linear = qkv_linear_kernel[grid](
                attn_output_flat, o_proj_weight, None, O,
                Bsz, S, 12288, D_out,
                attn_output_flat.stride(0), o_proj_weight.stride(1),
                O.stride(0), O.stride(1),
                num_warps=4, num_stages=2
            )

        output = O.view(Bsz, S, D_out)
        return output


def run(*args):
    return ModelNew()(*args)
