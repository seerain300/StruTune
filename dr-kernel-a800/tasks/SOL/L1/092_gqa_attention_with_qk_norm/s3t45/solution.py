import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: dense linear layer Y = X @ W^T + B
# X: [M, K] where M = B*S, K = input_dim
# W: [N, K] where N = output_dim
# B: [N]
# Y: [M, N]
@triton.jit
def linear_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_xm, stride_xk,
    stride_w0, stride_w1,
    stride_ym, stride_yn,
):
    m = tl.program_id(axis=0)  # row in X/Y
    n = tl.program_id(axis=1)  # output dim
    acc = tl.zeros((), dtype=tl.float32)
    # Accumulate over K in chunks of 64
    for k0 in range(0, K, 64):
        offs = k0 + tl.arange(0, 64)
        mask = offs < K
        x = tl.load(X_ptr + m * stride_xm + offs * stride_xk, mask=mask, other=0.0)  # [64]
        w = tl.load(W_ptr + n * stride_w0 + offs * stride_w1, mask=mask, other=0.0)  # [64]
        acc += tl.sum(x * w, axis=0)
    # Add bias
    b = tl.load(B_ptr + n)
    acc += b
    # Store result Y[m, n]
    tl.store(Y_ptr + m * stride_ym + n * stride_yn, acc)


# Triton kernel: RMSNorm per (b, h, s, d)
# X: [B, H, S, D], W: [D], Y: [B, H, S, D]
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


# Triton kernel: apply rotation (RoPE) for last dim D=128, split into two halves (64,64)
# X: [B, H, S, D], C: [S, D/2], S: [S, D/2], Y: [B, H, S, D]
@triton.jit
def rotate_half_kernel(
    X_ptr, C_ptr, S_ptr, Y_ptr,
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    stride_xb, stride_xh, stride_xs, stride_xd,
    stride_yb, stride_yh, stride_ys, stride_yd,
    stride_c0, stride_c1,     # cos strides: [S, D/2]
    stride_s0, stride_s1,     # sin strides: [S, D/2]
):
    b = tl.program_id(axis=0)
    h = tl.program_id(axis=1)
    s = tl.program_id(axis=2)
    d = tl.program_id(axis=3)
    # Load original X
    x = tl.load(X_ptr + b * stride_xb + h * stride_xh + s * stride_xs + d * stride_xd).to(tl.float32)
    # Split into q1 and q2
    q1 = x[:64]
    q2 = x[64:]
    # Load cos/sin for this s and d in [0, 64)
    c = tl.load(C_ptr + s * stride_c0 + d * stride_c1).to(tl.float32)
    sval = tl.load(S_ptr + s * stride_s0 + d * stride_s1).to(tl.float32)
    rotated_half = tl.cat([-q2, q1], axis=0)  # concatenate (-q2, q1)
    y = x * c + rotated_half * sval
    tl.store(Y_ptr + b * stride_yb + h * stride_yh + s * stride_ys + d * stride_yd, y)


# Triton kernel: compute attention scores S[b, h, i, j] = (Q[b,h,i] @ K[b,h,j]) * scaling
# Q: [B, H, S, D], K: [B, H, S, D], S: [B, H, S, S]
@triton.jit
def attn_scores_kernel(
    Q_ptr, K_ptr, S_ptr,
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_sb, stride_sh, stride_si, stride_sj,
    scaling: tl.float32,
):
    b = tl.program_id(axis=0)
    h = tl.program_id(axis=1)
    i = tl.program_id(axis=2)
    j = tl.program_id(axis=3)
    # Compute Q row and K column dot product
    score = tl.zeros((), dtype=tl.float32)
    for d0 in range(0, D, 64):
        offs = d0 + tl.arange(0, 64)
        q = tl.load(Q_ptr + b * stride_qb + h * stride_qh + i * stride_qs + offs * stride_qd, mask=offs < D, other=0.0)
        k = tl.load(K_ptr + b * stride_kb + h * stride_kh + j * stride_ks + offs * stride_kd, mask=offs < D, other=0.0)
        score += tl.sum(q * k, axis=0)
    score = score * scaling
    # Apply causal mask: if i >= j, set score to -inf
    if i >= j:
        score = -float('inf')
    tl.store(S_ptr + b * stride_sb + h * stride_sh + i * stride_si + j * stride_sj, score)


# Triton kernel: compute softmax over sequence dim for each (b, h) on matrix S[b,h, :, :]
# S_in: [B, H, S, S], Soft_out: [B, H, S, S]
@triton.jit
def softmax_cols_kernel(
    S_ptr, Soft_ptr,
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr,
    stride_sb, stride_sh, stride_si, stride_sj,
    stride_ob, stride_oh, stride_oi, stride_oj,
):
    b = tl.program_id(axis=0)
    h = tl.program_id(axis=1)
    # Compute max over j for row i
    max_val = tl.full((), -float('inf'), dtype=tl.float32)
    for j in range(0, S):
        s = tl.load(S_ptr + b * stride_sb + h * stride_sh + j * stride_sj)  # S[b,h, :, j] is not available; need to read per j
        # To implement row-wise softmax, we need to access S[b,h,i,j] for all j. Triton doesn't support indexing by non-constexpr in this context.
        # Instead, we implement softmax by reading S[i,j] per loop. However, Triton kernels operate on tiles, so we'll use a 1D grid over i and compute all j.
        # We need a loop over j in the kernel. Triton supports range loops; we can compute max and sum across j for each i.
        # For simplicity and correctness, we implement per i in host by launching one program per (b,h,i). But Triton kernel needs full S for each i; better to do softmax in PyTorch for now.
    # Since implementing full softmax robustly across 2D requires 2D tiling and reductions, we switch to PyTorch for softmax in this version to ensure correctness.
    # Placeholder: return without writing, as we won't use this kernel. We'll compute softmax in PyTorch.
    pass


# Triton kernel: compute attention output Y[b,h,i,:] = sum_j Soft[b,h,i,j] * V[b,h,j,:]
# Soft: [B, H, S, S], V: [B, H, S, D], Y_out: [B, H, S, D]
@triton.jit
def attn_output_kernel(
    Soft_ptr, V_ptr, Y_ptr,
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    stride_sb, stride_sh, stride_si, stride_sj,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_yb, stride_yh, stride_yi, stride_yd,
):
    b = tl.program_id(axis=0)
    h = tl.program_id(axis=1)
    i = tl.program_id(axis=2)
    # Compute output for position i
    for d0 in range(0, D, 64):
        offs = d0 + tl.arange(0, 64)
        acc = tl.zeros((), dtype=tl.float32)
        # Loop over j in [0, S)
        for j in range(0, S):
            soft_ij = tl.load(Soft_ptr + b * stride_sb + h * stride_sh + i * stride_si + j * stride_sj)
            vj = tl.load(V_ptr + b * stride_vb + h * stride_vh + j * stride_vs + offs * stride_vd, mask=offs < D, other=0.0)
            acc += soft_ij * tl.sum(vj, axis=0)  # sum over D vector to a scalar
        # Store acc to Y[b,h,i,:]
        tl.store(Y_ptr + b * stride_yb + h * stride_yh + i * stride_yi + (d0 + tl.arange(0, 64)) * stride_yd, acc, mask=(d0 + tl.arange(0, 64)) < D)
    # Note: The above stores a scalar acc for each d chunk; this is incorrect. Instead, compute and store per d:
    # We need per d output: acc_d = sum_j Soft[b,h,i,j] * V[b,h,j,d]
    # Do that by storing scalar acc per d:
    for d0 in range(0, D, 64):
        offs = d0 + tl.arange(0, 64)
        for j in range(0, S):
            soft_ij = tl.load(Soft_ptr + b * stride_sb + h * stride_sh + i * stride_si + j * stride_sj)
            vj = tl.load(V_ptr + b * stride_vb + h * stride_vh + j * stride_vs + offs * stride_vd, mask=offs < D, other=0.0)
            acc = soft_ij * tl.sum(vj, axis=0)
            tl.store(Y_ptr + b * stride_yb + h * stride_yh + i * stride_yi + offs * stride_yd, acc, mask=offs < D)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        # Fixed constants per the original code
        self.head_dim = 128
        self.num_attention_heads = 96
        self.num_key_value_heads = 8
        self.num_key_value_groups = 12
        self.scaling = 1.0 / (self.head_dim ** 0.5)

    def forward(self, hidden_states: torch.Tensor,
                q_proj_weight: torch.Tensor, q_proj_bias: torch.Tensor,
                k_proj_weight: torch.Tensor, k_proj_bias: torch.Tensor,
                v_proj_weight: torch.Tensor, v_proj_bias: torch.Tensor,
                o_proj_weight: torch.Tensor, o_proj_bias: torch.Tensor,
                q_norm_weight: torch.Tensor, k_norm_weight: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor,
                rms_norm_eps: float):
        # Ensure float32 for stable Triton math
        device = hidden_states.device
        B, S, _ = hidden_states.shape
        D = self.head_dim
        H = self.num_attention_heads
        KVH = self.num_key_value_heads
        KVG = self.num_key_value_groups

        # 1) Linear layers: Q, K, V
        B_S = B * S
        # Flatten X to [B*S, D]
        X = hidden_states.reshape(B_S, D)
        # Q
        Q = torch.empty((B_S, D), device=device, dtype=torch.float32)
        linear_kernel[(B_S, D)](
            X, q_proj_weight, q_proj_bias, Q,
            B_S, D, D,
            1, 1,  # stride_xm, stride_xk for contiguous [M,K]
            q_proj_weight.stride(0), q_proj_weight.stride(1),
            Q.stride(0), Q.stride(1),
        )
        # Reshape to [B, H, S, D]
        Q = Q.reshape(B, H, S, D).contiguous()
        # K and V similarly
        K = torch.empty((B_S, D), device=device, dtype=torch.float32)
        linear_kernel[(B_S, D)](
            X, k_proj_weight, k_proj_bias, K,
            B_S, D, D,
            1, 1,
            k_proj_weight.stride(0), k_proj_weight.stride(1),
            K.stride(0), K.stride(1),
        )
        V = torch.empty((B_S, D), device=device, dtype=torch.float32)
        linear_kernel[(B_S, D)](
            X, v_proj_weight, v_proj_bias, V,
            B_S, D, D,
            1, 1,
            v_proj_weight.stride(0), v_proj_weight.stride(1),
            V.stride(0), V.stride(1),
        )
        K = K.reshape(B, H, S, D).contiguous()
        V = V.reshape(B, H, S, D).contiguous()

        # 2) RMSNorm for Q and K
        Q_norm = torch.empty_like(Q, device=device, dtype=torch.float32)
        rmsnorm_kernel[(B, H, S, D)](
            Q, q_norm_weight, Q_norm,
            B, H, S, D,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(2), Q_norm.stride(3),
            rms_norm_eps,
        )
        K_norm = torch.empty_like(K, device=device, dtype=torch.float32)
        rmsnorm_kernel[(B, H, S, D)](
            K, k_norm_weight, K_norm,
            B, H, S, D,
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            K_norm.stride(0), K_norm.stride(1), K_norm.stride(2), K_norm.stride(3),
            rms_norm_eps,
        )

        # 3) Rotate Q and K (RoPE) using Triton
        Qr = torch.empty_like(Q_norm, device=device, dtype=torch.float32)
        rotate_half_kernel[(B, H, S, D)](
            Q_norm, cos, sin, Qr,
            B, H, S, D,
            Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(2), Q_norm.stride(3),
            Qr.stride(0), Qr.stride(1), Qr.stride(2), Qr.stride(3),
            cos.stride(0), cos.stride(1),    # cos is [S, D/2] contiguous
            sin.stride(0), sin.stride(1),    # sin is [S, D/2] contiguous
        )
        Kr = torch.empty_like(K_norm, device=device, dtype=torch.float32)
        rotate_half_kernel[(B, H, S, D)](
            K_norm, cos, sin, Kr,
            B, H, S, D,
            K_norm.stride(0), K_norm.stride(1), K_norm.stride(2), K_norm.stride(3),
            Kr.stride(0), Kr.stride(1), Kr.stride(2), Kr.stride(3),
            cos.stride(0), cos.stride(1),
            sin.stride(0), sin.stride(1),
        )

        # 4) Repeat KV heads for GQA: expand [B, KVH, S, D] -> [B, H, S, D]
        Krr = Krr = torch.empty((B, H, S, D), device=device, dtype=torch.float32)
        for hkv in range(KVH):
            # Map each kv head to a group of attention heads
            g = hkv // (H // KVH)  # groups of 12 since H=96, KVH=8 => H//KVH=12
            start = g * (H // KVH)
            end = start + (H // KVH)
            Krr[:, start:end, :, :] = Kr[:, hkv, :, :]
        # Similarly for V
        Vr = torch.empty((B, H, S, D), device=device, dtype=torch.float32)
        for hkv in range(KVH):
            g = hkv // (H // KVH)
            start = g * (H // KVH)
            end = start + (H // KVH)
            Vr[:, start:end, :, :] = V[:, hkv, :, :]

        # 5) Compute attention scores S[b,h,i,j] = (Qr[b,h,i] @ Kr[b,h,j]) * scaling
        S = torch.empty((B, H, S, S), device=device, dtype=torch.float32)
        # Launch one program per (b,h,i,j)
        grid = (B, H, S, S)
        attn_scores_kernel[grid](
            Qr, Kr, S,
            B, H, S, D,
            Qr.stride(0), Qr.stride(1), Qr.stride(2), Qr.stride(3),
            Kr.stride(0), Kr.stride(1), Kr.stride(2), Kr.stride(3),
            S.stride(0), S.stride(1), S.stride(2), S.stride(3),
            self.scaling,
        )
        # 6) Softmax over sequence dim (we implement in PyTorch for correctness)
        # Use causal mask
        causal_mask = torch.triu(
            torch.full((S, S), float('-inf'), device=device, dtype=torch.float32),
            diagonal=1
        )
        S = S + causal_mask  # broadcast over batch and heads
        Soft = F.softmax(S, dim=-1)  # [B, H, S, S]

        # 7) Compute attention output Y[b,h,i,:] = sum_j Soft[b,h,i,j] * Vr[b,h,j,:]
        Y = torch.empty((B, H, S, D), device=device, dtype=torch.float32)
        # Triton kernel to compute output per (b,h,i)
        for b_idx in range(B):
            for h_idx in range(H):
                # Compute for each i in S
                for i_idx in range(S):
                    attn_output_kernel[(1,)](
                        Soft[b_idx, h_idx, i_idx, :],  # Soft[b,h,i,:] is 1D length S
                        Vr[b_idx, h_idx, :, :],        # Vr[b,h, :, :] is [S, D]
                        Y[b_idx, h_idx, i_idx, :],
                        1, 1, S, D,
                        Soft.stride(0), Soft.stride(1), Soft.stride(2), Soft.stride(3),
                        Vr.stride(0), Vr.stride(1), Vr.stride(2), Vr.stride(3),
                        Y[b_idx, h_idx, i_idx, :].numel(), 0, 0, 0,  # dummy strides
                        num_warps=1, num_stages=1
                    )

        # 8) Output projection: Y_flat = Y @ o_proj_weight^T + o_proj_bias
        # Flatten Y to [B*S, D]
        Y_flat = Y.reshape(B_S, D)
        Output = torch.empty((B_S, o_proj_weight.shape[0]), device=device, dtype=torch.float32)
        linear_kernel[(B_S, o_proj_weight.shape[0])](
            Y_flat, o_proj_weight, (o_proj_bias if o_proj_bias is not None else torch.zeros(o_proj_weight.shape[0], device=device, dtype=torch.float32)),
            Output,
            B_S, o_proj_weight.shape[0], D,
            1, 1,
            o_proj_weight.stride(0), o_proj_weight.stride(1),
            Output.stride(0), Output.stride(1),
        )
        Output = Output.reshape(B, S, o_proj_weight.shape[0])
        return Output


def run(*args):
    return ModelNew()(*args)
