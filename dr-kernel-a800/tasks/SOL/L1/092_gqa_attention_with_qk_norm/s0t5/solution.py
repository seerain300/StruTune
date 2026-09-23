import torch
import triton
import triton.language as tl

# 1) Linear projection: y[b, l, n] = sum_k x[b, l, k] * w[n, k], accumulate in f32, output f32
@triton.jit
def linear_proj_kernel(
    x_ptr,           # *f16/f32, [B, L, H_in]
    w_ptr,           # *f16/f32, [N_out, H_in]
    bias_ptr,        # *f32, [N_out] (can be dummy if None)
    y_ptr,           # *f32, [B, L, N_out]
    B, L, H_in, N_out,
    x_bs0, x_bs1, x_bs2,
    w_bs0, w_bs1,
    y_bs0, y_bs1, y_bs2,
    BLOCK_K: tl.constexpr,
):
    # Each program computes y[b, l, n]
    b = tl.program_id(0)
    l = tl.program_id(1)
    n = tl.program_id(2)

    acc = tl.zeros((), dtype=tl.float32)
    for k0 in range(0, H_in, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < H_in

        x_ptrs = x_ptr + b * x_bs0 + l * x_bs1 + offs_k * x_bs2
        x_vals = tl.load(x_ptrs, mask=mask_k, other=0.0).to(tl.float32)

        w_ptrs = w_ptr + n * w_bs0 + offs_k * w_bs1
        w_vals = tl.load(w_ptrs, mask=mask_k, other=0.0).to(tl.float32)

        acc += tl.sum(x_vals * w_vals, axis=0)

    bias_val = tl.load(bias_ptr + n).to(tl.float32)
    out = acc + bias_val

    y_ptrs = y_ptr + b * y_bs0 + l * y_bs1 + n * y_bs2
    tl.store(y_ptrs, out)


# 2) RMSNorm for per-row: y = x * rsqrt(mean(x^2) + eps) * weight
@triton.jit
def rmsnorm_kernel(
    x_ptr,          # *f32, [B, L, H]
    weight_ptr,     # *f32, [H]
    y_ptr,          # *f32, [B, L, H]
    B, L, H,
    x_bs0, x_bs1, x_bs2,
    y_bs0, y_bs1, y_bs2,
    eps,
    BLOCK_H: tl.constexpr,
):
    b = tl.program_id(0)
    l = tl.program_id(1)
    # Reduce over H in chunks
    sumsq = tl.zeros((), dtype=tl.float32)
    for h0 in range(0, H, BLOCK_H):
        offs_h = h0 + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H
        x_ptrs = x_ptr + b * x_bs0 + l * x_bs1 + offs_h * x_bs2
        x_vals = tl.load(x_ptrs, mask=mask_h, other=0.0).to(tl.float32)
        sumsq += tl.sum(x_vals * x_vals, axis=0)

    mean = sumsq / H
    inv_rms = 1.0 / tl.sqrt(mean + eps)

    for h0 in range(0, H, BLOCK_H):
        offs_h = h0 + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H
        x_ptrs = x_ptr + b * x_bs0 + l * x_bs1 + offs_h * x_bs2
        x_vals = tl.load(x_ptrs, mask=mask_h, other=0.0).to(tl.float32)

        w_ptrs = weight_ptr + offs_h
        w_vals = tl.load(w_ptrs, mask=mask_h, other=1.0).to(tl.float32)

        y_vals = x_vals * inv_rms * w_vals
        y_ptrs = y_ptr + b * y_bs0 + l * y_bs1 + offs_h * y_bs2
        tl.store(y_ptrs, y_vals)


# 3) Q/K Rotation (RoPE): for each q[k] where k in [0,63], rotated = q1*cos + q2*(-sin)
@triton.jit
def rotate_qk_kernel(
    z_ptr,          # *f32, [B, L, H] input after RMSNorm
    cos_ptr,        # *f32, [L, H//2]
    sin_ptr,        # *f32, [L, H//2]
    y_ptr,          # *f32, [B, L, H] output rotated
    B, L, H,
    z_bs0, z_bs1, z_bs2,
    cos_bs0, cos_bs1,
    sin_bs0, sin_bs1,
    y_bs0, y_bs1, y_bs2,
    BLOCK_H: tl.constexpr,
):
    b = tl.program_id(0)
    l = tl.program_id(1)
    for h0 in range(0, H, BLOCK_H):
        offs_h = h0 + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H

        z_ptrs = z_ptr + b * z_bs0 + l * z_bs1 + offs_h * z_bs2
        z_vals = tl.load(z_ptrs, mask=mask_h, other=0.0).to(tl.float32)

        h1 = offs_h[:64]
        h2 = offs_h[64:]

        cos_ptrs1 = cos_ptr + l * cos_bs0 + h1 * cos_bs1
        sin_ptrs1 = sin_ptr + l * sin_bs0 + h1 * sin_bs1

        cos1 = tl.load(cos_ptrs1, mask=h1 < 64, other=1.0).to(tl.float32)
        sin1 = tl.load(sin_ptrs1, mask=h1 < 64, other=0.0).to(tl.float32)

        q1 = z_vals[:64]
        q2 = z_vals[64:]

        # rotated = q1 * cos + (-q2) * sin
        rotated1 = q1 * cos1 - q2 * sin1
        rotated = tl.zeros((H,), dtype=tl.float32)
        rotated[:64] = rotated1
        # h2 remains original (no rotation applied to h2 in the original semantics)
        rotated[64:] = z_vals[64:]

        y_ptrs = y_ptr + b * y_bs0 + l * y_bs1 + offs_h * y_bs2
        tl.store(y_ptrs, rotated)


# 4) Attention Score Matmul: out[b, qh, l1, l2] = sum_{hdim} query[b, qh, l1, hdim] * key[b, qh, l2, hdim]
@triton.jit
def attn_score_matmul_kernel(
    Q_ptr,           # *f32, [B, num_heads, L, head_dim]
    K_ptr,           # *f32, [B, num_heads, L, head_dim]
    Out_ptr,         # *f32, [B, num_heads, L, L]
    B, num_heads, L, head_dim,
    Q_bs0, Q_bs1, Q_bs2, Q_bs3,
    K_bs0, K_bs1, K_bs2, K_bs3,
    Out_bs0, Out_bs1, Out_bs2, Out_bs3,
    BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr,
):
    b = tl.program_id(0)
    qh = tl.program_id(1)
    l1 = tl.program_id(2)  # row index (query position)
    # We will fill entire Out row for this (b, qh, l1) across all l2
    # Accumulate in f32
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Loop over K-dimension (head_dim) in chunks
    for k0 in range(0, head_dim, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < head_dim

        # Load Q row slice: [BLOCK_M]
        Q_ptrs = Q_ptr + b * Q_bs0 + qh * Q_bs1 + l1 * Q_bs2 + offs_k * Q_bs3
        Q_row = tl.load(Q_ptrs, mask=mask_k, other=0.0).to(tl.float32)

        # Accumulate K across all l2 positions in chunks of BLOCK_M
        for l2_base in range(0, L, BLOCK_M):
            offs_l2 = l2_base + tl.arange(0, BLOCK_M)
            mask_l2 = offs_l2 < L

            K_ptrs = K_ptr + b * K_bs0 + qh * K_bs1 + offs_l2 * K_bs2 + offs_k * K_bs3
            K_block = tl.load(K_ptrs, mask=mask_l2[:, None] & mask_k[None, :], other=0.0).to(tl.float32)  # [BLOCK_M, BLOCK_K]
            # Dot product along K: [BLOCK_M] * [BLOCK_M, BLOCK_K]^T -> [BLOCK_M]
            prod = tl.sum(K_block * Q_row[None, :], axis=1)  # sum over K-block
            acc += prod

    # Store to Out
    Out_ptrs = Out_ptr + b * Out_bs0 + qh * Out_bs1 + l1 * Out_bs2 + tl.arange(0, L) * Out_bs3
    mask_all = tl.arange(0, L) < L
    tl.store(Out_ptrs, acc, mask=mask_all)


# 5) Softmax with causal mask (stable): apply mask where l2 <= l1 else -inf, then softmax per row
@triton.jit
def softmax_mask_kernel(
    scores_ptr,      # *f32, [B, num_heads, L, L]
    mask_ptr,        # *f32, [B, num_heads, L, L] (already filled with -inf or 0)
    out_ptr,         # *f32, [B, num_heads, L, L]
    B, num_heads, L,
    scores_bs0, scores_bs1, scores_bs2, scores_bs3,
    mask_bs0, mask_bs1, mask_bs2, mask_bs3,
    out_bs0, out_bs1, out_bs2, out_bs3,
    BLOCK_M: tl.constexpr,
):
    b = tl.program_id(0)
    qh = tl.program_id(1)
    l1 = tl.program_id(2)  # row index
    # We compute softmax across L for each row (l1 fixed). Use BLOCK_M chunking.
    # Find max for numerical stability
    max_val = -float('inf')
    for l2_base in range(0, L, BLOCK_M):
        offs_l2 = l2_base + tl.arange(0, BLOCK_M)
        mask_l2 = offs_l2 < L

        scores_ptrs = scores_ptr + b * scores_bs0 + qh * scores_bs1 + l1 * scores_bs2 + offs_l2 * scores_bs3
        scores_chunk = tl.load(scores_ptrs, mask=mask_l2, other=-float('inf')).to(tl.float32)
        # Apply mask: load mask and set negative where mask==0 (i.e., non-causal)
        mask_ptrs = mask_ptr + b * mask_bs0 + qh * mask_bs1 + l1 * mask_bs2 + offs_l2 * mask_bs3
        mask_chunk = tl.load(mask_ptrs, mask=mask_l2, other=0.0).to(tl.float32)
        scores_chunk = scores_chunk + mask_chunk * (-float('inf'))
        max_val = tl.maximum(max_val, tl.max(scores_chunk, axis=0))

    # Compute exponentials and normalize
    sum_exp = 0.0
    for l2_base in range(0, L, BLOCK_M):
        offs_l2 = l2_base + tl.arange(0, BLOCK_M)
        mask_l2 = offs_l2 < L

        scores_ptrs = scores_ptr + b * scores_bs0 + qh * scores_bs1 + l1 * scores_bs2 + offs_l2 * scores_bs3
        scores_chunk = tl.load(scores_ptrs, mask=mask_l2, other=-float('inf')).to(tl.float32)
        mask_ptrs = mask_ptr + b * mask_bs0 + qh * mask_bs1 + l1 * mask_bs2 + offs_l2 * mask_bs3
        mask_chunk = tl.load(mask_ptrs, mask=mask_l2, other=0.0).to(tl.float32)
        scores_chunk = scores_chunk + mask_chunk * (-float('inf'))
        exp_chunk = tl.exp(scores_chunk - max_val)
        sum_exp += tl.sum(exp_chunk, axis=0)

    for l2_base in range(0, L, BLOCK_M):
        offs_l2 = l2_base + tl.arange(0, BLOCK_M)
        mask_l2 = offs_l2 < L

        scores_ptrs = scores_ptr + b * scores_bs0 + qh * scores_bs1 + l1 * scores_bs2 + offs_l2 * scores_bs3
        scores_chunk = tl.load(scores_ptrs, mask=mask_l2, other=-float('inf')).to(tl.float32)
        mask_ptrs = mask_ptr + b * mask_bs0 + qh * mask_bs1 + l1 * mask_bs2 + offs_l2 * mask_bs3
        mask_chunk = tl.load(mask_ptrs, mask=mask_l2, other=0.0).to(tl.float32)
        scores_chunk = scores_chunk + mask_chunk * (-float('inf'))
        exp_chunk = tl.exp(scores_chunk - max_val) / sum_exp

        out_ptrs = out_ptr + b * out_bs0 + qh * out_bs1 + l1 * out_bs2 + offs_l2 * out_bs3
        tl.store(out_ptrs, exp_chunk, mask=mask_l2)


# 6) Output Matmul: out[b, qh, l, head_dim] = sum_{L} attn[b, qh, l, l2] * V[b, qh, l2, head_dim]
@triton.jit
def output_matmul_kernel(
    attn_ptr,        # *f32, [B, num_heads, L, L]
    V_ptr,           # *f32, [B, num_heads, L, head_dim]
    out_ptr,         # *f32, [B, num_heads, L, head_dim]
    B, num_heads, L, head_dim,
    attn_bs0, attn_bs1, attn_bs2, attn_bs3,
    V_bs0, V_bs1, V_bs2, V_bs3,
    out_bs0, out_bs1, out_bs2, out_bs3,
    BLOCK_L: tl.constexpr, BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)
    qh = tl.program_id(1)
    l = tl.program_id(2)
    acc = tl.zeros((head_dim,), dtype=tl.float32)
    for l2_base in range(0, L, BLOCK_L):
        offs_l2 = l2_base + tl.arange(0, BLOCK_L)
        mask_l2 = offs_l2 < L

        attn_ptrs = attn_ptr + b * attn_bs0 + qh * attn_bs1 + l * attn_bs2 + offs_l2 * attn_bs3
        attn_row = tl.load(attn_ptrs, mask=mask_l2, other=0.0).to(tl.float32)  # [BLOCK_L]

        for k0 in range(0, head_dim, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < head_dim

            V_ptrs = V_ptr + b * V_bs0 + qh * V_bs1 + offs_l2[:, None] * V_bs2 + offs_k[None, :] * V_bs3
            V_block = tl.load(V_ptrs, mask=mask_l2[:, None] & mask_k[None, :], other=0.0).to(tl.float32)  # [BLOCK_L, BLOCK_K]

            prod = tl.sum(V_block * attn_row[:, None], axis=0)  # sum over L-block -> [BLOCK_K]
            acc += prod

    out_ptrs = out_ptr + b * out_bs0 + qh * out_bs1 + l * out_bs2 + tl.arange(0, head_dim) * out_bs3
    mask_all = tl.arange(0, head_dim) < head_dim
    tl.store(out_ptrs, acc, mask=mask_all)


# 7) Final Linear Projection: out[b, l, hidden_dim] = sum_{n} attn_output[b, l, n] * o_proj_weight[hidden_dim, n]
@triton.jit
def final_linear_kernel(
    attn_flat_ptr,   # *f32, [B, L, num_heads*head_dim] flattened
    o_proj_ptr,      # *f32, [hidden_dim, num_heads*head_dim]
    out_ptr,         # *f32, [B, L, hidden_dim]
    B, L, hidden_dim, H_flat,
    attn_bs0, attn_bs1, attn_bs2,
    o_proj_bs0, o_proj_bs1,
    out_bs0, out_bs1, out_bs2,
    BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)
    l = tl.program_id(1)
    n_out = tl.program_id(2)  # hidden_dim index
    acc = tl.zeros((), dtype=tl.float32)

    for k0 in range(0, H_flat, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < H_flat

        attn_ptrs = attn_flat_ptr + b * attn_bs0 + l * attn_bs1 + offs_k * attn_bs2
        attn_vals = tl.load(attn_ptrs, mask=mask_k, other=0.0).to(tl.float32)  # [BLOCK_K]

        o_ptrs = o_proj_ptr + n_out * o_proj_bs0 + offs_k * o_proj_bs1
        o_vals = tl.load(o_ptrs, mask=mask_k, other=0.0).to(tl.float32)  # [BLOCK_K]

        acc += tl.sum(attn_vals * o_vals, axis=0)

    out_ptrs = out_ptr + b * out_bs0 + l * out_bs1 + n_out * out_bs2
    tl.store(out_ptrs, acc)


class ModelNew(torch.nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim  # e.g., 768

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_proj_weight: torch.Tensor,
        q_proj_bias: torch.Tensor,
        k_proj_weight: torch.Tensor,
        k_proj_bias: torch.Tensor,
        v_proj_weight: torch.Tensor,
        v_proj_bias: torch.Tensor,
        o_proj_weight: torch.Tensor,     # [hidden_dim, num_heads*head_dim]
        q_norm_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
        cos: torch.Tensor,               # [L, head_dim//2]
        sin: torch.Tensor,               # [L, head_dim//2]
        rms_norm_eps: float,
    ):
        # Shapes derived from input (original head_dim=128, num_heads=96, num_key_value_heads=8, num_key_value_groups=12)
        B, L, H_in = hidden_states.shape  # hidden_states: [B, L, 768]
        head_dim = 128
        num_heads = 96
        num_key_value_heads = 8
        num_key_value_groups = 12
        assert head_dim % 2 == 0, "head_dim must be even for split rotation."

        # Ensure device/dtype
        device = hidden_states.device
        dtype = hidden_states.dtype

        # 1) Q, K, V linear projection (compute in f32)
        # Cast weights to f32 for kernel math
        q_w = q_proj_weight.to(device=device, dtype=torch.float32)
        k_w = k_proj_weight.to(device=device, dtype=torch.float32)
        v_w = v_proj_weight.to(device=device, dtype=torch.float32)

        Q = torch.empty((B, L, head_dim), device=device, dtype=torch.float32)
        K = torch.empty((B, L, head_dim), device=device, dtype=torch.float32)
        V = torch.empty((B, L, head_dim), device=device, dtype=torch.float32)

        grid_q = (B, L, head_dim)
        linear_proj_kernel[grid_q](
            hidden_states, q_w, q_proj_bias, Q,
            B, L, H_in, head_dim,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            q_w.stride(0), q_w.stride(1),
            Q.stride(0), Q.stride(1), Q.stride(2),
            BLOCK_K=64, num_warps=2, num_stages=2
        )

        grid_k = (B, L, head_dim)
        linear_proj_kernel[grid_k](
            hidden_states, k_w, k_proj_bias, K,
            B, L, H_in, head_dim,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            k_w.stride(0), k_w.stride(1),
            K.stride(0), K.stride(1), K.stride(2),
            BLOCK_K=64, num_warps=2, num_stages=2
        )

        grid_v = (B, L, head_dim)
        linear_proj_kernel[grid_v](
            hidden_states, v_w, v_proj_bias, V,
            B, L, H_in, head_dim,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            v_w.stride(0), v_w.stride(1),
            V.stride(0), V.stride(1), V.stride(2),
            BLOCK_K=64, num_warps=2, num_stages=2
        )

        # 2) RMSNorm for Q and K (in f32)
        q_norm_w = q_norm_weight.to(device=device, dtype=torch.float32)
        k_norm_w = k_norm_weight.to(device=device, dtype=torch.float32)

        Q_norm = torch.empty_like(Q, dtype=torch.float32)
        K_norm = torch.empty_like(K, dtype=torch.float32)

        grid_norm = (B, L)
        rmsnorm_kernel[grid_norm](
            Q, q_norm_w, Q_norm, B, L, head_dim,
            Q.stride(0), Q.stride(1), Q.stride(2),
            Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(2),
            float(rms_norm_eps),
            BLOCK_H=128, num_warps=4, num_stages=2
        )

        rmsnorm_kernel[grid_norm](
            K, k_norm_w, K_norm, B, L, head_dim,
            K.stride(0), K.stride(1), K.stride(2),
            K_norm.stride(0), K_norm.stride(1), K_norm.stride(2),
            float(rms_norm_eps),
            BLOCK_H=128, num_warps=4, num_stages=2
        )

        # 3) Rotate Q and K (RoPE) in f32
        cos = cos.to(device=device, dtype=torch.float32)
        sin = sin.to(device=device, dtype=torch.float32)

        Q_rot = torch.empty_like(Q_norm, dtype=torch.float32)
        K_rot = torch.empty_like(K_norm, dtype=torch.float32)

        grid_rotate = (B, L)
        rotate_qk_kernel[grid_rotate](
            Q_norm, cos, sin, Q_rot,
            B, L, head_dim,
            Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(2),
            cos.stride(0), cos.stride(1),
            sin.stride(0), sin.stride(1),
            Q_rot.stride(0), Q_rot.stride(1), Q_rot.stride(2),
            BLOCK_H=128, num_warps=4, num_stages=2
        )

        rotate_qk_kernel[grid_rotate](
            K_norm, cos, sin, K_rot,
            B, L, head_dim,
            K_norm.stride(0), K_norm.stride(1), K_norm.stride(2),
            cos.stride(0), cos.stride(1),
            sin.stride(0), sin.stride(1),
            K_rot.stride(0), K_rot.stride(1), K_rot.stride(2),
            BLOCK_H=128, num_warps=4, num_stages=2
        )

        # 4) Compute attention scores: [B, num_heads, L, L]
        attn_scores = torch.empty((B, num_heads, L, L), device=device, dtype=torch.float32)
        grid_score = (B, num_heads, L)
        attn_score_matmul_kernel[grid_score](
            Q_rot, K_rot, attn_scores,
            B, num_heads, L, head_dim,
            Q_rot.stride(0), Q_rot.stride(1), Q_rot.stride(2), Q_rot.stride(3),
            K_rot.stride(0), K_rot.stride(1), K_rot.stride(2), K_rot.stride(3),
            attn_scores.stride(0), attn_scores.stride(1), attn_scores.stride(2), attn_scores.stride(3),
            BLOCK_K=64, BLOCK_M=64, num_warps=4, num_stages=2
        )

        # 5) Generate causal mask: [B, num_heads, L, L], causal: l2 <= l1 allowed, else -inf
        # We'll build it on host to avoid extra kernel complexity; tiny tensor.
        causal_mask = torch.empty((B, num_heads, L, L), device=device, dtype=torch.float32)
        for b_idx in range(B):
            for qh in range(num_heads):
                for l1 in range(L):
                    for l2 in range(L):
                        causal_mask[b_idx, qh, l1, l2] = 0.0 if (l2 <= l1) else (-float('inf'))

        # 6) Apply softmax with mask (stable) per (b, qh, l1)
        attn_probs = torch.empty_like(attn_scores)
        grid_softmax = (B, num_heads, L)
        softmax_mask_kernel[grid_softmax](
            attn_scores, causal_mask, attn_probs,
            B, num_heads, L,
            attn_scores.stride(0), attn_scores.stride(1), attn_scores.stride(2), attn_scores.stride(3),
            causal_mask.stride(0), causal_mask.stride(1), causal_mask.stride(2), causal_mask.stride(3),
            attn_probs.stride(0), attn_probs.stride(1), attn_probs.stride(2), attn_probs.stride(3),
            BLOCK_M=128, num_warps=4, num_stages=2
        )

        # 7) Compute attn_output = attn_probs @ V
        attn_out = torch.empty((B, num_heads, L, head_dim), device=device, dtype=torch.float32)
        grid_output = (B, num_heads, L)
        output_matmul_kernel[grid_output](
            attn_probs, V, attn_out,
            B, num_heads, L, head_dim,
            attn_probs.stride(0), attn_probs.stride(1), attn_probs.stride(2), attn_probs.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            attn_out.stride(0), attn_out.stride(1), attn_out.stride(2), attn_out.stride(3),
            BLOCK_L=64, BLOCK_K=64, num_warps=4, num_stages=2
        )

        # 8) Flatten to [B, L, num_heads*head_dim] then final linear projection to [B, L, hidden_dim]
        attn_flat = attn_out.reshape(B, L, num_heads * head_dim).contiguous()
        final_out = torch.empty((B, L, self.hidden_dim), device=device, dtype=torch.float32)

        # o_proj_weight: [hidden_dim, num_heads*head_dim]
        o_w = o_proj_weight.to(device=device, dtype=torch.float32)

        grid_final = (B, L, self.hidden_dim)
        final_linear_kernel[grid_final](
            attn_flat, o_w, final_out,
            B, L, self.hidden_dim, num_heads * head_dim,
            attn_flat.stride(0), attn_flat.stride(1), attn_flat.stride(2),
            o_w.stride(0), o_w.stride(1),
            final_out.stride(0), final_out.stride(1), final_out.stride(2),
            BLOCK_K=64, num_warps=2, num_stages=2
        )

        return final_out


def run(*args):
    return ModelNew()(*args)
