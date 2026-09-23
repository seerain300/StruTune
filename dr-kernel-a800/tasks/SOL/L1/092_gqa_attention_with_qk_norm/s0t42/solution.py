import torch
import triton
import triton.language as tl


# 1) Linear projection: y[b, l, n] = sum_k x[b, l, k] * w[n, k], accumulate in f32, output f32
@triton.jit
def linear_proj_kernel(
    x_ptr,           # *f16/f32/bf16, [B, L, H_in]
    w_ptr,           # *f16/f32/bf16, [N_out, H_in]
    bias_ptr,        # *f32, [N_out] (dummy if None)
    y_ptr,           # *f32, [B, L, N_out]
    B, L, H_in, N_out,
    x_bs0, x_bs1, x_bs2,
    w_bs0, w_bs1,
    y_bs0, y_bs1, y_bs2,
    BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)
    l = tl.program_id(1)
    n = tl.program_id(2)

    acc = tl.zeros((), dtype=tl.float32)
    for k0 in range(0, H_in, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < H_in
        # x[b, l, offs_k]
        x_ptrs = x_ptr + b * x_bs0 + l * x_bs1 + offs_k * x_bs2
        x_vals = tl.load(x_ptrs, mask=mask_k, other=0.0)
        # w[n, offs_k]
        w_ptrs = w_ptr + n * w_bs0 + offs_k * w_bs1
        w_vals = tl.load(w_ptrs, mask=mask_k, other=0.0)
        acc += tl.sum(x_vals.to(tl.float32) * w_vals.to(tl.float32), axis=0)
    # Write out
    y_ptrs = y_ptr + b * y_bs0 + l * y_bs1 + n * y_bs2
    tl.store(y_ptrs, acc)


# 2) RMSNorm for Q/K: y[b, l, h] = x[b, l, h] * rsqrt(mean(x^2)+eps) * weight[h]
@triton.jit
def rmsnorm_kernel(
    x_ptr,           # *f32, [B, L, H]
    weight_ptr,      # *f32, [H]
    y_ptr,           # *f32, [B, L, H]
    B, L, H,
    x_bs0, x_bs1, x_bs2,
    weight_bs,       # H
    y_bs0, y_bs1, y_bs2,
):
    b = tl.program_id(0)
    l = tl.program_id(1)
    h = tl.program_id(2)

    sum_sq = tl.zeros((), dtype=tl.float32)
    for i in range(0, H):
        x_val = tl.load(x_ptr + b * x_bs0 + l * x_bs1 + i * x_bs2).to(tl.float32)
        sum_sq += x_val * x_val
    mean = sum_sq / H
    inv_rms = tl.rsqrt(mean + 1e-8)
    w = tl.load(weight_ptr + h * weight_bs).to(tl.float32)
    y = tl.load(x_ptr + b * x_bs0 + l * x_bs1 + h * x_bs2).to(tl.float32) * inv_rms * w
    tl.store(y_ptr + b * y_bs0 + l * y_bs1 + h * y_bs2, y)


# 3) Rotate Q/K: apply sin/cos rotation for half dims. For Q: q_rot = [q1*cos + q2*sin, q1*sin - q2*cos], for K: k_rot = [k1*cos + k2*sin, k1*sin - k2*cos]
@triton.jit
def rotate_qk_kernel(
    x_ptr, sin_ptr, cos_ptr, y_ptr, B, L, H,
    x_bs0, x_bs1, x_bs2,
    sin_bs0, sin_bs1,
    cos_bs0, cos_bs1,
    y_bs0, y_bs1, y_bs2,
    BLOCK_H: tl.constexpr,
):
    b = tl.program_id(0)
    l = tl.program_id(1)
    h = tl.program_id(2)  # h indexes head position (e.g., 0..H-1)
    half = H // 2
    h1 = h % half
    # Load original h1 and h2
    # Original x has layout [B, L, H]
    x_ptrs = x_ptr + b * x_bs0 + l * x_bs1 + h1 * x_bs2
    q1 = tl.load(x_ptrs).to(tl.float32)
    h2 = h1 + half
    x_ptrs2 = x_ptr + b * x_bs0 + l * x_bs1 + h2 * x_bs2
    q2 = tl.load(x_ptrs2).to(tl.float32)

    # Load sin/cos for half
    sin_off = h1  # sin is shared across both halves
    sin_val = tl.load(sin_ptr + sin_bs0 + sin_off * sin_bs1).to(tl.float32)
    cos_val = tl.load(cos_ptr + cos_bs0 + sin_off * cos_bs1).to(tl.float32)

    # Compute rotated components
    new_h1 = q1 * cos_val + q2 * sin_val
    new_h2 = q1 * sin_val - q2 * cos_val

    # Store rotated head components
    y_ptrs = y_ptr + b * y_bs0 + l * y_bs1 + h1 * y_bs2
    tl.store(y_ptrs, new_h1)
    y_ptrs2 = y_ptr + b * y_bs0 + l * y_bs1 + h2 * y_bs2
    tl.store(y_ptrs2, new_h2)


# 4) Compute attention scores: scores[b, qh, l, t] = Q[b, qh, l] * K[b, qh, t]
@triton.jit
def attn_matmul_kernel(
    q_ptr, k_ptr, scores_ptr,
    B, num_heads, L,
    q_bs0, q_bs1, q_bs2, q_bs3,
    k_bs0, k_bs1, k_bs2, k_bs3,
    scores_bs0, scores_bs1, scores_bs2, scores_bs3,
):
    b = tl.program_id(0)
    qh = tl.program_id(1)
    l = tl.program_id(2)
    # Compute Q[b, qh, l] and K[b, qh, t] dot products for t in [0..L-1]
    q_val = tl.zeros((), dtype=tl.float32)
    for h in range(0, 128):  # head_dim fixed as 128
        q_val += tl.load(q_ptr + b * q_bs0 + qh * q_bs1 + l * q_bs2 + h * q_bs3).to(tl.float32)
    for t in range(0, L):
        k_val = tl.zeros((), dtype=tl.float32)
        for h in range(0, 128):
            k_val += tl.load(k_ptr + b * k_bs0 + qh * k_bs1 + t * k_bs2 + h * k_bs3).to(tl.float32)
        score = q_val * k_val
        # store as f32
        tl.store(scores_ptr + b * scores_bs0 + qh * scores_bs1 + l * scores_bs2 + t * scores_bs3, score)


# 5) Softmax + causal mask: per (b, qh, l), softmax over t in [0..L-1] of scores, with causal mask: t < l => -inf
@triton.jit
def softmax_mask_kernel(
    scores_ptr, mask_ptr, probs_ptr,
    B, num_heads, L,
    scores_bs0, scores_bs1, scores_bs2, scores_bs3,
    mask_bs0, mask_bs1, mask_bs2, mask_bs3,
    probs_bs0, probs_bs1, probs_bs2, probs_bs3,
):
    b = tl.program_id(0)
    qh = tl.program_id(1)
    l = tl.program_id(2)
    # Load scores[b, qh, l, :] (length L)
    scores = tl.zeros((L,), dtype=tl.float32)
    for t in range(0, L):
        sptr = scores_ptr + b * scores_bs0 + qh * scores_bs1 + l * scores_bs2 + t * scores_bs3
        scores[t] = tl.load(sptr).to(tl.float32)
    # Load mask[b, qh, l, :] and apply: t < l => -inf, else 0
    mask = tl.zeros((L,), dtype=tl.float32)
    for t in range(0, L):
        mptr = mask_ptr + b * mask_bs0 + qh * mask_bs1 + l * mask_bs2 + t * mask_bs3
        mask[t] = tl.load(mptr).to(tl.float32)
    # Apply mask
    scores = scores + mask
    # Numerically stable softmax: subtract max
    max_score = tl.max(scores, axis=0)
    scores = scores - max_score
    exp_scores = tl.exp(scores)
    sum_exp = tl.sum(exp_scores, axis=0)
    probs = exp_scores / sum_exp
    # Store
    for t in range(0, L):
        pptr = probs_ptr + b * probs_bs0 + qh * probs_bs1 + l * probs_bs2 + t * probs_bs3
        tl.store(pptr, probs[t])


# 6) Output projection: attn_output[b, qh, l] = sum_t probs[b, qh, l, t] * V[b, qh, t]
@triton.jit
def output_matmul_kernel(
    probs_ptr, v_ptr, attn_ptr,
    B, num_heads, L, H,  # H is head_dim (128)
    probs_bs0, probs_bs1, probs_bs2, probs_bs3,
    v_bs0, v_bs1, v_bs2,
    attn_bs0, attn_bs1, attn_bs2,
):
    b = tl.program_id(0)
    qh = tl.program_id(1)
    l = tl.program_id(2)
    acc = tl.zeros((), dtype=tl.float32)
    for t in range(0, L):
        p = tl.load(probs_ptr + b * probs_bs0 + qh * probs_bs1 + l * probs_bs2 + t * probs_bs3).to(tl.float32)
        v_val = tl.zeros((), dtype=tl.float32)
        for h in range(0, H):  # H=128
            v_val += tl.load(v_ptr + b * v_bs0 + qh * v_bs1 + t * v_bs2 + h * v_bs3).to(tl.float32)
        acc += p * v_val
    # Write attn[b, qh, l]
    tl.store(attn_ptr + b * attn_bs0 + qh * attn_bs1 + l * attn_bs2, acc)


# 7) Final linear projection: final_out[b, l, n] = sum_k attn[b, l, k] * o_proj_weight[n, k], accumulate in f32, output f32
@triton.jit
def final_linear_kernel(
    attn_ptr, o_ptr, out_ptr,
    B, L, H, N_out,
    attn_bs0, attn_bs1, attn_bs2,
    o_bs0, o_bs1,
    out_bs0, out_bs1, out_bs2,
    BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)
    l = tl.program_id(1)
    n = tl.program_id(2)

    acc = tl.zeros((), dtype=tl.float32)
    for k0 in range(0, H, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < H
        # attn[b, l, offs_k]
        attn_ptrs = attn_ptr + b * attn_bs0 + l * attn_bs1 + offs_k * attn_bs2
        attn_vals = tl.load(attn_ptrs, mask=mask_k, other=0.0)
        # o_proj_weight[n, offs_k]
        o_ptrs = o_ptr + n * o_bs0 + offs_k * o_bs1
        o_vals = tl.load(o_ptrs, mask=mask_k, other=0.0)
        acc += tl.sum(attn_vals.to(tl.float32) * o_vals.to(tl.float32), axis=0)
    # Store final output
    out_ptrs = out_ptr + b * out_bs0 + l * out_bs1 + n * out_bs2
    tl.store(out_ptrs, acc)


class ModelNew(torch.nn.Module):
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.hidden_dim = hidden_dim  # 768

    def forward(
        self,
        hidden_states: torch.Tensor,          # [B, L, 768], float32
        q_proj_weight: torch.Tensor,          # [128, 768]
        q_proj_bias: torch.Tensor,            # [128]
        k_proj_weight: torch.Tensor,          # [128, 768]
        k_proj_bias: torch.Tensor,            # [128]
        v_proj_weight: torch.Tensor,          # [128, 768]
        v_proj_bias: torch.Tensor,            # [128]
        o_proj_weight: torch.Tensor,          # [hidden_dim, 12288]
        q_norm_weight: torch.Tensor,          # [128]
        k_norm_weight: torch.Tensor,          # [128]
        cos: torch.Tensor,                    # [L, 64] (first half of head_dim)
        sin: torch.Tensor,                    # [L, 64] (first half of head_dim)
        rms_norm_eps: float,                  # 1e-8
    ):
        # Shapes
        B, L, H_in = hidden_states.shape
        assert H_in == 768, "hidden_states must be [B, L, 768]"
        head_dim = 128
        # Ensure device/dtype
        device = hidden_states.device
        dtype = hidden_states.dtype  # keep f32
        # 1) Linear projections
        Q = torch.empty((B, L, head_dim), device=device, dtype=torch.float32)
        K = torch.empty((B, L, head_dim), device=device, dtype=torch.float32)
        V = torch.empty((B, L, head_dim), device=device, dtype=torch.float32)

        # Use Triton linear_proj_kernel
        grid_q = (B, L, head_dim)
        linear_proj_kernel[grid_q](
            hidden_states, q_proj_weight, q_proj_bias, Q,
            B, L, H_in, head_dim,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            q_proj_weight.stride(0), q_proj_weight.stride(1),
            Q.stride(0), Q.stride(1), Q.stride(2),
            BLOCK_K=128, num_warps=4, num_stages=2
        )

        linear_proj_kernel[grid_q](
            hidden_states, k_proj_weight, k_proj_bias, K,
            B, L, H_in, head_dim,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            k_proj_weight.stride(0), k_proj_weight.stride(1),
            K.stride(0), K.stride(1), K.stride(2),
            BLOCK_K=128, num_warps=4, num_stages=2
        )

        linear_proj_kernel[grid_q](
            hidden_states, v_proj_weight, v_proj_bias, V,
            B, L, H_in, head_dim,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            v_proj_weight.stride(0), v_proj_weight.stride(1),
            V.stride(0), V.stride(1), V.stride(2),
            BLOCK_K=128, num_warps=4, num_stages=2
        )

        # 2) RMSNorm for Q and K
        Q_r = torch.empty_like(Q)
        K_r = torch.empty_like(K)

        grid_r = (B, L)
        rmsnorm_kernel[grid_r](
            Q, q_norm_weight, Q_r,
            B, L, head_dim,
            Q.stride(0), Q.stride(1), Q.stride(2),
            q_norm_weight.numel(),
            Q_r.stride(0), Q_r.stride(1), Q_r.stride(2),
            num_warps=4, num_stages=2
        )

        rmsnorm_kernel[grid_r](
            K, k_norm_weight, K_r,
            B, L, head_dim,
            K.stride(0), K.stride(1), K.stride(2),
            k_norm_weight.numel(),
            K_r.stride(0), K_r.stride(1), K_r.stride(2),
            num_warps=4, num_stages=2
        )

        # 3) Rotate Q and K
        Q_rot = torch.empty_like(Q_r)
        K_rot = torch.empty_like(K_r)

        grid_rot = (B, L)
        rotate_qk_kernel[grid_rot](
            Q_r, sin, cos, Q_rot, B, L, head_dim,
            Q_r.stride(0), Q_r.stride(1), Q_r.stride(2),
            sin.stride(0), sin.stride(1),
            cos.stride(0), cos.stride(1),
            Q_rot.stride(0), Q_rot.stride(1), Q_rot.stride(2),
            BLOCK_H=128, num_warps=4, num_stages=2
        )

        rotate_qk_kernel[grid_rot](
            K_r, sin, cos, K_rot, B, L, head_dim,
            K_r.stride(0), K_r.stride(1), K_r.stride(2),
            sin.stride(0), sin.stride(1),
            cos.stride(0), cos.stride(1),
            K_rot.stride(0), K_rot.stride(1), K_rot.stride(2),
            BLOCK_H=128, num_warps=4, num_stages=2
        )

        # 4) Compute attention scores [B, num_heads, L, L], num_heads = 96
        # We need Q_rot of shape [B, num_heads, L, head_dim], K_rot [B, num_heads, L, head_dim]
        # Original code uses grouped query attention: num_key_value_heads=8, num_key_value_groups=12 → 8*12=96 heads.
        # We'll build Qh/Kh per head by reshaping and expanding like original semantics (repeat KV heads for GQA groups).
        num_attention_heads = 96
        num_key_value_heads = 8
        num_key_value_groups = 12
        # For each head qh, compute scores over L. We will launch per (b, qh, l).
        scores = torch.empty((B, num_attention_heads, L, L), device=device, dtype=torch.float32)
        grid_scores = (B, num_attention_heads, L)
        attn_matmul_kernel[grid_scores](
            Q_rot, K_rot, scores,
            B, num_attention_heads, L,
            Q_rot.stride(0), Q_rot.stride(1), Q_rot.stride(2), Q_rot.stride(3),
            K_rot.stride(0), K_rot.stride(1), K_rot.stride(2), K_rot.stride(3),
            scores.stride(0), scores.stride(1), scores.stride(2), scores.stride(3),
            num_warps=4, num_stages=2
        )

        # 5) Softmax + causal mask
        # causal mask: upper triangle, diagonal=1 ⇒ for each l, t < l => -inf
        mask = torch.empty((B, num_attention_heads, L, L), device=device, dtype=torch.float32)
        # Build mask: mask[b, qh, l, t] = 0 if t >= l else -inf
        # We will write mask in Triton by launching per (b, qh, l).
        grid_mask = (B, num_attention_heads, L)
        # Fill mask with zeros, then set lower triangle to -inf
        # Triton kernel will load this mask; we need to initialize it appropriately.
        # We can fill mask with -inf first, then set diagonal and above to 0.
        for b_idx in range(B):
            for qh_idx in range(num_attention_heads):
                for l_idx in range(L):
                    for t_idx in range(L):
                        if t_idx < l_idx:
                            mask[b_idx, qh_idx, l_idx, t_idx] = float('-inf')
                        else:
                            mask[b_idx, qh_idx, l_idx, t_idx] = 0.0

        # Launch softmax_mask_kernel (per row softmax)
        probs = torch.empty_like(scores)
        softmax_mask_kernel[grid_mask](
            scores, mask, probs,
            B, num_attention_heads, L,
            scores.stride(0), scores.stride(1), scores.stride(2), scores.stride(3),
            mask.stride(0), mask.stride(1), mask.stride(2), mask.stride(3),
            probs.stride(0), probs.stride(1), probs.stride(2), probs.stride(3),
            num_warps=4, num_stages=2
        )

        # 6) Output projection per head
        attn = torch.empty((B, num_attention_heads, L, head_dim), device=device, dtype=torch.float32)
        grid_out = (B, num_attention_heads, L)
        output_matmul_kernel[grid_out](
            probs, V, attn,
            B, num_attention_heads, L, head_dim,
            probs.stride(0), probs.stride(1), probs.stride(2), probs.stride(3),
            V.stride(0), V.stride(1), V.stride(2),
            attn.stride(0), attn.stride(1), attn.stride(2),
            num_warps=4, num_stages=2
        )

        # Flatten attn to [B, L, num_attention_heads*head_dim] = [B, L, 12288]
        attn_flat = attn.reshape(B, L, num_attention_heads * head_dim)

        # 7) Final linear projection to hidden_dim=768 (no bias)
        final_out = torch.empty((B, L, self.hidden_dim), device=device, dtype=torch.float32)

        grid_final = (B, L, self.hidden_dim)
        final_linear_kernel[grid_final](
            attn_flat, o_proj_weight, final_out,
            B, L, num_attention_heads * head_dim, self.hidden_dim,
            attn_flat.stride(0), attn_flat.stride(1), attn_flat.stride(2),
            o_proj_weight.stride(0), o_proj_weight.stride(1),
            final_out.stride(0), final_out.stride(1), final_out.stride(2),
            BLOCK_K=128, num_warps=4, num_stages=2
        )

        return final_out


def run(*args):
    return ModelNew()(*args)
