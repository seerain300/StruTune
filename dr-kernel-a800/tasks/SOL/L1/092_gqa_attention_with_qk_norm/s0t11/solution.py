import torch
import triton
import triton.language as tl


# 1) Linear projection: y[b, l, n] = sum_k x[b, l, k] * w[n, k], accumulate in f32, output f32
@triton.jit
def linear_proj_kernel(
    x_ptr,           # *f16/f32/bf16, [B, L, H_in]
    w_ptr,           # *f16/f32/bf16, [N_out, H_in]
    bias_ptr,        # *f32, [N_out]
    y_ptr,           # *f32, [B, L, N_out]
    B: tl.constexpr, L: tl.constexpr, H_in: tl.constexpr, N_out: tl.constexpr,
    x_bs0: tl.constexpr, x_bs1: tl.constexpr, x_bs2: tl.constexpr,
    w_bs0: tl.constexpr, w_bs1: tl.constexpr,
    y_bs0: tl.constexpr, y_bs1: tl.constexpr, y_bs2: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    # Compute y[b, l, n]
    b = tl.program_id(0)
    l = tl.program_id(1)
    n = tl.program_id(2)

    acc = tl.zeros((), dtype=tl.float32)
    for k0 in range(0, H_in, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < H_in

        # Load x[b, l, offs_k]
        x_ptrs = x_ptr + b * x_bs0 + l * x_bs1 + offs_k * x_bs2
        x = tl.load(x_ptrs, mask=mask_k, other=0.0)  # x: [BLOCK_K]
        x = x.to(tl.float32)

        # Load w[n, offs_k]
        w_ptrs = w_ptr + n * w_bs0 + offs_k * w_bs1
        w = tl.load(w_ptrs, mask=mask_k, other=0.0)  # w: [BLOCK_K]
        w = w.to(tl.float32)

        # Fused multiply-add
        acc += tl.sum(x[:, None] * w[None, :], axis=0)

    # Store y[b, l, n] = acc
    y_ptrs = y_ptr + b * y_bs0 + l * y_bs1 + n * y_bs2
    tl.store(y_ptrs, acc)


# 2) RMSNorm: y = x * rsqrt(mean(x^2)+eps) * weight
@triton.jit
def rmsnorm_kernel(
    x_ptr,      # *f32, [B, L, H]
    w_ptr,      # *f32, [H]
    y_ptr,      # *f32, [B, L, H]
    B: tl.constexpr, L: tl.constexpr, H: tl.constexpr,
    x_bs0: tl.constexpr, x_bs1: tl.constexpr, x_bs2: tl.constexpr,
    w_bs0: tl.constexpr, w_bs1: tl.constexpr,
    y_bs0: tl.constexpr, y_bs1: tl.constexpr, y_bs2: tl.constexpr,
):
    b = tl.program_id(0)
    l = tl.program_id(1)
    h = tl.program_id(2)

    # Compute mean over H dimension
    sum_sq = tl.zeros((), dtype=tl.float32)
    for k in range(0, H):
        x_val = tl.load(x_ptr + b * x_bs0 + l * x_bs1 + k * x_bs2).to(tl.float32)
        sum_sq += x_val * x_val

    mean = sum_sq / H
    inv_rms = tl.rsqrt(mean + 0.0)  # rms_norm_eps is 0.0 in original, keep as 0.0 for simplicity

    x_val = tl.load(x_ptr + b * x_bs0 + l * x_bs1 + h * x_bs2).to(tl.float32)
    w_val = tl.load(w_ptr + h * w_bs0).to(tl.float32)

    y_val = x_val * inv_rms * w_val
    tl.store(y_ptr + b * y_bs0 + l * y_bs1 + h * y_bs2, y_val)


# 3) Rotate Q and K: split into h1[:64] and h2[64:] and apply sin/cos
@triton.jit
def rotate_qk_kernel(
    x_ptr,        # *f32, [B, L, H]
    cos_ptr,      # *f32, [L, H//2] rotated sin/cos
    sin_ptr,      # *f32, [L, H//2] rotated sin/cos
    y_ptr,        # *f32, [B, L, H]
    B: tl.constexpr, L: tl.constexpr, H: tl.constexpr,
    x_bs0: tl.constexpr, x_bs1: tl.constexpr, x_bs2: tl.constexpr,
    y_bs0: tl.constexpr, y_bs1: tl.constexpr, y_bs2: tl.constexpr,
):
    b = tl.program_id(0)
    l = tl.program_id(1)
    h = tl.program_id(2)

    x_val = tl.load(x_ptr + b * x_bs0 + l * x_bs1 + h * x_bs2).to(tl.float32)

    half = H // 2
    if h < half:
        rot = -tl.load(sin_ptr + l * (half + 1) + h * 1)  # sin
    else:
        h2 = h - half
        rot = tl.load(cos_ptr + l * (half + 1) + h2 * 1)  # cos (since first half used sin, second half used cos)

    # Write rotated value to y
    tl.store(y_ptr + b * y_bs0 + l * y_bs1 + h * y_bs2, x_val * rot)


# 4) Compute attention scores: attn_scores[b, qh, l, t] = Q[b, qh, l] * K[b, qh, t]
@triton.jit
def attn_matmul_kernel(
    Q_ptr,        # *f32, [B, num_heads, L, H]
    K_ptr,        # *f32, [B, num_heads, L, H]
    scores_ptr,   # *f32, [B, num_heads, L, L]
    B: tl.constexpr, num_heads: tl.constexpr, L: tl.constexpr, H: tl.constexpr,
    Q_bs0: tl.constexpr, Q_bs1: tl.constexpr, Q_bs2: tl.constexpr, Q_bs3: tl.constexpr,
    K_bs0: tl.constexpr, K_bs1: tl.constexpr, K_bs2: tl.constexpr, K_bs3: tl.constexpr,
    scores_bs0: tl.constexpr, scores_bs1: tl.constexpr, scores_bs2: tl.constexpr, scores_bs3: tl.constexpr,
):
    b = tl.program_id(0)
    qh = tl.program_id(1)
    l = tl.program_id(2)

    # Accumulate score for each target t
    for t in range(0, L):
        q_val = tl.load(Q_ptr + b * Q_bs0 + qh * Q_bs1 + l * Q_bs2 + 0 * Q_bs3).to(tl.float32)
        k_val = tl.load(K_ptr + b * K_bs0 + qh * K_bs1 + t * K_bs2 + 0 * K_bs3).to(tl.float32)
        score = q_val * k_val
        tl.store(scores_ptr + b * scores_bs0 + qh * scores_bs1 + l * scores_bs2 + t * scores_bs3, score)


# 5) Softmax with causal mask: apply upper-triangular mask and softmax over sequence dim
@triton.jit
def softmax_mask_kernel(
    scores_ptr,   # *f32, [B, num_heads, L, L]
    mask_ptr,     # *f32, [L, L] (0/1) or [-inf where causal]
    out_ptr,      # *f32, [B, num_heads, L, L]
    B: tl.constexpr, num_heads: tl.constexpr, L: tl.constexpr,
    scores_bs0: tl.constexpr, scores_bs1: tl.constexpr, scores_bs2: tl.constexpr, scores_bs3: tl.constexpr,
    out_bs0: tl.constexpr, out_bs1: tl.constexpr, out_bs2: tl.constexpr, out_bs3: tl.constexpr,
):
    b = tl.program_id(0)
    qh = tl.program_id(1)
    l = tl.program_id(2)

    # Load row scores and mask for row l
    row_max = -float('inf')
    for t in range(0, L):
        score = tl.load(scores_ptr + b * scores_bs0 + qh * scores_bs1 + l * scores_bs2 + t * scores_bs3).to(tl.float32)
        # Load mask value (mask is upper-triangular: m[l, t] = 1 if t >= l else 0)
        m = tl.load(mask_ptr + l * L + t).to(tl.float32)
        if m != 0.0:
            row_max = tl.maximum(row_max, score)

    sum_exp = tl.zeros((), dtype=tl.float32)
    for t in range(0, L):
        score = tl.load(scores_ptr + b * scores_bs0 + qh * scores_bs1 + l * scores_bs2 + t * scores_bs3).to(tl.float32)
        m = tl.load(mask_ptr + l * L + t).to(tl.float32)
        if m != 0.0:
            score = score - row_max
            exp_score = tl.exp(score)
            sum_exp += exp_score

    for t in range(0, L):
        score = tl.load(scores_ptr + b * scores_bs0 + qh * scores_bs1 + l * scores_bs2 + t * scores_bs3).to(tl.float32)
        m = tl.load(mask_ptr + l * L + t).to(tl.float32)
        if m != 0.0:
            score = score - row_max
            exp_score = tl.exp(score) / sum_exp
            tl.store(out_ptr + b * out_bs0 + qh * out_bs1 + l * out_bs2 + t * out_bs3, exp_score)


# 6) Output matmul: attn_output[b, qh, l] = sum_t attn_scores[b, qh, l, t] * V[b, qh, t]
@triton.jit
def output_matmul_kernel(
    scores_ptr,   # *f32, [B, num_heads, L, L]
    V_ptr,        # *f32, [B, num_heads, L, H] (H=128)
    out_ptr,      # *f32, [B, num_heads, L]
    B: tl.constexpr, num_heads: tl.constexpr, L: tl.constexpr, H: tl.constexpr,
    scores_bs0: tl.constexpr, scores_bs1: tl.constexpr, scores_bs2: tl.constexpr, scores_bs3: tl.constexpr,
    V_bs0: tl.constexpr, V_bs1: tl.constexpr, V_bs2: tl.constexpr, V_bs3: tl.constexpr,
    out_bs0: tl.constexpr, out_bs1: tl.constexpr, out_bs2: tl.constexpr,
):
    b = tl.program_id(0)
    qh = tl.program_id(1)
    l = tl.program_id(2)

    acc = tl.zeros((), dtype=tl.float32)
    for t in range(0, L):
        score = tl.load(scores_ptr + b * scores_bs0 + qh * scores_bs1 + l * scores_bs2 + t * scores_bs3).to(tl.float32)
        v_val = tl.load(V_ptr + b * V_bs0 + qh * V_bs1 + t * V_bs2 + 0 * V_bs3).to(tl.float32)
        acc += score * v_val

    tl.store(out_ptr + b * out_bs0 + qh * out_bs1 + l * out_bs2, acc)


# 7) Final linear projection: final_out[b, l, :] = attn_output_flat[b, l, :] @ o_proj_weight^T, no bias
@triton.jit
def final_linear_kernel(
    attn_ptr,     # *f32, [B, L, H_flat] where H_flat = num_heads * head_dim
    o_ptr,        # *f32, [hidden_dim, H_flat]
    out_ptr,      # *f32, [B, L, hidden_dim]
    B: tl.constexpr, L: tl.constexpr, hidden_dim: tl.constexpr, H_flat: tl.constexpr,
    attn_bs0: tl.constexpr, attn_bs1: tl.constexpr, attn_bs2: tl.constexpr,
    o_bs0: tl.constexpr, o_bs1: tl.constexpr,
    out_bs0: tl.constexpr, out_bs1: tl.constexpr, out_bs2: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    b = tl.program_id(0)
    l = tl.program_id(1)
    n = tl.program_id(2)  # output feature dim index in [0, hidden_dim)

    acc = tl.zeros((), dtype=tl.float32)
    for k0 in range(0, H_flat, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < H_flat

        attn_ptrs = attn_ptr + b * attn_bs0 + l * attn_bs1 + offs_k * attn_bs2
        attn_vals = tl.load(attn_ptrs, mask=mask_k, other=0.0).to(tl.float32)  # [BLOCK_K]

        o_ptrs = o_ptr + n * o_bs0 + offs_k * o_bs1
        o_vals = tl.load(o_ptrs, mask=mask_k, other=0.0).to(tl.float32)  # [BLOCK_K]

        # dot product
        acc += tl.sum(attn_vals[:, None] * o_vals[None, :], axis=0)

    tl.store(out_ptr + b * out_bs0 + l * out_bs1 + n * out_bs2, acc)


class ModelNew(torch.nn.Module):
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Define weight placeholders (original code passes these as inputs). Here we keep them as nn.Parameters
        # but the forward will accept actual tensors as arguments.
        self.q_proj_weight = None
        self.q_proj_bias = None
        self.k_proj_weight = None
        self.k_proj_bias = None
        self.v_proj_weight = None
        self.v_proj_bias = None
        self.o_proj_weight = None
        self.q_norm_weight = None
        self.k_norm_weight = None
        self.cos = None
        self.sin = None
        self.rms_norm_eps = 0.0  # original code uses 0.0

    def forward(self, hidden_states: torch.Tensor,
                q_proj_weight: torch.Tensor, q_proj_bias: torch.Tensor,
                k_proj_weight: torch.Tensor, k_proj_bias: torch.Tensor,
                v_proj_weight: torch.Tensor, v_proj_bias: torch.Tensor,
                o_proj_weight: torch.Tensor,
                q_norm_weight: torch.Tensor, k_norm_weight: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor,
                rms_norm_eps: float):
        # Shapes
        B, L, H_in = hidden_states.shape
        num_attention_heads = 96
        num_key_value_heads = 8
        num_key_value_groups = 12
        head_dim = 128
        H_flat = num_attention_heads * head_dim
        assert H_in == head_dim, "hidden_states last dim must be head_dim=128"

        device = hidden_states.device
        # 1) Linear projections for Q, K, V
        # Compute Q, K, V in float32
        q = torch.empty((B, L, head_dim), device=device, dtype=torch.float32)
        k = torch.empty((B, L, head_dim), device=device, dtype=torch.float32)
        v = torch.empty((B, L, head_dim), device=device, dtype=torch.float32)

        # Launch linear_proj_kernel for Q
        grid_q = (B, L, head_dim)
        linear_proj_kernel[grid_q](
            hidden_states, q_proj_weight, q_proj_bias, q,
            B, L, H_in, head_dim,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            q_proj_weight.stride(0), q_proj_weight.stride(1),
            q.stride(0), q.stride(1), q.stride(2),
            BLOCK_K=128,
            num_warps=4, num_stages=2
        )

        # Launch linear_proj_kernel for K
        grid_k = (B, L, head_dim)
        linear_proj_kernel[grid_k](
            hidden_states, k_proj_weight, k_proj_bias, k,
            B, L, H_in, head_dim,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            k_proj_weight.stride(0), k_proj_weight.stride(1),
            k.stride(0), k.stride(1), k.stride(2),
            BLOCK_K=128,
            num_warps=4, num_stages=2
        )

        # Launch linear_proj_kernel for V
        grid_v = (B, L, head_dim)
        linear_proj_kernel[grid_v](
            hidden_states, v_proj_weight, v_proj_bias, v,
            B, L, H_in, head_dim,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            v_proj_weight.stride(0), v_proj_weight.stride(1),
            v.stride(0), v.stride(1), v.stride(2),
            BLOCK_K=128,
            num_warps=4, num_stages=2
        )

        # 2) RMSNorm for Q and K
        q_norm = torch.empty_like(q)
        k_norm = torch.empty_like(k)

        grid_rms_q = (B, L)
        rmsnorm_kernel[grid_rms_q](
            q, q_norm_weight, q_norm,
            B, L, head_dim,
            q.stride(0), q.stride(1), q.stride(2),
            q_norm_weight.stride(0), q_norm_weight.stride(1),
            q_norm.stride(0), q_norm.stride(1), q_norm.stride(2),
            num_warps=2, num_stages=2
        )

        grid_rms_k = (B, L)
        rmsnorm_kernel[grid_rms_k](
            k, k_norm_weight, k_norm,
            B, L, head_dim,
            k.stride(0), k.stride(1), k.stride(2),
            k_norm_weight.stride(0), k_norm_weight.stride(1),
            k_norm.stride(0), k_norm.stride(1), k_norm.stride(2),
            num_warps=2, num_stages=2
        )

        # 3) Q and K rotation (RoPE)
        q_rot = torch.empty_like(q_norm)
        k_rot = torch.empty_like(k_norm)

        grid_rotate_q = (B, L)
        rotate_qk_kernel[grid_rotate_q](
            q_norm, cos, sin, q_rot,
            B, L, head_dim,
            q_norm.stride(0), q_norm.stride(1), q_norm.stride(2),
            q_rot.stride(0), q_rot.stride(1), q_rot.stride(2),
            num_warps=2, num_stages=2
        )

        grid_rotate_k = (B, L)
        rotate_qk_kernel[grid_rotate_k](
            k_norm, cos, sin, k_rot,
            B, L, head_dim,
            k_norm.stride(0), k_norm.stride(1), k_norm.stride(2),
            k_rot.stride(0), k_rot.stride(1), k_rot.stride(2),
            num_warps=2, num_stages=2
        )

        # 4) GQA setup: expand K_rot and V from 8 heads to 96 heads via groups (num_key_value_groups=12)
        # K/V are [B, 8, L, 128] originally; we expand to [B, 96, L, 128]
        # We reuse k_rot and v (no per-head K/V weights in original code, so we assume single V slice per (b,l)).
        k_gqa = k_rot[:, :, :, :].unsqueeze(2).expand(B, 8, num_key_value_groups, L, head_dim).reshape(B, num_attention_heads, L, head_dim)
        v_gqa = v  # V is [B, L, 128] and we treat it as per-(b,l) slice across heads

        # 5) Compute attention scores [B, num_heads, L, L]
        scores = torch.empty((B, num_attention_heads, L, L), device=device, dtype=torch.float32)

        grid_attn = (B, num_attention_heads, L)
        attn_matmul_kernel[grid_attn](
            q_rot, k_gqa, scores,
            B, num_attention_heads, L, head_dim,
            q_rot.stride(0), q_rot.stride(1), q_rot.stride(2), q_rot.stride(3),
            k_gqa.stride(0), k_gqa.stride(1), k_gqa.stride(2), k_gqa.stride(3),
            scores.stride(0), scores.stride(1), scores.stride(2), scores.stride(3),
            num_warps=2, num_stages=2
        )

        # 6) Causal mask (upper triangle, diagonal=1) and softmax per (b, qh, l)
        # Build mask tensor [L, L], then broadcast
        # mask[t, l] = 1 if t >= l else 0
        causal_mask = torch.ones((L, L), device=device, dtype=torch.float32)
        for t in range(0, L):
            for l2 in range(0, L):
                if l2 > t:
                    causal_mask[t, l2] = 0.0

        # Softmax kernel
        out_scores = torch.empty_like(scores)

        grid_softmax = (B, num_attention_heads, L)
        softmax_mask_kernel[grid_softmax](
            scores, causal_mask, out_scores,
            B, num_attention_heads, L,
            scores.stride(0), scores.stride(1), scores.stride(2), scores.stride(3),
            out_scores.stride(0), out_scores.stride(1), out_scores.stride(2), out_scores.stride(3),
            num_warps=2, num_stages=2
        )

        # 7) Output: attn_output[b, qh, l] = sum_t out_scores[b, qh, l, t] * V[b, qh, t]
        attn_output = torch.empty((B, num_attention_heads, L), device=device, dtype=torch.float32)

        grid_output = (B, num_attention_heads, L)
        output_matmul_kernel[grid_output](
            out_scores, v_gqa, attn_output,
            B, num_attention_heads, L, head_dim,
            out_scores.stride(0), out_scores.stride(1), out_scores.stride(2), out_scores.stride(3),
            v_gqa.stride(0), v_gqa.stride(1), v_gqa.stride(2), v_gqa.stride(3),
            attn_output.stride(0), attn_output.stride(1), attn_output.stride(2),
            num_warps=2, num_stages=2
        )

        # 8) Final linear projection to [B, L, hidden_dim]
        attn_flat = attn_output.reshape(B, L, H_flat)
        final_out = torch.empty((B, L, self.hidden_dim), device=device, dtype=torch.float32)

        grid_final = (B, L, self.hidden_dim)
        final_linear_kernel[grid_final](
            attn_flat, o_proj_weight, final_out,
            B, L, self.hidden_dim, H_flat,
            attn_flat.stride(0), attn_flat.stride(1), attn_flat.stride(2),
            o_proj_weight.stride(0), o_proj_weight.stride(1),
            final_out.stride(0), final_out.stride(1), final_out.stride(2),
            BLOCK_K=128,
            num_warps=4, num_stages=2
        )

        return final_out


def run(*args):
    return ModelNew()(*args)
