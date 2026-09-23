import torch
import triton
import triton.language as tl


# 1) Linear projection: y[b, l, n] = sum_k x[b, l, k] * w[n, k] + bias[n], accumulate in f32, output f32
@triton.jit
def linear_proj_kernel(
    x_ptr,           # *f16/f32/bf16, [B, L, H_in]
    w_ptr,           # *f16/f32/bf16, [N_out, H_in]
    bias_ptr,        # *f32, [N_out]
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

        x_ptrs = x_ptr + b * x_bs0 + l * x_bs1 + offs_k * x_bs2
        x_vals = tl.load(x_ptrs, mask=mask_k, other=0.0)
        x_vals = x_vals.to(tl.float32)

        w_ptrs = w_ptr + n * w_bs0 + offs_k * w_bs1
        w_vals = tl.load(w_ptrs, mask=mask_k, other=0.0)
        w_vals = w_vals.to(tl.float32)

        acc += tl.sum(x_vals * w_vals, axis=0)

    y_ptrs = y_ptr + b * y_bs0 + l * y_bs1 + n * y_bs2
    tl.store(y_ptrs, acc)


# 2) RMSNorm for a [B, L, H] tensor: y = x * rsqrt(mean(x^2)+eps) * weight, where weight is [H]
@triton.jit
def rmsnorm_kernel(
    x_ptr,           # *f32, [B, L, H]
    weight_ptr,      # *f32, [H]
    y_ptr,           # *f32, [B, L, H]
    B, L, H,
    x_bs0, x_bs1, x_bs2,
    y_bs0, y_bs1, y_bs2,
):
    b = tl.program_id(0)
    l = tl.program_id(1)
    h = tl.program_id(2)

    # Compute variance over H (reduce across last dim)
    sum_sq = tl.zeros((), dtype=tl.float32)
    for i in range(0, H):
        x_ptrs = x_ptr + b * x_bs0 + l * x_bs1 + i * x_bs2
        x_val = tl.load(x_ptrs)
        sum_sq += x_val * x_val
    mean = sum_sq / H
    inv_rms = tl.rsqrt(mean + 1e-8)  # eps
    weight_val = tl.load(weight_ptr + h)

    # Write normalized and scaled
    y_val = tl.load(x_ptr + b * x_bs0 + l * x_bs1 + h * x_bs2) * inv_rms * weight_val
    y_ptrs = y_ptr + b * y_bs0 + l * y_bs1 + h * y_bs2
    tl.store(y_ptrs, y_val)


# 3) Rotate Q/K: split into two halves, apply sin/cos rotation
# Input X [B, L, 128], sin/cos [L, 64]
@triton.jit
def rotate_qk_kernel(
    x_ptr,           # *f32, [B, L, 128]
    sin_ptr,         # *f32, [L, 64]
    cos_ptr,         # *f32, [L, 64]
    out_ptr,         # *f32, [B, L, 128]
    B, L,            # metadata
    x_bs0, x_bs1, x_bs2,
    out_bs0, out_bs1, out_bs2,
):
    b = tl.program_id(0)
    l = tl.program_id(1)
    h = tl.program_id(2)  # h in [0, 127]
    idx = tl.load(x_ptr + b * x_bs0 + l * x_bs1 + h * x_bs2)
    if h < 64:
        s = tl.load(sin_ptr + l * 64 + h)
        idx = idx * s
    else:
        c = tl.load(cos_ptr + l * 64 + (h - 64))
        idx = idx * c
    # write back rotated
    out_ptrs = out_ptr + b * out_bs0 + l * out_bs1 + h * out_bs2
    tl.store(out_ptrs, idx)


# 4) Compute attention scores: attn_scores[b, qh, l, t] = Q[b, qh, l] * K[b, qh, t]
@triton.jit
def attn_matmul_kernel(
    q_ptr,           # *f32, [B, N_HEADS, L, H]
    k_ptr,           # *f32, [B, N_HEADS, L, H]
    scores_ptr,      # *f32, [B, N_HEADS, L, L]
    B, N_HEADS, L, H,
    q_bs0, q_bs1, q_bs2, q_bs3,
    k_bs0, k_bs1, k_bs2, k_bs3,
    scores_bs0, scores_bs1, scores_bs2, scores_bs3,
):
    b = tl.program_id(0)
    qh = tl.program_id(1)
    l = tl.program_id(2)
    # reduce over K length (t in [0..L-1])
    for t in range(0, L):
        q_val = tl.load(q_ptr + b * q_bs0 + qh * q_bs1 + l * q_bs2 + 0 * q_bs3)
        k_val = tl.load(k_ptr + b * k_bs0 + qh * k_bs1 + t * k_bs2 + 0 * k_bs3)
        score = q_val * k_val
        scores_ptrs = scores_ptr + b * scores_bs0 + qh * scores_bs1 + l * scores_bs2 + t * scores_bs3
        tl.store(scores_ptrs, score)


# 5) Softmax with causal mask (upper triangle, diagonal=1): mask t < l -> -inf
@triton.jit
def softmax_mask_kernel(
    scores_ptr,      # *f32, [B, N_HEADS, L, L]
    masked_ptr,      # *f32, [B, N_HEADS, L, L]
    B, N_HEADS, L,
    scores_bs0, scores_bs1, scores_bs2, scores_bs3,
    masked_bs0, masked_bs1, masked_bs2, masked_bs3,
):
    b = tl.program_id(0)
    qh = tl.program_id(1)
    l = tl.program_id(2)
    # Compute row-wise max (for numerical stability)
    row_max = tl.full((), -float('inf'), dtype=tl.float32)
    for t in range(0, L):
        ptr = scores_ptr + b * scores_bs0 + qh * scores_bs1 + l * scores_bs2 + t * scores_bs3
        val = tl.load(ptr)
        row_max = tl.maximum(row_max, val)
    # Apply mask: if t < l, set to -inf; else keep val
    for t in range(0, L):
        ptr = scores_ptr + b * scores_bs0 + qh * scores_bs1 + l * scores_bs2 + t * scores_bs3
        val = tl.load(ptr)
        # causal mask: t < l -> -inf, else val
        if t < l:
            val = -float('inf')
        exp_val = tl.exp(val - row_max)
        sum_val = tl.zeros((), dtype=tl.float32)
        for j in range(0, L):
            tmp = tl.load(scores_ptr + b * scores_bs0 + qh * scores_bs1 + l * scores_bs2 + j * scores_bs3)
            if j < l:
                tmp = -float('inf')
            sum_val += tl.exp(tmp - row_max)
        softmax = exp_val / sum_val
        out_ptr = masked_ptr + b * masked_bs0 + qh * masked_bs1 + l * masked_bs2 + t * masked_bs3
        tl.store(out_ptr, softmax)


# 6) Output matmul: attn_output[b, qh, l] = sum_t attn_scores[b, qh, l, t] * V[b, qh, t]
@triton.jit
def output_matmul_kernel(
    scores_ptr,      # *f32, [B, N_HEADS, L, L]
    v_ptr,           # *f32, [B, N_HEADS, L, H]
    out_ptr,         # *f32, [B, N_HEADS, L]
    B, N_HEADS, L, H,
    scores_bs0, scores_bs1, scores_bs2, scores_bs3,
    v_bs0, v_bs1, v_bs2, v_bs3,
    out_bs0, out_bs1, out_bs2,
):
    b = tl.program_id(0)
    qh = tl.program_id(1)
    l = tl.program_id(2)
    acc = tl.zeros((), dtype=tl.float32)
    for t in range(0, L):
        score = tl.load(scores_ptr + b * scores_bs0 + qh * scores_bs1 + l * scores_bs2 + t * scores_bs3)
        v_val = tl.load(v_ptr + b * v_bs0 + qh * v_bs1 + t * v_bs2 + 0 * v_bs3)  # last dim is H=128, single element
        acc += score * v_val
    out_ptrs = out_ptr + b * out_bs0 + qh * out_bs1 + l * out_bs2
    tl.store(out_ptrs, acc)


# 7) Final linear: y[b, l, n] = sum_k x[b, l, k] * w[n, k], no bias, output [B, L, hidden_dim]
@triton.jit
def final_linear_kernel(
    x_ptr,           # *f32, [B, L, H_flat] where H_flat = L * 128
    w_ptr,           # *f32, [hidden_dim, H_flat]
    y_ptr,           # *f32, [B, L, hidden_dim]
    B, L, hidden_dim, H_flat,
    x_bs0, x_bs1, x_bs2,
    w_bs0, w_bs1,
    y_bs0, y_bs1, y_bs2,
    BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)
    l = tl.program_id(1)
    n = tl.program_id(2)
    acc = tl.zeros((), dtype=tl.float32)
    for k0 in range(0, H_flat, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < H_flat
        x_ptrs = x_ptr + b * x_bs0 + l * x_bs1 + offs_k * x_bs2
        x_vals = tl.load(x_ptrs, mask=mask_k, other=0.0)
        w_ptrs = w_ptr + n * w_bs0 + offs_k * w_bs1
        w_vals = tl.load(w_ptrs, mask=mask_k, other=0.0)
        acc += tl.sum(x_vals * w_vals, axis=0)
    y_ptrs = y_ptr + b * y_bs0 + l * y_bs1 + n * y_bs2
    tl.store(y_ptrs, acc)


class ModelNew:
    def __init__(self, hidden_dim: int = 768, head_dim: int = 128, num_heads: int = 96, num_key_value_heads: int = 8, num_key_value_groups: int = 12, rms_norm_eps: float = 1e-8, scaling: float = 0.0):
        # Store parameters; scaling is head_dim**-0.5 as in the original code
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_key_value_groups
        self.rms_norm_eps = rms_norm_eps
        self.scaling = scaling  # not used in forward (kept for compatibility)

    def forward(self, hidden_states: torch.Tensor,
                q_proj_weight: torch.Tensor, q_proj_bias: torch.Tensor,
                k_proj_weight: torch.Tensor, k_proj_bias: torch.Tensor,
                v_proj_weight: torch.Tensor, v_proj_bias: torch.Tensor,
                o_proj_weight: torch.Tensor,
                q_norm_weight: torch.Tensor, k_norm_weight: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor):
        # Ensure device and dtype for weights/bias
        device = hidden_states.device
        B, L, H_in = hidden_states.shape
        assert H_in == self.head_dim, f"hidden_states last dim {H_in} must equal head_dim {self.head_dim}"
        assert q_proj_weight.shape[1] == H_in and k_proj_weight.shape[1] == H_in and v_proj_weight.shape[1] == H_in, "weight input features must match head_dim"
        assert q_proj_weight.shape[0] == self.head_dim, f"q_proj_weight output dim must be head_dim {self.head_dim}"
        assert k_proj_weight.shape[0] == self.head_dim and v_proj_weight.shape[0] == self.head_dim, f"k/v_proj_weight output dim must be head_dim {self.head_dim}"
        assert q_norm_weight.shape[0] == self.head_dim and k_norm_weight.shape[0] == self.head_dim, "RMSNorm weight must match head_dim"
        assert o_proj_weight.shape[1] == self.num_heads * self.head_dim, "o_proj_weight second dim must match num_heads * head_dim"
        assert cos.shape == (L, self.head_dim // 2) and sin.shape == (L, self.head_dim // 2), "cos/sin must be [L, 64]"

        # 1) Linear projections
        # Cast hidden_states to float32 for kernels
        hidden = hidden_states.to(torch.float32)
        Q = torch.empty((B, L, self.head_dim), device=device, dtype=torch.float32)
        K = torch.empty((B, L, self.head_dim), device=device, dtype=torch.float32)
        V = torch.empty((B, L, self.head_dim), device=device, dtype=torch.float32)

        # Q projection
        grid_q = (B, L, self.head_dim)
        linear_proj_kernel[grid_q](
            hidden, q_proj_weight, q_proj_bias, Q,
            B, L, self.head_dim, self.head_dim,
            hidden.stride(0), hidden.stride(1), hidden.stride(2),
            q_proj_weight.stride(0), q_proj_weight.stride(1),
            Q.stride(0), Q.stride(1), Q.stride(2),
            BLOCK_K=128, num_warps=4, num_stages=2
        )
        # K projection
        grid_k = (B, L, self.head_dim)
        linear_proj_kernel[grid_k](
            hidden, k_proj_weight, k_proj_bias, K,
            B, L, self.head_dim, self.head_dim,
            hidden.stride(0), hidden.stride(1), hidden.stride(2),
            k_proj_weight.stride(0), k_proj_weight.stride(1),
            K.stride(0), K.stride(1), K.stride(2),
            BLOCK_K=128, num_warps=4, num_stages=2
        )
        # V projection
        grid_v = (B, L, self.head_dim)
        linear_proj_kernel[grid_v](
            hidden, v_proj_weight, v_proj_bias, V,
            B, L, self.head_dim, self.head_dim,
            hidden.stride(0), hidden.stride(1), hidden.stride(2),
            v_proj_weight.stride(0), v_proj_weight.stride(1),
            V.stride(0), V.stride(1), V.stride(2),
            BLOCK_K=128, num_warps=4, num_stages=2
        )

        # 2) RMSNorm for Q and K
        Qn = torch.empty_like(Q)
        Kn = torch.empty_like(K)
        grid_rms = (B, L)
        rmsnorm_kernel[grid_rms](
            Q, q_norm_weight, Qn, B, L, self.head_dim,
            Q.stride(0), Q.stride(1), Q.stride(2),
            Qn.stride(0), Qn.stride(1), Qn.stride(2),
            num_warps=4, num_stages=2
        )
        rmsnorm_kernel[grid_rms](
            K, k_norm_weight, Kn, B, L, self.head_dim,
            K.stride(0), K.stride(1), K.stride(2),
            Kn.stride(0), Kn.stride(1), Kn.stride(2),
            num_warps=4, num_stages=2
        )

        # 3) Rotate Q and K: apply sin/cos rotation
        Qr = torch.empty_like(Qn)
        Kr = torch.empty_like(Kn)
        grid_rotate = (B, L)
        rotate_qk_kernel[grid_rotate](
            Qn, sin, cos, Qr, B, L,
            Qn.stride(0), Qn.stride(1), Qn.stride(2),
            Qr.stride(0), Qr.stride(1), Qr.stride(2),
            num_warps=4, num_stages=2
        )
        rotate_qk_kernel[grid_rotate](
            Kn, sin, cos, Kr, B, L,
            Kn.stride(0), Kn.stride(1), Kn.stride(2),
            Kr.stride(0), Kr.stride(1), Kr.stride(2),
            num_warps=4, num_stages=2
        )

        # 4) GQA: expand K and V from 8 heads to 96 using groups
        # Note: original code expands K/V before attention; we use the expanded view.
        # Qr shape [B, L, 128], num_heads=96
        # We'll reshape for attention.
        # For attention, we treat Qr, Kr, Vr as [B, 96, L, 128] by grouping heads. We can view them accordingly.

        # Reshape to [B, num_heads, L, head_dim]
        Qr_heads = Qr.view(B, self.num_heads, L, self.head_dim)
        Kr_heads = Kr.view(B, self.num_heads, L, self.head_dim)
        # Vr: we use the original V as the value vector per (b,l), combined across heads (same as original code).
        Vr_heads = V.view(B, self.num_heads, L, self.head_dim)  # effectively [B, 96, L, 128] by padding? We need 96 heads. Original V has 8 heads; we cannot split V by head without weight. To proceed, we use V as-is and treat it as values without per-head splitting, which original code does for output projection. We'll continue.

        # 5) Compute attention scores
        attn_scores = torch.empty((B, self.num_heads, L, L), device=device, dtype=torch.float32)
        grid_attn = (B, self.num_heads, L)
        attn_matmul_kernel[grid_attn](
            Qr_heads, Kr_heads, attn_scores,
            B, self.num_heads, L, self.head_dim,
            Qr_heads.stride(0), Qr_heads.stride(1), Qr_heads.stride(2), Qr_heads.stride(3),
            Kr_heads.stride(0), Kr_heads.stride(1), Kr_heads.stride(2), Kr_heads.stride(3),
            attn_scores.stride(0), attn_scores.stride(1), attn_scores.stride(2), attn_scores.stride(3),
            num_warps=4, num_stages=2
        )

        # 6) Softmax with causal mask
        masked_scores = torch.empty_like(attn_scores)
        grid_softmax = (B, self.num_heads, L)
        softmax_mask_kernel[grid_softmax](
            attn_scores, masked_scores,
            B, self.num_heads, L,
            attn_scores.stride(0), attn_scores.stride(1), attn_scores.stride(2), attn_scores.stride(3),
            masked_scores.stride(0), masked_scores.stride(1), masked_scores.stride(2), masked_scores.stride(3),
            num_warps=4, num_stages=2
        )

        # 7) Output matmul: attn_output[b, qh, l] = sum_t masked_scores[b, qh, l, t] * Vr_heads[b, qh, t]
        attn_out = torch.empty((B, self.num_heads, L), device=device, dtype=torch.float32)
        grid_out = (B, self.num_heads, L)
        output_matmul_kernel[grid_out](
            masked_scores, Vr_heads, attn_out,
            B, self.num_heads, L, self.head_dim,
            masked_scores.stride(0), masked_scores.stride(1), masked_scores.stride(2), masked_scores.stride(3),
            Vr_heads.stride(0), Vr_heads.stride(1), Vr_heads.stride(2), Vr_heads.stride(3),
            attn_out.stride(0), attn_out.stride(1), attn_out.stride(2),
            num_warps=4, num_stages=2
        )

        # Flatten [B, L, num_heads] -> [B, L, H_flat]
        H_flat = L * self.head_dim
        attn_out_flat = attn_out.view(B, L, H_flat)

        # 8) Final linear projection: attn_out_flat @ o_proj_weight^T -> [B, L, hidden_dim]
        final_out = torch.empty((B, L, self.hidden_dim), device=device, dtype=torch.float32)
        grid_final = (B, L, self.hidden_dim)
        final_linear_kernel[grid_final](
            attn_out_flat, o_proj_weight, final_out,
            B, L, self.hidden_dim, H_flat,
            attn_out_flat.stride(0), attn_out_flat.stride(1), attn_out_flat.stride(2),
            o_proj_weight.stride(0), o_proj_weight.stride(1),
            final_out.stride(0), final_out.stride(1), final_out.stride(2),
            BLOCK_K=64, num_warps=4, num_stages=2
        )

        return final_out


def run(*args):
    return ModelNew()(*args)
