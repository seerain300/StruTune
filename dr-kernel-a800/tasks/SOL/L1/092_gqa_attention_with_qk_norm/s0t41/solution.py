import torch
import triton
import triton.language as tl


# 1) Linear projection: y[b, l, n] = sum_k x[b, l, k] * w[n, k], accumulate in f32, output f32
@triton.jit
def linear_proj_kernel(
    x_ptr,           # *f16/f32/bf16, [B, L, H_in]
    w_ptr,           # *f16/f32/bf16, [N_out, H_in]
    bias_ptr,        # *f32, [N_out] (ignored if not used)
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
        # Load x[b, l, offs_k]
        x_ptrs = x_ptr + b * x_bs0 + l * x_bs1 + offs_k * x_bs2
        x_vals = tl.load(x_ptrs, mask=mask_k, other=0.0)
        # Load w[n, offs_k]
        w_ptrs = w_ptr + n * w_bs0 + offs_k * w_bs1
        w_vals = tl.load(w_ptrs, mask=mask_k, other=0.0)
        # Accumulate dot product
        acc += tl.sum(x_vals.to(tl.float32) * w_vals.to(tl.float32), axis=0)
    # Store result
    y_ptrs = y_ptr + b * y_bs0 + l * y_bs1 + n * y_bs2
    tl.store(y_ptrs, acc)


# 2) RMSNorm: x[b, l, :] -> x * rsqrt(mean(x^2)+eps) * weight[:]
@triton.jit
def rmsnorm_kernel(
    x_ptr,        # *f32, [B, L, H]
    weight_ptr,   # *f32, [H]
    out_ptr,      # *f32, [B, L, H]
    B, L, H,
    x_bs0, x_bs1, x_bs2,
    w_bs0,
    out_bs0, out_bs1, out_bs2,
):
    b = tl.program_id(0)
    l = tl.program_id(1)
    for h in range(0, H):
        x_ptrs = x_ptr + b * x_bs0 + l * x_bs1 + h * x_bs2
        x_val = tl.load(x_ptrs).to(tl.float32)
        w_ptrs = weight_ptr + h * w_bs0
        w_val = tl.load(w_ptrs).to(tl.float32)
        mean_sq = tl.sum(x_val * x_val, axis=0) / H
        scale = tl.rsqrt(mean_sq + 1e-8)  # rms_norm_eps from the original
        y_val = x_val * scale * w_val
        out_ptrs = out_ptr + b * out_bs0 + l * out_bs1 + h * out_bs2
        tl.store(out_ptrs, y_val)


# 3) Rotate Q/K: per (b, l), split head into h1[:64], h2[64:], apply cos/sin and cat
@triton.jit
def rotate_qk_kernel(
    x_ptr,     # *f32, [B, L, H]
    cos_ptr,   # *f32, [L, H//2]
    sin_ptr,   # *f32, [L, H//2]
    out_ptr,   # *f32, [B, L, H]
    B, L, H,
    x_bs0, x_bs1, x_bs2,
    cos_bs0, cos_bs1,
    sin_bs0, sin_bs1,
    out_bs0, out_bs1, out_bs2,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    l = tl.program_id(1)
    for h in range(0, H):
        x_ptrs = x_ptr + b * x_bs0 + l * x_bs1 + h * x_bs2
        x_val = tl.load(x_ptrs).to(tl.float32)
        if h < 64:
            cos_ptrs = cos_ptr + l * cos_bs0 + h * cos_bs1
            sin_ptrs = sin_ptr + l * sin_bs0 + h * sin_bs1
            cos_val = tl.load(cos_ptrs).to(tl.float32)
            sin_val = tl.load(sin_ptrs).to(tl.float32)
            # y_h = x * cos - x @ sin (note: sin applied to the same h)
            # Here x_val is h in [0,63], so we rotate within this half
            # Apply rotation: y = x * cos - x * sin for h in [0,63]
            y_val = x_val * cos_val - x_val * sin_val
        else:
            offset = h - 64
            cos_ptrs = cos_ptr + l * cos_bs0 + offset * cos_bs1
            sin_ptrs = sin_ptr + l * sin_bs0 + offset * sin_bs1
            cos_val = tl.load(cos_ptrs).to(tl.float32)
            sin_val = tl.load(sin_ptrs).to(tl.float32)
            # For h in [64,127], rotate with -sin on the second half
            y_val = x_val * cos_val + x_val * sin_val
        out_ptrs = out_ptr + b * out_bs0 + l * out_bs1 + h * out_bs2
        tl.store(out_ptrs, y_val)


# 4) Attention score matmul: scores[b, qh, l, t] = Q[b, qh, l] * K[b, qh, t]
@triton.jit
def attn_matmul_kernel(
    q_ptr,   # *f32, [B, num_heads, L, H]
    k_ptr,   # *f32, [B, num_heads, L, H]
    scores_ptr,   # *f32, [B, num_heads, L, L]
    B, num_heads, L, H,
    q_bs0, q_bs1, q_bs2, q_bs3,
    k_bs0, k_bs1, k_bs2, k_bs3,
    s_bs0, s_bs1, s_bs2, s_bs3,
):
    b = tl.program_id(0)
    qh = tl.program_id(1)
    l = tl.program_id(2)
    # For each t, compute Q[b,qh,l] * K[b,qh,t]
    for t in range(0, L):
        q_row_ptrs = q_ptr + b * q_bs0 + qh * q_bs1 + l * q_bs2  # vector length H
        k_col_ptrs = k_ptr + b * k_bs0 + qh * k_bs1 + t * k_bs2  # vector length H
        q_row = tl.load(q_row_ptrs).to(tl.float32)                # shape [H]
        k_col = tl.load(k_col_ptrs).to(tl.float32)                # shape [H]
        score = tl.sum(q_row * k_col, axis=0)                     # scalar
        sptr = scores_ptr + b * s_bs0 + qh * s_bs1 + l * s_bs2 + t * s_bs3
        tl.store(sptr, score)


# 5) Softmax with causal mask (upper triangle, diagonal=1)
# probs[b, qh, l, t] = exp(scores[b, qh, l, t] - max_l) / sum_t exp(scores[b, qh, l, t] - max_l)
@triton.jit
def softmax_mask_kernel(
    scores_ptr,   # *f32, [B, num_heads, L, L]
    mask_ptr,     # *f32, [B, num_heads, L, L] (0 or -inf)
    probs_ptr,    # *f32, [B, num_heads, L, L]
    B, num_heads, L,
    scores_bs0, scores_bs1, scores_bs2, scores_bs3,
    mask_bs0, mask_bs1, mask_bs2, mask_bs3,
    probs_bs0, probs_bs1, probs_bs2, probs_bs3,
):
    b = tl.program_id(0)
    qh = tl.program_id(1)
    l = tl.program_id(2)
    # Load scores[b, qh, l, :]
    scores = tl.zeros((L,), dtype=tl.float32)
    for t in range(0, L):
        sptr = scores_ptr + b * scores_bs0 + qh * scores_bs1 + l * scores_bs2 + t * scores_bs3
        scores[t] = tl.load(sptr).to(tl.float32)
    # Load mask[b, qh, l, :]
    mask = tl.zeros((L,), dtype=tl.float32)
    for t in range(0, L):
        mptr = mask_ptr + b * mask_bs0 + qh * mask_bs1 + l * mask_bs2 + t * mask_bs3
        mask[t] = tl.load(mptr).to(tl.float32)
    # Apply mask: invalid positions (t<l) become -inf
    scores = scores + mask
    # Stable softmax
    max_score = tl.max(scores, axis=0)
    scores = scores - max_score
    exp_scores = tl.exp(scores)
    sum_exp = tl.sum(exp_scores, axis=0)
    probs = exp_scores / sum_exp
    # Store
    for t in range(0, L):
        pptr = probs_ptr + b * probs_bs0 + qh * probs_bs1 + l * probs_bs2 + t * probs_bs3
        tl.store(pptr, probs[t])


# 6) Output matmul: attn_output[b, qh, l] = sum_t probs[b, qh, l, t] * V[b, qh, t]
@triton.jit
def output_matmul_kernel(
    probs_ptr,  # *f32, [B, num_heads, L, L]
    v_ptr,      # *f32, [B, num_heads, L, H] (we use V as [B, L, H] in forward)
    out_ptr,    # *f32, [B, num_heads, L, H]
    B, num_heads, L, H,
    probs_bs0, probs_bs1, probs_bs2, probs_bs3,
    v_bs0, v_bs1, v_bs2, v_bs3,
    out_bs0, out_bs1, out_bs2, out_bs3,
):
    # Note: this kernel assumes V is already per-head as passed by forward. In our case, we pass V expanded to [B, num_heads, L, H].
    b = tl.program_id(0)
    qh = tl.program_id(1)
    l = tl.program_id(2)
    # Compute output[b, qh, l, :] = sum_t probs[b, qh, l, t] * V[b, qh, l, t]
    acc = tl.zeros((H,), dtype=tl.float32)
    for t in range(0, L):
        p = tl.load(probs_ptr + b * probs_bs0 + qh * probs_bs1 + l * probs_bs2 + t * probs_bs3).to(tl.float32)
        v_ptrs = v_ptr + b * v_bs0 + qh * v_bs1 + l * v_bs2 + t * v_bs3
        v_vals = tl.load(v_ptrs).to(tl.float32)
        acc = acc + p * v_vals
    out_ptrs = out_ptr + b * out_bs0 + qh * out_bs1 + l * out_bs2
    tl.store(out_ptrs, acc)


# 7) Final linear projection: y[b, l, n] = sum_k x[b, l, k] * w[n, k], no bias
@triton.jit
def final_linear_kernel(
    x_ptr,        # *f32, [B, L, H_in]
    w_ptr,        # *f32, [N_out, H_in]
    y_ptr,        # *f32, [B, L, N_out]
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
        w_ptrs = w_ptr + n * w_bs0 + offs_k * w_bs1
        x_vals = tl.load(x_ptrs, mask=mask_k, other=0.0)
        w_vals = tl.load(w_ptrs, mask=mask_k, other=0.0)
        acc += tl.sum(x_vals.to(tl.float32) * w_vals.to(tl.float32), axis=0)
    y_ptrs = y_ptr + b * y_bs0 + l * y_bs1 + n * y_bs2
    tl.store(y_ptrs, acc)


class ModelNew(torch.nn.Module):
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.hidden_dim = hidden_dim  # output projection size

    def forward(self,
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
        # Shapes from the original code
        B, L, _ = hidden_states.shape
        num_attention_heads = 96
        num_key_value_heads = 8
        head_dim = 128  # hidden_states last dim is 128
        num_key_value_groups = 12
        scaling = head_dim ** -0.5  # unused in the original math, kept for consistency

        device = hidden_states.device
        # 1) Linear projections for Q, K, V: [B, L, H] -> H=head_dim=128
        Q = torch.empty((B, L, head_dim), device=device, dtype=torch.float32)
        K = torch.empty((B, L, head_dim), device=device, dtype=torch.float32)
        V = torch.empty((B, L, head_dim), device=device, dtype=torch.float32)

        # Launch linear_proj_kernel for Q, K, V
        grid_q = (B, L, head_dim)
        linear_proj_kernel[grid_q](
            hidden_states, q_proj_weight, (q_proj_bias if q_proj_bias is not None else torch.zeros(1, device=device, dtype=torch.float32)), Q,
            B, L, hidden_states.shape[-1], head_dim,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            q_proj_weight.stride(0), q_proj_weight.stride(1),
            Q.stride(0), Q.stride(1), Q.stride(2),
            BLOCK_K=64, num_warps=4, num_stages=2
        )

        grid_k = (B, L, head_dim)
        linear_proj_kernel[grid_k](
            hidden_states, k_proj_weight, (k_proj_bias if k_proj_bias is not None else torch.zeros(1, device=device, dtype=torch.float32)), K,
            B, L, hidden_states.shape[-1], head_dim,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            k_proj_weight.stride(0), k_proj_weight.stride(1),
            K.stride(0), K.stride(1), K.stride(2),
            BLOCK_K=64, num_warps=4, num_stages=2
        )

        grid_v = (B, L, head_dim)
        linear_proj_kernel[grid_v](
            hidden_states, v_proj_weight, (v_proj_bias if v_proj_bias is not None else torch.zeros(1, device=device, dtype=torch.float32)), V,
            B, L, hidden_states.shape[-1], head_dim,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            v_proj_weight.stride(0), v_proj_weight.stride(1),
            V.stride(0), V.stride(1), V.stride(2),
            BLOCK_K=64, num_warps=4, num_stages=2
        )

        # 2) RMSNorm for Q and K: per (b, l), no bias
        Q_norm = torch.empty_like(Q)
        K_norm = torch.empty_like(K)

        grid_rms_q = (B, L)
        rmsnorm_kernel[grid_rms_q](
            Q, q_norm_weight, Q_norm,
            B, L, head_dim,
            Q.stride(0), Q.stride(1), Q.stride(2),
            q_norm_weight.stride(0),
            Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(2),
            num_warps=2, num_stages=2
        )

        grid_rms_k = (B, L)
        rmsnorm_kernel[grid_rms_k](
            K, k_norm_weight, K_norm,
            B, L, head_dim,
            K.stride(0), K.stride(1), K.stride(2),
            k_norm_weight.stride(0),
            K_norm.stride(0), K_norm.stride(1), K_norm.stride(2),
            num_warps=2, num_stages=2
        )

        # 3) Rotate Q and K using cos/sin (shape [L, head_dim//2])
        Q_rot = torch.empty_like(Q_norm)
        K_rot = torch.empty_like(K_norm)

        grid_rotate = (B, L)
        rotate_qk_kernel[grid_rotate](
            Q_norm, cos, sin, Q_rot,
            B, L, head_dim,
            Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(2),
            cos.stride(0), cos.stride(1),
            sin.stride(0), sin.stride(1),
            Q_rot.stride(0), Q_rot.stride(1), Q_rot.stride(2),
            BLOCK=64, num_warps=2, num_stages=2
        )

        rotate_qk_kernel[grid_rotate](
            K_norm, cos, sin, K_rot,
            B, L, head_dim,
            K_norm.stride(0), K_norm.stride(1), K_norm.stride(2),
            cos.stride(0), cos.stride(1),
            sin.stride(0), sin.stride(1),
            K_rot.stride(0), K_rot.stride(1), K_rot.stride(2),
            BLOCK=64, num_warps=2, num_stages=2
        )

        # 4) GQA expansion to num_attention_heads=96: reshape K_rot and V to [B, 96, L, head_dim]
        # We'll compute scores using these expanded K: scores[b, qh, l, t] where qh ∈ [0..8*12=96], t ∈ [0..L-1]
        # Note: original code expands K/V by groups; here we reuse K_rot across qh slots. This mimics original attention computation using expanded K/V groups.
        # We still need to produce attention_output of shape [B, L, 12288], so we will expand the output after reduction.

        # Build K_expanded and V_expanded: [B, 96, L, head_dim]
        # K/V for each qh head are the same as K_rot; groups logic is encoded in qh. We can just repeat K_rot across qh dimension.
        K_exp = torch.empty((B, num_attention_heads, L, head_dim), device=device, dtype=torch.float32)
        V_exp = torch.empty((B, num_attention_heads, L, head_dim), device=device, dtype=torch.float32)
        # Manually expand: K_exp[b, qh, l, :] = K_rot[b, l, :]
        for qh in range(num_attention_heads):
            # qh maps to key_value head qkv_h = qh % num_key_value_heads and group = qh // num_key_value_heads
            # But K_rot is already per (b, l); for GQA, we can reuse it across qh. The original uses expanded K/V; we mimic by repeating K_rot across qh.
            K_exp[:, qh, :, :] = K_rot
            V_exp[:, qh, :, :] = V  # original V projection is [B, L, head_dim], use as-is

        # 5) Compute attention scores: scores[b, qh, l, t] = Q_rot[b, qh, l] * K_exp[b, qh, t]
        scores = torch.empty((B, num_attention_heads, L, L), device=device, dtype=torch.float32)
        grid_attn = (B, num_attention_heads, L)
        attn_matmul_kernel[grid_attn](
            Q_rot, K_exp, scores,
            B, num_attention_heads, L, head_dim,
            Q_rot.stride(0), Q_rot.stride(1), Q_rot.stride(2),
            K_exp.stride(0), K_exp.stride(1), K_exp.stride(2),
            scores.stride(0), scores.stride(1), scores.stride(2), scores.stride(3),
            num_warps=2, num_stages=2
        )

        # 6) Softmax with causal mask (upper triangle, diagonal=1): apply mask and then softmax along L
        # Construct mask: mask[b, qh, l, t] = -inf if t < l else 0
        mask = torch.empty((B, num_attention_heads, L, L), device=device, dtype=torch.float32)
        for l_idx in range(L):
            for t_idx in range(L):
                if t_idx < l_idx:
                    mask[:, :, l_idx, t_idx].fill_(-float('inf'))
                else:
                    mask[:, :, l_idx, t_idx].fill_(0.0)

        probs = torch.empty_like(scores)

        grid_softmax = (B, num_attention_heads, L)
        softmax_mask_kernel[grid_softmax](
            scores, mask, probs,
            B, num_attention_heads, L,
            scores.stride(0), scores.stride(1), scores.stride(2), scores.stride(3),
            mask.stride(0), mask.stride(1), mask.stride(2), mask.stride(3),
            probs.stride(0), probs.stride(1), probs.stride(2), probs.stride(3),
            num_warps=2, num_stages=2
        )

        # 7) Output matmul: attn_output[b, qh, l, :] = sum_t probs[b, qh, l, t] * V_exp[b, qh, l, t]
        # V_exp shape: [B, num_attention_heads, L, head_dim]
        attn_out = torch.empty((B, num_attention_heads, L, head_dim), device=device, dtype=torch.float32)

        grid_output = (B, num_attention_heads, L)
        output_matmul_kernel[grid_output](
            probs, V_exp, attn_out,
            B, num_attention_heads, L, head_dim,
            probs.stride(0), probs.stride(1), probs.stride(2), probs.stride(3),
            V_exp.stride(0), V_exp.stride(1), V_exp.stride(2), V_exp.stride(3),
            attn_out.stride(0), attn_out.stride(1), attn_out.stride(2), attn_out.stride(3),
            num_warps=2, num_stages=2
        )

        # 8) Transpose and reshape attn_output to [B, L, num_attention_heads*head_dim] = [B, L, 12288]
        attn_flat = attn_out.transpose(1, 2).reshape(B, L, num_attention_heads * head_dim)

        # 9) Final linear projection to [B, L, hidden_dim] using o_proj_weight [hidden_dim, 12288], no bias
        output = torch.empty((B, L, self.hidden_dim), device=device, dtype=torch.float32)

        grid_final = (B, L, self.hidden_dim)
        final_linear_kernel[grid_final](
            attn_flat, o_proj_weight, output,
            B, L, num_attention_heads * head_dim, self.hidden_dim,
            attn_flat.stride(0), attn_flat.stride(1), attn_flat.stride(2),
            o_proj_weight.stride(0), o_proj_weight.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            BLOCK_K=128, num_warps=4, num_stages=2
        )

        return output


def run(*args):
    return ModelNew()(*args)
