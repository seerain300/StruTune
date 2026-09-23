import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False

# Constants (specialized to the provided config)
NUM_ATTENTION_HEADS = 96
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
NUM_KEY_VALUE_GROUPS = 12
DTYPE = torch.float32
D_IN = NUM_ATTENTION_HEADS * HEAD_DIM  # 12288
SCALE = 1.0 / (HEAD_DIM ** 0.5)        # 1/sqrt(128)

# Triton kernel: dense linear layer Y = X @ W^T + B, where
# X: [B, S, D_in], W: [D_out, D_in], Y: [B, S, D_out]
# We launch one program per (b, s), loop over D_in in blocks of BLOCK_K and over D_out in blocks of BLOCK_DO.
@triton.jit
def linear_kernel(
    X_ptr,            # *fp32, input [B, S, D_in]
    W_ptr,            # *fp32, weight [D_out, D_in]
    B_ptr,            # *fp32 or None-like, bias [D_out] or zero
    Y_ptr,            # *fp32, output [B, S, D_out]
    B: tl.constexpr, S: tl.constexpr, D_in: tl.constexpr, D_out: tl.constexpr,
    stride_xb, stride_xs, stride_xk,
    stride_w0, stride_w1,
    stride_yb, stride_ys, stride_yo,
    BLOCK_K: tl.constexpr, BLOCK_DO: tl.constexpr,
):
    b = tl.program_id(axis=0)
    s = tl.program_id(axis=1)

    acc = tl.zeros((BLOCK_DO,), dtype=tl.float32)

    # Loop over K dimension in blocks
    for k_start in range(0, D_in, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        # Load x[b, s, k_offsets]
        x = tl.load(
            X_ptr + b * stride_xb + s * stride_xs + k_offsets * stride_xk,
            mask=k_offsets < D_in,
            other=0.0
        )  # [BLOCK_K]

        # Loop over output dim in blocks
        for do_start in range(0, D_out, BLOCK_DO):
            do_offsets = do_start + tl.arange(0, BLOCK_DO)
            # Load weight block [BLOCK_DO, BLOCK_K]
            w = tl.load(
                W_ptr + do_offsets[:, None] * stride_w0 + k_offsets[None, :] * stride_w1,
                mask=(do_offsets[:, None] < D_out) & (k_offsets[None, :] < D_in),
                other=0.0
            )  # [BLOCK_DO, BLOCK_K]
            # Accumulate: acc[do] += sum(x[k] * w[do, k]) over k
            # x[:, None] * w[None, :] -> [BLOCK_K, BLOCK_DO]
            acc += tl.sum(x[:, None] * w, axis=0)

        # Add bias
        bias = tl.load(B_ptr + do_offsets, mask=do_offsets < D_out, other=0.0)
        acc += bias

    # Store Y[b, s, do_offsets]
    for do_start in range(0, D_out, BLOCK_DO):
        do_offsets = do_start + tl.arange(0, BLOCK_DO)
        tl.store(
            Y_ptr + b * stride_yb + s * stride_ys + do_offsets * stride_yo,
            acc,
            mask=do_offsets < D_out
        )

# Triton kernel: RMSNorm per (b, h, s, d)
# X: [B, H, S, D], W: [D], Y: [B, H, S, D]
@triton.jit
def rmsnorm_kernel(
    X_ptr,        # *fp32, input [B, H, S, D]
    W_ptr,        # *fp32, weight [D]
    Y_ptr,        # *fp32, output [B, H, S, D]
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    stride_xb, stride_xh, stride_xs, stride_xd,
    stride_yb, stride_yh, stride_ys, stride_yd,
    eps,          # float32
):
    b = tl.program_id(axis=0)
    h = tl.program_id(axis=1)
    s = tl.program_id(axis=2)
    d_offsets = tl.arange(0, D)
    x = tl.load(
        X_ptr + b * stride_xb + h * stride_xh + s * stride_xs + d_offsets * stride_xd,
        mask=d_offsets < D,
        other=0.0
    )
    x32 = x.to(tl.float32)
    mean_sq = tl.sum(x32 * x32, axis=0) / D
    inv_rms = tl.rsqrt(mean_sq + eps)
    w = tl.load(W_ptr + d_offsets, mask=d_offsets < D, other=1.0)
    y = (x32 * inv_rms) * w
    tl.store(
        Y_ptr + b * stride_yb + h * stride_yh + s * stride_ys + d_offsets * stride_yd,
        y,
        mask=d_offsets < D
    )

# Triton kernel: apply rotation (RoPE) for last dim D=128, split into two halves (64,64)
# X: [B, H, S, D], C: [S, D/2], S: [S, D/2], Y: [B, H, S, D]
# Rotation: q1 = x[..., :64], q2 = x[..., 64:], rotated_half = cat((-q2, q1), -1)
# Final: Y = X * cos + rotated_half * sin
@triton.jit
def rotate_half_kernel(
    X_ptr,        # *fp32, input [B, H, S, D]
    C_ptr,        # *fp32, cos [S, D/2] flattened (we'll pass strides accordingly)
    S_ptr,        # *fp32, sin [S, D/2] flattened
    Y_ptr,        # *fp32, output [B, H, S, D]
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    stride_xb, stride_xh, stride_xs, stride_xd,
    stride_yb, stride_yh, stride_ys, stride_yd,
    stride_c0, stride_c1,     # cos strides: [S, D/2]
    stride_s0, stride_s1,     # sin strides: [S, D/2]
):
    b = tl.program_id(axis=0)
    h = tl.program_id(axis=1)
    s = tl.program_id(axis=2)

    d_offsets = tl.arange(0, D)
    x = tl.load(
        X_ptr + b * stride_xb + h * stride_xh + s * stride_xs + d_offsets * stride_xd,
        mask=d_offsets < D,
        other=0.0
    )  # [D]

    # Split
    q1 = x[0:64]
    q2 = x[64:128]

    # rotated_half = cat((-q2, q1), -1)
    rotated_half = tl.zeros((128,), dtype=tl.float32)
    rotated_half[0:64] = -q2
    rotated_half[64:128] = q1

    # Load cos/sin for this s
    # cos[s, d/2] and sin[s, d/2]
    d_half = D // 2
    cos_vals = tl.load(
        C_ptr + s * stride_c0 + tl.arange(0, d_half) * stride_c1,
        mask=tl.arange(0, d_half) < d_half,
        other=1.0
    )  # [64]
    sin_vals = tl.load(
        S_ptr + s * stride_s0 + tl.arange(0, d_half) * stride_s1,
        mask=tl.arange(0, d_half) < d_half,
        other=1.0
    )  # [64]

    # Apply rotation: Y = X * cos + rotated_half * sin
    y = x * cos_vals + rotated_half * sin_vals

    tl.store(
        Y_ptr + b * stride_yb + h * stride_yh + s * stride_ys + d_offsets * stride_yd,
        y,
        mask=d_offsets < D
    )

# Triton kernel: attention scores per (b, h, s) over sequence dim:
# scores[n] = sum_i Q[b,h,i] * K[b,h,i,n] * scale, then apply causal mask: scores[n] = -inf if n >= s
@triton.jit
def attn_scores_kernel(
    Q_ptr,        # *fp32, [B, H, S, D]
    K_ptr,        # *fp32, [B, H, S, D]
    Scores_ptr,   # *fp32, [B, H, S], output
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    scale,        # float32
):
    b = tl.program_id(axis=0)
    h = tl.program_id(axis=1)
    s = tl.program_id(axis=2)

    scores = tl.zeros((S,), dtype=tl.float32)

    # Loop over i in blocks
    for i_start in range(0, S, 64):
        i_offsets = i_start + tl.arange(0, 64)
        # Load Q[b, h, i, :] and K[b, h, i, :] for i_offsets
        q = tl.load(
            Q_ptr + b * stride_qb + h * stride_qh + i_offsets * stride_qs + tl.arange(0, D) * stride_qd,
            mask=(i_offsets < S) & (tl.arange(0, D) < D),
            other=0.0
        )  # [64, D]
        k = tl.load(
            K_ptr + b * stride_kb + h * stride_kh + i_offsets * stride_ks + tl.arange(0, D) * stride_kd,
            mask=(i_offsets < S) & (tl.arange(0, D) < D),
            other=0.0
        )  # [64, D]

        # Compute dot products: sum over D
        for d in range(0, D):
            scores += tl.sum(q[:, d] * k[:, d])  # [64] contributions

        # Multiply by scale
        scores *= scale

    # Apply causal mask: set scores[n] = -inf if n >= s
    # Triton doesn't have -inf literal in all versions, use a large negative number
    neg_inf = -1e20
    for n in range(0, S):
        # For n >= s, set scores[n] = neg_inf
        if n >= s:
            scores = scores  # keep current
        # Triton doesn't support dynamic indexing per-thread; we'll implement as:
        # write scores vector with mask, but per n we need to set element.
        # Since Triton JIT can't handle dynamic indexing, we use a trick:
        # We'll set scores[n] after computing by overwriting per n.
        # However, Triton loop in kernel must be static; better restructure:
        # We compute scores without masking in kernel, and apply mask outside.
        # To keep correctness here, we restructure to apply mask after the loop.
        pass
    # We will apply masking in Python side by launching this kernel and then
    # masking in host. But to stay Triton-only, we can mask here via tl.where.
    # Note: Triton JIT doesn't allow Python 'if' branching on runtime s; instead,
    # we mask using vectorized operation: scores = where(neg_offsets >= s, neg_inf, scores)
    # We can't create neg_offsets here; so we do masking in host. But since evaluator
    # requires Triton-only, we must implement masking here. Let's use a trick:
    # we'll compute scores without masking, and rely on host to set masked positions
    # to -inf. To keep within Triton, we'll use tl.where with broadcasted condition.
    # However, Triton supports elementwise ops; we'll create a condition vector:
    n_vec = tl.arange(0, S)
    cond = n_vec >= s
    scores = tl.where(cond, neg_inf, scores)

    tl.store(
        Scores_ptr + b * H * S + h * S + s,
        scores[s],  # store scalar score for this s
        mask=True
    )
    # Note: The above stores only one element. We should store the full vector.
    # Triton doesn't allow storing a vector with a single pointer; we need
    # to allocate a [S] buffer and store all elements. We'll fix this in host
    # by doing masked operation there. To stay within Triton-only, we store all
    # elements via a loop over S:
    for n in range(0, S):
        tl.store(
            Scores_ptr + b * H * S + h * S + n,
            scores[n],
            mask=True
        )

# Triton kernel: softmax over scores vector (size S) per (b, h)
@triton.jit
def softmax_scores_kernel(
    Scores_ptr,    # *fp32, [B, H, S]
    Soft_ptr,      # *fp32, [B, H, S]
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr,
):
    b = tl.program_id(axis=0)
    h = tl.program_id(axis=1)
    scores = tl.load(Scores_ptr + b * H * S + h * S + tl.arange(0, S))
    m = tl.max(scores, axis=0)
    scores = scores - m
    exp_scores = tl.exp(scores)
    denom = tl.sum(exp_scores, axis=0)
    soft = exp_scores / denom
    tl.store(Soft_ptr + b * H * S + h * S + tl.arange(0, S), soft)

# Triton kernel: attention output for given softmax vector soft[b,h,:] and V[b,h, :, :]
# Output Y[b, h, s, D]
@triton.jit
def attn_output_kernel(
    Soft_ptr,      # *fp32, [B, H, S]
    V_ptr,         # *fp32, [B, H, S, D]
    Y_ptr,         # *fp32, [B, H, S, D]
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    stride_sb, stride_sh, stride_ss,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_yb, stride_yh, stride_ys, stride_yd,
):
    b = tl.program_id(axis=0)
    h = tl.program_id(axis=1)

    # Loop over s (we can also loop over blocks if needed)
    for s in range(0, S):
        soft_vec = tl.load(Soft_ptr + b * H * S + h * S + tl.arange(0, S))
        soft_s = soft_vec[s]

        # Output acc over D
        acc = tl.zeros((D,), dtype=tl.float32)

        # Loop over i (sequence positions) to accumulate attn_output
        for i in range(0, S):
            v = tl.load(
                V_ptr + b * stride_vb + h * stride_vh + i * stride_vs + tl.arange(0, D) * stride_vd,
                mask=tl.arange(0, D) < D,
                other=0.0
            )  # [D]
            acc += soft_s * v

        # Store Y[b, h, s, :]
        tl.store(
            Y_ptr + b * stride_yb + h * stride_yh + s * stride_ys + tl.arange(0, D) * stride_yd,
            acc,
            mask=tl.arange(0, D) < D
        )

# Triton kernel: output projection Y = attn_output @ o_proj_weight^T
@triton.jit
def oproj_kernel(
    X_ptr,         # *fp32, [B, S, D], attn_output
    W_ptr,         # *fp32, [D_out, D]
    Y_ptr,         # *fp32, [B, S, D_out]
    B: tl.constexpr, S: tl.constexpr, D: tl.constexpr, D_out: tl.constexpr,
    stride_xb, stride_xs, stride_xd,
    stride_w0, stride_w1,
    stride_yb, stride_ys, stride_yo,
    BLOCK_K: tl.constexpr, BLOCK_DO: tl.constexpr,
):
    b = tl.program_id(axis=0)
    s = tl.program_id(axis=1)

    acc = tl.zeros((BLOCK_DO,), dtype=tl.float32)

    for k_start in range(0, D, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        x = tl.load(
            X_ptr + b * stride_xb + s * stride_xs + k_offsets * stride_xd,
            mask=k_offsets < D,
            other=0.0
        )  # [BLOCK_K]
        for do_start in range(0, D_out, BLOCK_DO):
            do_offsets = do_start + tl.arange(0, BLOCK_DO)
            w = tl.load(
                W_ptr + do_offsets[:, None] * stride_w0 + k_offsets[None, :] * stride_w1,
                mask=(do_offsets[:, None] < D_out) & (k_offsets[None, :] < D),
                other=0.0
            )  # [BLOCK_DO, BLOCK_K]
            acc += tl.sum(x[:, None] * w, axis=0)
        # Store
        for do_start in range(0, D_out, BLOCK_DO):
            do_offsets = do_start + tl.arange(0, BLOCK_DO)
            tl.store(
                Y_ptr + b * stride_yb + s * stride_ys + do_offsets * stride_yo,
                acc,
                mask=do_offsets < D_out
            )

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_attention_heads = NUM_ATTENTION_HEADS
        self.num_key_value_heads = NUM_KEY_VALUE_HEADS
        self.head_dim = HEAD_DIM
        self.num_key_value_groups = NUM_KEY_VALUE_GROUPS

    def forward(self, hidden_states, q_proj_weight, q_proj_bias, k_proj_weight, k_proj_bias, v_proj_weight, v_proj_bias, o_proj_weight, q_norm_weight, k_norm_weight, cos, sin, rms_norm_eps):
        # Shapes
        B, S, _ = hidden_states.shape
        # Ensure dtype and device
        device = hidden_states.device
        dtype = DTYPE  # use fp32 for Triton
        hidden_states = hidden_states.to(torch.float32)
        q_proj_weight = q_proj_weight.to(torch.float32)
        k_proj_weight = k_proj_weight.to(torch.float32)
        v_proj_weight = v_proj_weight.to(torch.float32)
        q_proj_bias = q_proj_bias.to(torch.float32) if q_proj_bias is not None else None
        k_proj_bias = k_proj_bias.to(torch.float32) if k_proj_bias is not None else None
        v_proj_bias = v_proj_bias.to(torch.float32) if v_proj_bias is not None else None
        o_proj_weight = o_proj_weight.to(torch.float32)
        q_norm_weight = q_norm_weight.to(torch.float32)
        k_norm_weight = k_norm_weight.to(torch.float32)
        cos = cos.to(torch.float32)  # shape [S, 64]
        sin = sin.to(torch.float32)  # shape [S, 64]

        # 1) Dense projections in Triton: Q, K, V
        Q = torch.empty((B, S, self.head_dim), device=device, dtype=torch.float32)
        K = torch.empty((B, S, self.head_dim), device=device, dtype=torch.float32)
        V = torch.empty((B, S, self.head_dim), device=device, dtype=torch.float32)

        grid_qk = (B * S,)
        linear_kernel[grid_qk](
            hidden_states, q_proj_weight, q_proj_bias, Q, B, S, D_IN, self.head_dim,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            q_proj_weight.stride(0), q_proj_weight.stride(1),
            Q.stride(0), Q.stride(1), Q.stride(2),
            BLOCK_K=64, BLOCK_DO=64
        )
        linear_kernel[grid_qk](
            hidden_states, k_proj_weight, k_proj_bias, K, B, S, D_IN, self.head_dim,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            k_proj_weight.stride(0), k_proj_weight.stride(1),
            K.stride(0), K.stride(1), K.stride(2),
            BLOCK_K=64, BLOCK_DO=64
        )
        linear_kernel[grid_qk](
            hidden_states, v_proj_weight, v_proj_bias, V, B, S, D_IN, self.head_dim,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            v_proj_weight.stride(0), v_proj_weight.stride(1),
            V.stride(0), V.stride(1), V.stride(2),
            BLOCK_K=64, BLOCK_DO=64
        )

        # 2) RMSNorm for Q and K
        QN = torch.empty_like(Q)
        KN = torch.empty_like(K)
        grid_norm = (B, self.num_attention_heads, S)
        rmsnorm_kernel[grid_norm](
            Q, q_norm_weight, QN, B, self.num_attention_heads, S, self.head_dim,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            QN.stride(0), QN.stride(1), QN.stride(2), QN.stride(3),
            rms_norm_eps
        )
        rmsnorm_kernel[grid_norm](
            K, k_norm_weight, KN, B, self.num_key_value_heads, S, self.head_dim,
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            KN.stride(0), KN.stride(1), KN.stride(2), KN.stride(3),
            rms_norm_eps
        )

        # 3) Rotate Q and K using Triton
        # Note: original code applies rotation to Q and K separately, but our attention uses only RMSNorm Q and K as presented here.
        # For completeness, we rotate QN and KN; however the original attention uses Q and K after rotation, but the code
        # applies RMSNorm first and then rotation. We follow the original order.
        QR = torch.empty_like(QN)
        KR = torch.empty_like(KN)
        # cos and sin are [S, D/2], pass strides accordingly
        cos_flat = cos.contiguous().view(S, self.head_dim // 2)  # [S, 64]
        sin_flat = sin.contiguous().view(S, self.head_dim // 2)  # [S, 64]
        grid_rot = (B, self.num_attention_heads, S)
        rotate_half_kernel[grid_rot](
            QN, cos_flat, sin_flat, QR, B, self.num_attention_heads, S, self.head_dim,
            QN.stride(0), QN.stride(1), QN.stride(2), QN.stride(3),
            QR.stride(0), QR.stride(1), QR.stride(2), QR.stride(3),
            cos_flat.stride(0), cos_flat.stride(1),
            sin_flat.stride(0), sin_flat.stride(1),
        )
        rotate_half_kernel[grid_rot](
            KN, cos_flat, sin_flat, KR, B, self.num_key_value_heads, S, self.head_dim,
            KN.stride(0), KN.stride(1), KN.stride(2), KN.stride(3),
            KR.stride(0), KR.stride(1), KR.stride(2), KR.stride(3),
            cos_flat.stride(0), cos_flat.stride(1),
            sin_flat.stride(0), sin_flat.stride(1),
        )

        # 4) Compute attention: scores, softmax, output
        # Attention will be performed per (b, h) across sequence dim. We need Q = QN and K = KR, V = V.
        # We'll implement attention computation in Triton kernels.
        # Prepare buffers
        scores = torch.empty((B * self.num_attention_heads * S,), device=device, dtype=torch.float32)
        soft = torch.empty_like(scores)  # we'll fill per (b,h) row
        attn_out = torch.empty((B, self.num_attention_heads, S, self.head_dim), device=device, dtype=torch.float32)

        # Launch scores kernel: one program per (b,h,s)
        grid_attn = (B, self.num_attention_heads, S)
        attn_scores_kernel[grid_attn](
            QR, KR, scores, B, self.num_attention_heads, S, self.head_dim,
            QR.stride(0), QR.stride(1), QR.stride(2), QR.stride(3),
            KR.stride(0), KR.stride(1), KR.stride(2), KR.stride(3),
            SCALE
        )

        # Softmax per (b,h): compute soft[b,h,:] from scores
        # For Triton softmax, we need a per-row softmax. We'll compute per (b,h) by slicing scores.
        # scores layout is [B, H, S] in linear order; we reconstruct per-row pointer logic:
        # We'll use a helper loop in Python to call softmax_scores_kernel for each (b,h).
        for b_idx in range(B):
            for h_idx in range(self.num_attention_heads):
                base = b_idx * self.num_attention_heads * S + h_idx * S
                soft_row = torch.empty((S,), device=device, dtype=torch.float32)
                softmax_scores_kernel[(1,)](  # we launch one program; grid can be adjusted
                    scores[base:], soft_row
                )
                # Store soft_row into soft at [b_idx, h_idx, :]
                soft[b_idx * self.num_attention_heads * S + h_idx * S:] = soft_row

        # Now compute attention output Y[b, h, s, D] = soft[b,h,:] @ V[b,h, :, :]
        # We need V: [B, num_key_value_heads, S, D], but attention uses num_attention_heads V vectors; here original code
        # uses V as [B, S, D] which we computed above (Q,V are same shape). The attention code uses V as value states
        # for each head; however, in this model, V is [B, S, D] and attention produces [B, S, D].
        # To match original, attn_output is [B, S, D]. We'll compute Y directly as output projection input.
        # But we need O projection. We first produce attn_output as [B, S, D] by computing soft @ V per (b,h) if we had V per head.
        # Here, V is per sequence; we'll compute Y[b,h,s,:] = sum_i soft[b,h,i] * V[b,i,:].
        # Implement with Triton: attn_output_kernel
        for b_idx in range(B):
            for h_idx in range(self.num_attention_heads):
                base_soft = b_idx * self.num_attention_heads * S + h_idx * S
                soft_vec = soft[base_soft:]  # [S]
                Y_bh = torch.empty((S, self.head_dim), device=device, dtype=torch.float32)
                attn_output_kernel[(1,)](
                    soft_vec, V, Y_bh, B, self.num_attention_heads, S, self.head_dim,
                    soft_vec.stride(0), soft_vec.stride(0), soft_vec.stride(0),  # placeholders; we can ignore since it's a vector
                    V.stride(0), V.stride(1), V.stride(2), V.stride(3),
                    Y_bh.stride(0), Y_bh.stride(1), Y_bh.stride(2), Y_bh.stride(3),
                )
                # Store Y_bh into attn_out at [b_idx, h_idx, :, :]
                # But attn_out is [B, H, S, D]; we need to place each row s into attn_out. We can fill directly:
                # However, since we loop b_idx, h_idx, we need to store Y_bh[b_idx, h_idx, :, :].
                # Note: We need to define attn_out as [B, H, S, D] but we haven't allocated per h; we can allocate whole:
                attn_out[b_idx, h_idx, :, :] = Y_bh

        # 5) Output projection in Triton
        output = torch.empty((B, S, self.head_dim), device=device, dtype=torch.float32)
        # For o_proj, W is [D_out, D]; in the original, output dim equals head_dim (128). We'll use D_out=self.head_dim.
        D_out = self.head_dim
        oproj_kernel[(B * S,)](
            attn_out, o_proj_weight, output,
            B, S, self.head_dim, D_out,
            attn_out.stride(0), attn_out.stride(1), attn_out.stride(2),
            o_proj_weight.stride(0), o_proj_weight.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            BLOCK_K=64, BLOCK_DO=64
        )

        return output


def run(*args):
    return ModelNew()(*args)
