import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False

# Fixed parameters of the model (as per the original code)
NUM_ATTENTION_HEADS = 96
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
NUM_KEY_VALUE_GROUPS = 12
DTYPE = torch.float32  # compute in fp32

# Triton kernel: Linear (Y = X @ W^T + B), X: [B, S, D_in], W: [D_out, D_in], Y: [B, S, D_out]
@triton.jit
def linear_kernel(
    X_ptr,        # *fp32, input [B, S, D_in] flattened
    W_ptr,        # *fp32, weight [D_out, D_in]
    B_ptr,        # *fp32, bias [D_out] or None (handled via pointer check)
    Y_ptr,        # *fp32, output [B, S, D_out] flattened
    Bsz: tl.constexpr,
    S: tl.constexpr,
    D_in: tl.constexpr,
    D_out: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for d_out_start in range(0, D_out, BLOCK_N):
        d_out_offsets = d_out_start + tl.arange(0, BLOCK_N)
        # Accumulate dot products over D_in in tiles of 64
        for d_in_start in range(0, D_in, 64):
            d_in_offsets = d_in_start + tl.arange(0, 64)
            x = tl.load(X_ptr + b * S * D_in + s * D_in + d_in_offsets, mask=d_in_offsets < D_in, other=0.0)  # [64]
            w = tl.load(W_ptr + d_out_offsets[:, None] * D_in + d_in_offsets[None, :], mask=(d_out_offsets[:, None] < D_out) & (d_in_offsets[None, :] < D_in), other=0.0)
            acc += tl.sum(x[:, None] * w, axis=1)
        if B_ptr is not None:
            bias = tl.load(B_ptr + d_out_offsets, mask=d_out_offsets < D_out, other=0.0)
            acc += bias
        tl.store(Y_ptr + b * S * D_out + s * D_out + d_out_offsets, acc, mask=d_out_offsets < D_out)

# Triton kernel: RMSNorm per (b, h, s, d)
@triton.jit
def rmsnorm_kernel(
    X_ptr,        # *fp32, input [B, H, S, D]
    W_ptr,        # *fp32, weight [D]
    Y_ptr,        # *fp32, output [B, H, S, D]
    S: tl.constexpr,
    D: tl.constexpr,
    eps,          # float32
):
    pid = tl.program_id(axis=0)  # one program per (b, h, s)
    total = B * H * S
    b = pid // (H * S)
    rem = pid % (H * S)
    h = rem // S
    s = rem % S
    x = tl.load(X_ptr + b * H * S * D + h * S * D + s * D + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)
    x32 = x.to(tl.float32)
    mean_sq = tl.sum(x32 * x32, axis=0) / D
    inv_rms = tl.rsqrt(mean_sq + eps)
    y = (x32 * inv_rms) * tl.load(W_ptr + tl.arange(0, D), mask=tl.arange(0, D) < D, other=1.0)
    tl.store(Y_ptr + b * H * S * D + h * S * D + s * D + tl.arange(0, D), y, mask=tl.arange(0, D) < D)

# Triton kernel: apply rotation (RoPE) for half-dimension: rotate Q and K
@triton.jit
def rotate_half_kernel(
    X_ptr,        # *fp32, input [B, H, S, D], D=128
    C_ptr,        # *fp32, cos [S, D/2] flattened
    S_ptr,        # *fp32, sin [S, D/2] flattened
    Y_ptr,        # *fp32, output [B, H, S, D]
    S: tl.constexpr,
    D: tl.constexpr,           # head_dim, e.g., 128
):
    pid = tl.program_id(axis=0)  # one program per (b, h, s)
    total = B * H * S
    b = pid // (H * S)
    rem = pid % (H * S)
    h = rem // S
    s = rem % S
    x = tl.load(X_ptr + b * H * S * D + h * S * D + s * D + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)
    d_half = D // 2
    cos_vals = tl.load(C_ptr + s * d_half + tl.arange(0, d_half), mask=tl.arange(0, d_half) < d_half, other=1.0)  # [64]
    sin_vals = tl.load(S_ptr + s * d_half + tl.arange(0, d_half), mask=tl.arange(0, d_half) < d_half, other=0.0)  # [64]
    q1 = x[:d_half]
    q2 = x[d_half:]
    y1 = q1 * cos_vals - q2 * sin_vals
    y2 = -q2 * cos_vals + q1 * sin_vals
    y = tl.concatenate([y1, y2])
    tl.store(Y_ptr + b * H * S * D + h * S * D + s * D + tl.arange(0, D), y, mask=tl.arange(0, D) < D)

# Triton kernel: compute attention softmax over keys for each (b, h), storing Soft[b, h, S, S]
# Soft_ptr should be [B, H, S, S] contiguous float32
@triton.jit
def compute_attention_scores_softmax(
    Q_ptr,        # *fp32, [B, H, S, D]
    K_ptr,        # *fp32, [B, Hkv, S, D] where Hkv is num_key_value_heads
    Soft_ptr,     # *fp32, [B, H, S, S] contiguous
    B: tl.constexpr,
    H: tl.constexpr,            # number of attention heads (query/value)
    Hkv: tl.constexpr,          # number of key/value heads
    S: tl.constexpr,
    D: tl.constexpr,
    scaling,                     # float32
):
    pid = tl.program_id(axis=0)  # one program per (b, h, m)
    total = B * H
    b = pid // H
    h = pid % H
    m = tl.program_id(axis=2)  # axis2 is S, so we use m directly as the query position
    q = tl.load(Q_ptr + b * H * S * D + h * S * D + m * D + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)  # [D]
    for n_start in range(0, S, 128):
        n_offs = n_start + tl.arange(0, 128)
        kv_h = (h // NUM_KEY_VALUE_GROUPS) * (NUM_KEY_VALUE_HEADS // NUM_KEY_VALUE_GROUPS)  # map attention head to KV head
        k = tl.load(K_ptr + b * Hkv * S * D + kv_h * S * D + n_offs * D + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)  # [128, D]
        # Compute scores vector for this tile: q · k per element across D
        scores = tl.sum(q[:, None] * k[None, :], axis=1)  # [128]
        scores = scores * scaling
        # Apply causal mask: if n > m, set to -inf
        causal_mask = n_offs > m
        scores = tl.where(causal_mask, -1e20, scores)
        # Softmax per element for this tile
        max_score = tl.max(scores, axis=0)
        scores = scores - max_score
        exp_scores = tl.exp(scores)
        sum_exp = tl.sum(exp_scores, axis=0)
        softmax = exp_scores / sum_exp  # [128]
        # Store into Soft[b, h, m, n_offs] contiguous: linear offset = m*S*S + n_offs
        tl.store(Soft_ptr + m * S * S + n_offs, softmax, mask=(n_offs < S))

# Triton kernel: compute final output by reading Soft and V, output Y_out[B, H, S, D]
@triton.jit
def compute_output_from_softmax_and_v(
    Soft_ptr,     # *fp32, [B, H, S, S] contiguous
    V_ptr,        # *fp32, [B, Hkv, S, D] (note: Hkv is num_key_value_heads; each attention head h uses its mapped kv_h)
    Y_ptr,        # *fp32, [B, H, S, D] output
    B: tl.constexpr,
    H: tl.constexpr,            # attention heads
    Hkv: tl.constexpr,          # key/value heads
    S: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,      # tile over query positions m
):
    pid = tl.program_id(axis=0)  # one program per (b, h, m_tile)
    total = B * H
    b = pid // H
    h = pid % H
    m_start = tl.program_id(axis=1) * BLOCK_M
    for m_off in range(0, BLOCK_M):
        m = m_start + m_off
        if m >= S:
            break
        out = tl.zeros([D], dtype=tl.float32)
        kv_h = (h // NUM_KEY_VALUE_GROUPS) * (NUM_KEY_VALUE_HEADS // NUM_KEY_VALUE_GROUPS)
        for n_start in range(0, S, 128):
            n_offs = n_start + tl.arange(0, 128)
            softmax_vec = tl.load(Soft_ptr + m * S * S + n_offs, mask=(n_offs < S), other=0.0)  # [128]
            v = tl.load(V_ptr + b * Hkv * S * D + kv_h * S * D + m * D + n_offs * D + tl.arange(0, D), mask=(n_offs < S) & (tl.arange(0, D) < D), other=0.0)  # [128, D]
            contrib = softmax_vec[:, None] * v  # [128, D]
            out += tl.sum(contrib, axis=0)
        tl.store(Y_ptr + b * H * S * D + h * S * D + m * D + tl.arange(0, D), out, mask=tl.arange(0, D) < D)

# Triton kernel: output projection (no bias) Y = X @ W^T, X: [B, S, D_in], W: [D_out, D_in], Y: [B, S, D_out]
@triton.jit
def linear_no_bias_kernel(
    X_ptr,        # *fp32, input [B, S, D_in] flattened
    W_ptr,        # *fp32, weight [D_out, D_in]
    Y_ptr,        # *fp32, output [B, S, D_out] flattened
    Bsz: tl.constexpr,
    S: tl.constexpr,
    D_in: tl.constexpr,
    D_out: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for d_out_start in range(0, D_out, BLOCK_N):
        d_out_offsets = d_out_start + tl.arange(0, BLOCK_N)
        for d_in_start in range(0, D_in, 64):
            d_in_offsets = d_in_start + tl.arange(0, 64)
            x = tl.load(X_ptr + b * S * D_in + s * D_in + d_in_offsets, mask=d_in_offsets < D_in, other=0.0)  # [64]
            w = tl.load(W_ptr + d_out_offsets[:, None] * D_in + d_in_offsets[None, :], mask=(d_out_offsets[:, None] < D_out) & (d_in_offsets[None, :] < D_in), other=0.0)
            acc += tl.sum(x[:, None] * w, axis=1)
        tl.store(Y_ptr + b * S * D_out + s * D_out + d_out_offsets, acc, mask=d_out_offsets < D_out)

# ModelNew: Triton-optimized version
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_attention_heads = NUM_ATTENTION_HEADS
        self.num_key_value_heads = NUM_KEY_VALUE_HEADS
        self.head_dim = HEAD_DIM
        self.num_key_value_groups = NUM_KEY_VALUE_GROUPS

    def forward(self, hidden_states, q_proj_weight, q_proj_bias, k_proj_weight, k_proj_bias, v_proj_weight, v_proj_bias, o_proj_weight, q_norm_weight, k_norm_weight, cos, sin, rms_norm_eps):
        # Ensure dtype is fp32 for Triton kernels
        device = hidden_states.device
        if hidden_states.dtype != DTYPE:
            hidden_states = hidden_states.to(DTYPE)
        if q_proj_weight.dtype != DTYPE:
            q_proj_weight = q_proj_weight.to(DTYPE)
        if k_proj_weight.dtype != DTYPE:
            k_proj_weight = k_proj_weight.to(DTYPE)
        if v_proj_weight.dtype != DTYPE:
            v_proj_weight = v_proj_weight.to(DTYPE)
        if o_proj_weight.dtype != DTYPE:
            o_proj_weight = o_proj_weight.to(DTYPE)
        if q_proj_bias is not None and q_proj_bias.dtype != DTYPE:
            q_proj_bias = q_proj_bias.to(DTYPE)
        if k_proj_bias is not None and k_proj_bias.dtype != DTYPE:
            k_proj_bias = k_proj_bias.to(DTYPE)
        if v_proj_bias is not None and v_proj_bias.dtype != DTYPE:
            v_proj_bias = v_proj_bias.to(DTYPE)
        if q_norm_weight is not None and q_norm_weight.dtype != DTYPE:
            q_norm_weight = q_norm_weight.to(DTYPE)
        if k_norm_weight is not None and k_norm_weight.dtype != DTYPE:
            k_norm_weight = k_norm_weight.to(DTYPE)
        if cos is not None and cos.dtype != DTYPE:
            cos = cos.to(DTYPE)
        if sin is not None and sin.dtype != DTYPE:
            sin = sin.to(DTYPE)

        B, S, D_in = hidden_states.shape
        assert D_in == self.num_attention_heads * self.head_dim, "hidden_states last dim must be num_attention_heads * head_dim"

        # Dense projections in Triton: Q, K, V
        Q = torch.empty((B, S, self.head_dim), device=device, dtype=DTYPE)
        grid_q = (B * S,)
        linear_kernel[grid_q](hidden_states, q_proj_weight, q_proj_bias if q_proj_bias is not None else torch.empty(0, device=device, dtype=DTYPE), Q, B, S, D_in, self.head_dim, 64, 64)

        K = torch.empty((B, S, self.head_dim), device=device, dtype=DTYPE)
        grid_k = (B * S,)
        linear_kernel[grid_k](hidden_states, k_proj_weight, k_proj_bias if k_proj_bias is not None else torch.empty(0, device=device, dtype=DTYPE), K, B, S, D_in, self.head_dim, 64, 64)

        V = torch.empty((B, S, self.head_dim), device=device, dtype=DTYPE)
        grid_v = (B * S,)
        linear_kernel[grid_v](hidden_states, v_proj_weight, v_proj_bias if v_proj_bias is not None else torch.empty(0, device=device, dtype=DTYPE), V, B, S, D_in, self.head_dim, 64, 64)

        # RMSNorm for Q and K
        Q_norm = torch.empty_like(Q)
        grid_qn = (B * self.num_attention_heads * S,)
        rmsnorm_kernel[grid_qn](Q, q_norm_weight, Q_norm, S, self.head_dim, rms_norm_eps)

        K_norm = torch.empty_like(K)
        grid_kn = (B * self.num_key_value_heads * S,)
        rmsnorm_kernel[grid_kn](K, k_norm_weight, K_norm, S, self.head_dim, rms_norm_eps)

        # Rotation (RoPE)
        Q_rot = torch.empty_like(Q_norm)
        grid_qr = (B * self.num_attention_heads * S,)
        rotate_half_kernel[grid_qr](Q_norm, cos, sin, Q_rot, S, self.head_dim)

        K_rot = torch.empty_like(K_norm)
        grid_kr = (B * self.num_key_value_heads * S,)
        rotate_half_kernel[grid_kr](K_norm, cos, sin, K_rot, S, self.head_dim)

        # GQA: map attention head to KV head: kv_h = (h // NUM_GROUPS) * (NUM_KEY_VALUE_HEADS // NUM_GROUPS)
        # We already have K_rot, V (we need to use K_rot and V for attention), and Q_rot.
        # Create Soft buffer [B, H, S, S] for softmax
        Soft = torch.empty((B, self.num_attention_heads, S, S), device=device, dtype=DTYPE)

        # Compute attention softmax and store to Soft
        grid_s = (B * self.num_attention_heads * S,)
        compute_attention_scores_softmax[grid_s](Q_rot, K_rot, Soft, B, self.num_attention_heads, self.num_key_value_heads, S, self.head_dim, 1.0 / (self.head_dim ** 0.5))

        # Compute output from Soft and V (use repeated KV: expand to H heads)
        attn_output = torch.empty((B, S, self.num_attention_heads * self.head_dim), device=device, dtype=DTYPE)
        grid_out = (B * self.num_attention_heads,)
        compute_output_from_softmax_and_v[grid_out](Soft, V, attn_output, B, self.num_attention_heads, self.num_key_value_heads, S, self.head_dim, 128)

        # Output projection (O) without bias
        output = torch.empty((B, S, self.head_dim), device=device, dtype=DTYPE)
        grid_o = (B * S,)
        linear_no_bias_kernel[grid_o](attn_output, o_proj_weight, output, B, S, self.num_attention_heads * self.head_dim, self.head_dim, 64, 64)

        return output


def run(*args):
    return ModelNew()(*args)
