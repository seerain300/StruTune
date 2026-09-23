import torch
import triton
import triton.language as tl


# Constants derived from the original config (fixed in this implementation).
NUM_ATTENTION_HEADS = 96
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
GROUPS = 12  # NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS
SCALING = 1.0 / (HEAD_DIM ** 0.5)
RMS_NORM_EPS = 1e-5  # use the same default behavior as original


# 1) Triton dense linear for input [B, S, K] -> out[b, s, h] where h is a linear head index.
# Each program handles (b, h) and loops over s and K tiles.
@triton.jit
def triton_linear_bsh(input_ptr, weight_ptr, bias_ptr, out_ptr,
                      B, S, K,
                      input_stride0, input_stride1, input_stride2,
                      weight_stride0, weight_stride1,
                      bias_stride0,  # typically 1
                      out_stride0, out_stride1, out_stride2,
                      BLOCK_K: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    # For each sequence position s
    for s in range(0, S):
        acc = tl.zeros((), dtype=tl.float32)
        # Accumulate over K in tiles
        for k0 in range(0, K, BLOCK_K):
            kk = k0 + tl.arange(0, BLOCK_K)
            mask = kk < K
            # input[b, s, kk]
            base_in = b * input_stride0 + s * input_stride1
            x_vec = tl.load(input_ptr + base_in + kk * input_stride2, mask=mask, other=0.0)
            # weight[h, kk]
            w_vec = tl.load(weight_ptr + h * weight_stride0 + kk * weight_stride1, mask=mask, other=0.0)
            # dot product for this tile
            acc += tl.sum(x_vec * w_vec, axis=0)
        # add bias
        bias_val = tl.load(bias_ptr + h * bias_stride0)
        acc += bias_val
        # store
        tl.store(out_ptr + b * out_stride0 + s * out_stride1 + h * out_stride2, acc)


# 2) Triton RMSNorm per (b, h) across S: out[b, s, h] = x[b, s, h] * (weight[h] / sqrt(mean(x^2) + eps))
@triton.jit
def triton_rmsnorm_bsh(x_ptr, weight_ptr, out_ptr,
                       B, S,
                       x_stride0, x_stride1, x_stride2,
                       out_stride0, out_stride1, out_stride2,
                       eps,  # float
                       BLOCK_S: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    # First pass: compute sum of squares across S
    sum_sq = tl.zeros((), dtype=tl.float32)
    for s0 in range(0, S, BLOCK_S):
        ss = s0 + tl.arange(0, BLOCK_S)
        mask = ss < S
        x = tl.load(x_ptr + b * x_stride0 + ss * x_stride1 + h * x_stride2, mask=mask, other=0.0)
        sum_sq += tl.sum(x * x, axis=0)
    mean = sum_sq / S
    inv_rms = 1.0 / tl.sqrt(mean + eps)
    scale = inv_rms * tl.load(weight_ptr + h)
    # Second pass: write normalized and scaled
    for s0 in range(0, S, BLOCK_S):
        ss = s0 + tl.arange(0, BLOCK_S)
        mask = ss < S
        x = tl.load(x_ptr + b * x_stride0 + ss * x_stride1 + h * x_stride2, mask=mask, other=0.0)
        y = x * scale
        tl.store(out_ptr + b * out_stride0 + ss * out_stride1 + h * out_stride2, y, mask=mask)


# 3) Triton GQA expand: expand KV heads from KVH to H using groups mapping (GROUPS = H // KVH).
# For each (b, kh, g, j), copy K[b, kh, j, :] into K_out[b, kh*GROUPS + g, j, :]
# and similarly for V. We implement this as a Triton kernel with grid (B, KVH, GROUPS, S).
@triton.jit
def triton_expand_kv(K_ptr, V_ptr, Kout_ptr, Vout_ptr,
                     B, S, KVH, KD,
                     K_stride0, K_stride1, K_stride2, K_stride3,
                     V_stride0, V_stride1, V_stride2, V_stride3,
                     Kout_stride0, Kout_stride1, Kout_stride2, Kout_stride3,
                     Vout_stride0, Vout_stride1, Vout_stride2, Vout_stride3,
                     GROUPS: tl.constexpr):
    b = tl.program_id(0)
    kh = tl.program_id(1)
    g = tl.program_id(2)
    j = tl.program_id(3)
    h_target = kh * GROUPS + g
    # Copy K row
    k_val = tl.load(K_ptr + b * K_stride0 + j * K_stride1 + kh * K_stride2 + 0 * K_stride3)
    tl.store(Kout_ptr + b * Kout_stride0 + j * Kout_stride1 + h_target * Kout_stride2 + 0 * Kout_stride3, k_val)
    # Copy V row
    v_val = tl.load(V_ptr + b * V_stride0 + j * V_stride1 + kh * V_stride2 + 0 * V_stride3)
    tl.store(Vout_ptr + b * Vout_stride0 + j * Vout_stride1 + h_target * Vout_stride2 + 0 * Vout_stride3, v_val)


# 4) Triton attention compute:
# Input:
#   Q_norm: [B, S, H]
#   K_exp:  [B, S, H]
#   V_exp:  [B, S, H]
# Output:
#   Out:    [B, S, H]
# Steps:
# - Compute QK scores as Q @ K^T per (b, h) across S, scaled by SCALING.
# - Apply causal mask (j > i -> -inf).
# - Softmax along j.
# - Accumulate Out[i] += softmax_row[j] * V[j].
# Implementation uses one Triton program per (b, h). Inside, loops over i, j tiles with masks.
@triton.jit
def triton_attention_forward(Q_ptr, K_ptr, V_ptr, Out_ptr,
                             B, S, H,
                             Q_stride0, Q_stride1, Q_stride2,
                             K_stride0, K_stride1, K_stride2,
                             V_stride0, V_stride1, V_stride2,
                             Out_stride0, Out_stride1, Out_stride2,
                             BLOCK_S: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    # We will compute attention for all s positions: one program handles all s, but we iterate s and use masks for length.
    # Initialize output vector
    out_vec = tl.zeros((S,), dtype=tl.float32)
    # Loop over query positions i
    for i0 in range(0, S, BLOCK_S):
        ii = i0 + tl.arange(0, BLOCK_S)
        mask_i = ii < S
        # For each i, compute scores against all j
        # scores[ii, jj] = (Q[b, ii, h] * K[b, jj, h]) * SCALING
        # We'll compute q_vec and k_vec per tile, then pairwise multiply
        q_vec = tl.zeros((BLOCK_S,), dtype=tl.float32)
        # Load q vector for all ii
        for ii_idx in range(0, BLOCK_S):
            if mask_i[ii_idx]:
                q_vec[ii_idx] = tl.load(Q_ptr + b * Q_stride0 + ii[ii_idx] * Q_stride1 + h * Q_stride2)
        # Initialize scores matrix
        scores = tl.zeros((BLOCK_S, BLOCK_S), dtype=tl.float32)
        # For each j in tiles
        for j0 in range(0, S, BLOCK_S):
            jj = j0 + tl.arange(0, BLOCK_S)
            mask_j = jj < S
            k_vec = tl.zeros((BLOCK_S,), dtype=tl.float32)
            # Load k vector for all jj
            for jj_idx in range(0, BLOCK_S):
                if mask_j[jj_idx]:
                    k_vec[jj_idx] = tl.load(K_ptr + b * K_stride0 + jj[jj_idx] * K_stride1 + h * K_stride2)
            # Compute qk for each ii
            # Note: q_vec is vector; k_vec is vector. scores[ii, jj] = q_vec[ii] * k_vec[jj] * SCALING
            for ii_idx in range(0, BLOCK_S):
                if mask_i[ii_idx]:
                    qk = q_vec[ii_idx] * tl.sum(k_vec * k_vec, axis=0)  # not correct; we need dot product between q_vec[ii_idx] and k_vec
                    # Better approach: compute qk[ii, :] = q_vec[ii] * k_vec[:]; but k_vec is vector. Use pairwise:
                    # scores[ii, jj] = q_vec[ii] * k_vec[jj] * SCALING
                    # Implement by outer product
                    # For each ii, load q scalar and multiply with k_vec
                    q_scalar = q_vec[ii_idx]
                    scores[ii_idx, :] = q_scalar * k_vec * SCALING
        # Apply causal mask: j > i -> -inf
        # Make scores -inf where j > i
        for ii_idx in range(0, BLOCK_S):
            if mask_i[ii_idx]:
                for jj_idx in range(0, BLOCK_S):
                    if mask_j[jj_idx] and (jj_idx > ii_idx):
                        scores[ii_idx, jj_idx] = -1e20
        # Softmax along j axis
        for ii_idx in range(0, BLOCK_S):
            if mask_i[ii_idx]:
                exp_row = tl.exp(scores[ii_idx, :])
                sum_row = tl.sum(exp_row, axis=0)
                soft_row = exp_row / sum_row
                # Accumulate output: Out[i] += soft_row[jj] * V[b, jj, h]
                # Need to gather V vector
                v_vec = tl.zeros((BLOCK_S,), dtype=tl.float32)
                for jj_idx in range(0, BLOCK_S):
                    if mask_j[jj_idx]:
                        v_vec[jj_idx] = tl.load(V_ptr + b * V_stride0 + jj[jj_idx] * V_stride1 + h * V_stride2)
                out_vec[ii_idx] += tl.sum(soft_row * v_vec, axis=0)
    # Store output vector for this (b, h)
    for s in range(0, S):
        tl.store(Out_ptr + b * Out_stride0 + s * Out_stride1 + h * Out_stride2, out_vec[s])


# 5) Triton output projection: linear over last dimension (no bias).
# Input: attn_out [B, S, H], weight [H, K]; Output: out [B, S, H]
# We implement with grid (B, H, S): each program handles one s.
@triton.jit
def triton_o_proj(attn_ptr, weight_ptr, out_ptr,
                 B, S, H, K,
                 attn_stride0, attn_stride1, attn_stride2,
                 w_stride0, w_stride1,
                 out_stride0, out_stride1, out_stride2,
                 BLOCK_K: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    for s in range(0, S):
        acc = tl.zeros((), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            kk = k0 + tl.arange(0, BLOCK_K)
            mask = kk < K
            attn_vec = tl.load(attn_ptr + b * attn_stride0 + s * attn_stride1 + kk * attn_stride2, mask=mask, other=0.0)
            w_vec = tl.load(weight_ptr + h * w_stride0 + kk * w_stride1, mask=mask, other=0.0)
            acc += tl.sum(attn_vec * w_vec, axis=0)
        tl.store(out_ptr + b * out_stride0 + s * out_stride1 + h * out_stride2, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # We do not use nn.Parameters here; all heavy work is done in Triton kernels.

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
                rms_norm_eps: float):
        # hidden_states: [B, S, K_in] (K_in may be any, but we linearly produce H)
        B, S, K_in = hidden_states.shape

        # 0) Allocate intermediates
        device = hidden_states.device
        dtype = hidden_states.dtype  # assume float32 for stability; all kernels operate in this dtype

        # 1) Dense linear for Q, K, V: produce [B, S, H]
        # Q
        Q = torch.empty((B, S, NUM_ATTENTION_HEADS), device=device, dtype=dtype)
        triton_linear_bsh[triton.cdiv(B * NUM_ATTENTION_HEADS, 1)](  # launch grid: enough programs, masked inside
            hidden_states, q_proj_weight, q_proj_bias, Q,
            B, S, K_in,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            q_proj_weight.stride(0), q_proj_weight.stride(1),
            q_proj_bias.stride(0),
            Q.stride(0), Q.stride(1), Q.stride(2),
            BLOCK_K=128, num_warps=4)

        # K
        K = torch.empty((B, S, NUM_KEY_VALUE_HEADS), device=device, dtype=dtype)
        triton_linear_bsh[triton.cdiv(B * NUM_KEY_VALUE_HEADS, 1)](
            hidden_states, k_proj_weight, k_proj_bias, K,
            B, S, K_in,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            k_proj_weight.stride(0), k_proj_weight.stride(1),
            k_proj_bias.stride(0),
            K.stride(0), K.stride(1), K.stride(2),
            BLOCK_K=128, num_warps=4)

        # V
        V = torch.empty((B, S, NUM_KEY_VALUE_HEADS), device=device, dtype=dtype)
        triton_linear_bsh[triton.cdiv(B * NUM_KEY_VALUE_HEADS, 1)](
            hidden_states, v_proj_weight, v_proj_bias, V,
            B, S, K_in,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            v_proj_weight.stride(0), v_proj_weight.stride(1),
            v_proj_bias.stride(0),
            V.stride(0), V.stride(1), V.stride(2),
            BLOCK_K=128, num_warps=4)

        # 2) RMSNorm for Q and K per (b, h)
        Q_norm = torch.empty_like(Q)
        triton_rmsnorm_bsh[(B, NUM_ATTENTION_HEADS)](
            Q, q_norm_weight, Q_norm,
            B, S,
            Q.stride(0), Q.stride(1), Q.stride(2),
            Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(2),
            rms_norm_eps,
            BLOCK_S=64, num_warps=4
        )
        K_norm = torch.empty_like(K)
        triton_rmsnorm_bsh[(B, NUM_KEY_VALUE_HEADS)](
            K, k_norm_weight, K_norm,
            B, S,
            K.stride(0), K.stride(1), K.stride(2),
            K_norm.stride(0), K_norm.stride(1), K_norm.stride(2),
            rms_norm_eps,
            BLOCK_S=64, num_warps=4
        )

        # 3) Rotate Position Embedding (RoPE): expand cos/sin to match head dim and apply rotation for Q and K.
        # Here we apply rotation to Q_norm and K_norm in-place. We implement a Triton kernel over (B, H, S).
        # We assume cos/sin are [1, 128] vectors and we broadcast them across batch and sequence.
        # This is done in-place on Q_norm and K_norm.
        # Note: original code used sin/cos vectors of length 64 (first half), but we mirror the entire 128-dim behavior by using cos/sin full vectors.
        # We load them and apply rotation as before.
        # We call a Triton kernel to perform rotation for each (b, h, s).
        for h in range(NUM_ATTENTION_HEADS):
            triton_rope_row[(B, 1, S)](
                Q_norm, cos, sin, Q_norm,
                B, NUM_ATTENTION_HEADS, S, HEAD_DIM,
                Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(2),
                cos.stride(0), cos.stride(1),
                sin.stride(0), sin.stride(1),
                Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(2),
                BLOCK_D=128, num_warps=4
            )
        for h in range(NUM_KEY_VALUE_HEADS):
            triton_rope_row[(B, 1, S)](
                K_norm, cos, sin, K_norm,
                B, NUM_KEY_VALUE_HEADS, S, HEAD_DIM,
                K_norm.stride(0), K_norm.stride(1), K_norm.stride(2),
                cos.stride(0), cos.stride(1),
                sin.stride(0), sin.stride(1),
                K_norm.stride(0), K_norm.stride(1), K_norm.stride(2),
                BLOCK_D=128, num_warps=4
            )

        # 4) Expand KV from NUM_KEY_VALUE_HEADS to NUM_ATTENTION_HEADS with groups mapping
        K_exp = torch.empty((B, S, NUM_ATTENTION_HEADS), device=device, dtype=dtype)
        V_exp = torch.empty((B, S, NUM_ATTENTION_HEADS), device=device, dtype=dtype)
        triton_expand_kv[(B, NUM_KEY_VALUE_HEADS, GROUPS, S)](
            K_norm, V, K_exp, V_exp,
            B, S, NUM_KEY_VALUE_HEADS, HEAD_DIM,
            K_norm.stride(0), K_norm.stride(1), K_norm.stride(2), K_norm.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            K_exp.stride(0), K_exp.stride(1), K_exp.stride(2), K_exp.stride(3),
            V_exp.stride(0), V_exp.stride(1), V_exp.stride(2), V_exp.stride(3),
            GROUPS=GROUPS, num_warps=1
        )

        # 5) Attention compute: Out = attention(Q_norm, K_exp, V_exp) per (b, h)
        Out = torch.empty((B, S, NUM_ATTENTION_HEADS), device=device, dtype=dtype)
        triton_attention_forward[(B, NUM_ATTENTION_HEADS)](
            Q_norm, K_exp, V_exp, Out,
            B, S, NUM_ATTENTION_HEADS,
            Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(2),
            K_exp.stride(0), K_exp.stride(1), K_exp.stride(2),
            V_exp.stride(0), V_exp.stride(1), V_exp.stride(2),
            Out.stride(0), Out.stride(1), Out.stride(2),
            BLOCK_S=64, num_warps=4
        )

        # 6) Output projection (no bias): Out_proj = linear(Out, o_proj_weight)
        Output = torch.empty((B, S, NUM_ATTENTION_HEADS * HEAD_DIM), device=device, dtype=dtype)
        triton_o_proj[(B, NUM_ATTENTION_HEADS, S)](
            Out, o_proj_weight, Output,
            B, S, NUM_ATTENTION_HEADS, o_proj_weight.shape[1],
            Out.stride(0), Out.stride(1), Out.stride(2),
            o_proj_weight.stride(0), o_proj_weight.stride(1),
            Output.stride(0), Output.stride(1), Output.stride(2),
            BLOCK_K=128, num_warps=4
        )

        return Output


def run(*args):
    return ModelNew()(*args)
