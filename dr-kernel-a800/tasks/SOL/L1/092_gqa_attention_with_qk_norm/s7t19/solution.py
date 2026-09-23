import torch
import triton
import triton.language as tl


# 1) Dense linear: out[b, s, h] = sum_k input[b, :, k] * weight[h, k] + bias[h]
@triton.jit
def triton_linear_bsh(x_ptr, weight_ptr, bias_ptr, out_ptr,
                       B, S, H, K,
                       x_stride0, x_stride1,  # input: [B, S, K]
                       weight_stride0, weight_stride1,  # weight: [H, K]
                       bias_stride0,  # bias: [H]
                       out_stride0, out_stride1, out_stride2,  # out: [B, H, S]
                       BLOCK_K: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    acc = 0.0
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        mask_k = k < K
        base_x = b * x_stride0 + s * x_stride1
        x_row = tl.load(x_ptr + base_x + k, mask=mask_k, other=0.0)
        w_row = tl.load(weight_ptr + h * weight_stride0 + k * weight_stride1, mask=mask_k, other=0.0)
        acc += tl.sum(x_row * w_row, axis=0)
    b_val = tl.load(bias_ptr + h * bias_stride0)
    acc += b_val
    tl.store(out_ptr + b * out_stride0 + h * out_stride1 + s * out_stride2, acc)


# 2) RMSNorm per (b, h) across S: y[b, h, s] = x[b, h, s] * (w[h] / sqrt(mean(x^2) + eps))
@triton.jit
def triton_rmsnorm_bhs(x_ptr, weight_ptr, out_ptr,
                       B, S, H,
                       x_stride0, x_stride1, x_stride2,
                       out_stride0, out_stride1, out_stride2,
                       eps: tl.constexpr,
                       BLOCK_S: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    row_start = b * S
    sum_sq = 0.0
    for s0 in range(0, S, BLOCK_S):
        s = s0 + tl.arange(0, BLOCK_S)
        mask = s < S
        x = tl.load(x_ptr + row_start + s * x_stride1, mask=mask, other=0.0)
        sum_sq += tl.sum(x * x, axis=0)
    mean = sum_sq / S
    inv_rms = 1.0 / tl.sqrt(mean + eps)
    scale = tl.load(weight_ptr + h) * inv_rms
    for s0 in range(0, S, BLOCK_S):
        s = s0 + tl.arange(0, BLOCK_S)
        mask = s < S
        x = tl.load(x_ptr + row_start + s * x_stride1, mask=mask, other=0.0)
        y = x * scale
        tl.store(out_ptr + b * out_stride0 + h * out_stride1 + s * out_stride2, y, mask=mask)


# 3) Triton RoPE: rotate Q and K rows (split 128 into 64+64)
@triton.jit
def triton_rope_bh(x_ptr, cos_ptr, sin_ptr, out_ptr,
                   B, H, S, head_dim,
                   x_stride0, x_stride1, x_stride2,
                   cos_stride0, cos_stride1,
                   sin_stride0, sin_stride1,
                   out_stride0, out_stride1, out_stride2,
                   BLOCK_D: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    base_x = b * x_stride0 + s * x_stride1 + h * x_stride2
    base_out = b * out_stride0 + s * out_stride1 + h * out_stride2

    for d0 in range(0, head_dim, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        mask = d < head_dim
        q = tl.load(x_ptr + base_x + d, mask=mask, other=0.0)
        cos_vec = tl.load(cos_ptr + d, mask=mask, other=0.0)
        sin_vec = tl.load(sin_ptr + d, mask=mask, other=0.0)
        q1 = q[:64]
        q2 = q[64:]
        rotated = -q2 + q1
        q_out = q * cos_vec + rotated * sin_vec
        tl.store(out_ptr + base_out + d, q_out, mask=mask)


# 4) Triton GQA expand: expand KV heads from KVH to H via GROUPS = H // KVH
# Copy K/V rows (for each j) from kh into target_h = kh * GROUPS + g
@triton.jit
def triton_expand_kv_bkgs(K_ptr, V_ptr, K_out_ptr, V_out_ptr,
                          B, S, KVH, KD,
                          K_stride0, K_stride1, K_stride2, K_stride3,
                          V_stride0, V_stride1, V_stride2, V_stride3,
                          Kout_stride0, Kout_stride1, Kout_stride2, Kout_stride3,
                          Vout_stride0, Vout_stride1, Vout_stride2, Vout_stride3,
                          GROUPS: tl.constexpr):
    for kh in range(0, KVH):
        for g in range(0, GROUPS):
            h_target = kh * GROUPS + g
            for j in range(0, S):
                # K: load K[b, kh, j, :]
                k_row = tl.load(K_ptr + b * K_stride0 + j * K_stride1 + kh * K_stride2,  # KD is the last dim, omitted since it's 1
                                mask=True, other=0.0)
                tl.store(K_out_ptr + b * Kout_stride0 + j * Kout_stride1 + h_target * Kout_stride2, k_row)
                # V: load V[b, kh, j, :]
                v_row = tl.load(V_ptr + b * V_stride0 + j * V_stride1 + kh * V_stride2, mask=True, other=0.0)
                tl.store(V_out_ptr + b * Vout_stride0 + j * Vout_stride1 + h_target * Vout_stride2, v_row)


# 5) Attention compute per (b, h): for each i in S, compute scores[i, j] = Q[i]·K[j]^T * scaling,
#    apply causal mask, softmax along j, accumulate attn_output[i] = sum_j scores[i, j] * V[j].
@triton.jit
def attention_compute_bh(Q_ptr, K_ptr, V_ptr, Out_ptr,
                         B, S, H, head_dim,
                         Q_stride0, Q_stride1, Q_stride2, Q_stride3,
                         K_stride0, K_stride1, K_stride2, K_stride3,
                         V_stride0, V_stride1, V_stride2, V_stride3,
                         Out_stride0, Out_stride1, Out_stride2,
                         scaling: tl.constexpr,
                         BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)

    # accum output per i
    out_row = tl.zeros([S], dtype=tl.float32)

    # tiling over i
    for i0 in range(0, S, BLOCK_I):
        i = i0 + tl.arange(0, BLOCK_I)
        mask_i = i < S

        # scores[i, :] for all j
        scores = tl.zeros([BLOCK_I], dtype=tl.float32)

        for j0 in range(0, S, BLOCK_J):
            j = j0 + tl.arange(0, BLOCK_J)
            mask_j = j < S

            # load Q[i] and K[j]
            q_vec = tl.load(Q_ptr + b * Q_stride0 + i * Q_stride1 + h * Q_stride2, mask=mask_i, other=0.0)
            k_rows = tl.load(K_ptr + b * K_stride0 + j * K_stride1 + h * K_stride2, mask=mask_j, other=0.0)  # [BLOCK_J, head_dim]

            # compute dot: scores[i, j_block] = Q[i, :] · K[j, :]
            dot = tl.sum(q_vec[:, None] * k_rows[None, :], axis=1)  # [BLOCK_J]
            scores += dot

        scores = scores * scaling

        # causal mask: attn[s, t] = -inf if t > s else 0
        for i_idx in range(0, BLOCK_I):
            if mask_i[i_idx]:
                for j_idx in range(0, BLOCK_J):
                    if (i0 + i_idx) > (j0 + j_idx):
                        scores[i_idx] = -float('inf')

        # softmax along j for each i
        exp_scores = tl.exp(scores)
        sum_exp = tl.sum(exp_scores, axis=0)
        softmax = exp_scores / sum_exp

        # accumulate output: out_row[i] += sum_j softmax * V[j]
        for j0 in range(0, S, BLOCK_J):
            j = j0 + tl.arange(0, BLOCK_J)
            mask_j = j < S
            v_rows = tl.load(V_ptr + b * V_stride0 + j * V_stride1 + h * V_stride2, mask=mask_j, other=0.0)  # [BLOCK_J]
            # multiply and sum over j
            out_row[i0 + tl.arange(0, BLOCK_I)] += tl.sum(softmax[None, :] * v_rows[None, :], axis=1)

    # store out_row
    for i in range(0, S):
        tl.store(Out_ptr + b * Out_stride0 + h * Out_stride1 + i * Out_stride2, out_row[i])


# 6) Final output projection: out[b, s, h] = sum_k attn_output[b, s, k] * o_proj_weight[h, k]
@triton.jit
def triton_linear_out(x_ptr, weight_ptr, out_ptr,
                       B, S, H, K,
                       x_stride0, x_stride1,  # input: [B, S, K]
                       weight_stride0, weight_stride1,  # weight: [H, K]
                       out_stride0, out_stride1, out_stride2,  # out: [B, H, S]
                       BLOCK_K: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    acc = 0.0
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        mask_k = k < K
        x_row = tl.load(x_ptr + b * x_stride0 + s * x_stride1 + k, mask=mask_k, other=0.0)
        w_row = tl.load(weight_ptr + h * weight_stride0 + k * weight_stride1, mask=mask_k, other=0.0)
        acc += tl.sum(x_row * w_row, axis=0)
    tl.store(out_ptr + b * out_stride0 + h * out_stride1 + s * out_stride2, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

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
        # Shapes (fixed in evaluator configs)
        B = hidden_states.shape[0]
        S = hidden_states.shape[1]
        H = 96  # num_attention_heads
        KVH = 8  # num_key_value_heads
        KD = 128  # head_dim
        groups = H // KVH  # 12

        # 0) Allocate outputs and expanded tensors (all Triton)
        # Q, K, V projections
        Q = torch.empty((B, H, S), dtype=hidden_states.dtype, device=hidden_states.device)
        K = torch.empty((B, KVH, S), dtype=hidden_states.dtype, device=hidden_states.device)
        V = torch.empty((B, KVH, S), dtype=hidden_states.dtype, device=hidden_states.device)

        # Expand K/V to H for attention
        K_exp = torch.empty((B, H, S), dtype=hidden_states.dtype, device=hidden_states.device)
        V_exp = torch.empty((B, H, S), dtype=hidden_states.dtype, device=hidden_states.device)

        # 1) Q, K, V dense linear (GEMV)
        grid_linear = lambda META: (B, H, S)
        triton_linear_bsh[grid_linear](
            hidden_states, q_proj_weight, q_proj_bias, Q,
            B, S, H, q_proj_weight.shape[1],
            hidden_states.stride(0), hidden_states.stride(1),
            q_proj_weight.stride(0), q_proj_weight.stride(1),
            q_proj_bias.stride(0),
            Q.stride(0), Q.stride(1), Q.stride(2),
            BLOCK_K=64
        )

        triton_linear_bsh[grid_linear](
            hidden_states, k_proj_weight, k_proj_bias, K,
            B, S, KVH, k_proj_weight.shape[1],
            hidden_states.stride(0), hidden_states.stride(1),
            k_proj_weight.stride(0), k_proj_weight.stride(1),
            k_proj_bias.stride(0),
            K.stride(0), K.stride(1), K.stride(2),
            BLOCK_K=64
        )

        triton_linear_bsh[grid_linear](
            hidden_states, v_proj_weight, v_proj_bias, V,
            B, S, KVH, v_proj_weight.shape[1],
            hidden_states.stride(0), hidden_states.stride(1),
            v_proj_weight.stride(0), v_proj_weight.stride(1),
            v_proj_bias.stride(0),
            V.stride(0), V.stride(1), V.stride(2),
            BLOCK_K=64
        )

        # 2) RMSNorm for Q and K
        grid_rms = lambda META: (B, H)
        triton_rmsnorm_bhs[grid_rms](
            Q, q_norm_weight, Q,
            B, S, H,
            Q.stride(0), Q.stride(1), Q.stride(2),
            Q.stride(0), Q.stride(1), Q.stride(2),
            rms_norm_eps, BLOCK_S=128
        )

        triton_rmsnorm_bhs[grid_rms](
            K, k_norm_weight, K,
            B, S, KVH,
            K.stride(0), K.stride(1), K.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            rms_norm_eps, BLOCK_S=128
        )

        # 3) Apply RoPE to Q and K
        grid_rope = lambda META: (B, H, S)
        triton_rope_bh[grid_rope](
            Q, cos, sin, Q,
            B, H, S, 128,
            Q.stride(0), Q.stride(1), Q.stride(2),
            cos.stride(0), cos.stride(1),
            sin.stride(0), sin.stride(1),
            Q.stride(0), Q.stride(1), Q.stride(2),
            BLOCK_D=64
        )

        triton_rope_bh[grid_rope](
            K, cos, sin, K,
            B, KVH, S, 128,
            K.stride(0), K.stride(1), K.stride(2),
            cos.stride(0), cos.stride(1),
            sin.stride(0), sin.stride(1),
            K.stride(0), K.stride(1), K.stride(2),
            BLOCK_D=64
        )

        # 4) GQA expansion: K/V from 8 heads to 96 heads
        grid_expand = lambda META: (B, KVH, groups, S)
        triton_expand_kv_bkgs[grid_expand](
            K, V, K_exp, V_exp,
            B, S, KVH, KD,
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            K_exp.stride(0), K_exp.stride(1), K_exp.stride(2), K_exp.stride(3),
            V_exp.stride(0), V_exp.stride(1), V_exp.stride(2), V_exp.stride(3),
            GROUPS=groups
        )

        # 5) Attention compute: Out[b, s, h] = softmax(Q*R^T) @ V
        #    We compute per (b, h), tiling over i and j.
        grid_attn = lambda META: (B, H)
        Out = torch.empty((B, H, S), dtype=hidden_states.dtype, device=hidden_states.device)
        attention_compute_bh[grid_attn](
            Q, K_exp, V_exp, Out,
            B, S, H, 128,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K_exp.stride(0), K_exp.stride(1), K_exp.stride(2), K_exp.stride(3),
            V_exp.stride(0), V_exp.stride(1), V_exp.stride(2), V_exp.stride(3),
            Out.stride(0), Out.stride(1), Out.stride(2),
            scaling=(128 ** -0.5),
            BLOCK_I=64, BLOCK_J=64
        )

        # 6) Output projection
        grid_out = lambda META: (B, H, S)
        Output = torch.empty((B, H, S), dtype=hidden_states.dtype, device=hidden_states.device)
        triton_linear_out[grid_out](
            Out, o_proj_weight, Output,
            B, S, H, o_proj_weight.shape[1],
            Out.stride(0), Out.stride(1),
            o_proj_weight.stride(0), o_proj_weight.stride(1),
            Output.stride(0), Output.stride(1), Output.stride(2),
            BLOCK_K=64
        )

        return Output


def run(*args):
    return ModelNew()(*args)
