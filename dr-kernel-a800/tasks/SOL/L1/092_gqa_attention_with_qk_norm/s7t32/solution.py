import torch
import triton
import triton.language as tl

# Constants from the original configuration
NUM_ATTENTION_HEADS = 96
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
GROUPS = 12  # NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS
SCALING = 1.0 / (HEAD_DIM ** 0.5)
RMS_NORM_EPS = 1e-5


# Triton kernel: dense linear for input [B, S, K] -> out[b, s, h], where h is a linear head index.
# Grid: (B, H). Each program handles (b, h) and loops over s and K tiles.
@triton.jit
def triton_linear_bsh(input_ptr, weight_ptr, bias_ptr, out_ptr,
                      B, S, K,
                      input_stride0, input_stride1, input_stride2,
                      weight_stride0, weight_stride1,
                      bias_stride0,
                      out_stride0, out_stride1, out_stride2,
                      BLOCK_K: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    # Loop over sequence positions
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
            acc += tl.sum(x_vec * w_vec, axis=0)
        # add bias
        bias_val = tl.load(bias_ptr + h * bias_stride0)
        acc += bias_val
        # store result
        tl.store(out_ptr + b * out_stride0 + s * out_stride1 + h * out_stride2, acc)


# Triton kernel: RMSNorm per (b, h) over S. Output writes normalized vector for each s.
# Grid: (B, H). Each program handles (b, h) and loops over S to compute mean and then normalize and apply weight[h].
@triton.jit
def triton_rmsnorm(x_ptr, weight_ptr, out_ptr,
                   B, S,
                   x_stride0, x_stride1, x_stride2,
                   out_stride0, out_stride1, out_stride2,
                   weight_stride0,
                   BLOCK_S: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    # Compute mean of x^2 across S
    sum_sq = tl.zeros((), dtype=tl.float32)
    for s0 in range(0, S, BLOCK_S):
        ss = s0 + tl.arange(0, BLOCK_S)
        mask = ss < S
        x_vec = tl.load(x_ptr + b * x_stride0 + ss * x_stride1 + h * x_stride2, mask=mask, other=0.0)
        sum_sq += tl.sum(x_vec * x_vec, axis=0)
    mean = sum_sq / S
    inv_rms = 1.0 / tl.sqrt(mean + RMS_NORM_EPS)
    # scale factor
    scale = tl.load(weight_ptr + h * weight_stride0) * inv_rms
    # normalize and write
    for s0 in range(0, S, BLOCK_S):
        ss = s0 + tl.arange(0, BLOCK_S)
        mask = ss < S
        x_vec = tl.load(x_ptr + b * x_stride0 + ss * x_stride1 + h * x_stride2, mask=mask, other=0.0)
        y = x_vec * scale
        tl.store(out_ptr + b * out_stride0 + ss * out_stride1 + h * out_stride2, y, mask=mask)


# Triton kernel: Apply Rotary Position Embedding to a [B, H, S] tensor (split 128 into 64+64).
# Grid: (B, H, S). Each program handles (b, h, s) and rotates the row.
@triton.jit
def triton_rope_row(q_ptr, cos_ptr, sin_ptr, out_ptr,
                    B, H, S, head_dim,
                    q_stride0, q_stride1, q_stride2,
                    cos_stride0, cos_stride1,
                    sin_stride0, sin_stride1,
                    out_stride0, out_stride1, out_stride2,
                    BLOCK_D: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)
    base_q = b * q_stride0 + s * q_stride1 + h * q_stride2
    base_out = b * out_stride0 + s * out_stride1 + h * out_stride2
    for d0 in range(0, head_dim, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        mask = d < head_dim
        q = tl.load(q_ptr + base_q + d, mask=mask, other=0.0)
        cos_vec = tl.load(cos_ptr + d, mask=mask, other=0.0)
        sin_vec = tl.load(sin_ptr + d, mask=mask, other=0.0)
        q1 = q[:64]
        q2 = q[64:]
        rotated = -q2 + q1  # rotate last half: [-q2, q1]
        q_out = q * cos_vec + rotated * sin_vec
        tl.store(out_ptr + base_out + d, q_out, mask=mask)


# Triton kernel: Expand KV from num_key_value_heads (KVH) to num_attention_heads (H) using groups mapping.
# Grid: (B, KVH, GROUPS, S). Each program handles one original head kh, group g, and position j, then writes to target head h_target = kh * GROUPS + g.
@triton.jit
def triton_expand_kv(K_ptr, V_ptr, K_out_ptr, V_out_ptr,
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
    # K: load K[b, kh, j, :]
    k_val = tl.load(K_ptr + b * K_stride0 + j * K_stride1 + kh * K_stride2 + 0 * K_stride3)
    # V: load V[b, kh, j, :]
    v_val = tl.load(V_ptr + b * V_stride0 + j * V_stride1 + kh * V_stride2 + 0 * V_stride3)
    # Store to expanded tensors at target head
    tl.store(K_out_ptr + b * Kout_stride0 + j * Kout_stride1 + h_target * Kout_stride2 + 0 * Kout_stride3, k_val)
    tl.store(V_out_ptr + b * Vout_stride0 + j * Vout_stride1 + h_target * Vout_stride2 + 0 * Vout_stride3, v_val)


# Triton kernel: Attention compute for each (b, h): scores = Q[b,h] @ K[b,h]^T, apply scaling, causal mask, softmax along j, and accumulate attn_output = scores @ V[b,h].
# Grid: (B, H). Each program handles (b, h) and iterates over i and j in tiles. Softmax is computed row-wise for each i over j.
@triton.jit
def triton_attention_compute(Q_ptr, K_ptr, V_ptr, Out_ptr,
                             B, S, H,
                             Q_stride0, Q_stride1, Q_stride2, Q_stride3,
                             K_stride0, K_stride1, K_stride2, K_stride3,
                             V_stride0, V_stride1, V_stride2, V_stride3,
                             Out_stride0, Out_stride1, Out_stride2,
                             BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    # Prepare output vector for this (b, h)
    # Out[b, :, h] is length S
    # We will compute out[b, i, h] for i in 0..S-1
    # Initialize accumulator
    # Since we don't have direct vector store for all i, we will compute each i's output in a loop and store it.
    # We'll use masks to handle partial tiles.
    for i0 in range(0, S, BLOCK_I):
        ii = i0 + tl.arange(0, BLOCK_I)
        mask_i = ii < S
        out_vec = tl.zeros((BLOCK_I,), dtype=tl.float32)
        # Compute scores matrix for this tile of i
        scores = tl.zeros((BLOCK_I, BLOCK_J), dtype=tl.float32)
        # First compute q_vec for each i in the tile
        q_vec = tl.zeros((BLOCK_I,), dtype=tl.float32)
        for i_idx in range(0, BLOCK_I):
            if mask_i[i_idx]:
                # Load Q[b, h, i_idx, :]
                # We need to form a vector for this row. In Triton, we load per element using pointer arithmetic.
                q_row = tl.load(Q_ptr + b * Q_stride0 + i_idx * Q_stride1 + h * Q_stride2 + 0 * Q_stride3)
                q_vec[i_idx] = tl.sum(q_row, axis=0)  # if q_row is 128-dim, sum to scalar
            else:
                q_vec[i_idx] = 0.0
        # For each j in tiles, load k_vec, compute scores = q_vec * k_vec * SCALING, and store in scores
        for j0 in range(0, S, BLOCK_J):
            jj = j0 + tl.arange(0, BLOCK_J)
            mask_j = jj < S
            # Load k_vec for this head h over jj
            k_vec = tl.zeros((BLOCK_J,), dtype=tl.float32)
            for j_idx in range(0, BLOCK_J):
                if mask_j[j_idx]:
                    k_row = tl.load(K_ptr + b * K_stride0 + jj[j_idx] * K_stride1 + h * K_stride2 + 0 * K_stride3)
                    k_vec[j_idx] = tl.sum(k_row, axis=0)
                else:
                    k_vec[j_idx] = 0.0
            # Compute scores row-wise
            for ii_idx in range(0, BLOCK_I):
                if mask_i[ii_idx]:
                    q_scalar = q_vec[ii_idx]
                    scores[ii_idx, :] = q_scalar * k_vec * SCALING
            # Apply causal mask: j > i -> -inf
            for ii_idx in range(0, BLOCK_I):
                if mask_i[ii_idx]:
                    for jj_idx in range(0, BLOCK_J):
                        if mask_j[jj_idx] and (jj_idx > ii_idx):
                            scores[ii_idx, jj_idx] = -1e20
            # Softmax along j axis
            for ii_idx in range(0, BLOCK_I):
                if mask_i[ii_idx]:
                    exp_row = tl.exp(scores[ii_idx, :])
                    sum_row = tl.sum(exp_row, axis=0)
                    soft_row = exp_row / sum_row
                    # Accumulate output: Out[i] += soft_row[jj] * V[b, jj, h]
                    v_vec = tl.zeros((BLOCK_J,), dtype=tl.float32)
                    for jj_idx in range(0, BLOCK_J):
                        if mask_j[jj_idx]:
                            v_row = tl.load(V_ptr + b * V_stride0 + jj[jj_idx] * V_stride1 + h * V_stride2 + 0 * V_stride3)
                            v_vec[jj_idx] = tl.sum(v_row, axis=0)
                    out_vec[ii_idx] += tl.sum(soft_row * v_vec, axis=0)
        # Store output vector for this tile
        for i_idx in range(0, BLOCK_I):
            if mask_i[i_idx]:
                tl.store(Out_ptr + b * Out_stride0 + (i0 + i_idx) * Out_stride1 + h * Out_stride2, out_vec[i_idx])


# Triton kernel: Output projection: linear over last dimension (no bias).
# Input: attn_out [B, S, H], weight [H, K]; Output: out [B, S, H]
# Grid: (B, H). Each program handles (b, h) and loops over S and K tiles.
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
            attn_row = tl.load(attn_ptr + b * attn_stride0 + s * attn_stride1 + kk * attn_stride2, mask=mask, other=0.0)
            w_vec = tl.load(weight_ptr + h * w_stride0 + kk * w_stride1, mask=mask, other=0.0)
            acc += tl.sum(attn_row * w_vec, axis=0)
        tl.store(out_ptr + b * out_stride0 + s * out_stride1 + h * out_stride2, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # We keep constants; weights/bias are expected to be passed at call-time to ensure generality.
        # Triton will read from provided tensors.

    def forward(self,
                hidden_states: torch.Tensor,
                q_proj_weight: torch.Tensor,  # [H, K]
                q_proj_bias: torch.Tensor,    # [H]
                k_proj_weight: torch.Tensor,  # [KVH, K]
                k_proj_bias: torch.Tensor,    # [KVH]
                v_proj_weight: torch.Tensor,  # [KVH, K]
                v_proj_bias: torch.Tensor,    # [KVH]
                o_proj_weight: torch.Tensor,  # [H, K]  (K can be H or head_dim; here H=96*128=12288)
                q_norm_weight: torch.Tensor,  # [H]
                k_norm_weight: torch.Tensor,  # [KVH]
                cos: torch.Tensor,            # [head_dim] cos values
                sin: torch.Tensor,            # [head_dim] sin values
                rms_norm_eps: float = RMS_NORM_EPS):
        """
        hidden_states: [B, S, K], typically K=HEAD_DIM (128) for these configs.
        All weights/bias: must be on the same device as hidden_states and dtype float32 for numerical stability.
        """
        assert hidden_states.is_cuda, "Input must be on CUDA for Triton kernels."
        B, S, K = hidden_states.shape
        device = hidden_states.device

        # 1) Q, K, V dense linear via Triton (grid over B, H)
        H = NUM_ATTENTION_HEADS
        KVH = NUM_KEY_VALUE_HEADS

        # Prepare output tensors for linear
        Q = torch.empty((B, S, H), device=device, dtype=torch.float32)
        Kt = torch.empty((B, S, KVH), device=device, dtype=torch.float32)
        Vt = torch.empty((B, S, KVH), device=device, dtype=torch.float32)

        # Launch triton_linear_bsh for Q
        grid_q = (B, H)
        triton_linear_bsh[grid_q](
            hidden_states, q_proj_weight, q_proj_bias, Q,
            B, S, K,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            q_proj_weight.stride(0), q_proj_weight.stride(1),
            q_proj_bias.stride(0),
            Q.stride(0), Q.stride(1), Q.stride(2),
            BLOCK_K=64,
        )

        # Launch triton_linear_bsh for K
        triton_linear_bsh[grid_q](
            hidden_states, k_proj_weight, k_proj_bias, Kt,
            B, S, K,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            k_proj_weight.stride(0), k_proj_weight.stride(1),
            k_proj_bias.stride(0),
            Kt.stride(0), Kt.stride(1), Kt.stride(2),
            BLOCK_K=64,
        )

        # Launch triton_linear_bsh for V
        triton_linear_bsh[grid_q](
            hidden_states, v_proj_weight, v_proj_bias, Vt,
            B, S, K,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            v_proj_weight.stride(0), v_proj_weight.stride(1),
            v_proj_bias.stride(0),
            Vt.stride(0), Vt.stride(1), Vt.stride(2),
            BLOCK_K=64,
        )

        # 2) RMSNorm for Q and K (grid over B, H)
        Q_norm = torch.empty_like(Q)
        K_norm = torch.empty_like(Kt)

        grid_rms = (B, H)
        triton_rmsnorm[grid_rms](
            Q, q_norm_weight, Q_norm,
            B, S,
            Q.stride(0), Q.stride(1), Q.stride(2),
            Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(2),
            q_norm_weight.stride(0),
            BLOCK_S=128,
        )

        grid_rms[0] = (B, KVH)
        triton_rmsnorm[grid_rms](
            Kt, k_norm_weight, K_norm,
            B, S,
            Kt.stride(0), Kt.stride(1), Kt.stride(2),
            K_norm.stride(0), K_norm.stride(1), K_norm.stride(2),
            k_norm_weight.stride(0),
            BLOCK_S=128,
        )

        # 3) RoPE for Q and K (grid over B, H, S)
        Q_rot = torch.empty_like(Q_norm)
        K_rot = torch.empty_like(K_norm)

        grid_rope = (B, H, S)
        triton_rope_row[grid_rope](
            Q_norm, cos, sin, Q_rot,
            B, H, S, HEAD_DIM,
            Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(2),
            cos.stride(0), cos.stride(0),  # cos has 1D, stride1=1
            sin.stride(0), sin.stride(0),
            Q_rot.stride(0), Q_rot.stride(1), Q_rot.stride(2),
            BLOCK_D=128,
        )

        triton_rope_row[grid_rope](
            K_norm, cos, sin, K_rot,
            B, KVH, S, HEAD_DIM,  # passing KVH for H here is fine; cos/sin are [head_dim] scalars
            K_norm.stride(0), K_norm.stride(1), K_norm.stride(2),
            cos.stride(0), cos.stride(0),
            sin.stride(0), sin.stride(0),
            K_rot.stride(0), K_rot.stride(1), K_rot.stride(2),
            BLOCK_D=128,
        )

        # 4) GQA expand: K_rot, Vt -> expanded to H heads (grid over B, KVH, GROUPS, S)
        S_int = S
        K_out = torch.empty((B, S_int, H), device=device, dtype=torch.float32)
        V_out = torch.empty((B, S_int, H), device=device, dtype=torch.float32)

        grid_expand = (B, KVH, GROUPS, S_int)
        triton_expand_kv[grid_expand](
            K_rot, Vt, K_out, V_out,
            B, S_int, KVH, HEAD_DIM,
            K_rot.stride(0), K_rot.stride(1), K_rot.stride(2), K_rot.stride(3),
            Vt.stride(0), Vt.stride(1), Vt.stride(2), Vt.stride(3),
            K_out.stride(0), K_out.stride(1), K_out.stride(2), K_out.stride(3),
            V_out.stride(0), V_out.stride(1), V_out.stride(2), V_out.stride(3),
            GROUPS=GROUPS,
        )

        # 5) Attention compute (grid over B, H)
        attn_out = torch.empty((B, S_int, H), device=device, dtype=torch.float32)
        grid_attn = (B, H)
        triton_attention_compute[grid_attn](
            Q_rot, K_out, V_out, attn_out,
            B, S_int, H,
            Q_rot.stride(0), Q_rot.stride(1), Q_rot.stride(2), Q_rot.stride(3),
            K_out.stride(0), K_out.stride(1), K_out.stride(2), K_out.stride(3),
            V_out.stride(0), V_out.stride(1), V_out.stride(2), V_out.stride(3),
            attn_out.stride(0), attn_out.stride(1), attn_out.stride(2),
            BLOCK_I=64, BLOCK_J=64,
        )

        # 6) Output projection (grid over B, H)
        output = torch.empty((B, S_int, H), device=device, dtype=torch.float32)
        grid_proj = (B, H)
        # o_proj_weight shape is [H, K]. Here K can be H or head_dim; we let it be inferred by torch if provided.
        triton_o_proj[grid_proj](
            attn_out, o_proj_weight, output,
            B, S_int, H, o_proj_weight.shape[1],
            attn_out.stride(0), attn_out.stride(1), attn_out.stride(2),
            o_proj_weight.stride(0), o_proj_weight.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            BLOCK_K=64,
        )

        return output


def run(*args):
    return ModelNew()(*args)
