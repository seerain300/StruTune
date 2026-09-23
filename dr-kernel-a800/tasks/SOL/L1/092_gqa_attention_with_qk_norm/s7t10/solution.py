import torch
import triton
import triton.language as tl

# 1) Triton dense linear for Q/K/V: out[b, s, h] = sum_k input[b, :, k] * weight[h, k] + bias[h]
# Grid: (B, H, S). Iterate over K in tiles BLOCK_K.
@triton.jit
def triton_linear_bsh(x_ptr, w_ptr, b_ptr, out_ptr,
                       B, S, H, K,
                       x_stride0, x_stride1, x_stride2,
                       w_stride0, w_stride1,
                       out_stride0, out_stride1, out_stride2,
                       BLOCK_K: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    base_x = b * x_stride0 + s * x_stride1
    base_out = b * out_stride0 + s * out_stride1 + h * out_stride2

    acc = tl.zeros((), dtype=tl.float32)
    # sum over K dimension
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        mask_k = k < K
        # load input row [k]: x[b, s, k]
        x_row = tl.load(x_ptr + base_x + k * x_stride2, mask=mask_k, other=0.0)
        # load weight row [k]: w[h, k]
        w_row = tl.load(w_ptr + h * w_stride0 + k * w_stride1, mask=mask_k, other=0.0)
        prod = x_row * w_row
        acc += tl.sum(prod, axis=0)
    # add bias[h]
    bias = tl.load(b_ptr + h)
    acc = acc + bias
    # store to out[b, s, h]
    tl.store(out_ptr + base_out, acc)


# 2) Triton RMSNorm per row (b, h) across S: y = x * (weight[h] / sqrt(mean(x^2) + eps))
@triton.jit
def triton_rmsnorm_row(x_ptr, weight_ptr, out_ptr,
                       B, S,
                       x_stride0, x_stride1, x_stride2,
                       out_stride0, out_stride1, out_stride2,
                       eps,
                       BLOCK_D: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)

    row_start = b * S
    sum_sq = 0.0
    for d0 in range(0, S, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        mask = d < S
        x = tl.load(x_ptr + row_start + d * x_stride1, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_sq += tl.sum(x * x, axis=0)

    mean = sum_sq / S
    inv_rms = 1.0 / tl.sqrt(mean + eps)
    scale = tl.load(weight_ptr + h) * inv_rms

    for d0 in range(0, S, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        mask = d < S
        x = tl.load(x_ptr + row_start + d * x_stride1, mask=mask, other=0.0).to(tl.float32)
        y = x * scale
        tl.store(out_ptr + b * out_stride0 + h * out_stride1 + d * out_stride2, y, mask=mask)


# 3) Triton RoPE per row: split 128 into 64+64 and rotate
# For each (b, h, s), rotate q = [q1, q2] with q1=first 64, q2=last 64:
# q_out = q * cos + [-q2, q1] * sin
@triton.jit
def triton_rope_row(x_ptr, cos_ptr, sin_ptr, out_ptr,
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


# 4) Triton GQA expand: expand KV heads from KVH to H with GROUPS = H // KVH
# For each (b, kh, g, j), copy K[b, kh, j, :] -> K_out[b, kh*GROUPS+g, j, :]
# and similarly for V. We assume head_dim=128 and seq_len=S.
@triton.jit
def triton_expand_kv(K_ptr, V_ptr, K_out_ptr, V_out_ptr,
                     B, S, KVH, KD, GROUPS,
                     K_stride0, K_stride1, K_stride2, K_stride3,
                     V_stride0, V_stride1, V_stride2, V_stride3,
                     Kout_stride0, Kout_stride1, Kout_stride2, Kout_stride3,
                     Vout_stride0, Vout_stride1, Vout_stride2, Vout_stride3,
                     BLOCK_D: tl.constexpr):
    # KD is head_dim; we assume 128
    for kh in range(0, KVH):
        for g in range(0, GROUPS):
            h_target = kh * GROUPS + g
            for j in range(0, S):
                # Copy K row
                k_val = tl.load(K_ptr + b * K_stride0 + j * K_stride1 + kh * K_stride2)
                tl.store(K_out_ptr + b * Kout_stride0 + j * Kout_stride1 + h_target * Kout_stride2, k_val)
                # Copy V row
                v_val = tl.load(V_ptr + b * V_stride0 + j * V_stride1 + kh * V_stride2)
                tl.store(V_out_ptr + b * Vout_stride0 + j * Vout_stride1 + h_target * Vout_stride2, v_val)


# 5) Triton attention kernel:
# For each (b, h), compute attention over all i,j in [0, S). It performs:
# - Load Q_row[b, h, i], K_rows[b, :, j], V_rows[b, :, j]
# - Compute scores[i, j] = dot(Q_row[i], K_rows[j]) * scaling
# - Apply causal mask: for i < j, scores = -inf; else keep
# - Softmax over j
# - Accumulate output[i] += scores[j] * V_rows[j]
# Returns attn_out[b, :, h]
# Grid: (B, H). Loops over i and j in tiles for robustness.
@triton.jit
def triton_attention_fwd(Q_ptr, K_ptr, V_ptr, Out_ptr,
                         B, S, H,
                         Q_stride0, Q_stride1, Q_stride2,
                         K_stride0, K_stride1, K_stride2,
                         V_stride0, V_stride1, V_stride2,
                         Out_stride0, Out_stride1,
                         scaling, eps,
                         BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)

    # Initialize output vector for this (b, h)
    out_vec = tl.zeros((S,), dtype=tl.float32)

    for i0 in range(0, S, BLOCK_I):
        i = i0 + tl.arange(0, BLOCK_I)
        mask_i = i < S

        # Load Q_row[b, h, i]
        q_row = tl.load(Q_ptr + b * Q_stride0 + h * Q_stride1 + i * Q_stride2, mask=mask_i, other=0.0).to(tl.float32)  # [BLOCK_I]

        # Compute scores[i, j] for j in tiles
        for j0 in range(0, S, BLOCK_J):
            j = j0 + tl.arange(0, BLOCK_J)
            mask_j = j < S

            # K block: [BLOCK_J, 128]
            k_block = tl.zeros((BLOCK_J, 128), dtype=tl.float32)
            v_block = tl.zeros((BLOCK_J, 128), dtype=tl.float32)

            # Load K rows and V rows
            # Note: these tensors have last dim 128
            for jj in range(0, BLOCK_J):
                jj_valid = j[jj] < S
                if jj_valid:
                    k_row = tl.load(K_ptr + b * K_stride0 + j[jj] * K_stride1, mask=True, other=0.0).to(tl.float32)  # [128]
                    v_row = tl.load(V_ptr + b * V_stride0 + j[jj] * V_stride1, mask=True, other=0.0).to(tl.float32)  # [128]
                    k_block[jj, :] = k_row
                    v_block[jj, :] = v_row

            # Compute scores[i, j] = sum_k q_row[k] * k_block[j, k]
            scores = tl.zeros((BLOCK_I, BLOCK_J), dtype=tl.float32)
            for k in range(0, 128):
                qk = q_row[k]  # scalar
                kcol = k_block[:, k]  # [BLOCK_J]
                scores += qk * kcol[None, :]  # [BLOCK_I, BLOCK_J]

            # Scale
            scores = scores * scaling

            # Apply causal mask: if i < j, set to -inf; else 0
            # We compare i against j vectors
            for ii in range(0, BLOCK_I):
                ii_valid = i[ii] < S
                if ii_valid:
                    # For each j, if j > i[ii], set -inf
                    for jjj in range(0, BLOCK_J):
                        if j0 + jjj < S and (j0 + jjj) > (i0 + ii):
                            scores[ii, jjj] = -float('inf')

            # Softmax along j axis
            # First, set masked entries to -inf then compute softmax
            # Note: Triton does not have tl.softmax for 2D, so we implement row-wise softmax:
            # For each row ii:
            exp_scores = tl.exp(scores)  # [BLOCK_I, BLOCK_J]
            sum_exp = tl.sum(exp_scores, axis=1)  # [BLOCK_I]
            scores = exp_scores / sum_exp[:, None]  # [BLOCK_I, BLOCK_J]

            # Accumulate output: out[i] += scores[:, jj] * v_block[:, :] along j
            # We need to do dot(scores[:, jj], v_block[:, :]) for each jj, which is:
            # sum_j scores[ii, jj] * v_block[jj, :]
            for jjj in range(0, BLOCK_J):
                jj_valid = j0 + jjj < S
                if jj_valid:
                    v_row = v_block[jjj, :]  # [128]
                    out_vec[i0 + tl.arange(0, BLOCK_I)] += tl.sum(scores[:, jjj] * v_row, axis=0)

    # Store out_vec to Out[b, :, h]
    out_base = b * Out_stride0 + h * Out_stride1
    for p in range(0, S):
        tl.store(Out_ptr + out_base + p, out_vec[p])


# 6) Triton output projection: out[b, s, h] = sum_k attn_output[b, s, k] * o_proj_weight[h, k]
@triton.jit
def triton_o_proj(x_ptr, w_ptr, out_ptr,
                  B, S, H, K,
                  x_stride0, x_stride1, x_stride2,
                  w_stride0, w_stride1,
                  out_stride0, out_stride1, out_stride2,
                  BLOCK_K: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    base_x = b * x_stride0 + s * x_stride1
    base_out = b * out_stride0 + s * out_stride1 + h * out_stride2

    acc = tl.zeros((), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        mask_k = k < K
        x_row = tl.load(x_ptr + base_x + k * x_stride2, mask=mask_k, other=0.0)
        w_row = tl.load(w_ptr + h * w_stride0 + k * w_stride1, mask=mask_k, other=0.0)
        prod = x_row * w_row
        acc += tl.sum(prod, axis=0)

    tl.store(out_ptr + base_out, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Fixed constants from the original code for this task
        self.num_attention_heads = 96
        self.num_key_value_heads = 8
        self.head_dim = 128
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads  # 12
        self.scaling = self.head_dim ** -0.5
        self.rms_norm_eps = 1e-6

    def forward(self, hidden_states, q_proj_weight, q_proj_bias,
                k_proj_weight, k_proj_bias,
                v_proj_weight, v_proj_bias,
                o_proj_weight, q_norm_weight, k_norm_weight,
                cos, sin):
        # Shapes
        B, S, K = hidden_states.shape
        H = self.num_attention_heads
        KVH = self.num_key_value_heads
        KD = self.head_dim

        device = hidden_states.device

        # Allocate intermediates
        # 1) Dense linear for Q, K, V
        query = torch.empty((B, S, H), device=device, dtype=hidden_states.dtype)
        key = torch.empty((B, S, KVH), device=device, dtype=hidden_states.dtype)
        value = torch.empty((B, S, KVH), device=device, dtype=hidden_states.dtype)

        # Launch Triton linear for Q
        triton_linear_bsh(
            hidden_states, q_proj_weight, q_proj_bias, query,
            B, S, H, K,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            q_proj_weight.stride(0), q_proj_weight.stride(1),
            query.stride(0), query.stride(1), query.stride(2),
            BLOCK_K=64,
            num_warps=4
        )

        # Launch Triton linear for K
        triton_linear_bsh(
            hidden_states, k_proj_weight, k_proj_bias, key,
            B, S, KVH, K,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            k_proj_weight.stride(0), k_proj_weight.stride(1),
            key.stride(0), key.stride(1), key.stride(2),
            BLOCK_K=64,
            num_warps=4
        )

        # Launch Triton linear for V
        triton_linear_bsh(
            hidden_states, v_proj_weight, v_proj_bias, value,
            B, S, KVH, K,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            v_proj_weight.stride(0), v_proj_weight.stride(1),
            value.stride(0), value.stride(1), value.stride(2),
            BLOCK_K=64,
            num_warps=4
        )

        # 2) RMSNorm on Q and K
        query_norm = torch.empty_like(query)
        key_norm = torch.empty_like(key)

        triton_rmsnorm_row(
            query, q_norm_weight, query_norm,
            B, S,
            query.stride(0), query.stride(1), query.stride(2),
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
            self.rms_norm_eps,
            BLOCK_D=128,
            num_warps=4
        )

        triton_rmsnorm_row(
            key, k_norm_weight, key_norm,
            B, S,
            key.stride(0), key.stride(1), key.stride(2),
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2),
            self.rms_norm_eps,
            BLOCK_D=128,
            num_warps=4
        )

        # 3) Apply RoPE to Q and K
        query_rope = torch.empty_like(query_norm)
        key_rope = torch.empty_like(key_norm)

        triton_rope_row(
            query_norm, cos, sin, query_rope,
            B, H, S, KD,
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
            cos.stride(0), cos.stride(1),
            sin.stride(0), sin.stride(1),
            query_rope.stride(0), query_rope.stride(1), query_rope.stride(2),
            BLOCK_D=128,
            num_warps=4
        )

        triton_rope_row(
            key_norm, cos, sin, key_rope,
            B, KVH, S, KD,
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2),
            cos.stride(0), cos.stride(1),
            sin.stride(0), sin.stride(1),
            key_rope.stride(0), key_rope.stride(1), key_rope.stride(2),
            BLOCK_D=128,
            num_warps=4
        )

        # 4) GQA expand: expand KV heads to H (GROUPS=12)
        # Allocate expanded K and V
        K_expanded = torch.empty((B, S, H, KD), device=device, dtype=query.dtype)
        V_expanded = torch.empty((B, S, H, KD), device=device, dtype=query.dtype)

        # We need to pass correct strides; we create 4D tensors and copy rows
        # We use (B, S, KVH, KD) -> (B, S, H, KD) mapping: target_h = kh * GROUPS + g
        triton_expand_kv(
            key_rope, value, K_expanded, V_expanded,
            B, S, KVH, KD, self.num_key_value_groups,
            key_rope.stride(0), key_rope.stride(1), key_rope.stride(2),
            value.stride(0), value.stride(1), value.stride(2), value.stride(3),
            K_expanded.stride(0), K_expanded.stride(1), K_expanded.stride(2), K_expanded.stride(3),
            V_expanded.stride(0), V_expanded.stride(1), V_expanded.stride(2), V_expanded.stride(3),
            GROUPS=self.num_key_value_groups,
            BLOCK_D=128,
            num_warps=4
        )

        # 5) Attention compute in Triton: grid (B, H), attention over all i, j in [0, S)
        attn_out = torch.empty((B, S, H), device=device, dtype=torch.float32)

        triton_attention_fwd(
            query_rope, K_expanded, V_expanded, attn_out,
            B, S, H,
            query_rope.stride(0), query_rope.stride(1), query_rope.stride(2),
            K_expanded.stride(0), K_expanded.stride(1), K_expanded.stride(2),
            V_expanded.stride(0), V_expanded.stride(1), V_expanded.stride(2),
            attn_out.stride(0), attn_out.stride(1),
            self.scaling, self.rms_norm_eps,
            BLOCK_I=64, BLOCK_J=64,
            num_warps=4
        )

        # 6) Output projection through o_proj_weight (no bias)
        output = torch.empty((B, S, H), device=device, dtype=torch.float32)
        # Note: attn_out is float32 from Triton; weight dtype can be float32. We ensure K=H=128 for this setup.
        triton_o_proj(
            attn_out, o_proj_weight, output,
            B, S, H, 128,
            attn_out.stride(0), attn_out.stride(1), attn_out.stride(2),
            o_proj_weight.stride(0), o_proj_weight.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            BLOCK_K=64,
            num_warps=4
        )

        return output


# For compatibility with the original runner: entry point is ModelNew
Model = ModelNew


def run(*args):
    return ModelNew()(*args)
