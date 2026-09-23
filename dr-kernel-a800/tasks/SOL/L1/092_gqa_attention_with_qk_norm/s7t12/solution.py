import torch
import triton
import triton.language as tl

# Dense linear over last dim: out[b, s, h] = sum_k x[b, s, k] * w[h, k] + b[h]
# x shape: [B, S, K], w shape: [H, K], out shape: [B, S, H]
@triton.jit
def triton_linear_bsh(x_ptr, w_ptr, b_ptr, out_ptr,
                       B, S, K, H,
                       x_stride0, x_stride1, x_stride2,
                       w_stride0, w_stride1,
                       out_stride0, out_stride1, out_stride2,
                       BLOCK_K: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)
    base_x = b * x_stride0 + s * x_stride1
    base_out = b * out_stride0 + s * out_stride1 + h * out_stride2

    acc = tl.zeros([1], dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + tl.arange(0, BLOCK_K)
        mask_k = kk < K
        x_vec = tl.load(x_ptr + base_x + kk * x_stride2, mask=mask_k, other=0.0)
        w_vec = tl.load(w_ptr + h * w_stride0 + kk * w_stride1, mask=mask_k, other=0.0)
        acc += tl.sum(x_vec * w_vec, axis=0)
    acc += tl.load(b_ptr + h)
    tl.store(out_ptr + base_out, acc)


# RMSNorm per (b, h) over last dim S: y[b, s, h] = x[b, s, h] * (w[h] / sqrt(mean(x^2) + eps))
@triton.jit
def triton_rmsnorm(x_ptr, w_ptr, out_ptr,
                    B, S, H,
                    x_stride0, x_stride1, x_stride2,
                    out_stride0, out_stride1, out_stride2,
                    eps,
                    BLOCK_S: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    row_start_x = b * x_stride0 + h * x_stride2
    row_start_out = b * out_stride0 + h * out_stride2

    sum_sq = 0.0
    for s0 in range(0, S, BLOCK_S):
        ss = s0 + tl.arange(0, BLOCK_S)
        mask = ss < S
        x = tl.load(x_ptr + row_start_x + ss * x_stride1, mask=mask, other=0.0)
        sum_sq += tl.sum(x * x, axis=0)
    mean = sum_sq / S
    inv_rms = 1.0 / tl.sqrt(mean + eps)
    scale = tl.load(w_ptr + h) * inv_rms
    for s0 in range(0, S, BLOCK_S):
        ss = s0 + tl.arange(0, BLOCK_S)
        mask = ss < S
        x = tl.load(x_ptr + row_start_x + ss * x_stride1, mask=mask, other=0.0)
        y = x * scale
        tl.store(out_ptr + row_start_out + ss * out_stride1, y, mask=mask)


# Rotate each (b, h, s) row: head_dim must be 128; split into 64+64 halves
# q_out = q*cos + [-q2, q1]*sin
@triton.jit
def triton_rope_row(x_ptr, cos_ptr, sin_ptr, out_ptr,
                    B, S, H,
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

    for d0 in range(0, 128, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        mask = d < 128
        x = tl.load(x_ptr + base_x + d, mask=mask, other=0.0)
        cos_vec = tl.load(cos_ptr + d, mask=mask, other=0.0)
        sin_vec = tl.load(sin_ptr + d, mask=mask, other=0.0)
        q1 = x[:64]
        q2 = x[64:]
        rotated = -q2 + q1
        y = x * cos_vec + rotated * sin_vec
        tl.store(out_ptr + base_out + d, y, mask=mask)


# Expand KV heads from KVH to H using groups mapping: target_h = kh * GROUPS + g
# Copy K[V] rows (for each position j) from kh into target_h slots in expanded tensors.
@triton.jit
def triton_expand_kv(K_ptr, V_ptr, K_out_ptr, V_out_ptr,
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
                k_val = tl.load(K_ptr + b * K_stride0 + j * K_stride1 + kh * K_stride2 + 0 * K_stride3)
                # Store to K_out at target_h
                tl.store(K_out_ptr + b * Kout_stride0 + j * Kout_stride1 + h_target * Kout_stride2 + 0 * Kout_stride3, k_val)
                # V: load V[b, kh, j, :]
                v_val = tl.load(V_ptr + b * V_stride0 + j * V_stride1 + kh * V_stride2 + 0 * V_stride3)
                # Store to V_out at target_h
                tl.store(V_out_ptr + b * Vout_stride0 + j * Vout_stride1 + h_target * Vout_stride2 + 0 * Vout_stride3, v_val)


# Attention compute per (b, h): loop over i (query positions) and j (key positions), compute scores, softmax, and output
# Constants: head_dim=128, scaling=0.07905694 (1/sqrt(128))
@triton.jit
def triton_attention_compute(Q_ptr, K_ptr, V_ptr, Out_ptr,
                             B, S, H,
                             Q_stride0, Q_stride1, Q_stride2, Q_stride3,
                             K_stride0, K_stride1, K_stride2, K_stride3,
                             V_stride0, V_stride1, V_stride2, V_stride3,
                             Out_stride0, Out_stride1, Out_stride2,
                             scaling, BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)

    # For each query position i
    for i0 in range(0, S, BLOCK_I):
        ii = i0 + tl.arange(0, BLOCK_I)
        mask_i = ii < S

        # Accumulator for output at these i's
        out_acc = tl.zeros([BLOCK_I], dtype=tl.float32)

        # For each key position j
        for j0 in range(0, S, BLOCK_J):
            jj = j0 + tl.arange(0, BLOCK_J)
            mask_j = jj < S

            # Load Q[:, i, :] and K[:, j, :]
            q_vec = tl.load(Q_ptr + b * Q_stride0 + ii * Q_stride1 + h * Q_stride2, mask=mask_i, other=0.0)
            k_vec = tl.load(K_ptr + b * K_stride0 + jj * K_stride1 + h * K_stride2, mask=mask_j, other=0.0)

            # Dot product (head_dim = 128, implicit)
            # Compute scores = q * k^T
            scores = tl.sum(q_vec[:, None] * k_vec[None, :], axis=1)  # shape [BLOCK_I]
            scores = scores * scaling

            # Causal mask: lower-triangular with diagonal=1 => j >= i
            # Create mask for each i against jj
            for ii_idx in range(0, BLOCK_I):
                if ii_idx < S:
                    ii_val = ii[ii_idx]
                    # For this i, valid j are jj >= ii_val
                    valid = jj >= ii_val
                    # Apply negative infinity where not valid
                    scores[ii_idx] = tl.where(valid, scores[ii_idx], -1e20)

            # Softmax along j for each i
            max_score = tl.max(scores, axis=0)
            scores = scores - max_score
            exp_scores = tl.exp(scores)
            sum_exp = tl.sum(exp_scores, axis=0)
            attn = exp_scores / sum_exp  # shape [BLOCK_I]

            # Load V[:, j, :] and accumulate
            v_vec = tl.load(V_ptr + b * V_stride0 + jj * V_stride1 + h * V_stride2, mask=mask_j, other=0.0)
            out_acc += attn * v_vec

        # Store output for these i
        tl.store(Out_ptr + b * Out_stride0 + ii * Out_stride1 + h * Out_stride2, out_acc, mask=mask_i)


# Output projection: out[b, s, h] = sum_k out[b, s, k] * w[h, k]
@triton.jit
def triton_linear_out(x_ptr, w_ptr, out_ptr,
                      B, S, K, H,
                      x_stride0, x_stride1, x_stride2,
                      w_stride0, w_stride1,
                      out_stride0, out_stride1, out_stride2,
                      BLOCK_K: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)
    base_x = b * x_stride0 + s * x_stride1
    base_out = b * out_stride0 + s * out_stride1 + h * out_stride2

    acc = tl.zeros([1], dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + tl.arange(0, BLOCK_K)
        mask_k = kk < K
        x_vec = tl.load(x_ptr + base_x + kk * x_stride2, mask=mask_k, other=0.0)
        w_vec = tl.load(w_ptr + h * w_stride0 + kk * w_stride1, mask=mask_k, other=0.0)
        acc += tl.sum(x_vec * w_vec, axis=0)
    tl.store(out_ptr + base_out, acc)


class ModelNew(torch.nn.Module):
    def __init__(self, batch_size: int, seq_len: int, head_dim: int = 128):
        super().__init__()
        # Constants for this configuration
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.num_attention_heads = 96
        self.num_key_value_heads = 8
        self.num_key_value_groups = 12
        self.scaling = head_dim ** -0.5  # 0.07905694 for head_dim=128

    def forward(self,
                hidden_states: torch.Tensor,
                q_proj_weight: torch.Tensor,  # [H, K]
                q_proj_bias: torch.Tensor,    # [H]
                k_proj_weight: torch.Tensor,  # [KVH, K]
                k_proj_bias: torch.Tensor,    # [KVH]
                v_proj_weight: torch.Tensor,  # [KVH, K]
                v_proj_bias: torch.Tensor,    # [KVH]
                o_proj_weight: torch.Tensor,  # [H, K]
                q_norm_weight: torch.Tensor,  # [H]
                k_norm_weight: torch.Tensor,  # [KVH]
                cos: torch.Tensor,            # [head_dim]
                sin: torch.Tensor,            # [head_dim]
                rms_norm_eps: float = 1e-6):
        assert hidden_states.is_cuda, "Triton kernels require CUDA tensors"
        assert hidden_states.dtype in (torch.float32,), "Expect float32 input"
        B = self.batch_size
        S = self.seq_len
        H = self.num_attention_heads
        KVH = self.num_key_value_heads
        GROUPS = self.num_key_value_groups

        # 1) Linear for Q, K, V: input is hidden_states [B, S, K]
        # Prepare outputs
        dtype = hidden_states.dtype
        K = hidden_states.shape[2]
        # Allocate Q, K, V
        query = torch.empty((B, S, H), device=hidden_states.device, dtype=dtype)
        key = torch.empty((B, S, KVH), device=hidden_states.device, dtype=dtype)
        value = torch.empty((B, S, KVH), device=hidden_states.device, dtype=dtype)

        # Launch triton_linear_bsh for Q, K, V
        # Grid (B, H, S)
        grid_q = (B, H, S)
        triton_linear_bsh[grid_q](
            hidden_states, q_proj_weight, q_proj_bias, query,
            B, S, K, H,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            q_proj_weight.stride(0), q_proj_weight.stride(1),
            query.stride(0), query.stride(1), query.stride(2),
            BLOCK_K=64
        )
        triton_linear_bsh[grid_q](
            hidden_states, k_proj_weight, k_proj_bias, key,
            B, S, K, KVH,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            k_proj_weight.stride(0), k_proj_weight.stride(1),
            key.stride(0), key.stride(1), key.stride(2),
            BLOCK_K=64
        )
        triton_linear_bsh[grid_q](
            hidden_states, v_proj_weight, v_proj_bias, value,
            B, S, K, KVH,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            v_proj_weight.stride(0), v_proj_weight.stride(1),
            value.stride(0), value.stride(1), value.stride(2),
            BLOCK_K=64
        )

        # 2) RMSNorm for Q and K
        query_norm = torch.empty_like(query)
        key_norm = torch.empty_like(key)
        grid_rms_q = (B, H)
        grid_rms_k = (B, KVH)
        triton_rmsnorm[grid_rms_q](
            query, q_norm_weight, query_norm,
            B, S, H,
            query.stride(0), query.stride(1), query.stride(2),
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
            rms_norm_eps,
            BLOCK_S=128
        )
        triton_rmsnorm[grid_rms_k](
            key, k_norm_weight, key_norm,
            B, S, KVH,
            key.stride(0), key.stride(1), key.stride(2),
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2),
            rms_norm_eps,
            BLOCK_S=128
        )

        # 3) Rotary Position Embedding for Q and K
        query_rot = torch.empty_like(query_norm)
        key_rot = torch.empty_like(key_norm)
        grid_rope = (B, H, S)
        triton_rope_row[grid_rope](
            query_norm, cos, sin, query_rot,
            B, S, H,
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
            cos.stride(0), cos.stride(1),
            sin.stride(0), sin.stride(1),
            query_rot.stride(0), query_rot.stride(1), query_rot.stride(2),
            BLOCK_D=16
        )
        triton_rope_row[grid_rope](
            key_norm, cos, sin, key_rot,
            B, S, KVH,
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2),
            cos.stride(0), cos.stride(1),
            sin.stride(0), sin.stride(1),
            key_rot.stride(0), key_rot.stride(1), key_rot.stride(2),
            BLOCK_D=16
        )

        # 4) GQA expansion: KVH -> H via groups=12
        K_expanded = torch.empty((B, S, H), device=hidden_states.device, dtype=dtype)
        V_expanded = torch.empty((B, S, H), device=hidden_states.device, dtype=dtype)
        grid_gqa = (B, KVH, GROUPS, S)
        triton_expand_kv[grid_gqa](
            key_rot, value, K_expanded, V_expanded,
            B, S, KVH, 128,
            key_rot.stride(0), key_rot.stride(1), key_rot.stride(2), 0,
            value.stride(0), value.stride(1), value.stride(2), 0,
            K_expanded.stride(0), K_expanded.stride(1), K_expanded.stride(2), 0,
            V_expanded.stride(0), V_expanded.stride(1), V_expanded.stride(2), 0,
            GROUPS=GROUPS
        )

        # 5) Attention compute: (B, H) grid, loop over i/j in tiles
        attn_out = torch.empty((B, S, H), device=hidden_states.device, dtype=dtype)
        grid_attn = (B, H)
        triton_attention_compute[grid_attn](
            query_rot, K_expanded, V_expanded, attn_out,
            B, S, H,
            query_rot.stride(0), query_rot.stride(1), query_rot.stride(2), 0,
            K_expanded.stride(0), K_expanded.stride(1), K_expanded.stride(2), 0,
            V_expanded.stride(0), V_expanded.stride(1), V_expanded.stride(2), 0,
            attn_out.stride(0), attn_out.stride(1), attn_out.stride(2),
            self.scaling,
            BLOCK_I=64, BLOCK_J=64
        )

        # 6) Output projection
        output = torch.empty((B, S, H), device=hidden_states.device, dtype=dtype)
        grid_out = (B, H, S)
        triton_linear_out[grid_out](
            attn_out, o_proj_weight, output,
            B, S, H, H,
            attn_out.stride(0), attn_out.stride(1), attn_out.stride(2),
            o_proj_weight.stride(0), o_proj_weight.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            BLOCK_K=64
        )

        return output


def run(*args):
    return ModelNew()(*args)
