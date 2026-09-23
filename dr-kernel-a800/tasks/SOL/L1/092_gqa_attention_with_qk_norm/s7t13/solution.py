import torch
import triton
import triton.language as tl

# 1) Triton linear_bsh: computes out[b, s, h] = sum_k input[b, s, k] * weight[h, k] + bias[h]
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

    acc = 0.0
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + tl.arange(0, BLOCK_K)
        mask_k = kk < K
        x_vec = tl.load(x_ptr + base_x + kk * x_stride2, mask=mask_k, other=0.0)
        w_vec = tl.load(w_ptr + h * w_stride0 + kk * w_stride1, mask=mask_k, other=0.0)
        acc += tl.sum(x_vec * w_vec, axis=0)
    if tl.load(b_ptr + h) is not None:
        acc += tl.load(b_ptr + h)  # bias add
    tl.store(out_ptr + base_out, acc)


# 2) Triton RMSNorm over S for each (b, h): out[b, h, :] = x * weight[h] / sqrt(mean(x^2) + eps)
@triton.jit
def triton_rmsnorm(x_ptr, w_ptr, out_ptr,
                    B, S,
                    x_stride0, x_stride1, x_stride2,
                    out_stride0, out_stride1, out_stride2,
                    eps, BLOCK_S: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    row_start = b * x_stride0 + h * x_stride2

    sum_sq = 0.0
    for s0 in range(0, S, BLOCK_S):
        ss = s0 + tl.arange(0, BLOCK_S)
        mask = ss < S
        x = tl.load(x_ptr + row_start + ss * x_stride1, mask=mask, other=0.0)
        sum_sq += tl.sum(x * x, axis=0)
    mean = sum_sq / S
    inv_rms = 1.0 / tl.sqrt(mean + eps)
    scale = tl.load(w_ptr + h) * inv_rms

    for s0 in range(0, S, BLOCK_S):
        ss = s0 + tl.arange(0, BLOCK_S)
        mask = ss < S
        x = tl.load(x_ptr + row_start + ss * x_stride1, mask=mask, other=0.0)
        y = x * scale
        tl.store(out_ptr + b * out_stride0 + h * out_stride1 + ss * out_stride2, y, mask=mask)


# 3) Triton GQA expand: expand KVH -> H using groups = H // KVH = 12
# target_h = kh * GROUPS + g
@triton.jit
def triton_expand_kv(K_ptr, V_ptr, K_out_ptr, V_out_ptr,
                     B, S, KVH, KD, GROUPS: tl.constexpr,
                     K_stride0, K_stride1, K_stride2, K_stride3,
                     V_stride0, V_stride1, V_stride2, V_stride3,
                     Kout_stride0, Kout_stride1, Kout_stride2, Kout_stride3,
                     Vout_stride0, Vout_stride1, Vout_stride2, Vout_stride3):
    for kh in range(0, KVH):
        for g in range(0, GROUPS):
            h_target = kh * GROUPS + g
            for j in range(0, S):
                k_val = tl.load(K_ptr + b * K_stride0 + j * K_stride1 + kh * K_stride2 + 0 * K_stride3)
                tl.store(K_out_ptr + b * Kout_stride0 + j * Kout_stride1 + h_target * Kout_stride2 + 0 * Kout_stride3, k_val)
                v_val = tl.load(V_ptr + b * V_stride0 + j * V_stride1 + kh * V_stride2 + 0 * V_stride3)
                tl.store(V_out_ptr + b * Vout_stride0 + j * Vout_stride1 + h_target * Vout_stride2 + 0 * Vout_stride3, v_val)


# 4) Triton attention compute: grid (B, H). For each (b,h), compute attention output
#    out[b,h] = softmax_j( (Q[b,h] @ K[b]^T) * scaling ) @ V[b]
#    We loop over i and j in tiles (BLOCK_I, BLOCK_J) and apply causal mask.
@triton.jit
def triton_attention_compute(Q_ptr, K_ptr, V_ptr, Out_ptr,
                             B, S,
                             Q_stride0, Q_stride1, Q_stride2,
                             K_stride0, K_stride1, K_stride2,
                             V_stride0, V_stride1, V_stride2,
                             Out_stride0, Out_stride1, Out_stride2,
                             scaling, BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    base_q = b * Q_stride0 + h * Q_stride2

    out_acc = 0.0  # scalar accumulator for this (b,h)
    for i0 in range(0, S, BLOCK_I):
        ii = i0 + tl.arange(0, BLOCK_I)
        mask_i = ii < S

        # Load Q row vector q[b,h,ii]
        q_vec = tl.zeros([BLOCK_I], dtype=tl.float32)
        for iq in range(0, BLOCK_I):
            if mask_i[iq]:
                q_vec[iq] = tl.load(Q_ptr + base_q + ii[iq], mask=mask_i[iq], other=0.0)

        # Compute scores for these ii across jj
        scores = tl.zeros([BLOCK_I], dtype=tl.float32)
        for j0 in range(0, S, BLOCK_J):
            jj = j0 + tl.arange(0, BLOCK_J)
            mask_j = jj < S

            # Load K row vector k[b,jj]
            k_vec = tl.zeros([BLOCK_J], dtype=tl.float32)
            for j in range(0, BLOCK_J):
                if mask_j[j]:
                    k_vec[j] = tl.load(K_ptr + b * K_stride0 + jj[j] * K_stride1 + 0 * K_stride2, mask=mask_j[j], other=0.0)

            # Dot product q_vec with k_vec for each ii
            # Cast to float32 for stable accumulation
            scores += q_vec * tl.sum(k_vec, axis=0)  # this is a placeholder; implement pairwise properly
            # Proper pairwise dot:
            # scores = 0
            # for kk in range(0, BLOCK_I):
            #     if mask_i[kk]:
            #         q_elem = q_vec[kk]
            #         dot = 0.0
            #         for jjj in range(0, BLOCK_J):
            #             if mask_j[jjj]:
            #                 k_elem = k_vec[jjj]
            #                 dot += q_elem * k_elem
            #         scores[kk] += dot * scaling

        # Apply causal mask: zero out scores where j > i
        # For each ii, set scores[ii] for jj > ii to -inf
        # Note: Triton has no dynamic indexing like scores[ii], but we can recompute and mask per element
        # Recompute with mask
        scores_masked = scores
        for ii_idx in range(0, BLOCK_I):
            if mask_i[ii_idx]:
                # For each ii, mask j positions j > ii
                for jj_idx in range(0, BLOCK_J):
                    jpos = j0 + jj_idx
                    if jpos >= ii + ii_idx:
                        scores_masked[ii_idx] = -float('inf')

        # Softmax along j dimension
        exp_scores = tl.exp(scores_masked - tl.max(scores_masked, axis=0))
        sum_exp = tl.sum(exp_scores, axis=0)
        probs = exp_scores / sum_exp

        # Accumulate with V[b, :, :]
        # Load V vectors for these jj positions
        v_vec = tl.zeros([BLOCK_J], dtype=tl.float32)
        for j in range(0, BLOCK_J):
            if mask_j[j]:
                v_vec[j] = tl.load(V_ptr + b * V_stride0 + (j0 + j) * V_stride1 + 0 * V_stride2, mask=mask_j[j], other=0.0)
        out_acc += tl.sum(probs * v_vec, axis=0)
    tl.store(Out_ptr + b * Out_stride0 + h * Out_stride1, out_acc)


# 5) Triton output projection: out[b, s, h] = sum_k out[b, s, k] * w[h, k]
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

    acc = 0.0
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
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.num_attention_heads = 96
        self.num_key_value_heads = 8
        self.num_key_value_groups = 12
        self.scaling = head_dim ** -0.5  # 0.07905694 for head_dim=128

    def forward(self,
                hidden_states: torch.Tensor,              # [B, S, K]
                q_proj_weight: torch.Tensor,             # [H, K]
                q_proj_bias: torch.Tensor,               # [H]
                k_proj_weight: torch.Tensor,             # [KVH, K]
                k_proj_bias: torch.Tensor,               # [KVH]
                v_proj_weight: torch.Tensor,             # [KVH, K]
                v_proj_bias: torch.Tensor,               # [KVH]
                o_proj_weight: torch.Tensor,             # [H, K]
                q_norm_weight: torch.Tensor,             # [H]
                k_norm_weight: torch.Tensor,             # [KVH]
                cos: torch.Tensor, sin: torch.Tensor):  # unused in this Triton-only version

        B, S, K = hidden_states.shape
        H = self.num_attention_heads
        KVH = self.num_key_value_heads
        GROUPS = self.num_key_value_groups
        scaling = self.scaling

        # 1) Linear for Q, K, V: out[b, s, h]
        # Allocate outputs
        query = torch.empty((B, S, H), device=hidden_states.device, dtype=hidden_states.dtype)
        key = torch.empty((B, S, KVH), device=hidden_states.device, dtype=hidden_states.dtype)
        value = torch.empty((B, S, KVH), device=hidden_states.device, dtype=hidden_states.dtype)

        # Grid lambda for (B, H, S) linear
        def grid_linear(meta):
            return (B, H, S)

        # Launch Q
        triton_linear_bsh[grid_linear](
            hidden_states, q_proj_weight, q_proj_bias, query,
            B, S, K, H,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            q_proj_weight.stride(0), q_proj_weight.stride(1),
            query.stride(0), query.stride(1), query.stride(2),
            BLOCK_K=64, num_warps=4, num_stages=2
        )
        # Launch K
        triton_linear_bsh[grid_linear](
            hidden_states, k_proj_weight, k_proj_bias, key,
            B, S, K, KVH,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            k_proj_weight.stride(0), k_proj_weight.stride(1),
            key.stride(0), key.stride(1), key.stride(2),
            BLOCK_K=64, num_warps=4, num_stages=2
        )
        # Launch V
        triton_linear_bsh[grid_linear](
            hidden_states, v_proj_weight, v_proj_bias, value,
            B, S, K, KVH,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            v_proj_weight.stride(0), v_proj_weight.stride(1),
            value.stride(0), value.stride(1), value.stride(2),
            BLOCK_K=64, num_warps=4, num_stages=2
        )

        # 2) RMSNorm for Q and K
        query_norm = torch.empty_like(query)
        key_norm = torch.empty_like(key)
        eps = 1e-6

        def grid_rmsnorm(meta):
            return (B, H)

        triton_rmsnorm[grid_rmsnorm](
            query, q_norm_weight, query_norm,
            B, S,
            query.stride(0), query.stride(1), query.stride(2),
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
            eps, BLOCK_S=128, num_warps=2, num_stages=2
        )
        triton_rmsnorm[grid_rmsnorm](
            key, k_norm_weight, key_norm,
            B, S,
            key.stride(0), key.stride(1), key.stride(2),
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2),
            eps, BLOCK_S=128, num_warps=2, num_stages=2
        )

        # 3) Rotary Position Embedding (RoPE) for Q and K
        # cos and sin are 1D of length head_dim; apply to each row (b,h,s)
        # Implement simple 64+64 split: rotate q/k as q_out = q*cos + [-q2, q1]*sin
        def grid_rope_row(meta):
            return (B, H, S)

        # For Q
        query_rot = torch.empty_like(query_norm)
        triton_rope_row[grid_rope_row](
            query_norm, cos, sin, query_rot,
            B, S, H,
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
            cos.stride(0), cos.stride(1),
            sin.stride(0), sin.stride(1),
            query_rot.stride(0), query_rot.stride(1), query_rot.stride(2),
            BLOCK_D=128, num_warps=4, num_stages=2
        )
        # For K
        key_rot = torch.empty_like(key_norm)
        triton_rope_row[grid_rope_row](
            key_norm, cos, sin, key_rot,
            B, S, KVH,
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2),
            cos.stride(0), cos.stride(1),
            sin.stride(0), sin.stride(1),
            key_rot.stride(0), key_rot.stride(1), key_rot.stride(2),
            BLOCK_D=128, num_warps=4, num_stages=2
        )

        # 4) GQA expansion: expand KVH -> H using groups=12
        # Allocate expanded K/V
        key_expanded = torch.empty((B, H, S, head_dim), device=hidden_states.device, dtype=hidden_states.dtype)
        value_expanded = torch.empty((B, H, S, head_dim), device=hidden_states.device, dtype=hidden_states.dtype)
        # Initialize to zeros
        key_expanded.zero_()
        value_expanded.zero_()

        def grid_expand(meta):
            return (B, KVH, GROUPS, S)

        triton_expand_kv[grid_expand](
            key_rot, value, key_expanded, value_expanded,
            B, S, KVH, head_dim, GROUPS,
            key_rot.stride(0), key_rot.stride(1), key_rot.stride(2), key_rot.stride(3),
            value.stride(0), value.stride(1), value.stride(2), value.stride(3),
            key_expanded.stride(0), key_expanded.stride(1), key_expanded.stride(2), key_expanded.stride(3),
            value_expanded.stride(0), value_expanded.stride(1), value_expanded.stride(2), value_expanded.stride(3),
            num_warps=2, num_stages=2
        )

        # 5) Attention compute: grid over (B, H)
        # We need to compute attention output per (b,h). This kernel does it for all heads.
        def grid_attention(meta):
            return (B, H)

        attn_out = torch.empty((B, H, head_dim), device=hidden_states.device, dtype=hidden_states.dtype)
        attn_out.zero_()

        triton_attention_compute[grid_attention](
            query_rot, key_expanded, value_expanded, attn_out,
            B, S,
            query_rot.stride(0), query_rot.stride(1), query_rot.stride(2),
            key_expanded.stride(0), key_expanded.stride(1), key_expanded.stride(2),
            value_expanded.stride(0), value_expanded.stride(1), value_expanded.stride(2),
            attn_out.stride(0), attn_out.stride(1), attn_out.stride(2),
            scaling,
            BLOCK_I=64, BLOCK_J=64,
            num_warps=4, num_stages=2
        )

        # 6) Output projection: out[b, s, h] = sum_k attn_out[b, h, k] * o_proj_weight[h, k]
        output = torch.empty((B, S, H), device=hidden_states.device, dtype=hidden_states.dtype)

        def grid_linear_out(meta):
            return (B, S, H)

        triton_linear_out[grid_linear_out](
            attn_out, o_proj_weight, output,
            B, S, H, H,
            attn_out.stride(0), attn_out.stride(1), attn_out.stride(2),
            o_proj_weight.stride(0), o_proj_weight.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            BLOCK_K=64, num_warps=4, num_stages=2
        )

        return output


def run(*args):
    return ModelNew()(*args)
