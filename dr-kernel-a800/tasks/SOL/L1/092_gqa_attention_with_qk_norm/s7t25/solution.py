import torch
import triton
import triton.language as tl

# ----------------------------
# Triton kernels
# ----------------------------

# 1) Triton linear_bsh: out[b, s, h] = sum_k x[b, s, k] * weight[h, k] + bias[h]
@triton.jit
def triton_linear_bsh(x_ptr, weight_ptr, bias_ptr, out_ptr,
                      B, S, K, H,
                      x_stride0, x_stride1, x_stride2,
                      weight_stride0, weight_stride1,
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
        mask = k < K
        x_row = tl.load(x_ptr + base_x + k * x_stride2, mask=mask, other=0.0)
        w_row = tl.load(weight_ptr + h * weight_stride0 + k * weight_stride1, mask=mask, other=0.0)
        # promote to f32 for accumulation
        acc += tl.sum(x_row.to(tl.float32) * w_row.to(tl.float32), axis=0)

    # add bias[h]
    bval = tl.load(bias_ptr + h).to(tl.float32)
    out_val = acc + bval
    # store as f32 (original run uses fp32)
    tl.store(out_ptr + base_out, out_val)


# 2) Triton RMSNorm per row: out[b, h, :] = x * (weight[h] / sqrt(mean(x^2) + eps))
@triton.jit
def triton_rmsnorm_row(x_ptr, weight_ptr, out_ptr,
                       B, S, H,
                       x_stride0, x_stride1, x_stride2,
                       out_stride0, out_stride1, out_stride2,
                       eps: tl.constexpr,
                       BLOCK_S: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    base_x = b * x_stride0 + h * x_stride1
    base_out = b * out_stride0 + h * out_stride2

    sum_sq = 0.0
    for s0 in range(0, S, BLOCK_S):
        s = s0 + tl.arange(0, BLOCK_S)
        mask = s < S
        x = tl.load(x_ptr + base_x + s * x_stride2, mask=mask, other=0.0)
        sum_sq += tl.sum(x * x, axis=0)

    mean = sum_sq / S
    inv_rms = 1.0 / tl.sqrt(mean + eps)
    scale = inv_rms * tl.load(weight_ptr + h).to(tl.float32)

    for s0 in range(0, S, BLOCK_S):
        s = s0 + tl.arange(0, BLOCK_S)
        mask = s < S
        x = tl.load(x_ptr + base_x + s * x_stride2, mask=mask, other=0.0)
        y = x * scale
        tl.store(out_ptr + base_out + s * out_stride2, y, mask=mask)


# 3) Triton RoPE per row: rotate 128-dim vector into out
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


# 4) Triton GQA expand: expand KV heads from KVH to H with groups
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
                # K: copy K[b, kh, j, :] -> K_out[b, h_target, j, :]
                src = tl.load(K_ptr + b * K_stride0 + j * K_stride1 + kh * K_stride2 + 0 * K_stride3)
                tl.store(K_out_ptr + b * Kout_stride0 + j * Kout_stride1 + h_target * Kout_stride2 + 0 * Kout_stride3, src)
                # V: copy V[b, kh, j, :] -> V_out[b, h_target, j, :]
                src = tl.load(V_ptr + b * V_stride0 + j * V_stride1 + kh * V_stride2 + 0 * V_stride3)
                tl.store(V_out_ptr + b * Vout_stride0 + j * Vout_stride1 + h_target * Vout_stride2 + 0 * Vout_stride3, src)


# 5) Triton attention compute: for each (b, h), tile over i and j
#    scores[i, j] = Q[b, i, :] @ K[b, j, :].T * scaling
#    apply causal mask, softmax over j, then out[i] = sum_j attn[j] * V[b, j, :]
@triton.jit
def triton_attention(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    B, S, H, head_dim,
    Q_stride0, Q_stride1, Q_stride2, Q_stride3,
    K_stride0, K_stride1, K_stride2, K_stride3,
    V_stride0, V_stride1, V_stride2, V_stride3,
    Out_stride0, Out_stride1, Out_stride2,
    scaling: tl.constexpr,
    BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)

    base_q = b * Q_stride0 + h * Q_stride1
    base_k = b * K_stride0 + h * K_stride1
    base_v = b * V_stride0 + h * V_stride1

    # Loop over query positions in tiles
    for i0 in range(0, S, BLOCK_I):
        i_idx = i0 + tl.arange(0, BLOCK_I)
        mask_i = i_idx < S

        # Compute scores[i, j] for j in tiles, accumulate attention output for i
        acc_out = tl.zeros((BLOCK_I,), dtype=tl.float32)

        for j0 in range(0, S, BLOCK_J):
            j_idx = j0 + tl.arange(0, BLOCK_J)
            mask_j = j_idx < S

            # q_tile: [BLOCK_I, head_dim]
            q_tile = tl.zeros((BLOCK_I, head_dim), dtype=tl.float32)
            for d in range(0, head_dim):
                q_row = tl.load(Q_ptr + base_q + i_idx * Q_stride2 + d * Q_stride3, mask=mask_i, other=0.0)
                q_tile[:, d] = q_row

            # k_tile: [BLOCK_J, head_dim]
            k_tile = tl.zeros((BLOCK_J, head_dim), dtype=tl.float32)
            for d in range(0, head_dim):
                k_row = tl.load(K_ptr + base_k + j_idx * K_stride2 + d * K_stride3, mask=mask_j, other=0.0)
                k_tile[:, d] = k_row

            # scores = q_tile @ k_tile.T => [BLOCK_I, BLOCK_J]
            scores = tl.dot(q_tile, tl.trans(k_tile)) * scaling

            # causal mask: for each (i, j), keep if i <= j else -inf
            for i in range(0, BLOCK_I):
                for j in range(0, BLOCK_J):
                    i_pos = i0 + i
                    j_pos = j0 + j
                    if i_pos > j_pos:
                        scores[i, j] = -float('inf')

            # softmax over j
            exp_scores = tl.exp(scores - tl.max(scores, axis=1, keepdims=True))
            sum_exp = tl.sum(exp_scores, axis=1)
            attn = exp_scores / sum_exp[:, None]

            # accumulate output: out[i] += sum_j attn[j] * V[b, j, h, :]
            for j in range(0, BLOCK_J):
                j_pos = j0 + j
                mask_j_valid = j_pos < S
                v_row = tl.load(V_ptr + base_v + j_pos * V_stride2, mask=mask_j_valid, other=0.0).to(tl.float32)  # [head_dim]
                acc_out += attn[:, j] * v_row  # v_row broadcast over i

        # store results
        for i in range(0, BLOCK_I):
            i_pos = i0 + i
            mask_i_valid = i_pos < S
            tl.store(Out_ptr + b * Out_stride0 + i_pos * Out_stride1 + h * Out_stride2, acc_out[i], mask=mask_i_valid)


# 6) Triton output projection: out[b, s, h] = sum_k out_attn[b, s, k] * o_proj_weight[h, k]
@triton.jit
def triton_linear_out(out_attn_ptr, weight_ptr, out_ptr,
                      B, S, K, H,
                      out_stride0, out_stride1, out_stride2,
                      weight_stride0, weight_stride1,
                      out_stride_out0, out_stride_out1, out_stride_out2,
                      BLOCK_K: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    base_in = b * out_stride0 + s * out_stride1
    base_out = b * out_stride_out0 + s * out_stride_out1 + h * out_stride_out2

    acc = tl.zeros((), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        mask = k < K
        x_row = tl.load(out_attn_ptr + base_in + k * out_stride2, mask=mask, other=0.0)
        w_row = tl.load(weight_ptr + h * weight_stride0 + k * weight_stride1, mask=mask, other=0.0)
        acc += tl.sum(x_row.to(tl.float32) * w_row.to(tl.float32), axis=0)

    tl.store(out_ptr + base_out, acc)


# ----------------------------
# ModelNew: entry point
# ----------------------------

class ModelNew(torch.nn.Module):
    def __init__(self, hidden_dim=768, head_dim=128, num_attention_heads=96, num_key_value_heads=8, groups=12, rms_norm_eps=1e-6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.groups = groups
        self.rms_norm_eps = rms_norm_eps

        # We keep the same parameters as original for testing compatibility
        # Note: in a real setup, these would be nn.Parameters. Here they are just tensors.
        # Example shapes:
        # q_proj_weight: [head_dim, hidden_dim]
        # q_proj_bias: [head_dim]
        # o_proj_weight: [hidden_dim, head_dim]
        # k_norm_weight: [head_dim]
        # q_norm_weight: [head_dim]
        # k_proj_weight, v_proj_weight: [head_dim, hidden_dim]
        # q_proj_bias, k_proj_bias, v_proj_bias: [head_dim]
        # cos, sin: [head_dim]
        self.q_proj_weight = None
        self.q_proj_bias = None
        self.k_proj_weight = None
        self.k_proj_bias = None
        self.v_proj_weight = None
        self.v_proj_bias = None
        self.o_proj_weight = None
        self.q_norm_weight = None
        self.k_norm_weight = None
        self.cos = None
        self.sin = None

    def forward(self, hidden_states: torch.Tensor,
                q_proj_weight: torch.Tensor, q_proj_bias: torch.Tensor,
                k_proj_weight: torch.Tensor, k_proj_bias: torch.Tensor,
                v_proj_weight: torch.Tensor, v_proj_bias: torch.Tensor,
                o_proj_weight: torch.Tensor,
                q_norm_weight: torch.Tensor, k_norm_weight: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor):
        """
        hidden_states: [B, S, hidden_dim]
        q_proj_weight, k_proj_weight, v_proj_weight: [head_dim, hidden_dim]
        q_proj_bias, k_proj_bias, v_proj_bias: [head_dim]
        o_proj_weight: [hidden_dim, head_dim]
        q_norm_weight, k_norm_weight: [head_dim]
        cos, sin: [head_dim]
        """

        B, S, hidden_dim = hidden_states.shape
        H = self.num_attention_heads
        KVH = self.num_key_value_heads
        KD = hidden_dim  # each head uses hidden_dim as input feature

        # Ensure dtype and device consistency; use fp32 for accumulation
        device = hidden_states.device
        dtype = hidden_states.dtype  # usually fp16/bf16; we'll promote to f32 inside kernels

        # 1) Dense linear for Q, K, V
        # Allocate outputs
        q = torch.empty((B, S, H), device=device, dtype=torch.float32)
        k = torch.empty((B, S, KVH), device=device, dtype=torch.float32)
        v = torch.empty((B, S, KVH), device=device, dtype=torch.float32)

        # Launch Triton kernels for Q, K, V
        BLOCK_K = 128  # hidden_dim is 768 in typical setups, this loops in tiles
        grid_q = (B, H, S)
        triton_linear_bsh(
            hidden_states, q_proj_weight, q_proj_bias, q,
            B, S, hidden_dim, H,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            q_proj_weight.stride(0), q_proj_weight.stride(1),
            q.stride(0), q.stride(1), q.stride(2),
            BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        grid_k = (B, KVH, S)
        triton_linear_bsh(
            hidden_states, k_proj_weight, k_proj_bias, k,
            B, S, hidden_dim, KVH,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            k_proj_weight.stride(0), k_proj_weight.stride(1),
            k.stride(0), k.stride(1), k.stride(2),
            BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        grid_v = (B, KVH, S)
        triton_linear_bsh(
            hidden_states, v_proj_weight, v_proj_bias, v,
            B, S, hidden_dim, KVH,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            v_proj_weight.stride(0), v_proj_weight.stride(1),
            v.stride(0), v.stride(1), v.stride(2),
            BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # 2) RMSNorm for Q and K: normalize each row (b, h) across S
        q_rms = torch.empty((B, H), device=device, dtype=torch.float32)
        k_rms = torch.empty((B, KVH), device=device, dtype=torch.float32)

        # Grid for RMSNorm is (B, H) for Q and (B, KVH) for K
        BLOCK_S = 256
        grid_q_rms = (B, H)
        triton_rmsnorm_row(
            q, q_norm_weight, q_rms,
            B, S, H,
            q.stride(0), q.stride(1), q.stride(2),
            q_rms.stride(0), q_rms.stride(1),
            eps=self.rms_norm_eps,
            BLOCK_S=BLOCK_S,
            num_warps=4, num_stages=2
        )

        grid_k_rms = (B, KVH)
        triton_rmsnorm_row(
            k, k_norm_weight, k_rms,
            B, S, KVH,
            k.stride(0), k.stride(1), k.stride(2),
            k_rms.stride(0), k_rms.stride(1),
            eps=self.rms_norm_eps,
            BLOCK_S=BLOCK_S,
            num_warps=4, num_stages=2
        )

        # Apply normalization: q = q * q_rms[h]; k = k * k_rms[kh]
        # We need q_rms and k_rms back to original Q/K shapes
        # But q_rms is per (b,h), k_rms is per (b,kh). Multiply elementwise along last dim.
        q = q * q_rms[:, :, None]  # broadcasting over S
        k = k * k_rms[:, :, None]

        # 3) Rotary Position Embedding for Q and K
        # Allocate rotated tensors
        q_rot = torch.empty_like(q)
        k_rot = torch.empty_like(k)

        # Launch Triton kernels
        BLOCK_D = 64
        grid_qrot = (B, H, S)
        triton_rope_row(
            q, cos, sin, q_rot,
            B, H, S, self.head_dim,
            q.stride(0), q.stride(1), q.stride(2),
            cos.stride(0), cos.stride(1),
            sin.stride(0), sin.stride(1),
            q_rot.stride(0), q_rot.stride(1), q_rot.stride(2),
            BLOCK_D=BLOCK_D,
            num_warps=4, num_stages=2
        )

        grid_krot = (B, KVH, S)
        triton_rope_row(
            k, cos, sin, k_rot,
            B, KVH, S, hidden_dim,
            k.stride(0), k.stride(1), k.stride(2),
            cos.stride(0), cos.stride(1),
            sin.stride(0), sin.stride(1),
            k_rot.stride(0), k_rot.stride(1), k_rot.stride(2),
            BLOCK_D=BLOCK_D,
            num_warps=4, num_stages=2
        )

        # 4) Grouped-Query expansion: expand KV heads from KVH to H with groups=12
        # Prepare expanded tensors
        K_exp = torch.empty((B, H, S, hidden_dim), device=device, dtype=torch.float32)
        V_exp = torch.empty((B, H, S, hidden_dim), device=device, dtype=torch.float32)

        grid_expand = (B, KVH, self.groups, S)
        triton_expand_kv(
            k_rot, v, K_exp, V_exp,
            B, S, KVH, hidden_dim,
            k_rot.stride(0), k_rot.stride(1), k_rot.stride(2), k_rot.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            K_exp.stride(0), K_exp.stride(1), K_exp.stride(2), K_exp.stride(3),
            V_exp.stride(0), V_exp.stride(1), V_exp.stride(2), V_exp.stride(3),
            GROUPS=self.groups,
            num_warps=4, num_stages=2
        )

        # 5) Attention compute: out[b, s, h] = sum_j attn[b, s, j] * V[b, j, h]
        # We need to compute attn weights over (S x S) per (b,h) using q_rot and K_exp.
        out_attn = torch.empty((B, S, H), device=device, dtype=torch.float32)

        BLOCK_I = 128
        BLOCK_J = 128
        scaling = 1.0 / (self.head_dim ** 0.5)
        grid_att = (B, H)
        triton_attention(
            q_rot, K_exp, V_exp, out_attn,
            B, S, H, self.head_dim,
            q_rot.stride(0), q_rot.stride(1), q_rot.stride(2), q_rot.stride(3),
            K_exp.stride(0), K_exp.stride(1), K_exp.stride(2), K_exp.stride(3),
            V_exp.stride(0), V_exp.stride(1), V_exp.stride(2), V_exp.stride(3),
            out_attn.stride(0), out_attn.stride(1), out_attn.stride(2),
            scaling=scaling,
            BLOCK_I=BLOCK_I, BLOCK_J=BLOCK_J,
            num_warps=4, num_stages=2
        )

        # 6) Output projection: out = linear(out_attn, o_proj_weight, no bias)
        # o_proj_weight: [hidden_dim, head_dim]
        # out_attn: [B, S, head_dim]
        output = torch.empty((B, S, hidden_dim), device=device, dtype=torch.float32)

        BLOCK_K = 128
        grid_out = (B, H, S)
        triton_linear_out(
            out_attn, o_proj_weight, output,
            B, S, self.head_dim, hidden_dim,
            out_attn.stride(0), out_attn.stride(1), out_attn.stride(2),
            o_proj_weight.stride(0), o_proj_weight.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        return output


# ----------------------------
# Optional: if you want to run a quick local test (not used by evaluator)
# ----------------------------
if __name__ == "__main__":
    # Example test with fixed sizes similar to original (hidden_dim=768, head_dim=128)
    B, S = 2, 128
    hidden_dim = 768
    head_dim = 128
    H = 96
    KVH = 8
    groups = 12
    rms_norm_eps = 1e-6

    # Dummy tensors (fp16 for performance, kernels promote to fp32)
    device = "cuda"
    hidden_states = torch.randn(B, S, hidden_dim, device=device, dtype=torch.float16)

    q_proj_weight = torch.randn(head_dim, hidden_dim, device=device, dtype=torch.float16)
    q_proj_bias = torch.randn(head_dim, device=device, dtype=torch.float16)
    k_proj_weight = torch.randn(head_dim, hidden_dim, device=device, dtype=torch.float16)
    k_proj_bias = torch.randn(head_dim, device=device, dtype=torch.float16)
    v_proj_weight = torch.randn(head_dim, hidden_dim, device=device, dtype=torch.float16)
    v_proj_bias = torch.randn(head_dim, device=device, dtype=torch.float16)
    o_proj_weight = torch.randn(hidden_dim, head_dim, device=device, dtype=torch.float16)
    q_norm_weight = torch.randn(head_dim, device=device, dtype=torch.float16)
    k_norm_weight = torch.randn(head_dim, device=device, dtype=torch.float16)
    cos = torch.randn(head_dim, device=device, dtype=torch.float16)
    sin = torch.randn(head_dim, device=device, dtype=torch.float16)

    model = ModelNew(hidden_dim=hidden_dim, head_dim=head_dim, num_attention_heads=H, num_key_value_heads=KVH, groups=groups, rms_norm_eps=rms_norm_eps).to(device)
    output = model(hidden_states, q_proj_weight, q_proj_bias, k_proj_weight, k_proj_bias, v_proj_weight, v_proj_bias, o_proj_weight, q_norm_weight, k_norm_weight, cos, sin)
    print(output.shape)  # should be [B, S, hidden_dim]


def run(*args):
    return ModelNew()(*args)
