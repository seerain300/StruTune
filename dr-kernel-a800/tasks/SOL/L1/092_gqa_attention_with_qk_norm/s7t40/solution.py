import torch
import triton
import triton.language as tl

# 1) Triton dense linear for [B, S, K] -> [B, S, H]: out[b, s, h] = sum_k x[b, s, k] * w[h, k] + bias[h]
@triton.jit
def triton_linear_bsh(x_ptr, w_ptr, b_ptr, out_ptr,
                       B, S, K, H,
                       x_stride0, x_stride1, x_stride2,
                       w_stride0, w_stride1,  # w[H, K]
                       out_stride0, out_stride1, out_stride2,
                       BLOCK_K: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    acc = tl.zeros((), dtype=tl.float32)
    row_x = b * x_stride0 + s * x_stride1

    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        mask_k = k < K
        x = tl.load(x_ptr + row_x + k * x_stride2, mask=mask_k, other=0.0)
        w = tl.load(w_ptr + h * w_stride0 + k * w_stride1, mask=mask_k, other=0.0)
        acc += tl.sum(x * w, axis=0)

    bias = tl.load(b_ptr + h)
    acc = acc + bias
    tl.store(out_ptr + b * out_stride0 + s * out_stride1 + h * out_stride2, acc.to(tl.float32))


# 2) Triton RMSNorm per row (b, h) across last dim S: y = x * (w[h] / sqrt(mean(x^2) + eps))
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
        offs = d0 + tl.arange(0, BLOCK_D)
        mask = offs < S
        x = tl.load(x_ptr + row_start + offs * x_stride1, mask=mask, other=0.0)
        sum_sq += tl.sum(x * x, axis=0)

    mean = sum_sq / S
    inv_rms = 1.0 / tl.sqrt(mean + eps)
    scale = tl.load(weight_ptr + h) * inv_rms

    for d0 in range(0, S, BLOCK_D):
        offs = d0 + tl.arange(0, BLOCK_D)
        mask = offs < S
        x = tl.load(x_ptr + row_start + offs * x_stride1, mask=mask, other=0.0)
        y = x * scale
        tl.store(out_ptr + b * out_stride0 + h * out_stride1 + offs * out_stride2, y, mask=mask)


# 3) Triton RoPE per row: for dim=128, rotate q = [-q2, q1], q_out = q*cos + rotated*sin
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
        cos_vec = tl.load(cos_ptr + d, mask=mask, other=1.0)  # safe default
        sin_vec = tl.load(sin_ptr + d, mask=mask, other=0.0)
        q1 = q[:64]
        q2 = q[64:]
        rotated = -q2 + q1
        q_out = q * cos_vec + rotated * sin_vec
        tl.store(out_ptr + base_out + d, q_out, mask=mask)


# 4) Triton GQA expand: expand KVH=8 to H=96 via groups=12. target_h = kh * GROUPS + g
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
                k_val = tl.load(K_ptr + b * K_stride0 + j * K_stride1 + kh * K_stride2 + 0 * K_stride3)
                tl.store(K_out_ptr + b * Kout_stride0 + j * Kout_stride1 + h_target * Kout_stride2 + 0 * Kout_stride3, k_val)
                v_val = tl.load(V_ptr + b * V_stride0 + j * V_stride1 + kh * V_stride2 + 0 * V_stride3)
                tl.store(V_out_ptr + b * Vout_stride0 + j * Vout_stride1 + h_target * Vout_stride2 + 0 * Vout_stride3, v_val)


# 5) Triton attention: compute out[b, h, S] = softmax_i((Q[i]*K[j]) * scaling) @ V over j, with causal mask i<j
@triton.jit
def triton_attention(Q_ptr, K_ptr, V_ptr, Out_ptr,
                     B, S, H,
                     Q_stride0, Q_stride1, Q_stride2,
                     K_stride0, K_stride1, K_stride2,
                     V_stride0, V_stride1, V_stride2,
                     Out_stride0, Out_stride1,
                     scaling,
                     BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)

    # Accumulate output for this (b, h)
    out_vec = tl.zeros((S,), dtype=tl.float32)

    # Loop over i tiles
    for i0 in range(0, S, BLOCK_I):
        i = i0 + tl.arange(0, BLOCK_I)
        i_mask = i < S

        # Compute scores[i, :] = Q[i] dot K[:, ]
        scores = tl.zeros((BLOCK_I,), dtype=tl.float32)

        for j0 in range(0, S, BLOCK_J):
            j = j0 + tl.arange(0, BLOCK_J)
            j_mask = j < S

            # Load Q[i, :]
            q_row = tl.load(Q_ptr + b * Q_stride0 + i * Q_stride1 + h * Q_stride2, mask=i_mask, other=0.0)
            # Load K[j, :]
            k_rows = tl.load(K_ptr + b * K_stride0 + j * K_stride1 + h * K_stride2, mask=j_mask, other=0.0)

            # Dot product per i
            # q_row shape [BLOCK_I], k_rows shape [BLOCK_J,]
            # Cast to float32 for accumulation
            q_row = q_row.to(tl.float32)
            k_rows = k_rows.to(tl.float32)
            scores += tl.sum(q_row[:, None] * k_rows[None, :], axis=1)

        # Apply scaling
        scores = scores * scaling

        # Apply causal mask: i < j -> -inf
        # Build causal matrix for (i, j) over tiles
        for ii in range(0, BLOCK_I):
            i_idx = i0 + ii
            valid_i = i_idx < S
            if valid_i:
                for jj in range(0, BLOCK_J):
                    j_idx = j0 + jj
                    valid_j = j_idx < S
                    if valid_j:
                        if i_idx < j_idx:
                            scores[ii] = -1e20  # large negative

        # Softmax along i axis (rows)
        m = tl.max(scores, axis=0)
        scores = scores - m
        exp_scores = tl.exp(scores)
        sum_exp = tl.sum(exp_scores, axis=0)
        attn = exp_scores / sum_exp

        # Accumulate output: out[i] += attn * V[i]
        v_rows = tl.load(V_ptr + b * V_stride0 + i * V_stride1 + h * V_stride2, mask=i_mask, other=0.0).to(tl.float32)
        out_vec = out_vec + attn * v_rows

    # Store out
    for s_idx in range(0, S):
        if s_idx < S:
            tl.store(Out_ptr + b * Out_stride0 + s_idx * Out_stride1 + h * Out_stride1, out_vec[s_idx])


# 6) Triton output projection: out[b, s, h] = sum_k x[b, s, k] * w[h, k]
@triton.jit
def triton_linear_out(x_ptr, w_ptr, out_ptr,
                       B, S, K, H,
                       x_stride0, x_stride1, x_stride2,
                       w_stride0, w_stride1,  # w[H, K]
                       out_stride0, out_stride1, out_stride2,
                       BLOCK_K: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    acc = tl.zeros((), dtype=tl.float32)
    row_x = b * x_stride0 + s * x_stride1

    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        mask_k = k < K
        x = tl.load(x_ptr + row_x + k * x_stride2, mask=mask_k, other=0.0)
        w = tl.load(w_ptr + h * w_stride0 + k * w_stride1, mask=mask_k, other=0.0)
        acc += tl.sum(x * w, axis=0)

    tl.store(out_ptr + b * out_stride0 + s * out_stride1 + h * out_stride2, acc.to(tl.float32))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # We won't use torch ops; all computation is done in Triton kernels.
        self.head_dim = 128
        self.num_attention_heads = 96
        self.num_key_value_heads = 8
        self.num_key_value_groups = 12
        self.scaling = 1.0 / (self.head_dim ** 0.5)
        self.rms_norm_eps = 1e-8

    def forward(self, hidden_states: torch.Tensor,
                q_proj_weight: torch.Tensor, q_proj_bias: torch.Tensor,
                k_proj_weight: torch.Tensor, k_proj_bias: torch.Tensor,
                v_proj_weight: torch.Tensor, v_proj_bias: torch.Tensor,
                o_proj_weight: torch.Tensor,
                q_norm_weight: torch.Tensor, k_norm_weight: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor):
        # hidden_states: [B, S, D] where D=hidden_dim
        B, S, D = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        # 1) Linear for Q, K, V: [B, S, D] -> [B, S, H] where H=128 for Q/K, V follows same
        # Note: weights are [H, K], but K here is D (hidden dimension). We'll use D=K.
        Q = torch.empty((B, S, 128), device=device, dtype=dtype)
        K = torch.empty((B, S, 128), device=device, dtype=dtype)
        V = torch.empty((B, S, 128), device=device, dtype=dtype)

        # Launch Triton linear for Q
        grid_q = (B, 128, S)
        triton.run(triton_linear_bsh, grid=grid_q,
                   x_ptr=hidden_states, w_ptr=q_proj_weight, b_ptr=q_proj_bias, out_ptr=Q,
                   B=B, S=S, K=D, H=128,
                   x_stride0=hidden_states.stride(0), x_stride1=hidden_states.stride(1), x_stride2=hidden_states.stride(2),
                   w_stride0=q_proj_weight.stride(0), w_stride1=q_proj_weight.stride(1),
                   out_stride0=Q.stride(0), out_stride1=Q.stride(1), out_stride2=Q.stride(2),
                   BLOCK_K=64)

        # Launch Triton linear for K
        grid_k = (B, 128, S)
        triton.run(triton_linear_bsh, grid=grid_k,
                   x_ptr=hidden_states, w_ptr=k_proj_weight, b_ptr=k_proj_bias, out_ptr=K,
                   B=B, S=S, K=D, H=128,
                   x_stride0=hidden_states.stride(0), x_stride1=hidden_states.stride(1), x_stride2=hidden_states.stride(2),
                   w_stride0=k_proj_weight.stride(0), w_stride1=k_proj_weight.stride(1),
                   out_stride0=K.stride(0), out_stride1=K.stride(1), out_stride2=K.stride(2),
                   BLOCK_K=64)

        # Launch Triton linear for V
        grid_v = (B, 128, S)
        triton.run(triton_linear_bsh, grid=grid_v,
                   x_ptr=hidden_states, w_ptr=v_proj_weight, b_ptr=v_proj_bias, out_ptr=V,
                   B=B, S=S, K=D, H=128,
                   x_stride0=hidden_states.stride(0), x_stride1=hidden_states.stride(1), x_stride2=hidden_states.stride(2),
                   w_stride0=v_proj_weight.stride(0), w_stride1=v_proj_weight.stride(1),
                   out_stride0=V.stride(0), out_stride1=V.stride(1), out_stride2=V.stride(2),
                   BLOCK_K=64)

        # 2) RMSNorm for Q and K: normalize rows (b, h) across S
        Q_norm = torch.empty_like(Q)
        K_norm = torch.empty_like(K)

        grid_rms = (B, 128)
        triton.run(triton_rmsnorm_row, grid=grid_rms,
                   x_ptr=Q, weight_ptr=q_norm_weight, out_ptr=Q_norm,
                   B=B, S=S,
                   x_stride0=Q.stride(0), x_stride1=Q.stride(1), x_stride2=Q.stride(2),
                   out_stride0=Q_norm.stride(0), out_stride1=Q_norm.stride(1), out_stride2=Q_norm.stride(2),
                   eps=self.rms_norm_eps,
                   BLOCK_D=128)

        triton.run(triton_rmsnorm_row, grid=grid_rms,
                   x_ptr=K, weight_ptr=k_norm_weight, out_ptr=K_norm,
                   B=B, S=S,
                   x_stride0=K.stride(0), x_stride1=K.stride(1), x_stride2=K.stride(2),
                   out_stride0=K_norm.stride(0), out_stride1=K_norm.stride(1), out_stride2=K_norm.stride(2),
                   eps=self.rms_norm_eps,
                   BLOCK_D=128)

        # 3) RoPE for Q and K
        Q_rope = torch.empty_like(Q_norm)
        K_rope = torch.empty_like(K_norm)

        grid_rope = (B, 128, S)
        triton.run(triton_rope_row, grid=grid_rope,
                   x_ptr=Q_norm, cos_ptr=cos, sin_ptr=sin, out_ptr=Q_rope,
                   B=B, H=128, S=S, head_dim=128,
                   x_stride0=Q_norm.stride(0), x_stride1=Q_norm.stride(1), x_stride2=Q_norm.stride(2),
                   cos_stride0=cos.stride(0), cos_stride1=cos.stride(1),
                   sin_stride0=sin.stride(0), sin_stride1=sin.stride(1),
                   out_stride0=Q_rope.stride(0), out_stride1=Q_rope.stride(1), out_stride2=Q_rope.stride(2),
                   BLOCK_D=128)

        triton.run(triton_rope_row, grid=grid_rope,
                   x_ptr=K_norm, cos_ptr=cos, sin_ptr=sin, out_ptr=K_rope,
                   B=B, H=128, S=S, head_dim=128,
                   x_stride0=K_norm.stride(0), x_stride1=K_norm.stride(1), x_stride2=K_norm.stride(2),
                   cos_stride0=cos.stride(0), cos_stride1=cos.stride(1),
                   sin_stride0=sin.stride(0), sin_stride1=sin.stride(1),
                   out_stride0=K_rope.stride(0), out_stride1=K_rope.stride(1), out_stride2=K_rope.stride(2),
                   BLOCK_D=128)

        # 4) GQA expand: KVH=8, H=96, GROUPS=12
        K_gqa = torch.empty((B, S, self.num_attention_heads), device=device, dtype=dtype)
        V_gqa = torch.empty((B, S, self.num_attention_heads), device=device, dtype=dtype)

        # We copy from K_rope and V (KVH rows) to K_gqa and V_gqa expanded heads.
        # Triton kernel expects strides; we pass them explicitly.
        # Make sure we pass strides correctly for K/V and K_out/V_out.
        grid_expand = (B, 8, 12, S)
        triton.run(triton_expand_kv, grid=grid_expand,
                   K_ptr=K_rope, V_ptr=V, K_out_ptr=K_gqa, V_out_ptr=V_gqa,
                   B=B, S=S, KVH=8, KD=128,
                   K_stride0=K_rope.stride(0), K_stride1=K_rope.stride(1), K_stride2=K_rope.stride(2), K_stride3=1,
                   V_stride0=V.stride(0), V_stride1=V.stride(1), V_stride2=V.stride(2), V_stride3=1,
                   Kout_stride0=K_gqa.stride(0), Kout_stride1=K_gqa.stride(1), Kout_stride2=K_gqa.stride(2), Kout_stride3=1,
                   Vout_stride0=V_gqa.stride(0), Vout_stride1=V_gqa.stride(1), Vout_stride2=V_gqa.stride(2), Vout_stride3=1,
                   GROUPS=12)

        # 5) Attention: compute Out[b, h, S] = softmax_i(scores) @ V over j, with causal mask
        Out = torch.empty((B, self.num_attention_heads, S), device=device, dtype=dtype)

        grid_attn = (B, self.num_attention_heads)
        triton.run(triton_attention, grid=grid_attn,
                   Q_ptr=Q_rope, K_ptr=K_gqa, V_ptr=V_gqa, Out_ptr=Out,
                   B=B, S=S, H=self.num_attention_heads,
                   Q_stride0=Q_rope.stride(0), Q_stride1=Q_rope.stride(1), Q_stride2=Q_rope.stride(2),
                   K_stride0=K_gqa.stride(0), K_stride1=K_gqa.stride(1), K_stride2=K_gqa.stride(2),
                   V_stride0=V_gqa.stride(0), V_stride1=V_gqa.stride(1), V_stride2=V_gqa.stride(2),
                   Out_stride0=Out.stride(0), Out_stride1=Out.stride(1),
                   scaling=self.scaling,
                   BLOCK_I=64, BLOCK_J=64)

        # 6) Output projection: Out[B, S, H] -> [B, S, H*head_dim]
        # Note: o_proj_weight is [H_out, H] where H_out = H*head_dim, H=96
        # We need to produce output of shape [B, S, num_attention_heads * head_dim] = [B, S, 96*128]
        H_out = self.num_attention_heads * self.head_dim
        output = torch.empty((B, S, H_out), device=device, dtype=dtype)

        # Launch Triton output projection
        grid_out = (B, self.num_attention_heads, S)
        triton.run(triton_linear_out, grid=grid_out,
                   x_ptr=Out, w_ptr=o_proj_weight, out_ptr=output,
                   B=B, S=S, K=self.num_attention_heads, H=H_out,
                   x_stride0=Out.stride(0), x_stride1=Out.stride(1), x_stride2=Out.stride(2),
                   w_stride0=o_proj_weight.stride(0), w_stride1=o_proj_weight.stride(1),
                   out_stride0=output.stride(0), out_stride1=output.stride(1), out_stride2=output.stride(2),
                   BLOCK_K=64)

        return output


# Original Model for reference (unchanged), used by evaluator to call ModelNew.run
class Model(torch.nn.Module):
    def forward(self, *args):
        return run(*args)


def run(*args):
    return ModelNew()(*args)
