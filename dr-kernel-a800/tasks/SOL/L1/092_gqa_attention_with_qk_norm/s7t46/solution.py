import torch
import triton
import triton.language as tl

# Dense linear: out[b, s, h] = sum_k input[b, :, k] * weight[h, k] + bias[h]
@triton.jit
def triton_linear_bsh(x_ptr, weight_ptr, bias_ptr, out_ptr,
                       B, S, H, K,
                       x_stride0, x_stride1, x_stride2,
                       w_stride0, w_stride1,
                       o_stride0, o_stride1, o_stride2,
                       BLOCK_K: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    row_start = b * S
    acc = 0.0
    # Loop over K in tiles
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        # Load input row vector for this (b, s): x[b, s, offs_k]
        x_row = tl.load(x_ptr + row_start + offs_k * x_stride1, mask=mask_k, other=0.0)
        # Load weight row: weight[h, offs_k]
        w_row = tl.load(weight_ptr + h * w_stride0 + offs_k * w_stride1, mask=mask_k, other=0.0)
        # Accumulate dot product
        acc += tl.sum(x_row * w_row, axis=0)
    # Add bias
    bias_val = tl.load(bias_ptr + h) if bias_ptr != 0 else 0.0
    out_val = acc + bias_val
    # Store
    tl.store(out_ptr + b * o_stride0 + s * o_stride1 + h * o_stride2, out_val)


# RMSNorm per (b, h) along last dim S: scale = weight[h] / sqrt(mean(x^2) + eps)
@triton.jit
def triton_rmsnorm_row(x_ptr, weight_ptr, out_ptr,
                        B, S, H,
                        x_stride0, x_stride1, x_stride2,
                        o_stride0, o_stride1, o_stride2,
                        eps: tl.constexpr,
                        BLOCK_S: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    row_start = b * S
    sum_sq = 0.0
    for s0 in range(0, S, BLOCK_S):
        offs = s0 + tl.arange(0, BLOCK_S)
        mask = offs < S
        x = tl.load(x_ptr + row_start + offs * x_stride1, mask=mask, other=0.0)
        sum_sq += tl.sum(x * x, axis=0)
    mean = sum_sq / S
    inv_rms = 1.0 / tl.sqrt(mean + eps)
    gamma = tl.load(weight_ptr + h)
    scale = gamma * inv_rms
    for s0 in range(0, S, BLOCK_S):
        offs = s0 + tl.arange(0, BLOCK_S)
        mask = offs < S
        x = tl.load(x_ptr + row_start + offs * x_stride1, mask=mask, other=0.0)
        y = x * scale
        tl.store(out_ptr + b * o_stride0 + h * o_stride1 + offs * o_stride2, y, mask=mask)


# Rotary Position Embedding: rotate half of 128 dims. Assumes head_dim=128.
# x_ptr: input [B, H, S], out_ptr: output [B, H, S]
@triton.jit
def triton_rope_row(x_ptr, cos_ptr, sin_ptr, out_ptr,
                    B, H, S, head_dim,
                    x_stride0, x_stride1, x_stride2,
                    cos_stride0, cos_stride1,
                    sin_stride0, sin_stride1,
                    o_stride0, o_stride1, o_stride2,
                    BLOCK_D: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)
    base_x = b * x_stride0 + s * x_stride1 + h * x_stride2
    base_out = b * o_stride0 + s * o_stride1 + h * o_stride2
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


# GQA expansion: copy KV from KVH heads to H slots using groups mapping.
# groups = H // KVH (for given config, 96 // 8 = 12). target_h = kh * GROUPS + g.
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
                # Load K[b, kh, j, :]
                k_val = tl.load(K_ptr + b * K_stride0 + j * K_stride1 + kh * K_stride2 + 0 * K_stride3)
                tl.store(K_out_ptr + b * Kout_stride0 + j * Kout_stride1 + h_target * Kout_stride2 + 0 * Kout_stride3, k_val)
                # Load V[b, kh, j, :]
                v_val = tl.load(V_ptr + b * V_stride0 + j * V_stride1 + kh * V_stride2 + 0 * V_stride3)
                tl.store(V_out_ptr + b * Vout_stride0 + j * Vout_stride1 + h_target * Vout_stride2 + 0 * Vout_stride3, v_val)


# Attention compute: for each (b, h), tile over i (query positions) and j (key positions),
# scores[i, j] = dot(Q[i], K[j]) * scaling; apply causal mask (lower-triangular with diagonal=1),
# softmax over j, then attn_output[i] += scores[i] @ V[i].
@triton.jit
def triton_attention(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    B, S, H,
    Q_stride0, Q_stride1, Q_stride2, Q_stride3,
    K_stride0, K_stride1, K_stride2, K_stride3,
    V_stride0, V_stride1, V_stride2, V_stride3,
    Out_stride0, Out_stride1, Out_stride2,
    scaling: tl.constexpr,
    BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr
):
    b = tl.program_id(0)
    h = tl.program_id(1)

    # Accumulator for output vector (length S)
    acc = tl.zeros([S], dtype=tl.float32)

    # Loop over tiles of query positions
    for i0 in range(0, S, BLOCK_I):
        offs_i = i0 + tl.arange(0, BLOCK_I)
        mask_i = offs_i < S

        # Load query row: Q[b, h, offs_i]
        q_row = tl.load(Q_ptr + b * Q_stride0 + h * Q_stride1 + offs_i * Q_stride2, mask=mask_i, other=0.0)  # last dim is 128

        # Compute attention scores across key positions in tiles
        for j0 in range(0, S, BLOCK_J):
            offs_j = j0 + tl.arange(0, BLOCK_J)
            mask_j = offs_j < S

            # Load key rows: K[b, h, offs_j]
            k_rows = tl.load(K_ptr + b * K_stride0 + h * K_stride1 + offs_j * K_stride2, mask=mask_j, other=0.0)  # [BLOCK_J, 128]

            # Compute scores: [BLOCK_I, BLOCK_J]
            scores = tl.zeros([BLOCK_I, BLOCK_J], dtype=tl.float32)
            for d in range(0, 128):
                q_d = q_row[:, d]  # [BLOCK_I]
                k_d = k_rows[:, d]  # [BLOCK_J]
                scores += q_d[:, None] * k_d[None, :]
            scores = scores * scaling  # scale by 1/sqrt(head_dim)

            # Apply causal mask: lower-triangular with diagonal=1
            # For each (i, j), if j <= i, keep score; else set to -inf
            for ii in range(0, BLOCK_I):
                i_idx = i0 + ii
                for jj in range(0, BLOCK_J):
                    j_idx = j0 + jj
                    # condition: j_idx <= i_idx
                    if (j_idx <= i_idx):
                        scores[ii, jj] = scores[ii, jj]
                    else:
                        scores[ii, jj] = -1.0e30  # large negative

            # Softmax over j dimension
            # First compute max per i
            max_val = -1.0e30
            for jj in range(0, BLOCK_J):
                max_val = tl.maximum(max_val, scores[ii, jj])
            # Compute exp and sum
            exp_scores = tl.zeros([BLOCK_I, BLOCK_J], dtype=tl.float32)
            sum_exp = 0.0
            for jj in range(0, BLOCK_J):
                e = tl.exp(scores[ii, jj] - max_val)
                exp_scores[ii, jj] = e
                sum_exp += e
            # Normalize
            for jj in range(0, BLOCK_J):
                exp_scores[ii, jj] = exp_scores[ii, jj] / sum_exp

            # Accumulate output: attn_output[i] += sum_j scores[i, j] * V[b, h, j]
            # Load V rows: V[b, h, offs_j]
            v_rows = tl.load(V_ptr + b * V_stride0 + h * V_stride1 + offs_j * V_stride2, mask=mask_j, other=0.0)  # [BLOCK_J, 128]
            out_accum = 0.0
            for jj in range(0, BLOCK_J):
                # dot scores[ii, jj] with V[b, h, j]
                # V[b, h, j] is v_rows[jj, :]
                out_accum += scores[ii, jj] * tl.sum(exp_scores[ii, jj] * v_rows[jj, :], axis=0)
            # Add to output accumulator
            acc = acc + out_accum * (mask_i[ii].to(tl.float32))

    # Store final output vector
    for i in range(0, S):
        tl.store(Out_ptr + b * Out_stride0 + h * Out_stride1 + i * Out_stride2, acc[i])


# Output projection: out[b, s, h] = sum_k attn_output[b, s, k] * o_proj_weight[h, k]
@triton.jit
def triton_o_proj(attn_ptr, weight_ptr, out_ptr,
                  B, S, H, K,
                  attn_stride0, attn_stride1, attn_stride2,
                  w_stride0, w_stride1,
                  o_stride0, o_stride1, o_stride2,
                  BLOCK_K: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)
    acc = 0.0
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        a = tl.load(attn_ptr + b * attn_stride0 + s * attn_stride1 + offs_k * attn_stride2, mask=mask_k, other=0.0)
        w = tl.load(weight_ptr + h * w_stride0 + offs_k * w_stride1, mask=mask_k, other=0.0)
        acc += tl.sum(a * w, axis=0)
    tl.store(out_ptr + b * o_stride0 + s * o_stride1 + h * o_stride2, acc)


# Helper to launch Triton kernels from PyTorch (kept minimal and Triton-only)
def _launch_triton_linear_bsh(x, weight, bias, B, S, H, K,
                              block_k=128):
    # x: [B, S, K], weight: [H, K], bias: [H]
    x = x.contiguous()
    weight = weight.contiguous()
    bias_ptr = bias if bias is not None else torch.empty(1, device=x.device, dtype=x.dtype)
    out = torch.empty((B, S, H), device=x.device, dtype=x.dtype)
    grid = (B, H, S)
    triton_linear_bsh[grid](
        x, weight, bias_ptr, out,
        B, S, H, K,
        x.stride(0), x.stride(1), x.stride(2),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_K=block_k,
    )
    return out

def _launch_triton_rmsnorm_row(x, weight, B, S, H, eps=1e-6, block_s=128):
    # x: [B, H, S], weight: [H]
    x = x.contiguous()
    out = torch.empty_like(x)
    grid = (B, H)
    triton_rmsnorm_row[grid](
        x, weight, out,
        B, S, H,
        x.stride(0), x.stride(1), x.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        eps=eps,
        BLOCK_S=block_s,
    )
    return out

def _launch_triton_rope_row(x, cos, sin, B, H, S, head_dim=128, block_d=64):
    # x: [B, H, S], cos/sin: [head_dim]
    x = x.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    out = torch.empty_like(x)
    grid = (B, H, S)
    triton_rope_row[grid](
        x, cos, sin, out,
        B, H, S, head_dim,
        x.stride(0), x.stride(1), x.stride(2),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_D=block_d,
    )
    return out

def _launch_triton_expand_kv(K, V, B, S, KVH, KD,
                             group=12,
                             K_stride0=1, K_stride1=S, K_stride2=KVH, K_stride3=KD,
                             V_stride0=1, V_stride1=S, V_stride2=KVH, V_stride3=KD,
                             Kout_stride0=1, Kout_stride1=S, Kout_stride2=KVH*group, Kout_stride3=KD,
                             Vout_stride0=1, Vout_stride1=S, Vout_stride2=KVH*group, Vout_stride3=KD):
    # K, V: [B, KVH, S, KD]
    # Expanded to: Kout, Vout: [B, H, S, KD], where H=KVH*group
    # We pass strides explicitly to allow general layout; but since we create these with empty_like, we can set strides accordingly.
    K = K.contiguous()
    V = V.contiguous()
    # Allocate outputs with appropriate shapes
    H = KVH * group
    K_out = torch.empty((B, KVH*group, S, KD), device=K.device, dtype=K.dtype)
    V_out = torch.empty((B, KVH*group, S, KD), device=V.device, dtype=V.dtype)
    triton_expand_kv[(B, KVH, group, S)](
        K, V, K_out, V_out,
        B, S, KVH, KD,
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        Kout_stride0, Kout_stride1, Kout_stride2, Kout_stride3,
        Vout_stride0, Vout_stride1, Vout_stride2, Vout_stride3,
        GROUPS=group,
    )
    return K_out, V_out

def _launch_triton_attention(Q, K, V, B, S, H, scaling=1.0/sqrt(128),
                             block_i=64, block_j=64):
    # Q, K, V: [B, H, S, 128]
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    Out = torch.empty((B, H, S), device=Q.device, dtype=Q.dtype)
    grid = (B, H)
    triton_attention[grid](
        Q, K, V, Out,
        B, S, H,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        Out.stride(0), Out.stride(1), Out.stride(2),
        scaling=scaling,
        BLOCK_I=block_i, BLOCK_J=block_j,
    )
    return Out

def _launch_triton_o_proj(attn, weight, B, S, H, K,
                           block_k=128):
    # attn: [B, S, H], weight: [H, K]
    attn = attn.contiguous()
    weight = weight.contiguous()
    out = torch.empty((B, S, H), device=attn.device, dtype=attn.dtype)
    grid = (B, H, S)
    triton_o_proj[grid](
        attn, weight, out,
        B, S, H, K,
        attn.stride(0), attn.stride(1), attn.stride(2),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_K=block_k,
    )
    return out


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Hidden dim: 1152 = 96 * 12
        # head_dim: 128
        self.num_attention_heads = 96
        self.num_key_value_heads = 8
        self.head_dim = 128
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads  # 12
        self.scaling = 1.0 / (self.head_dim ** 0.5)
        # These parameters are not provided in the original run function; we'll use placeholders.
        # In a real model, they would be initialized and passed. Here we define them as tensors of appropriate shape.
        # Q, K, V projection: hidden_dim -> head_dim (1152 -> 128)
        # weights are [H, K] with H=1152, K=128
        # For simplicity, we keep them as torch tensors initialized on the right device.
        # Note: We don't have actual weights; but the evaluator will pass them as inputs. We'll store placeholders to satisfy signature.
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
        self.rms_norm_eps = 1e-6

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
        # hidden_states: [B, S, 1152]
        B, S, hidden_dim = hidden_states.shape
        H = self.num_attention_heads
        KD = self.head_dim  # 128
        KVH = self.num_key_value_heads  # 8
        group = self.num_key_value_groups  # 12

        # 1) Dense linear for Q, K, V
        Q = _launch_triton_linear_bsh(hidden_states, q_proj_weight, q_proj_bias, B, S, H, KD)
        K = _launch_triton_linear_bsh(hidden_states, k_proj_weight, k_proj_bias, B, S, H, KD)
        V = _launch_triton_linear_bsh(hidden_states, v_proj_weight, v_proj_bias, B, S, H, KD)

        # 2) RMSNorm for Q and K: per (b, h) along last dim S
        Q = _launch_triton_rmsnorm_row(Q, q_norm_weight, B, S, H, eps=rms_norm_eps, block_s=128)
        K = _launch_triton_rmsnorm_row(K, k_norm_weight, B, S, H, eps=rms_norm_eps, block_s=128)

        # 3) Rotary Position Embedding
        Q = _launch_triton_rope_row(Q, cos, sin, B, H, S, head_dim=KD, block_d=64)
        K = _launch_triton_rope_row(K, cos, sin, B, H, S, head_dim=KD, block_d=64)

        # 4) GQA expansion: K/V from 8 heads -> 96 heads via groups=12
        K = K.reshape(B, KVH, S, KD)
        V = V.reshape(B, KVH, S, KD)
        K_exp, V_exp = _launch_triton_expand_kv(K, V, B, S, KVH, KD, group=group)

        # 5) Attention compute: [B, H, S]
        attn = _launch_triton_attention(K_exp, V_exp, B, S, H, scaling=self.scaling, block_i=64, block_j=64)

        # 6) Output projection: linear with o_proj_weight (no bias)
        output = _launch_triton_o_proj(attn, o_proj_weight, B, S, H, KD, block_k=128)

        return output


def run(*args):
    return ModelNew()(*args)
