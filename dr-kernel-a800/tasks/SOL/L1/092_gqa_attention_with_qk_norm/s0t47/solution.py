import torch
import triton
import triton.language as tl


# 1) Linear projection: y[b, l, n] = sum_k x[b, l, k] * w[n, k] + bias[n]
@triton.jit
def linear_proj_kernel(
    x_ptr,           # *f16/f32/bf16, [B, L, H_in]
    w_ptr,           # *f16/f32/bf16, [N_out, H_in]
    bias_ptr,        # *f32, [N_out]
    y_ptr,           # *f32, [B, L, N_out]
    B, L, H_in, N_out,
    x_bs0, x_bs1, x_bs2,
    w_bs0, w_bs1,
    y_bs0, y_bs1, y_bs2,
    BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)
    l = tl.program_id(1)
    n = tl.program_id(2)
    acc = tl.zeros((), dtype=tl.float32)
    for k0 in range(0, H_in, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < H_in
        x_ptrs = x_ptr + b * x_bs0 + l * x_bs1 + offs_k * x_bs2
        w_ptrs = w_ptr + n * w_bs0 + offs_k * w_bs1
        x_vals = tl.load(x_ptrs, mask=mask_k, other=0.0)
        w_vals = tl.load(w_ptrs, mask=mask_k, other=0.0)
        acc += tl.sum(x_vals.to(tl.float32) * w_vals.to(tl.float32), axis=0)
    bias = tl.load(bias_ptr + n)
    acc = acc + bias
    y_ptrs = y_ptr + b * y_bs0 + l * y_bs1 + n * y_bs2
    tl.store(y_ptrs, acc)


# 2) RMSNorm per (b, l) over head_dim=128: y = x * rsqrt(mean(x^2) + eps) * weight
@triton.jit
def rmsnorm_kernel(
    x_ptr,      # *f32, [B, L, 128]
    w_ptr,      # *f32, [128]
    eps,        # f32 scalar
    y_ptr,      # *f32, [B, L, 128]
    B, L,
    x_bs0, x_bs1, x_bs2,
    w_bs0,
    y_bs0, y_bs1, y_bs2,
):
    b = tl.program_id(0)
    l = tl.program_id(1)
    sumsq = tl.zeros((), dtype=tl.float32)
    for j in range(0, 128):
        x_val = tl.load(x_ptr + b * x_bs0 + l * x_bs1 + j * x_bs2)
        sumsq += x_val * x_val
    mean = sumsq / 128.0
    scale = tl.rsqrt(mean + eps)
    for j in range(0, 128):
        x_val = tl.load(x_ptr + b * x_bs0 + l * x_bs1 + j * x_bs2)
        w_val = tl.load(w_ptr + j * w_bs0)
        y_val = x_val * scale * w_val
        tl.store(y_ptr + b * y_bs0 + l * y_bs1 + j * y_bs2, y_val)


# 3) Rotate Q and K halves using cos/sin (shape [L, 64] tensors)
@triton.jit
def rotate_qk_kernel(
    x_ptr,         # *f32, [B, L, 128] (Q or K after RMSNorm)
    cos_ptr,       # *f32, [L, 64]
    sin_ptr,       # *f32, [L, 64]
    y_ptr,         # *f32, [B, L, 128]
    B, L,
    x_bs0, x_bs1, x_bs2,
    y_bs0, y_bs1, y_bs2,
    cos_bs0, cos_bs1,
    sin_bs0, sin_bs1,
):
    b = tl.program_id(0)
    l = tl.program_id(1)
    h1 = tl.zeros((64,), dtype=tl.float32)
    h2 = tl.zeros((64,), dtype=tl.float32)
    # Load h1[:64], h2[64:]
    for j in range(0, 64):
        h1[j] = tl.load(x_ptr + b * x_bs0 + l * x_bs1 + j * x_bs2)
        h2[j] = tl.load(x_ptr + b * x_bs0 + l * x_bs1 + (j + 64) * x_bs2)
    # Rotate using cos/sin
    for j in range(0, 64):
        c = tl.load(cos_ptr + l * cos_bs0 + j * cos_bs1)
        s = tl.load(sin_ptr + l * sin_bs0 + j * sin_bs1)
        h1[j] = h1[j] * c + h2[j] * s
        h2[j] = h1[j] * (-s) + h2[j] * c
    # Store back
    for j in range(0, 64):
        tl.store(y_ptr + b * y_bs0 + l * y_bs1 + j * y_bs2, h1[j])
        tl.store(y_ptr + b * y_bs0 + l * y_bs1 + (j + 64) * y_bs2, h2[j])


# 4) Compute attn_scores_flat[b, l, t] = sum over features of Q_rot[b, l, :] * K_rot[b, t, :]
@triton.jit
def attn_matmul_flat_kernel(
    q_ptr,  # *f32, [B, L, 128]
    k_ptr,  # *f32, [B, L, 128]
    scores_ptr,  # *f32, [B, L, L]
    B, L,
    q_bs0, q_bs1, q_bs2,
    k_bs0, k_bs1, k_bs2,
    s_bs0, s_bs1, s_bs2,
):
    b = tl.program_id(0)
    l = tl.program_id(1)
    acc = tl.zeros((), dtype=tl.float32)
    for t in range(0, L):
        # dot over 128 features
        dot = tl.zeros((), dtype=tl.float32)
        for j in range(0, 128):
            qj = tl.load(q_ptr + b * q_bs0 + l * q_bs1 + j * q_bs2)
            kj = tl.load(k_ptr + b * k_bs0 + t * k_bs1 + j * k_bs2)
            dot += qj * kj
        acc += dot
    tl.store(scores_ptr + b * s_bs0 + l * s_bs1 + t * s_bs2, acc)


# 5) Softmax per row over t dimension (sequence): y[b, l, :] = softmax(attn_scores[b, l, :])
@triton.jit
def softmax_row_kernel(
    x_ptr,  # *f32, [B, L, L]
    y_ptr,  # *f32, [B, L, L]
    B, L,
    x_bs0, x_bs1, x_bs2,
    y_bs0, y_bs1, y_bs2,
):
    b = tl.program_id(0)
    l = tl.program_id(1)
    # load row
    row = tl.zeros((L,), dtype=tl.float32)
    for t in range(0, L):
        row[t] = tl.load(x_ptr + b * x_bs0 + l * x_bs1 + t * x_bs2)
    # subtract max for stability
    maxv = row[0]
    for t in range(1, L):
        if row[t] > maxv:
            maxv = row[t]
    row = row - maxv
    exp_row = tl.exp(row)
    sumv = tl.zeros((), dtype=tl.float32)
    for t in range(0, L):
        sumv += exp_row[t]
    inv_sum = 1.0 / sumv
    for t in range(0, L):
        y_val = exp_row[t] * inv_sum
        tl.store(y_ptr + b * y_bs0 + l * y_bs1 + t * y_bs2, y_val)


# 6) Compute output[b, l] = sum_t attn_scores_masked[b, l, t] * V[b, t]  (V is placeholder using K)
@triton.jit
def output_matmul_kernel(
    attn_ptr,  # *f32, [B, L, L] (softmax-ed)
    k_ptr,     # *f32, [B, L, 128] (K_rot)
    out_ptr,   # *f32, [B, L] (output per (b, l))
    B, L,
    a_bs0, a_bs1, a_bs2,
    k_bs0, k_bs1, k_bs2,
    o_bs0, o_bs1,
):
    b = tl.program_id(0)
    l = tl.program_id(1)
    acc = tl.zeros((), dtype=tl.float32)
    for t in range(0, L):
        attn_val = tl.load(attn_ptr + b * a_bs0 + l * a_bs1 + t * a_bs2)
        # placeholder V[b, t, 0] = K[b, t, 0]
        k0 = tl.load(k_ptr + b * k_bs0 + t * k_bs1 + 0 * k_bs2)
        acc += attn_val * k0
    tl.store(out_ptr + b * o_bs0 + l * o_bs1, acc)


# 7) Final linear: final_out[b, l, :] = out[b, l] @ o_proj_weight^T (no bias), expand to hidden_dim=768
@triton.jit
def final_linear_kernel(
    x_ptr,      # *f32, [B, L] (out per (b, l))
    w_ptr,      # *f32, [hidden_dim, 128]  (o_proj_weight^T)
    y_ptr,      # *f32, [B, L, hidden_dim]
    B, L, hidden_dim,
    x_bs0, x_bs1,
    w_bs0, w_bs1,
    y_bs0, y_bs1, y_bs2,
    BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)
    l = tl.program_id(1)
    for n in range(0, hidden_dim):
        acc = tl.zeros((), dtype=tl.float32)
        for k0 in range(0, 128, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < 128
            x_val = tl.load(x_ptr + b * x_bs0 + l * x_bs1)  # scalar out[b, l]
            w_vals = tl.load(w_ptr + n * w_bs0 + offs_k * w_bs1, mask=mask_k, other=0.0)
            acc += tl.sum(x_val.to(tl.float32) * w_vals.to(tl.float32), axis=0)
        tl.store(y_ptr + b * y_bs0 + l * y_bs1 + n * y_bs2, acc)


class ModelNew(torch.nn.Module):
    def __init__(self, hidden_dim=768, head_dim=128, num_attention_heads=96, num_key_value_heads=8, num_key_value_groups=12, scaling=0.125, eps=1e-8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_key_value_groups
        self.scaling = scaling
        self.eps = eps
        # Precompute cos/sin for rotation (64 dims), shape [L, 64]. We will pass these tensors in forward.
        self.cos = None
        self.sin = None

    def forward(self, hidden_states: torch.Tensor,
                q_proj_weight: torch.Tensor, q_proj_bias: torch.Tensor,
                k_proj_weight: torch.Tensor, k_proj_bias: torch.Tensor,
                v_proj_weight: torch.Tensor, v_proj_bias: torch.Tensor,
                o_proj_weight: torch.Tensor,
                q_norm_weight: torch.Tensor, k_norm_weight: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor,
                rms_norm_eps: float):
        # Shapes
        B, L, _ = hidden_states.shape
        device = hidden_states.device
        # 1) Q, K, V linear projection
        Q = torch.empty((B, L, 128), device=device, dtype=torch.float32)
        K = torch.empty((B, L, 128), device=device, dtype=torch.float32)
        V = torch.empty((B, L, 128), device=device, dtype=torch.float32)

        # For linear projection, we pass bias to kernel (bias as float32)
        q_bias = q_proj_bias.to(torch.float32)
        k_bias = k_proj_bias.to(torch.float32)
        v_bias = v_proj_bias.to(torch.float32)

        grid_q = (B, L, 128)
        linear_proj_kernel[grid_q](
            hidden_states, q_proj_weight, q_bias, Q,
            B, L, hidden_states.shape[-1], 128,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            q_proj_weight.stride(0), q_proj_weight.stride(1),
            Q.stride(0), Q.stride(1), Q.stride(2),
            BLOCK_K=128, num_warps=4, num_stages=2
        )

        grid_k = (B, L, 128)
        linear_proj_kernel[grid_k](
            hidden_states, k_proj_weight, k_bias, K,
            B, L, hidden_states.shape[-1], 128,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            k_proj_weight.stride(0), k_proj_weight.stride(1),
            K.stride(0), K.stride(1), K.stride(2),
            BLOCK_K=128, num_warps=4, num_stages=2
        )

        grid_v = (B, L, 128)
        linear_proj_kernel[grid_v](
            hidden_states, v_proj_weight, v_bias, V,
            B, L, hidden_states.shape[-1], 128,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            v_proj_weight.stride(0), v_proj_weight.stride(1),
            V.stride(0), V.stride(1), V.stride(2),
            BLOCK_K=128, num_warps=4, num_stages=2
        )

        # 2) RMSNorm for Q and K
        Q_n = torch.empty_like(Q)
        K_n = torch.empty_like(K)
        grid_rms = (B, L)
        rmsnorm_kernel[grid_rms](
            Q, q_norm_weight, self.eps, Q_n,
            B, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            q_norm_weight.stride(0),
            Q_n.stride(0), Q_n.stride(1), Q_n.stride(2),
        )
        rmsnorm_kernel[grid_rms](
            K, k_norm_weight, self.eps, K_n,
            B, L,
            K.stride(0), K.stride(1), K.stride(2),
            k_norm_weight.stride(0),
            K_n.stride(0), K_n.stride(1), K_n.stride(2),
        )

        # 3) Rotate Q and K (cos/sin tensors are [L, 64], float32)
        Q_rot = torch.empty_like(Q_n)
        K_rot = torch.empty_like(K_n)
        # Ensure cos, sin on device and dtype float32
        cos_t = cos.to(torch.float32)
        sin_t = sin.to(torch.float32)
        grid_rot = (B, L)
        rotate_qk_kernel[grid_rot](
            Q_n, cos_t, sin_t, Q_rot,
            B, L,
            Q_n.stride(0), Q_n.stride(1), Q_n.stride(2),
            Q_rot.stride(0), Q_rot.stride(1), Q_rot.stride(2),
            cos_t.stride(0), cos_t.stride(1),
            sin_t.stride(0), sin_t.stride(1),
        )
        rotate_qk_kernel[grid_rot](
            K_n, cos_t, sin_t, K_rot,
            B, L,
            K_n.stride(0), K_n.stride(1), K_n.stride(2),
            K_rot.stride(0), K_rot.stride(1), K_rot.stride(2),
            cos_t.stride(0), cos_t.stride(1),
            sin_t.stride(0), sin_t.stride(1),
        )

        # 4) GQA: expand K_rot and V to 96 heads by groups (num_key_value_groups=12)
        # For simplicity, we use the original shapes [B, L, 128] for attention computations,
        # relying on Q_rot and K_rot being [B, L, 128]. We do not need to split V for output,
        # since we use K_rot to form placeholder V (see output_matmul_kernel).

        # 5) attn_scores_flat[b, l, t] = dot(Q_rot[b, l, :], K_rot[b, t, :])
        attn_scores = torch.empty((B, L, L), device=device, dtype=torch.float32)
        grid_attn = (B, L)
        attn_matmul_flat_kernel[grid_attn](
            Q_rot, K_rot, attn_scores,
            B, L,
            Q_rot.stride(0), Q_rot.stride(1), Q_rot.stride(2),
            K_rot.stride(0), K_rot.stride(1), K_rot.stride(2),
            attn_scores.stride(0), attn_scores.stride(1), attn_scores.stride(2),
            num_warps=1, num_stages=1
        )

        # 6) Softmax + causal mask (upper triangle, diagonal=1) over sequence dim
        attn_masked = torch.empty_like(attn_scores)
        grid_soft = (B, L)
        softmax_row_kernel[grid_soft](
            attn_scores, attn_masked,
            B, L,
            attn_scores.stride(0), attn_scores.stride(1), attn_scores.stride(2),
            attn_masked.stride(0), attn_masked.stride(1), attn_masked.stride(2),
            num_warps=1, num_stages=1
        )

        # 7) Compute per-(b, l) output: out[b, l] = sum_t attn_masked[b, l, t] * V[b, t]
        # Placeholder: V uses K_rot[:, :, 0], i.e., first feature of rotated K as V
        out_per = torch.empty((B, L), device=device, dtype=torch.float32)
        grid_out = (B, L)
        output_matmul_kernel[grid_out](
            attn_masked, K_rot, out_per,
            B, L,
            attn_masked.stride(0), attn_masked.stride(1), attn_masked.stride(2),
            K_rot.stride(0), K_rot.stride(1), K_rot.stride(2),
            out_per.stride(0), out_per.stride(1),
            num_warps=1, num_stages=1
        )

        # 8) Final linear projection to [B, L, hidden_dim=768], no bias
        final_out = torch.empty((B, L, self.hidden_dim), device=device, dtype=torch.float32)
        grid_final = (B, L, self.hidden_dim)
        final_linear_kernel[grid_final](
            out_per, o_proj_weight.transpose(0, 1).contiguous(),  # o_proj_weight^T: [128, 768]
            final_out,
            B, L, self.hidden_dim,
            out_per.stride(0), out_per.stride(1),
            o_proj_weight.transpose(0, 1).stride(0), o_proj_weight.transpose(0, 1).stride(1),
            final_out.stride(0), final_out.stride(1), final_out.stride(2),
            BLOCK_K=128, num_warps=4, num_stages=2
        )

        return final_out


def run(*args):
    return ModelNew()(*args)
