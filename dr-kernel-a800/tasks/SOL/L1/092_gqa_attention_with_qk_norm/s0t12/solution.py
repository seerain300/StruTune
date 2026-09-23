import torch
import triton
import triton.language as tl

# 1) Linear projection: y[b, l, n] = sum_k x[b, l, k] * w[n, k], accumulate in f32, output f32
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

        # cast to f32 for accumulate
        x_vals = x_vals.to(tl.float32)
        w_vals = w_vals.to(tl.float32)

        acc += tl.sum(x_vals * w_vals, axis=0)

    # add bias
    bval = tl.load(bias_ptr + n)
    acc += bval  # bias is f32

    tl.store(y_ptr + b * y_bs0 + l * y_bs1 + n * y_bs2, acc)


# 2) RMSNorm per (b, l): y = x * rsqrt(mean(x^2) + eps) * weight
@triton.jit
def rmsnorm_kernel(
    x_ptr,   # *f32, [B, L, H]
    w_ptr,   # *f32, [H]
    y_ptr,   # *f32, [B, L, H]
    B, L, H,
    x_bs0, x_bs1, x_bs2,
    w_bs0,  # scalar (H), not used in indexing
    y_bs0, y_bs1, y_bs2,
    eps: tl.constexpr,
):
    b = tl.program_id(0)
    l = tl.program_id(1)
    h = tl.program_id(2)

    x_vec = tl.load(x_ptr + b * x_bs0 + l * x_bs1 + h * x_bs2)  # scalar
    mean = tl.sum(x_vec * x_vec, axis=0) / H
    inv = 1.0 / tl.sqrt(mean + eps)
    w_val = tl.load(w_ptr + h)  # scalar
    y_val = x_vec * inv * w_val
    tl.store(y_ptr + b * y_bs0 + l * y_bs1 + h * y_bs2, y_val)


# 3) Rotate Q/K: split head_dim/2
@triton.jit
def rotate_half_kernel(
    inp_ptr,  # *f32, [B, L, H]
    cos_ptr,  # *f32, [L, H/2]
    sin_ptr,  # *f32, [L, H/2]
    out_ptr,  # *f32, [B, L, H]
    B, L, H,
    inp_bs0, inp_bs1, inp_bs2,
    cos_bs0, cos_bs1,
    sin_bs0, sin_bs1,
    out_bs0, out_bs1, out_bs2,
    BLOCK_H: tl.constexpr,
):
    b = tl.program_id(0)
    l = tl.program_id(1)
    h = tl.program_id(2)

    half = H // 2
    if h < half:
        # h1: use cos
        in_val = tl.load(inp_ptr + b * inp_bs0 + l * inp_bs1 + h * inp_bs2)
        c = tl.load(cos_ptr + l * cos_bs0 + h * cos_bs1)
        s = tl.load(sin_ptr + l * sin_bs0 + h * sin_bs1)
        out_val = in_val * c + (tl.load(inp_ptr + b * inp_bs0 + l * inp_bs1 + (h + half) * inp_bs2)) * s
    else:
        # h2: use -sin
        in_val = tl.load(inp_ptr + b * inp_bs0 + l * inp_bs1 + h * inp_bs2)
        c = tl.load(cos_ptr + l * cos_bs0 + (h - half) * cos_bs1)
        s = tl.load(sin_ptr + l * sin_bs0 + (h - half) * sin_bs1)
        out_val = in_val * c - (tl.load(inp_ptr + b * inp_bs0 + l * inp_bs1 + (h - half) * inp_bs2)) * s

    tl.store(out_ptr + b * out_bs0 + l * out_bs1 + h * out_bs2, out_val)


# 4) Attn score matmul: attn_scores[b, qh, l, t] = Q[b, qh, l] * K[b, qh, t]
@triton.jit
def attn_score_kernel(
    Q_ptr, K_ptr,  # *f32, [B, num_heads, L, H]
    attn_ptr,      # *f32, [B, num_heads, L, L]
    B, num_heads, L, H,
    Q_bs0, Q_bs1, Q_bs2, Q_bs3,
    K_bs0, K_bs1, K_bs2, K_bs3,
    attn_bs0, attn_bs1, attn_bs2, attn_bs3,
):
    # Grid = (B, num_heads, L) -> each program handles one (b, qh, l)
    b = tl.program_id(0)
    qh = tl.program_id(1)
    l = tl.program_id(2)

    acc = tl.zeros((), dtype=tl.float32)
    # t in [0, L-1]
    for t in range(0, L):
        q = tl.load(Q_ptr + b * Q_bs0 + qh * Q_bs1 + l * Q_bs2 + 0 * Q_bs3)
        k = tl.load(K_ptr + b * K_bs0 + qh * K_bs1 + t * K_bs2 + 0 * K_bs3)
        acc += q * k

    # scale by 1/sqrt(H)
    # store attn_scores[b, qh, l, t]
    tl.store(attn_ptr + b * attn_bs0 + qh * attn_bs1 + l * attn_bs2 + t * attn_bs3, acc)


# 5) Softmax with causal mask (upper triangle, diag=1) over sequence dim t
@triton.jit
def softmax_mask_kernel(
    attn_in_ptr,    # *f32, [B, num_heads, L, L]
    attn_out_ptr,   # *f32, [B, num_heads, L, L]
    B, num_heads, L,
    attn_in_bs0, attn_in_bs1, attn_in_bs2, attn_in_bs3,
    attn_out_bs0, attn_out_bs1, attn_out_bs2, attn_out_bs3,
):
    # Grid = (B, num_heads, L) -> each program handles one row l of each (b, qh)
    b = tl.program_id(0)
    qh = tl.program_id(1)
    l = tl.program_id(2)

    # Initialize row max
    max_val = -float('inf')
    for t in range(0, L):
        ptr = attn_in_ptr + b * attn_in_bs0 + qh * attn_in_bs1 + l * attn_in_bs2 + t * attn_in_bs3
        val = tl.load(ptr)
        if val > max_val:
            max_val = val

    # Apply mask: set lower-triangular (t < l) to -inf
    for t in range(0, L):
        ptr_in = attn_in_ptr + b * attn_in_bs0 + qh * attn_in_bs1 + l * attn_in_bs2 + t * attn_in_bs3
        val = tl.load(ptr_in)
        if t < l:
            val = -float('inf')
        # softmax: exp(val - max), sum, normalize
        exp_val = tl.exp(val - max_val)
        sum_val = 0.0
        for tt in range(0, L):
            v = tl.load(attn_in_ptr + b * attn_in_bs0 + qh * attn_in_bs1 + l * attn_in_bs2 + tt * attn_in_bs3)
            if tt < l:
                v = -float('inf')
            sum_val += tl.exp(v - max_val)
        out = exp_val / sum_val
        ptr_out = attn_out_ptr + b * attn_out_bs0 + qh * attn_out_bs1 + l * attn_out_bs2 + t * attn_out_bs3
        tl.store(ptr_out, out)


# 6) Output matmul: attn_output[b, qh, l] = sum_t attn[b, qh, l, t] * V[b, qh, t]
@triton.jit
def output_matmul_kernel(
    attn_ptr, V_ptr, out_ptr,
    B, num_heads, L, H,
    attn_bs0, attn_bs1, attn_bs2, attn_bs3,
    V_bs0, V_bs1, V_bs2,
    out_bs0, out_bs1, out_bs2,
):
    # Grid = (B, num_heads, L)
    b = tl.program_id(0)
    qh = tl.program_id(1)
    l = tl.program_id(2)

    acc = tl.zeros((), dtype=tl.float32)
    for t in range(0, L):
        attn_val = tl.load(attn_ptr + b * attn_bs0 + qh * attn_bs1 + l * attn_bs2 + t * attn_bs3)
        V_val = tl.load(V_ptr + b * V_bs0 + qh * V_bs1 + t * V_bs2)
        acc += attn_val * V_val

    tl.store(out_ptr + b * out_bs0 + qh * out_bs1 + l * out_bs2, acc)


# 7) Final linear: y[b, l, n] = sum_k x[b, l, k] * w[n, k], output f32
@triton.jit
def final_linear_kernel(
    x_ptr, w_ptr, y_ptr,
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

        x_vals = tl.load(x_ptrs, mask=mask_k, other=0.0).to(tl.float32)
        w_vals = tl.load(w_ptrs, mask=mask_k, other=0.0).to(tl.float32)

        acc += tl.sum(x_vals * w_vals, axis=0)

    tl.store(y_ptr + b * y_bs0 + l * y_bs1 + n * y_bs2, acc)


class ModelNew:
    def __init__(self, hidden_dim=768):
        self.hidden_dim = hidden_dim  # output dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_proj_weight: torch.Tensor,
        q_proj_bias: torch.Tensor,
        k_proj_weight: torch.Tensor,
        k_proj_bias: torch.Tensor,
        v_proj_weight: torch.Tensor,
        v_proj_bias: torch.Tensor,
        o_proj_weight: torch.Tensor,    # [hidden_dim, 12288] from original output projection
        q_norm_weight: torch.Tensor,    # [128]
        k_norm_weight: torch.Tensor,    # [128]
        cos: torch.Tensor,              # [L, 64], fp32
        sin: torch.Tensor,              # [L, 64], fp32
        rms_norm_eps: float,
    ):
        # Shapes from original model: hidden_states [B, L, H] with H=128
        B, L, H = hidden_states.shape  # L varies across workloads, H=128
        num_attention_heads = 96
        num_key_value_heads = 8
        num_key_value_groups = 12

        # Ensure device and dtype
        device = hidden_states.device
        dtype = hidden_states.dtype  # keep input dtype; we can cast in kernels to f32 for compute

        # 1) Linear projections for Q, K, V
        # Cast weights to f32 for accumulation; keep outputs in f32
        Q = torch.empty((B, L, H), device=device, dtype=torch.float32)
        K = torch.empty((B, L, H), device=device, dtype=torch.float32)
        V = torch.empty((B, L, H), device=device, dtype=torch.float32)

        # q_proj
        grid_q = (B, L, H)
        linear_proj_kernel[grid_q](
            hidden_states, q_proj_weight, q_proj_bias, Q,
            B, L, H, H,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            q_proj_weight.stride(0), q_proj_weight.stride(1),
            Q.stride(0), Q.stride(1), Q.stride(2),
            BLOCK_K=64, num_warps=4, num_stages=2
        )

        # k_proj
        grid_k = (B, L, H)
        linear_proj_kernel[grid_k](
            hidden_states, k_proj_weight, k_proj_bias, K,
            B, L, H, H,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            k_proj_weight.stride(0), k_proj_weight.stride(1),
            K.stride(0), K.stride(1), K.stride(2),
            BLOCK_K=64, num_warps=4, num_stages=2
        )

        # v_proj
        grid_v = (B, L, H)
        linear_proj_kernel[grid_v](
            hidden_states, v_proj_weight, v_proj_bias, V,
            B, L, H, H,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            v_proj_weight.stride(0), v_proj_weight.stride(1),
            V.stride(0), V.stride(1), V.stride(2),
            BLOCK_K=64, num_warps=4, num_stages=2
        )

        # 2) RMSNorm for Q and K
        # Q
        Q_n = torch.empty_like(Q)  # [B, L, H], f32
        grid_qnorm = (B, L)
        rmsnorm_kernel[grid_qnorm](
            Q, q_norm_weight, Q_n,
            B, L, H,
            Q.stride(0), Q.stride(1), Q.stride(2),
            q_norm_weight.stride(0),
            Q_n.stride(0), Q_n.stride(1), Q_n.stride(2),
            eps=rms_norm_eps
        )
        Q = Q_n

        # K
        K_n = torch.empty_like(K)  # [B, L, H], f32
        grid_knorm = (B, L)
        rmsnorm_kernel[grid_knorm](
            K, k_norm_weight, K_n,
            B, L, H,
            K.stride(0), K.stride(1), K.stride(2),
            k_norm_weight.stride(0),
            K_n.stride(0), K_n.stride(1), K_n.stride(2),
            eps=rms_norm_eps
        )
        K = K_n

        # 3) Rotate Q and K (head_dim=128, split 64)
        Qr = torch.empty_like(Q)  # rotated Q
        Kr = torch.empty_like(K)  # rotated K

        grid_rot_q = (B, L)
        rotate_half_kernel[grid_rot_q](
            Q, cos, sin, Qr,
            B, L, H,
            Q.stride(0), Q.stride(1), Q.stride(2),
            cos.stride(0), cos.stride(1),
            sin.stride(0), sin.stride(1),
            Qr.stride(0), Qr.stride(1), Qr.stride(2),
            BLOCK_H=64, num_warps=4, num_stages=2
        )

        grid_rot_k = (B, L)
        rotate_half_kernel[grid_rot_k](
            K, cos, sin, Kr,
            B, L, H,
            K.stride(0), K.stride(1), K.stride(2),
            cos.stride(0), cos.stride(1),
            sin.stride(0), sin.stride(1),
            Kr.stride(0), Kr.stride(1), Kr.stride(2),
            BLOCK_H=64, num_warps=4, num_stages=2
        )

        # 4) GQA: expand K/V to 96 heads by groups, then reshape to [B, 96, L, H]
        # Note: original code expands K/V to 96 heads without per-head projection.
        K96 = K[:, :, None, :].expand(B, num_key_value_heads, num_key_value_groups, L, H).reshape(B, num_key_value_heads * num_key_value_groups, L, H)  # [B, 96, L, H]
        V96 = V[:, :, None, :].expand(B, num_key_value_heads, num_key_value_groups, L, H).reshape(B, num_key_value_heads * num_key_value_groups, L, H)

        # Qr already [B, L, H]; expand to [B, 96, L, H]
        Q96 = Qr[:, :, None, :].expand(B, num_attention_heads, L, H).reshape(B, num_attention_heads, L, H)

        # 5) Compute attention scores [B, num_heads, L, L]
        attn = torch.empty((B, num_attention_heads, L, L), device=device, dtype=torch.float32)

        # Launch attn score kernel: grid over (B, num_heads, L)
        grid_attn = (B, num_attention_heads, L)
        attn_score_kernel[grid_attn](
            Q96, K96, attn,
            B, num_attention_heads, L, H,
            Q96.stride(0), Q96.stride(1), Q96.stride(2), Q96.stride(3),
            K96.stride(0), K96.stride(1), K96.stride(2), K96.stride(3),
            attn.stride(0), attn.stride(1), attn.stride(2), attn.stride(3),
        )

        # 6) Softmax with causal mask (upper triangle, diag=1)
        attn_out = torch.empty_like(attn)

        grid_softmax = (B, num_attention_heads, L)
        softmax_mask_kernel[grid_softmax](
            attn, attn_out,
            B, num_attention_heads, L,
            attn.stride(0), attn.stride(1), attn.stride(2), attn.stride(3),
            attn_out.stride(0), attn_out.stride(1), attn_out.stride(2), attn_out.stride(3),
        )

        # 7) Output matmul: attn_output[b, qh, l] = sum_t attn_out[b, qh, l, t] * V96[b, qh, t]
        attn_output = torch.empty((B, num_attention_heads, L, H), device=device, dtype=torch.float32)

        grid_output = (B, num_attention_heads, L)
        output_matmul_kernel[grid_output](
            attn_out, V96, attn_output,
            B, num_attention_heads, L, H,
            attn_out.stride(0), attn_out.stride(1), attn_out.stride(2), attn_out.stride(3),
            V96.stride(0), V96.stride(1), V96.stride(2),
            attn_output.stride(0), attn_output.stride(1), attn_output.stride(2),
        )

        # 8) Final linear projection to [B, L, hidden_dim]
        attn_output_flat = attn_output.reshape(B, L, num_attention_heads * H)  # [B, L, 12288]
        final_out = torch.empty((B, L, self.hidden_dim), device=device, dtype=torch.float32)

        grid_final = (B, L, self.hidden_dim)
        final_linear_kernel[grid_final](
            attn_output_flat, o_proj_weight, final_out,
            B, L, num_attention_heads * H, self.hidden_dim,
            attn_output_flat.stride(0), attn_output_flat.stride(1), attn_output_flat.stride(2),
            o_proj_weight.stride(0), o_proj_weight.stride(1),
            final_out.stride(0), final_out.stride(1), final_out.stride(2),
            BLOCK_K=64, num_warps=4, num_stages=2
        )

        return final_out


def run(*args):
    return ModelNew()(*args)
