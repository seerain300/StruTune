import torch
import triton
import triton.language as tl


# 1) Linear projection: y[b, l, n] = sum_k x[b, l, k] * w[n, k], accumulate in f32, output f32
@triton.jit
def linear_proj_kernel(
    x_ptr,           # *f16/f32/bf16, [B, L, H_in]
    w_ptr,           # *f16/f32/bf16, [N_out, H_in]
    bias_ptr,        # *f32 (ignored if None), [N_out]
    y_ptr,           # *f32, [B, L, N_out]
    B, L, H_in, N_out,
    x_bs0, x_bs1, x_bs2,
    w_bs0, w_bs1,
    y_bs0, y_bs1, y_bs2,
    BLOCK_K: tl.constexpr,
):
    # Each program computes one output element y[b, l, n]
    b = tl.program_id(0)
    l = tl.program_id(1)
    n = tl.program_id(2)

    acc = tl.zeros((), dtype=tl.float32)

    # Loop over input features H_in in chunks
    for k0 in range(0, H_in, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < H_in

        # Load x[b, l, offs_k] -> vector
        x_ptrs = x_ptr + b * x_bs0 + l * x_bs1 + offs_k * x_bs2
        x_vals = tl.load(x_ptrs, mask=mask_k, other=0.0).to(tl.float32)

        # Load w[n, offs_k] -> vector
        w_ptrs = w_ptr + n * w_bs0 + offs_k * w_bs1
        w_vals = tl.load(w_ptrs, mask=mask_k, other=0.0).to(tl.float32)

        # Accumulate dot product for this chunk
        acc += tl.sum(x_vals * w_vals, axis=0)

    # Store result to y[b, l, n]
    y_ptrs = y_ptr + b * y_bs0 + l * y_bs1 + n * y_bs2
    tl.store(y_ptrs, acc)


# 2) RMSNorm per (b, l, h): y = x * rsqrt(mean(x^2) + eps) * weight
@triton.jit
def rmsnorm_kernel(
    x_ptr,          # *f32, [B, L, H]
    weight_ptr,     # *f32, [H]
    y_ptr,          # *f32, [B, L, H]
    B, L, H,
    x_bs0, x_bs1, x_bs2,
    w_bs0,
    y_bs0, y_bs1, y_bs2,
    eps,            # float32
    BLOCK_H: tl.constexpr
):
    # One program per (b, l, h)
    b = tl.program_id(0)
    l = tl.program_id(1)
    h = tl.program_id(2)

    # Compute variance over last dim for this (b, l)
    var = tl.zeros((), dtype=tl.float32)
    for k in range(0, H, BLOCK_H):
        offs = k + tl.arange(0, BLOCK_H)
        mask = offs < H
        x_ptrs = x_ptr + b * x_bs0 + l * x_bs1 + offs * x_bs2
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        var += tl.sum(x_vals * x_vals, axis=0)
    var = var / H
    scale = tl.rsqrt(var + eps)

    # Apply normalization and weight
    for k in range(0, H, BLOCK_H):
        offs = k + tl.arange(0, BLOCK_H)
        mask = offs < H
        x_ptrs = x_ptr + b * x_bs0 + l * x_bs1 + offs * x_bs2
        w_ptrs = weight_ptr + offs * w_bs0
        y_ptrs = y_ptr + b * y_bs0 + l * y_bs1 + offs * y_bs2
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        w_vals = tl.load(w_ptrs, mask=mask, other=1.0).to(tl.float32)
        y_vals = x_vals * scale * w_vals
        tl.store(y_ptrs, y_vals)


# 3) Rotate Q/K using sin/cos (split rotation)
# rotate_qk_kernel expects: x [B, L, H], cos [L, H//2], sin [L, H//2], out [B, L, H]
@triton.jit
def rotate_qk_kernel(
    x_ptr,      # *f32, [B, L, H]
    cos_ptr,    # *f32, [L, H//2]
    sin_ptr,    # *f32, [L, H//2]
    out_ptr,    # *f32, [B, L, H]
    B, L, H,
    x_bs0, x_bs1, x_bs2,
    cos_bs0, cos_bs1,
    sin_bs0, sin_bs1,
    out_bs0, out_bs1, out_bs2,
    BLOCK_H: tl.constexpr
):
    b = tl.program_id(0)
    l = tl.program_id(1)
    h = tl.program_id(2)

    # H is 128; we operate in chunks
    for k in range(0, H, BLOCK_H):
        offs = k + tl.arange(0, BLOCK_H)
        mask = offs < H

        x_ptrs = x_ptr + b * x_bs0 + l * x_bs1 + offs * x_bs2
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

        # split
        h1 = x_vals[:64]
        h2 = x_vals[64:]

        # load cos/sin for h1 index
        cos_ptrs = cos_ptr + l * cos_bs0 + tl.arange(0, 64) * cos_bs1
        sin_ptrs = sin_ptr + l * sin_bs0 + tl.arange(0, 64) * sin_bs1
        cos_vals = tl.load(cos_ptrs).to(tl.float32)
        sin_vals = tl.load(sin_ptrs).to(tl.float32)

        # rotated = cat(-h2, h1) * cos + h1 * sin
        rotated = tl.concatenate([-h2, h1], axis=0) * cos_vals + h1 * sin_vals

        out_ptrs = out_ptr + b * out_bs0 + l * out_bs1 + offs * out_bs2
        tl.store(out_ptrs, rotated, mask=mask)


# 4) Attention matmul: attn_scores[b, qh, l, t] = Q[b, qh, l] * K[b, qh, t]
@triton.jit
def attn_matmul_kernel(
    Q_ptr,  # *f32, [B, NH, L, H]
    K_ptr,  # *f32, [B, NH, L, H]
    attn_ptr,  # *f32, [B, NH, L, L]
    B, NH, L, H,
    Q_bs0, Q_bs1, Q_bs2, Q_bs3,
    K_bs0, K_bs1, K_bs2, K_bs3,
    attn_bs0, attn_bs1, attn_bs2, attn_bs3,
):
    # grid = (B, NH, L) -> each program computes one row (l) and writes L outputs
    b = tl.program_id(0)
    qh = tl.program_id(1)
    l_row = tl.program_id(2)

    # compute Q row: Q[b, qh, l_row]
    Q_row_ptrs = Q_ptr + b * Q_bs0 + qh * Q_bs1 + l_row * Q_bs2  # + 0 * Q_bs3
    # Q_row is a vector of length H
    Q_row = tl.load(Q_row_ptrs + tl.arange(0, H) * Q_bs3, mask=tl.arange(0, H) < H, other=0.0).to(tl.float32)

    # compute dot with each K column
    for t in range(0, L):
        K_col_ptrs = K_ptr + b * K_bs0 + qh * K_bs1 + t * K_bs2  # + 0 * K_bs3
        K_col = tl.load(K_col_ptrs + tl.arange(0, H) * K_bs3, mask=tl.arange(0, H) < H, other=0.0).to(tl.float32)
        score = tl.sum(Q_row * K_col, axis=0)
        attn_ptrs = attn_ptr + b * attn_bs0 + qh * attn_bs1 + l_row * attn_bs2 + t * attn_bs3
        tl.store(attn_ptrs, score)


# 5) Softmax + causal mask (upper triangle, diagonal=1)
@triton.jit
def softmax_mask_kernel(
    attn_ptr,        # *f32, [B, NH, L, L]
    mask_ptr,        # *f32, [B, NH, L, L] (already filled with -inf/0)
    out_ptr,         # *f32, [B, NH, L, L]
    B, NH, L,
    attn_bs0, attn_bs1, attn_bs2, attn_bs3,
    mask_bs0, mask_bs1, mask_bs2, mask_bs3,
    out_bs0, out_bs1, out_bs2, out_bs3,
):
    # grid = (B, NH, L) -> each program computes softmax over the last dim (L) for a given (b, nh, l_row)
    b = tl.program_id(0)
    nh = tl.program_id(1)
    l_row = tl.program_id(2)

    # 1) find max
    max_val = tl.zeros((), dtype=tl.float32)
    for t in range(0, L):
        attn_ptrs = attn_ptr + b * attn_bs0 + nh * attn_bs1 + l_row * attn_bs2 + t * attn_bs3
        val = tl.load(attn_ptrs)
        if t == 0:
            max_val = val
        else:
            max_val = tl.maximum(max_val, val)

    # 2) subtract max and exp
    sum_exp = tl.zeros((), dtype=tl.float32)
    for t in range(0, L):
        attn_ptrs = attn_ptr + b * attn_bs0 + nh * attn_bs1 + l_row * attn_bs2 + t * attn_bs3
        val = tl.load(attn_ptrs)
        e = tl.exp(val - max_val)
        # apply mask: load mask and set e to 0 where mask is -inf
        mask_ptrs = mask_ptr + b * mask_bs0 + nh * mask_bs1 + l_row * mask_bs2 + t * mask_bs3
        m = tl.load(mask_ptrs)  # m is -inf or 0
        e = tl.where(m == -float('inf'), 0.0, e)
        sum_exp += e

    inv_sum = 1.0 / sum_exp

    # 3) write normalized output
    for t in range(0, L):
        attn_ptrs = attn_ptr + b * attn_bs0 + nh * attn_bs1 + l_row * attn_bs2 + t * attn_bs3
        val = tl.load(attn_ptrs)
        e = tl.exp(val - max_val)
        mask_ptrs = mask_ptr + b * mask_bs0 + nh * mask_bs1 + l_row * mask_bs2 + t * mask_bs3
        m = tl.load(mask_ptrs)  # -inf or 0
        e = tl.where(m == -float('inf'), 0.0, e)
        out_ptrs = out_ptr + b * out_bs0 + nh * out_bs1 + l_row * out_bs2 + t * out_bs3
        tl.store(out_ptrs, e * inv_sum)


# 6) Output matmul: attn_output[b, qh, l] = sum_t attn_probs[b, qh, l, t] * V[b, qh, t]
@triton.jit
def output_matmul_kernel(
    attn_ptr,   # *f32, [B, NH, L, L]
    V_ptr,      # *f32, [B, NH, L, H]
    out_ptr,    # *f32, [B, NH, L, H] (we'll store into final output later as [B, L, H_flat])
    B, NH, L, H,
    attn_bs0, attn_bs1, attn_bs2, attn_bs3,
    V_bs0, V_bs1, V_bs2, V_bs3,
    out_bs0, out_bs1, out_bs2, out_bs3,
):
    # grid = (B, NH, L) -> each program computes one row reduction
    b = tl.program_id(0)
    qh = tl.program_id(1)
    l_row = tl.program_id(2)

    acc = tl.zeros((H,), dtype=tl.float32)

    for t in range(0, L):
        attn_ptrs = attn_ptr + b * attn_bs0 + qh * attn_bs1 + l_row * attn_bs2 + t * attn_bs3
        attn_val = tl.load(attn_ptrs).to(tl.float32)

        V_col_ptrs = V_ptr + b * V_bs0 + qh * V_bs1 + t * V_bs2  # + 0 * V_bs3
        V_col = tl.load(V_col_ptrs + tl.arange(0, H) * V_bs3, mask=tl.arange(0, H) < H, other=0.0).to(tl.float32)

        acc += attn_val * V_col

    out_ptrs = out_ptr + b * out_bs0 + qh * out_bs1 + l_row * out_bs2  # + 0 * out_bs3
    tl.store(out_ptrs + tl.arange(0, H) * out_bs3, acc, mask=tl.arange(0, H) < H)


# 7) Final linear projection: final_out[b, l, n] = sum_k out[b, l, k] * w[n, k]
@triton.jit
def final_linear_kernel(
    x_ptr,           # *f32, [B, L, H_out] (we pass attn_output_flat here)
    w_ptr,           # *f32, [N_out, H_out] = [hidden_dim, 12288]
    y_ptr,           # *f32, [B, L, N_out]
    B, L, H_out, N_out,
    x_bs0, x_bs1, x_bs2,
    w_bs0, w_bs1,
    y_bs0, y_bs1, y_bs2,
    BLOCK_K: tl.constexpr,
):
    # Each program computes one output element y[b, l, n]
    b = tl.program_id(0)
    l = tl.program_id(1)
    n = tl.program_id(2)

    acc = tl.zeros((), dtype=tl.float32)

    # Loop over H_out in chunks
    for k0 in range(0, H_out, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < H_out

        x_ptrs = x_ptr + b * x_bs0 + l * x_bs1 + offs_k * x_bs2
        x_vals = tl.load(x_ptrs, mask=mask_k, other=0.0).to(tl.float32)

        w_ptrs = w_ptr + n * w_bs0 + offs_k * w_bs1
        w_vals = tl.load(w_ptrs, mask=mask_k, other=0.0).to(tl.float32)

        acc += tl.sum(x_vals * w_vals, axis=0)

    y_ptrs = y_ptr + b * y_bs0 + l * y_bs1 + n * y_bs2
    tl.store(y_ptrs, acc)


class ModelNew(torch.nn.Module):
    def __init__(self, hidden_dim: int = 768, rms_norm_eps: float = 1e-6):
        super().__init__()
        # Store parameters to match the original signature
        self.hidden_dim = hidden_dim
        self.rms_norm_eps = rms_norm_eps

        # The original code passes q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight, q_norm_weight, k_norm_weight, cos, sin.
        # We will define placeholders as nn.Parameter to satisfy __init__ signature, but they will be set later via model.hook or constructor.
        # Here, we assume they are provided to forward. We'll keep them as buffers/parameters in the module for consistency.

    def forward(self,
                hidden_states: torch.Tensor,
                q_proj_weight: torch.Tensor,
                q_proj_bias: torch.Tensor,
                k_proj_weight: torch.Tensor,
                k_proj_bias: torch.Tensor,
                v_proj_weight: torch.Tensor,
                v_proj_bias: torch.Tensor,
                o_proj_weight: torch.Tensor,           # [hidden_dim, num_heads*head_dim] = [768, 12288]
                q_norm_weight: torch.Tensor,          # [128]
                k_norm_weight: torch.Tensor,          # [128]
                cos: torch.Tensor,                    # [L, head_dim//2] = [L, 64]
                sin: torch.Tensor,                    # [L, head_dim//2] = [L, 64]
                ):
        # Ensure device and dtype for Triton
        device = hidden_states.device
        dtype = hidden_states.dtype
        B, L, H_in = hidden_states.shape
        head_dim = 128
        num_attention_heads = 96
        num_key_value_heads = 8
        num_key_value_groups = 12
        scaling = head_dim ** -0.5

        # 1) Q, K, V projection (linear) using Triton kernel
        Q = torch.empty((B, L, head_dim), device=device, dtype=torch.float32)
        K = torch.empty((B, L, head_dim), device=device, dtype=torch.float32)
        V = torch.empty((B, L, head_dim), device=device, dtype=torch.float32)

        # Cast weights to float32 for kernel computation
        q_w = q_proj_weight.to(torch.float32)
        q_b = q_proj_bias.to(torch.float32) if q_proj_bias is not None else torch.zeros(head_dim, device=device, dtype=torch.float32)
        k_w = k_proj_weight.to(torch.float32)
        k_b = k_proj_bias.to(torch.float32) if k_proj_bias is not None else torch.zeros(head_dim, device=device, dtype=torch.float32)
        v_w = v_proj_weight.to(torch.float32)
        v_b = v_proj_bias.to(torch.float32) if v_proj_bias is not None else torch.zeros(head_dim, device=device, dtype=torch.float32)

        # Launch Q projection
        grid_q = (B, L, head_dim)
        linear_proj_kernel[grid_q](
            hidden_states.to(torch.float32), q_w, q_b, Q,
            B, L, H_in, head_dim,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            q_w.stride(0), q_w.stride(1),
            Q.stride(0), Q.stride(1), Q.stride(2),
            BLOCK_K=128, num_warps=4, num_stages=2
        )

        # Launch K projection
        grid_k = (B, L, head_dim)
        linear_proj_kernel[grid_k](
            hidden_states.to(torch.float32), k_w, k_b, K,
            B, L, H_in, head_dim,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            k_w.stride(0), k_w.stride(1),
            K.stride(0), K.stride(1), K.stride(2),
            BLOCK_K=128, num_warps=4, num_stages=2
        )

        # Launch V projection
        grid_v = (B, L, head_dim)
        linear_proj_kernel[grid_v](
            hidden_states.to(torch.float32), v_w, v_b, V,
            B, L, H_in, head_dim,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            v_w.stride(0), v_w.stride(1),
            V.stride(0), V.stride(1), V.stride(2),
            BLOCK_K=128, num_warps=4, num_stages=2
        )

        # 2) RMSNorm for Q and K using Triton kernel
        # Prepare weights
        q_norm_w = q_norm_weight.to(torch.float32)
        k_norm_w = k_norm_weight.to(torch.float32)

        Q_norm = torch.empty_like(Q)
        K_norm = torch.empty_like(K)

        grid_norm = (B, L)
        rmsnorm_kernel[grid_norm](
            Q, q_norm_w, Q_norm,
            B, L, head_dim,
            Q.stride(0), Q.stride(1), Q.stride(2),
            q_norm_w.stride(0),
            Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(2),
            self.rms_norm_eps,
            BLOCK_H=128,
            num_warps=4, num_stages=2
        )

        rmsnorm_kernel[grid_norm](
            K, k_norm_w, K_norm,
            B, L, head_dim,
            K.stride(0), K.stride(1), K.stride(2),
            k_norm_w.stride(0),
            K_norm.stride(0), K_norm.stride(1), K_norm.stride(2),
            self.rms_norm_eps,
            BLOCK_H=128,
            num_warps=4, num_stages=2
        )

        # 3) Rotate Q and K using Triton rotate kernel
        # Ensure cos, sin are float32 on device
        cos_f32 = cos.to(device=device, dtype=torch.float32)
        sin_f32 = sin.to(device=device, dtype=torch.float32)

        Q_rot = torch.empty_like(Q_norm)
        K_rot = torch.empty_like(K_norm)

        grid_rotate = (B, L)
        rotate_qk_kernel[grid_rotate](
            Q_norm, cos_f32, sin_f32, Q_rot,
            B, L, head_dim,
            Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(2),
            cos_f32.stride(0), cos_f32.stride(1),
            sin_f32.stride(0), sin_f32.stride(1),
            Q_rot.stride(0), Q_rot.stride(1), Q_rot.stride(2),
            BLOCK_H=128,
            num_warps=4, num_stages=2
        )

        rotate_qk_kernel[grid_rotate](
            K_norm, cos_f32, sin_f32, K_rot,
            B, L, head_dim,
            K_norm.stride(0), K_norm.stride(1), K_norm.stride(2),
            cos_f32.stride(0), cos_f32.stride(1),
            sin_f32.stride(0), sin_f32.stride(1),
            K_rot.stride(0), K_rot.stride(1), K_rot.stride(2),
            BLOCK_H=128,
            num_warps=4, num_stages=2
        )

        # 4) GQA: expand K_rot and V for attention heads
        # K_rot: [B, 8, L, 128] -> [B, 8, 12, L, 128] -> [B, 96, L, 128]
        # V: [B, L, 128] -> [B, 96, L, 128] (since V[b, l, :] shared across heads)
        K_rot_exp = K_rot.unsqueeze(2).expand(B, 8, 12, L, 128).reshape(B, 96, L, 128)
        V_exp = V.unsqueeze(1).expand(B, 96, L, 128)

        # 5) Compute attention scores using Triton attn_matmul_kernel: [B, 96, L, L]
        attn_scores = torch.empty((B, 96, L, L), device=device, dtype=torch.float32)

        grid_am = (B, 96, L)
        attn_matmul_kernel[grid_am](
            Q_rot, K_rot_exp, attn_scores,
            B, 96, L, head_dim,
            Q_rot.stride(0), Q_rot.stride(1), Q_rot.stride(2), Q_rot.stride(3),
            K_rot_exp.stride(0), K_rot_exp.stride(1), K_rot_exp.stride(2), K_rot_exp.stride(3),
            attn_scores.stride(0), attn_scores.stride(1), attn_scores.stride(2), attn_scores.stride(3),
            num_warps=4, num_stages=2
        )

        # 6) Apply causal mask and softmax using Triton softmax_mask_kernel
        # Build mask: [B, 96, L, L], upper triangle with diagonal=1, masked positions = -inf
        causal_mask = torch.empty((B, 96, L, L), device=device, dtype=torch.float32)
        for b_i in range(B):
            for nh in range(96):
                for l_row in range(L):
                    for l_col in range(L):
                        causal_mask[b_i, nh, l_row, l_col] = -float('inf') if l_col < l_row else 0.0

        attn_masked = attn_scores.clone()
        attn_out = torch.empty_like(attn_scores)

        grid_s = (B, 96, L)
        softmax_mask_kernel[grid_s](
            attn_masked, causal_mask, attn_out,
            B, 96, L,
            attn_masked.stride(0), attn_masked.stride(1), attn_masked.stride(2), attn_masked.stride(3),
            causal_mask.stride(0), causal_mask.stride(1), causal_mask.stride(2), causal_mask.stride(3),
            attn_out.stride(0), attn_out.stride(1), attn_out.stride(2), attn_out.stride(3),
            num_warps=4, num_stages=2
        )

        # 7) Compute attention output using Triton output_matmul_kernel: [B, 96, L, 128]
        attn_output = torch.empty((B, 96, L, head_dim), device=device, dtype=torch.float32)

        grid_out = (B, 96, L)
        output_matmul_kernel[grid_out](
            attn_out, V_exp, attn_output,
            B, 96, L, head_dim,
            attn_out.stride(0), attn_out.stride(1), attn_out.stride(2), attn_out.stride(3),
            V_exp.stride(0), V_exp.stride(1), V_exp.stride(2), V_exp.stride(3),
            attn_output.stride(0), attn_output.stride(1), attn_output.stride(2), attn_output.stride(3),
            num_warps=4, num_stages=2
        )

        # Flatten for final linear projection: [B, L, 96*128]
        attn_flat = attn_output.reshape(B, L, 96 * head_dim)

        # 8) Final linear projection using Triton final_linear_kernel: [B, L, hidden_dim]
        final_out = torch.empty((B, L, self.hidden_dim), device=device, dtype=torch.float32)

        grid_final = (B, L, self.hidden_dim)
        # o_proj_weight: [hidden_dim, 12288], input attn_flat: [B, L, 12288]
        # We pass attn_flat as x_ptr, w_ptr=o_proj_weight^T (already [12288, hidden_dim]), y_ptr=final_out
        # Note: we compute attn_flat as [B, L, 12288] previously via output_matmul_kernel; here we just launch final projection.
        # Ensure o_proj_weight is [hidden_dim, 12288] as provided.
        # Launch final_linear_kernel
        final_linear_kernel[grid_final](
            attn_flat, o_proj_weight.to(torch.float32),
            final_out,
            B, L, 96 * head_dim, self.hidden_dim,
            attn_flat.stride(0), attn_flat.stride(1), attn_flat.stride(2),
            o_proj_weight.stride(0), o_proj_weight.stride(1),
            final_out.stride(0), final_out.stride(1), final_out.stride(2),
            BLOCK_K=64,
            num_warps=4, num_stages=2
        )

        return final_out


def run(*args):
    return ModelNew()(*args)
