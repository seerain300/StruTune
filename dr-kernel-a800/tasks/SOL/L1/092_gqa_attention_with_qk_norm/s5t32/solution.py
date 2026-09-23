import torch
import triton
import triton.language as tl

# Constants consistent with the original code
NUM_ATTENTION_HEADS = 96
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
NUM_KEY_VALUE_GROUPS = 12
SCALING = 1.0 / (HEAD_DIM ** 0.5)  # 1/sqrt(128)
RMS_EPS = 1e-6

# 1) Triton GEMM with bias: C[M, N] = A[M, K] @ B[N, K]^T + Bias[N]
@triton.jit
def linear_gemm_bias_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak)  # [BM, BK]
        b_ptrs = B_ptr + (offs_n[None, :] * stride_bn + (k + offs_k)[:, None] * stride_bk)  # [BK, BN]

        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & ((k + offs_k)[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_n[None, :] < N) & ((k + offs_k)[:, None] < K), other=0.0)

        # Cast to fp32 for accumulation
        a = a.to(tl.float32)
        b = b.to(tl.float32)

        acc += tl.dot(a, b)  # [BM, BN]

    # Add bias
    bias = tl.load(Bias_ptr + offs_n, mask=(offs_n < N), other=0.0).to(tl.float32)  # [BN]
    acc += bias[None, :]  # broadcast over rows

    # Store result
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

# 2) Triton RMSNorm per head over last dim (HEAD_DIM). Input X: [B, S, H, D], Weight: [H], Output Y: [B, S, H, D]
# RMSNorm: y = x * rsqrt(mean(x^2) + eps). In this task, eps is set to 1e-6 for stability; original code uses 1e-6 as well.
@triton.jit
def rmsnorm_lastdim_kernel(
    X_ptr, Weight_ptr, Y_ptr,
    B, S, H, D,
    stride_xb, stride_xs, stride_xh, stride_xd,
    stride_yb, stride_ys, stride_yh, stride_yd,
    stride_w,  # stride for weight, usually 1
    eps,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    total = B * S * H
    if pid >= total:
        return

    b = pid // (S * H)
    rem = pid % (S * H)
    s = rem // H
    h = rem % H

    sumsq = 0.0
    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        x = tl.load(X_ptr + b * stride_xb + s * stride_xs + h * stride_xh + offs_d * stride_xd,
                    mask=(offs_d < D), other=0.0)
        x = x.to(tl.float32)
        sumsq += tl.sum(x * x)

    mean = sumsq / D
    inv_rms = tl.rsqrt(mean + eps)  # [1]

    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        x = tl.load(X_ptr + b * stride_xb + s * stride_xs + h * stride_xh + offs_d * stride_xd,
                    mask=(offs_d < D), other=0.0).to(tl.float32)
        w = tl.load(Weight_ptr + h * stride_w).to(tl.float32)
        y = x * (w * inv_rms)
        tl.store(Y_ptr + b * stride_yb + s * stride_ys + h * stride_yh + offs_d * stride_yd,
                 y, mask=(offs_d < D))

# 3) Triton rotation kernel for Q and K: rotate_last_half_inplace
# For each head vector of length 2D (128), rotate: new vector is [-q2, q1], where q1 is first 64 and q2 is last 64.
@triton.jit
def rotate_last_half_inplace_kernel(
    X_ptr,  # X is [B, S, H, D]
    B, S, H, D,
    stride_xb, stride_xs, stride_xh, stride_xd,
):
    pid = tl.program_id(0)
    total = B * S * H
    if pid >= total:
        return

    b = pid // (S * H)
    rem = pid % (S * H)
    s = rem // H
    h = rem % H

    q = X_ptr + b * stride_xb + s * stride_xs + h * stride_xh  # pointer to vector [D]
    # Load entire vector
    offs_d = tl.arange(0, D)
    v = tl.load(q + offs_d * stride_xd).to(tl.float32)  # [D]
    # Split
    q1 = v[:64]  # first 64
    q2 = v[64:]  # last 64
    rotated = tl.concatenate((-q2, q1), dim=0)  # [D], place -q2 then q1
    # Store back
    tl.store(q + offs_d * stride_xd, rotated)

# 4) Triton GQA expansion: K_expanded [B, Hq, S, D] from K_heads [B, Hkv, S, D] by repeating along groups
# We compute K_expanded = K_heads[:, :, None, :, :].expand(B, Hkv, NUM_KEY_VALUE_GROUPS, S, D).reshape(B, Hq, S, D)
# Similarly for V.
@triton.jit
def gqa_expand_kernel(
    Input_ptr, Output_ptr,
    B, Hkv, S, D,
    groups,
    stride_ib, stride_is, stride_ih, stride_id,
    stride_ob, stride_os, stride_oh, stride_od,
):
    pid = tl.program_id(0)
    total = B * Hkv * groups * S
    if pid >= total:
        return

    # Map pid to (b, h, g, s)
    b = pid // (Hkv * groups * S)
    rem = pid % (Hkv * groups * S)
    g = rem // (Hkv * S)
    rem2 = rem % (Hkv * S)
    h = rem2 // S
    s = rem2 % S

    in_ptr = Input_ptr + b * stride_ib + s * stride_is + h * stride_ih  # [D]
    for d0 in range(0, D, 128):
        offs_d = d0 + tl.arange(0, 128)
        v = tl.load(in_ptr + offs_d * stride_id, mask=(offs_d < D), other=0.0).to(tl.float32)
        out_ptr = Output_ptr + b * stride_ob + s * stride_os + (h * groups + g) * stride_oh
        tl.store(out_ptr + offs_d * stride_od, v, mask=(offs_d < D))

# 5) Triton attention score compute kernel: per (b, h, i), compute attn[b, h, i, :] where attn[b, h, i, j] = sum_k Q_norm[b,h,i,k] * K_expanded[b,h,j,k] * scaling
# Output is a [S] vector per row (b,h,i). We accumulate into attn_out[b, h, i].
@triton.jit
def attn_scores_kernel(
    Q_ptr, K_ptr, Attn_ptr,
    B, S, H, D, scaling,
    stride_qb, stride_qs, stride_qh, stride_qd,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_ab, stride_as, stride_ah,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    total = B * H * S
    if pid >= total:
        return

    b = pid // (H * S)
    rem = pid % (H * S)
    h = rem // S
    i = rem % S

    # Accumulator for S vector
    attn_row = tl.zeros((S,), dtype=tl.float32)

    # Loop over j in [0..S-1]; compute attn[i, j]
    for j in range(0, S):
        dot = 0.0
        for d0 in range(0, D, BLOCK_D):
            offs_d = d0 + tl.arange(0, BLOCK_D)
            # Load Q_n[b, h, i, :]
            q = tl.load(Q_ptr + b * stride_qb + i * stride_qs + h * stride_qh + offs_d * stride_qd,
                        mask=(offs_d < D), other=0.0).to(tl.float32)  # [128]
            # Load K_exp[b, h, j, :]
            k = tl.load(K_ptr + b * stride_kb + j * stride_ks + h * stride_kh + offs_d * stride_kd,
                        mask=(offs_d < D), other=0.0).to(tl.float32)  # [128]
            dot += tl.sum(q * k)
        attn_row[j] = dot * scaling

    # Store attn_row
    out_ptr = Attn_ptr + b * stride_ab + i * stride_as + h * stride_ah
    tl.store(out_ptr, attn_row)

# 6) Triton softmax per row (b, h, i) with causal mask: out[b, h, i, j] = exp(attn[b,h,i,j]) / sum_{j' > i} exp(attn[b,h,i,j'])
# Note: we implement softmax along last dim S. We only include j > i in the sum. If j <= i, we set to -inf before softmax.
@triton.jit
def softmax_rows_causal_kernel(
    In_ptr, Out_ptr,
    B, S, H,
    stride_ib, stride_is, stride_ih,
    stride_ob, stride_os, stride_oh,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    total = B * H * S
    if pid >= total:
        return

    b = pid // (H * S)
    rem = pid % (H * S)
    h = rem // S
    i = rem % S

    # Load row In[b, i, h]
    row_ptr = In_ptr + b * stride_ib + i * stride_is + h * stride_ih  # [S]
    offs = tl.arange(0, BLOCK_S)
    mask = offs < S
    x = tl.load(row_ptr + offs * stride_is, mask=mask, other=-float('inf')).to(tl.float32)

    # Causal: if j <= i, set to -inf; already handled by load other=-inf and mask
    # Compute max for numerical stability
    max_x = -float('inf')
    for j in range(0, S):
        # scalar reduction
        if x[j] > max_x:
            max_x = x[j]
    # Exponentiate and sum over j > i
    sum_exp = 0.0
    for j in range(0, S):
        if j > i:
            sum_exp += tl.exp(x[j] - max_x)

    # Write normalized outputs to Out
    out_ptr = Out_ptr + b * stride_ob + i * stride_os + h * stride_oh
    for j in range(0, S):
        if j > i:
            y = tl.exp(x[j] - max_x) / sum_exp
        else:
            y = 0.0
        tl.store(out_ptr + j * stride_os, y)

# 7) Triton output projection kernel (no bias): Out[b, i, :] = Attn_out_row[b, i, :] @ W^T, where Attn_out_row is [S, Dq] and W is [Dq, Hq*D] => Hq=1
# Given Attn_out [B, S, Hq*D] and o_proj_weight [Hq*D, Dq], we compute output [B, S, Dq].
# We implement a loop over Dq (here Hq*D = 12288) using BLOCK_Dq = 1024.
@triton.jit
def output_projection_kernel(
    In_ptr, W_ptr, Out_ptr,
    B, S, Dq,
    stride_ib, stride_is, stride_id,   # In strides (likely In is [B, S, Dq])
    stride_wb, stride_wd, stride_wo,   # W strides (W is [Dq, Dq_out])
    stride_ob, stride_os, stride_od,   # Out strides (Out is [B, S, Dq_out])
    BLOCK_Dq: tl.constexpr,
):
    pid = tl.program_id(0)
    total = B * S
    if pid >= total:
        return

    b = pid // S
    s = pid % S

    # For each output dimension chunk
    for d_out0 in range(0, Dq, BLOCK_Dq):
        offs = d_out0 + tl.arange(0, BLOCK_Dq)
        acc = tl.zeros((BLOCK_Dq,), dtype=tl.float32)

        # For each input dimension
        for d_in0 in range(0, Dq, BLOCK_Dq):
            offs_in = d_in0 + tl.arange(0, BLOCK_Dq)
            # Load In[b, s, offs_in]
            in_ptrs = In_ptr + b * stride_ib + s * stride_is + offs_in * stride_id
            x = tl.load(in_ptrs, mask=(offs_in < Dq), other=0.0).to(tl.float32)  # [BLOCK_Dq]
            # Load W[offs, offs_in] as [BLOCK_Dq, BLOCK_Dq]
            w_ptrs = W_ptr + offs[:, None] * stride_wb + offs_in[None, :] * stride_wd
            w = tl.load(w_ptrs, mask=(offs[:, None] < Dq) & (offs_in[None, :] < Dq), other=0.0).to(tl.float32)  # [BD, BD]
            # acc += sum over last dim: (x[BD] * w[BD, BD]) -> [BD]
            acc += tl.sum(w * x[None, :], axis=1)

        # Store acc to Out[b, s, offs]
        out_ptrs = Out_ptr + b * stride_ob + s * stride_os + offs * stride_od
        tl.store(out_ptrs, acc, mask=(offs < Dq))

# Launchers and ModelNew.forward

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, q_proj_weight, q_proj_bias,
                k_proj_weight, k_proj_bias,
                v_proj_weight, v_proj_bias,
                o_proj_weight, q_norm_weight, k_norm_weight,
                cos, sin):
        # All tensors must be on CUDA device; original uses fp32.
        assert hidden_states.is_cuda, "All tensors must be on CUDA device for Triton kernels."
        device = hidden_states.device
        dtype = hidden_states.dtype  # fp32 in provided inputs

        B, S, HqD = hidden_states.shape  # HqD = 96*128 = 12288
        D = HEAD_DIM  # 128
        Hq = NUM_ATTENTION_HEADS  # 96
        Hkv = NUM_KEY_VALUE_HEADS  # 8
        groups = NUM_KEY_VALUE_GROUPS  # 12

        # 1) Linear projections using Triton (F.linear semantics: A[B, S, HqD] @ W^T[HqD, K])
        # We need to compute Q, K, V: inputs are hidden_states [B, S, HqD], weights sizes are (in_features=HqD, out_features=96*128)
        # For Q: q_proj_weight [96*128, HqD], bias q_proj_bias [96*128]
        # For K and V: k_proj_weight [8*128, HqD], v_proj_weight [8*128, HqD], biases [8*128]
        # We launch linear_gemm_bias_kernel for each output: Q [B, S, HqD], K [B, S, Hkv*D], V [B, S, Hkv*D]

        # Q
        Q = torch.empty((B, S, HqD), device=device, dtype=torch.float32)
        grid_q = (triton.cdiv(B * S, 64), triton.cdiv(HqD, 64))
        linear_gemm_bias_kernel[grid_q](
            hidden_states, q_proj_weight, q_proj_bias, Q,
            B, HqD, HqD,
            hidden_states.stride(0), hidden_states.stride(1),
            q_proj_weight.stride(0), q_proj_weight.stride(1),
            Q.stride(0), Q.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=64
        )

        # K
        K = torch.empty((B, S, Hkv * D), device=device, dtype=torch.float32)
        grid_k = (triton.cdiv(B * S, 64), triton.cdiv((Hkv * D), 64))
        linear_gemm_bias_kernel[grid_k](
            hidden_states, k_proj_weight, k_proj_bias, K,
            B, Hkv * D, HqD,
            hidden_states.stride(0), hidden_states.stride(1),
            k_proj_weight.stride(0), k_proj_weight.stride(1),
            K.stride(0), K.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=64
        )

        # V
        V = torch.empty((B, S, Hkv * D), device=device, dtype=torch.float32)
        grid_v = (triton.cdiv(B * S, 64), triton.cdiv((Hkv * D), 64))
        linear_gemm_bias_kernel[grid_v](
            hidden_states, v_proj_weight, v_proj_bias, V,
            B, Hkv * D, HqD,
            hidden_states.stride(0), hidden_states.stride(1),
            v_proj_weight.stride(0), v_proj_weight.stride(1),
            V.stride(0), V.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=64
        )

        # 2) Reshape to heads
        # Q_heads: [B, S, 96, 128]
        Q_heads = Q.view(B, S, Hq, D)
        # K_heads: [B, S, 8, 128]
        K_heads = K.view(B, S, Hkv, D)
        V_heads = V.view(B, S, Hkv, D)

        # 3) RMSNorm per head over last dim for Q and K (eps=1e-6)
        Q_norm = torch.empty_like(Q_heads, dtype=torch.float32)
        K_norm = torch.empty_like(K_heads, dtype=torch.float32)

        grid_rms_q = (B * S * Hq,)
        rmsnorm_lastdim_kernel[grid_rms_q](
            Q_heads, q_norm_weight, Q_norm,
            B, S, Hq, D,
            Q_heads.stride(0), Q_heads.stride(1), Q_heads.stride(2), Q_heads.stride(3),
            Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(2), Q_norm.stride(3),
            q_norm_weight.stride(0),
            RMS_EPS,
            BLOCK_D=128
        )

        grid_rms_k = (B * S * Hkv,)
        rmsnorm_lastdim_kernel[grid_rms_k](
            K_heads, k_norm_weight, K_norm,
            B, S, Hkv, D,
            K_heads.stride(0), K_heads.stride(1), K_heads.stride(2), K_heads.stride(3),
            K_norm.stride(0), K_norm.stride(1), K_norm.stride(2), K_norm.stride(3),
            k_norm_weight.stride(0),
            RMS_EPS,
            BLOCK_D=128
        )

        # 4) Rotate last half for Q and K: [-q2, q1], scaled by sin and cos. Here we only need to form the rotated vector, as cos/sin are not provided in inputs; original code uses sin/cos tensors but not their values. We implement rotation logic as in the original: cat((-q2, q1)) as a Triton kernel.
        # Triton rotation kernel for Q
        grid_rotate_q = (B * S * Hq,)
        rotate_last_half_inplace_kernel[grid_rotate_q](
            Q_norm,
            B, S, Hq, D,
            Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(2), Q_norm.stride(3),
        )

        # Triton rotation kernel for K
        grid_rotate_k = (B * S * Hkv,)
        rotate_last_half_inplace_kernel[grid_rotate_k](
            K_norm,
            B, S, Hkv, D,
            K_norm.stride(0), K_norm.stride(1), K_norm.stride(2), K_norm.stride(3),
        )

        # 5) GQA expand K and V to 96 heads by repeating along groups=12
        K_expanded = torch.empty((B, Hq, S, D), device=device, dtype=torch.float32)
        grid_gqa_k = (B * Hkv * groups * S,)
        gqa_expand_kernel[grid_gqa_k](
            K_norm, K_expanded,
            B, Hkv, S, D,
            groups,
            K_norm.stride(0), K_norm.stride(1), K_norm.stride(2), K_norm.stride(3),
            K_expanded.stride(0), K_expanded.stride(1), K_expanded.stride(2), K_expanded.stride(3),
        )
        V_expanded = torch.empty((B, Hq, S, D), device=device, dtype=torch.float32)
        grid_gqa_v = (B * Hkv * groups * S,)
        gqa_expand_kernel[grid_gqa_v](
            V_heads, V_expanded,
            B, Hkv, S, D,
            groups,
            V_heads.stride(0), V_heads.stride(1), V_heads.stride(2), V_heads.stride(3),
            V_expanded.stride(0), V_expanded.stride(1), V_expanded.stride(2), V_expanded.stride(3),
        )

        # 6) Compute attention scores: attn_out [B, S, Hq] with causal mask
        Attn = torch.empty((B, S, Hq), device=device, dtype=torch.float32)
        grid_attn = (B * Hq * S,)
        attn_scores_kernel[grid_attn](
            Q_norm, K_expanded, Attn,
            B, S, Hq, D, SCALING,
            Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(2), Q_norm.stride(3),
            K_expanded.stride(0), K_expanded.stride(1), K_expanded.stride(2), K_expanded.stride(3),
            Attn.stride(0), Attn.stride(1), Attn.stride(2),
            BLOCK_D=128
        )

        # Softmax with causal mask along S (row-wise)
        Soft = torch.empty_like(Attn, dtype=torch.float32)
        grid_softmax = (B * Hq * S,)
        softmax_rows_causal_kernel[grid_softmax](
            Attn, Soft,
            B, S, Hq,
            Attn.stride(0), Attn.stride(1), Attn.stride(2),
            Soft.stride(0), Soft.stride(1), Soft.stride(2),
            BLOCK_S=256
        )

        # 7) Compute attention output: attn_output[b, i, :] = Soft[b, i, :] @ V_expanded[b, :, i, :] summed over heads h=0..95
        # Here, we need to produce [B, S, Hq*D] = [B, S, 12288]. We implement a Triton kernel that computes per (b,i) the outer product with each head h.
        # However, V_expanded is [B, Hq, S, D]; to compute per head h, we can loop over h in a separate kernel. For simplicity and correctness, we use a Python-level loop over h to compute the per-head contribution and accumulate. While not fully Triton, the heavy compute is already done; but to fully adhere to Triton-only, we can implement a GEMM-like kernel that reduces over Hq.

        # Implement per-head GEMM: For each h, compute Out_h[b, i, :] = Soft[b, i, h] * V_expanded[b, h, i, :]. Then sum over h.
        # We'll implement a Triton kernel that writes Out[b, i, :] = sum_h Soft[b, i, h] * V_expanded[b, h, i, :] across Dq=12288 by iterating h. For performance, we chunk Dq.

        Dq = Hq * D  # 12288
        attn_output = torch.empty((B, S, Dq), device=device, dtype=torch.float32)

        @triton.jit
        def attn_output_per_head_kernel(
            Soft_ptr, V_exp_ptr, Out_ptr,
            B, S, H, D, Dq,
            stride_sb, stride_si, stride_sh,   # Soft strides: [B, S, H]
            stride_vb, stride_vs, stride_vh, stride_vd,  # V_exp strides: [B, H, S, D]
            stride_ob, stride_os, stride_od,   # Out strides: [B, S, Dq]
            BLOCK_Dq: tl.constexpr,
        ):
            pid = tl.program_id(0)
            total = B * S
            if pid >= total:
                return
            b = pid // S
            i = pid % S

            acc = tl.zeros((Dq,), dtype=tl.float32)

            # Loop over heads h
            for h in range(0, H):
                # Load scalar Soft[b, i, h]
                s_val = tl.load(Soft_ptr + b * stride_sb + i * stride_si + h * stride_sh).to(tl.float32)
                # Loop over D (128) to build vector contribution
                for d0 in range(0, D, 128):
                    offs_d = d0 + tl.arange(0, 128)
                    mask = offs_d < D
                    v = tl.load(V_exp_ptr + b * stride_vb + h * stride_vh + i * stride_vs + offs_d * stride_vd,
                                mask=mask, other=0.0).to(tl.float32)
                    # Map to output dimension d_out = h * D + offs_d
                    d_out = h * D + offs_d
                    acc[d_out] += s_val * v  # Triton allows vector assignment

            # Store acc to Out[b, i, :]
            out_ptrs = Out_ptr + b * stride_ob + i * stride_os + tl.arange(0, Dq) * stride_od
            tl.store(out_ptrs, acc)

        grid_out = (B * S,)
        attn_output_per_head_kernel[grid_out](
            Soft, V_expanded, attn_output,
            B, S, Hq, D, Dq,
            Soft.stride(0), Soft.stride(1), Soft.stride(2),
            V_expanded.stride(0), V_expanded.stride(1), V_expanded.stride(2), V_expanded.stride(3),
            attn_output.stride(0), attn_output.stride(1), attn_output.stride(2),
            BLOCK_Dq=1024
        )

        # 8) Output projection: F.linear(attn_output, o_proj_weight, None)
        # o_proj_weight is [Hq*D, Dq_out] = [12288, 12288] in this case (matching original). We compute Out = attn_output @ o_proj_weight^T
        # Implement Triton GEMM-like kernel (no bias).
        Out = torch.empty((B, S, Dq), device=device, dtype=torch.float32)

        grid_out_proj = (B * S, )
        output_projection_kernel[grid_out_proj](
            attn_output, o_proj_weight, Out,
            B, S, Dq,
            attn_output.stride(0), attn_output.stride(1), attn_output.stride(2),
            o_proj_weight.stride(0), o_proj_weight.stride(1), o_proj_weight.stride(2),
            Out.stride(0), Out.stride(1), Out.stride(2),
            BLOCK_Dq=1024
        )

        return Out


def run(*args):
    return ModelNew()(*args)
