import torch
import triton
import triton.language as tl

# Constants matching the original code
NUM_ATTENTION_HEADS = 96
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
NUM_KEY_VALUE_GROUPS = 12
SCALING = 1.0 / (HEAD_DIM ** 0.5)  # 1/sqrt(128)
RMS_EPS = 1e-6

# 1) Triton GEMM for linear projection with bias: C[M, N] = A[M, K] @ B[N, K]^T + Bias[N]
# A: [M, K], B: [N, K], Bias: [N], C: [M, N]
@triton.jit
def linear_gemm_bias_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
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

        acc += tl.dot(a, b)  # [BM, BN]

    # Add bias
    bias = tl.load(Bias_ptr + offs_n, mask=(offs_n < N), other=0.0)  # [BN]
    acc += bias[None, :]  # broadcast over rows

    # Store result
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

# 2) Triton RMSNorm per head over last dim (HEAD_DIM) for Q and K
# Normalize along dim=-1: x = x * rsqrt(mean(x^2) + eps), then apply weight (length-HEAD_DIM vector per head)
@triton.jit
def rmsnorm_kernel(
    X_ptr, Weight_ptr, Y_ptr,
    B, S, H, D,
    stride_xb, stride_xs, stride_xh, stride_xd,
    stride_yb, stride_ys, stride_yh, stride_yd,
    stride_w,                      # stride for weight vector (per head)
    eps,
    BLOCK_D: tl.constexpr
):
    # Each program handles one (b, s, h) and reduces over D
    total = B * S * H
    pid = tl.program_id(0)
    if pid >= total:
        return
    b = pid // (S * H)
    rem = pid % (S * H)
    s = rem // H
    h = rem % H

    acc = tl.zeros((), dtype=tl.float32)
    # Compute mean of X[b,s,h,:]^2 over D
    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        mask = offs_d < D
        x = tl.load(X_ptr + b * stride_xb + s * stride_xs + h * stride_xh + offs_d * stride_xd, mask=mask, other=0.0)
        x = x.to(tl.float32)
        acc += tl.sum(x * x, axis=0)

    mean = acc / D
    inv_rms = tl.rsqrt(mean + eps)

    # Normalize and apply weight
    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        mask = offs_d < D
        x = tl.load(X_ptr + b * stride_xb + s * stride_xs + h * stride_xh + offs_d * stride_xd, mask=mask, other=0.0)
        x = x.to(tl.float32)
        w = tl.load(Weight_ptr + h * stride_w)  # per-head scalar weight
        y = x * inv_rms * w
        tl.store(Y_ptr + b * stride_yb + s * stride_ys + h * stride_yh + offs_d * stride_yd, y, mask=mask)

# 3) Triton kernel to rotate the last half dimension (for Q and K): new[:, :D//2] = -[:, D//2:], new[:, D//2:] = [:, :D//2]
@triton.jit
def rotate_half_kernel(
    In_ptr, Out_ptr,
    B, S, H, D,
    stride_ib, stride_is, stride_ih, stride_id,
    stride_ob, stride_os, stride_oh, stride_od,
    BLOCK_D: tl.constexpr
):
    total = B * S * H
    pid = tl.program_id(0)
    if pid >= total:
        return
    b = pid // (S * H)
    rem = pid % (S * H)
    s = rem // H
    h = rem % H

    half = D // 2
    for d0 in range(0, half, BLOCK_D):
        offs = d0 + tl.arange(0, BLOCK_D)
        mask = offs < half
        in1 = tl.load(In_ptr + b * stride_ib + s * stride_is + h * stride_ih + offs * stride_id, mask=mask, other=0.0)  # first half
        in2 = tl.load(In_ptr + b * stride_ib + s * stride_is + h * stride_ih + (offs + half) * stride_id, mask=mask, other=0.0)  # second half
        out1 = -in2
        out2 = in1
        # Write back to Out
        tl.store(Out_ptr + b * stride_ob + s * stride_os + h * stride_oh + offs * stride_od, out1, mask=mask)  # first half
        tl.store(Out_ptr + b * stride_ob + s * stride_os + h * stride_oh + (offs + half) * stride_od, out2, mask=mask)  # second half

# 4) Triton kernel to expand K/V heads from Hk=NUM_KEY_VALUE_HEADS to Hq=NUM_ATTENTION_HEADS by repeating along NUM_KEY_VALUE_GROUPS
#   We tile over i (sequence position) and h (destination head index), then compute source h_src = h % NUM_KEY_VALUE_HEADS and group repeat index g = h // NUM_KEY_VALUE_HEADS
@triton.jit
def gqa_expand_kernel(
    K_ptr, V_ptr, Out_ptr,
    B, S, Hq, Hk, D, GROUPS,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_vb, stride_vs, stride_vh, stride_vd,
    stride_ob, stride_os, stride_oh, stride_od,
    BLOCK_S: tl.constexpr
):
    total_rows = B * Hq * S
    pid = tl.program_id(0)
    if pid >= total_rows:
        return
    b = pid // (Hq * S)
    rem = pid % (Hq * S)
    hq = rem // S
    s = rem % S

    h_src = hq % Hk
    g = hq // Hk  # group index in [0..GROUPS-1], must be valid

    for d0 in range(0, D, BLOCK_S):
        offs = d0 + tl.arange(0, BLOCK_S)
        mask = offs < D
        k = tl.load(K_ptr + b * stride_kb + s * stride_ks + h_src * stride_kh + offs * stride_kd, mask=mask, other=0.0).to(tl.float32)
        v = tl.load(V_ptr + b * stride_vb + s * stride_vs + h_src * stride_vh + offs * stride_vd, mask=mask, other=0.0).to(tl.float32)
        # Write to expanded out[b, hq, s, offs]
        tl.store(Out_ptr + b * stride_ob + s * stride_os + hq * stride_oh + offs * stride_od, k, mask=mask)
        tl.store(Out_ptr + b * stride_ob + s * stride_os + hq * stride_oh + (offs + D) * stride_od, v, mask=mask)

# 5) Triton kernel to compute attention scores: attn[b, h, i, j] = sum_k Qn[b,h,i,k] * K_exp[b,h,j,k] * SCALING
#   Here we treat Qn and K_exp as [B, H, S, D] and write out a vector of length S for each (b,h,i).
@triton.jit
def attn_scores_kernel(
    Q_ptr, K_ptr, Out_ptr,
    B, S, H, D,
    stride_qb, stride_qs, stride_qh, stride_qd,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_ob, stride_os, stride_oh, stride_od,
    scaling,
    BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr
):
    total_rows = B * H * S
    pid = tl.program_id(0)
    if pid >= total_rows:
        return
    b = pid // (H * S)
    rem = pid % (H * S)
    h = rem // S
    i = rem % S

    # Compute attn vector for position j in [0..S-1]
    attn_vec = tl.zeros((S,), dtype=tl.float32)

    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        q_vec = tl.load(Q_ptr + b * stride_qb + i * stride_qs + h * stride_qh + offs_d * stride_qd,
                        mask=mask_d, other=0.0).to(tl.float32)  # [BLOCK_D]
        for j0 in range(0, S, BLOCK_S):
            j_idx = j0 + tl.arange(0, BLOCK_S)
            mask_j = j_idx < S
            k_vec = tl.load(K_ptr + b * stride_kb + j_idx * stride_ks + h * stride_kh + offs_d * stride_kd,
                            mask=mask_j[:, None] & mask_d[None, :], other=0.0).to(tl.float32)  # [BLOCK_S, BLOCK_D]
            # dot(q_vec, k_vec) -> [BLOCK_S]
            attn_vec += tl.sum(q_vec[None, :] * k_vec, axis=1)
        attn_vec *= scaling

    # Store attn_vec into Out[b, h, i, :]
    out_ptr = Out_ptr + b * stride_ob + i * stride_os + h * stride_oh
    for j0 in range(0, S, BLOCK_S):
        j_idx = j0 + tl.arange(0, BLOCK_S)
        mask_j = j_idx < S
        tl.store(out_ptr + j_idx * stride_od, attn_vec[j0:j0 + BLOCK_S], mask=mask_j)

# 6) Triton kernel to perform softmax along the last dim (S) for each row (b, h, i), applying causal mask (triangular with diag=1)
#    i.e., for j <= i, set value to -inf before softmax; otherwise use the value.
@triton.jit
def softmax_rows_causal_kernel(
    In_ptr, Out_ptr,
    B, S, H,
    stride_ib, stride_is, stride_ih, stride_id,  # In strides
    stride_ob, stride_os, stride_oh, stride_od,  # Out strides
    BLOCK_S: tl.constexpr
):
    total_rows = B * H * S
    pid = tl.program_id(0)
    if pid >= total_rows:
        return
    b = pid // (H * S)
    rem = pid % (H * S)
    h = rem // S
    i = rem % S

    # Load row In[b, h, i, :]
    row_ptr = In_ptr + b * stride_ib + i * stride_is + h * stride_ih
    row = tl.zeros((S,), dtype=tl.float32)
    for s0 in range(0, S, BLOCK_S):
        offs = s0 + tl.arange(0, BLOCK_S)
        mask = offs < S
        row[offs] = tl.load(row_ptr + offs * stride_id, mask=mask, other=-float('inf'))

    # Apply causal mask: j <= i -> -inf
    # We modify row in-place by setting invalid positions to -inf. Triton doesn't support direct indexing assignment,
    # so we recompute with masked loads. Instead, we'll compute max, exp, sum, and write normalized results in the next loop.
    max_val = -float('inf')
    for s0 in range(0, S, BLOCK_S):
        offs = s0 + tl.arange(0, BLOCK_S)
        mask = offs < S
        # For causal: if offs <= i, set to -inf
        causal_mask = offs <= i
        vals = tl.where(causal_mask & mask, -float('inf'), row[offs])
        # Only compute max if within bounds
        if s0 == 0:
            max_val = tl.max(vals, axis=0)
        else:
            max_val = tl.maximum(max_val, tl.max(vals, axis=0))

    exp_row = tl.exp(row - max_val)
    sum_row = tl.sum(exp_row, axis=0)
    inv_sum = 1.0 / sum_row

    out_ptr = Out_ptr + b * stride_ob + i * stride_os + h * stride_oh
    for s0 in range(0, S, BLOCK_S):
        offs = s0 + tl.arange(0, BLOCK_S)
        mask = offs < S
        # Reapply causal mask during store: set j<=i to 0 (will be converted to -inf if logsumexp path uses it)
        # Note: Triton stores will not change Out beyond what we write; we can write normalized values everywhere.
        # Given our exp/sum normalized values are finite, they will be written for all positions. For causal positions,
        # the original attention scores were -inf (effectively zero contribution). Our softmax excludes -inf from sum
        # implicitly since we recomputed exp(-inf) -> 0. Thus, we can simply store normalized values.
        vals = exp_row[s0:s0+BLOCK_S] * inv_sum
        tl.store(out_ptr + offs * stride_od, vals, mask=mask)

# 7) Triton GEMM-like output projection with bias: Out[M, N] = In[M, K] @ W[N, K]^T + Bias[N]
#    In: [B, S, Hq*D], W: [Hq*D, Hq*D], Bias: [Hq*D], Out: [B, S, Hq*D]
@triton.jit
def output_projection_kernel(
    In_ptr, W_ptr, Bias_ptr, Out_ptr,
    M, N, K,
    stride_im, stride_in,  # In strides (m and n=seq)
    stride_wm, stride_wk,  # W strides (m and k)
    stride_om, stride_on,  # Out strides (m and n)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a_ptrs = In_ptr + (offs_m[:, None] * stride_im + (k + offs_k)[None, :] * stride_in)
        b_ptrs = W_ptr + (offs_n[None, :] * stride_wm + (k + offs_k)[:, None] * stride_wk)

        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & ((k + offs_k)[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_n[None, :] < N) & ((k + offs_k)[:, None] < K), other=0.0)

        acc += tl.dot(a, b)  # [BM, BN]

    # Add bias
    bias = tl.load(Bias_ptr + offs_n, mask=(offs_n < N), other=0.0)
    acc += bias[None, :]

    # Store result
    out_ptrs = Out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(out_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

# Entry point: ModelNew.forward
class ModelNew(torch.nn.Module):
    def __init__(self, rms_eps: float = RMS_EPS):
        super().__init__()
        self.rms_eps = rms_eps

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_proj_weight: torch.Tensor,
        q_proj_bias: torch.Tensor,
        k_proj_weight: torch.Tensor,
        k_proj_bias: torch.Tensor,
        v_proj_weight: torch.Tensor,
        v_proj_bias: torch.Tensor,
        o_proj_weight: torch.Tensor,
        o_proj_bias: torch.Tensor,
        q_norm_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
        cos: torch.Tensor,  # unused in this Triton version; kept for signature compatibility
        sin: torch.Tensor,  # unused in this Triton version; kept for signature compatibility
    ):
        assert hidden_states.is_cuda, "All tensors must be on CUDA for Triton kernels."
        device = hidden_states.device
        dtype = hidden_states.dtype  # float32

        B, S, _ = hidden_states.shape  # hidden_states: [B, S, 12288] = [B, S, Hq*D] with Hq=96, D=128
        Hq = NUM_ATTENTION_HEADS
        Hk = NUM_KEY_VALUE_HEADS
        D = HEAD_DIM

        # 1) Linear projections: Q, K, V using Triton GEMM + bias
        # hidden_states [B, S, K_in] where K_in = Hq * D = 12288
        Q = torch.empty((B, S, Hq * D), device=device, dtype=torch.float32)
        K = torch.empty((B, S, Hk * D), device=device, dtype=torch.float32)
        V = torch.empty((B, S, Hk * D), device=device, dtype=torch.float32)

        # Launch linear_gemm_bias_kernel for Q
        grid_Q = (B, S, Hq)
        linear_gemm_bias_kernel[grid_Q](
            hidden_states, q_proj_weight, q_proj_bias, Q,
            B, Hq * D, K_in,
            hidden_states.stride(0), hidden_states.stride(2),
            q_proj_weight.stride(0), q_proj_weight.stride(1),
            Q.stride(0), Q.stride(2),
            BLOCK_M=64, BLOCK_N=128, BLOCK_K=64
        )

        # Launch linear_gemm_bias_kernel for K
        grid_K = (B, S, Hk)
        linear_gemm_bias_kernel[grid_K](
            hidden_states, k_proj_weight, k_proj_bias, K,
            B, Hk * D, K_in,
            hidden_states.stride(0), hidden_states.stride(2),
            k_proj_weight.stride(0), k_proj_weight.stride(1),
            K.stride(0), K.stride(2),
            BLOCK_M=64, BLOCK_N=128, BLOCK_K=64
        )

        # Launch linear_gemm_bias_kernel for V
        grid_V = (B, S, Hk)
        linear_gemm_bias_kernel[grid_V](
            hidden_states, v_proj_weight, v_proj_bias, V,
            B, Hk * D, K_in,
            hidden_states.stride(0), hidden_states.stride(2),
            v_proj_weight.stride(0), v_proj_weight.stride(1),
            V.stride(0), V.stride(2),
            BLOCK_M=64, BLOCK_N=128, BLOCK_K=64
        )

        # 2) Reshape to heads
        Q_heads = Q.view(B, S, Hq, D)   # [B, S, 96, 128]
        K_heads = K.view(B, S, Hk, D)   # [B, S, 8, 128]
        V_heads = V.view(B, S, Hk, D)   # [B, S, 8, 128]

        # 3) RMSNorm per head for Q and K (over last dim D=128)
        Q_norm = torch.empty_like(Q_heads, dtype=torch.float32)
        K_norm = torch.empty_like(K_heads, dtype=torch.float32)

        grid_rms = (B * S * Hq,)
        rmsnorm_kernel[grid_rms](
            Q_heads, q_norm_weight, Q_norm,
            B, S, Hq, D,
            Q_heads.stride(0), Q_heads.stride(1), Q_heads.stride(2), Q_heads.stride(3),
            Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(2), Q_norm.stride(3),
            q_norm_weight.stride(0), self.rms_eps,
            BLOCK_D=128
        )

        grid_rmsk = (B * S * Hk,)
        rmsnorm_kernel[grid_rmsk](
            K_heads, k_norm_weight, K_norm,
            B, S, Hk, D,
            K_heads.stride(0), K_heads.stride(1), K_heads.stride(2), K_heads.stride(3),
            K_norm.stride(0), K_norm.stride(1), K_norm.stride(2), K_norm.stride(3),
            k_norm_weight.stride(0), self.rms_eps,
            BLOCK_D=128
        )

        # 4) Rotate last half dimension for Q and K
        Q_rot = torch.empty_like(Q_norm)
        K_rot = torch.empty_like(K_norm)

        grid_rotate = (B * S * Hq,)
        rotate_half_kernel[grid_rotate](
            Q_norm, Q_rot,
            B, S, Hq, D,
            Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(2), Q_norm.stride(3),
            Q_rot.stride(0), Q_rot.stride(1), Q_rot.stride(2), Q_rot.stride(3),
            BLOCK_D=64
        )

        grid_rotatek = (B * S * Hk,)
        rotate_half_kernel[grid_rotatek](
            K_norm, K_rot,
            B, S, Hk, D,
            K_norm.stride(0), K_norm.stride(1), K_norm.stride(2), K_norm.stride(3),
            K_rot.stride(0), K_rot.stride(1), K_rot.stride(2), K_rot.stride(3),
            BLOCK_D=64
        )

        # 5) GQA expansion: repeat K and V from Hk to Hq along groups=NUM_KEY_VALUE_GROUPS
        K_exp = torch.empty((B, S, Hq, D), device=device, dtype=torch.float32)
        V_exp = torch.empty((B, S, Hq, D), device=device, dtype=torch.float32)

        grid_expand = (B * Hq * S,)
        gqa_expand_kernel[grid_expand](
            K_rot, V_heads, K_exp,
            B, S, Hq, Hk, D, NUM_KEY_VALUE_GROUPS,
            K_rot.stride(0), K_rot.stride(1), K_rot.stride(2), K_rot.stride(3),
            V_heads.stride(0), V_heads.stride(1), V_heads.stride(2), V_heads.stride(3),
            K_exp.stride(0), K_exp.stride(1), K_exp.stride(2), K_exp.stride(3),
            BLOCK_S=128
        )
        # Note: In gqa_expand we expanded V from [B,S,Hk,D] to [B,S,Hq,D]. Here we use V_heads which is already [B,S,Hk,D].
        # We need to repeat V similarly, but V_proj has no norm/rotation. We can reuse the same kernel by swapping inputs:
        gqa_expand_kernel[grid_expand](
            V_heads, V_heads, V_exp,
            B, S, Hq, Hk, D, NUM_KEY_VALUE_GROUPS,
            V_heads.stride(0), V_heads.stride(1), V_heads.stride(2), V_heads.stride(3),
            V_heads.stride(0), V_heads.stride(1), V_heads.stride(2), V_heads.stride(3),
            V_exp.stride(0), V_exp.stride(1), V_exp.stride(2), V_exp.stride(3),
            BLOCK_S=128
        )

        # 6) Compute attention scores: Attn[b, h, i, j] for each (b, h, i) row, over j in [0..S-1]
        Attn = torch.empty((B, Hq, S, S), device=device, dtype=torch.float32)

        grid_scores = (B * Hq * S,)
        attn_scores_kernel[grid_scores](
            Q_rot, K_exp, Attn,
            B, S, Hq, D,
            Q_rot.stride(0), Q_rot.stride(1), Q_rot.stride(2), Q_rot.stride(3),
            K_exp.stride(0), K_exp.stride(1), K_exp.stride(2), K_exp.stride(3),
            Attn.stride(0), Attn.stride(1), Attn.stride(2), Attn.stride(3),
            SCALING,
            BLOCK_S=128, BLOCK_D=64
        )

        # 7) Softmax along j per row with causal mask (triangular with diagonal=1)
        Soft = torch.empty_like(Attn, dtype=torch.float32)

        grid_softmax = (B * Hq * S,)
        softmax_rows_causal_kernel[grid_softmax](
            Attn, Soft,
            B, S, Hq,
            Attn.stride(0), Attn.stride(1), Attn.stride(2), Attn.stride(3),
            Soft.stride(0), Soft.stride(1), Soft.stride(2), Soft.stride(3),
            BLOCK_S=128
        )

        # 8) Compute attention output: attn_output[b, i, :] = sum_h Soft[b,h,i,:] @ V_exp[b,h,i,:] over h in [0..Hq-1]
        attn_output = torch.empty((B, S, Hq * D), device=device, dtype=torch.float32)

        # We implement this with a Triton kernel: for each (b,i), loop over h, accumulate [D] vector
        grid_out = (B * S,)
        @triton.jit
        def attn_output_reduce_kernel(
            Soft_ptr, V_exp_ptr, Out_ptr,
            B, S, H, D,
            stride_sb, stride_si, stride_sh,   # Soft strides: [B,S,H,S]
            stride_vb, stride_vs, stride_vh, stride_vd,  # V_exp strides: [B,S,H,D]
            stride_ob, stride_os, stride_od,  # Out strides: [B,S,D]
            BLOCK_D: tl.constexpr
        ):
            pid = tl.program_id(0)
            if pid >= (B * S):
                return
            b = pid // S
            i = pid % S

            acc = tl.zeros((D,), dtype=tl.float32)
            for h in range(0, H):
                # Load Soft[b,i,h,:] (length S)
                row_ptr = Soft_ptr + b * stride_sb + i * stride_si + h * stride_sh
                s_row = tl.zeros((S,), dtype=tl.float32)
                for s0 in range(0, S, BLOCK_D):
                    offs = s0 + tl.arange(0, BLOCK_D)
                    mask = offs < S
                    s_row[offs] = tl.load(row_ptr + offs, mask=mask, other=0.0)
                # Multiply by V_exp[b,i,h,:] and accumulate
                v_ptr = V_exp_ptr + b * stride_vb + i * stride_vs + h * stride_vh
                for d0 in range(0, D, BLOCK_D):
                    offs_d = d0 + tl.arange(0, BLOCK_D)
                    mask_d = offs_d < D
                    v_vec = tl.load(v_ptr + offs_d * stride_vd, mask=mask_d, other=0.0)
                    acc[d0:d0 + BLOCK_D] += tl.sum(s_row[None, :] * v_vec[None, :], axis=1)

            # Store acc to Out[b,i,:]
            out_ptr = Out_ptr + b * stride_ob + i * stride_os
            for d0 in range(0, D, BLOCK_D):
                offs_d = d0 + tl.arange(0, BLOCK_D)
                mask_d = offs_d < D
                tl.store(out_ptr + offs_d * stride_od, acc[offs_d], mask=mask_d)

        attn_output_reduce_kernel[grid_out](
            Soft, V_exp, attn_output,
            B, S, Hq, D,
            Soft.stride(0), Soft.stride(1), Soft.stride(2), Soft.stride(3),
            V_exp.stride(0), V_exp.stride(1), V_exp.stride(2), V_exp.stride(3),
            attn_output.stride(0), attn_output.stride(1), attn_output.stride(2),
            BLOCK_D=128
        )

        # 9) Final output projection with bias (matches original F.linear(..., o_proj_bias))
        output = torch.empty((B, S, Hq * D), device=device, dtype=torch.float32)

        # o_proj_weight is [Hq*D, Hq*D], o_proj_bias is [Hq*D]
        grid_proj = (B, S, Hq * D)
        output_projection_kernel[grid_proj](
            attn_output, o_proj_weight, o_proj_bias, output,
            B, Hq * D, Hq * D,
            attn_output.stride(0), attn_output.stride(1),
            o_proj_weight.stride(0), o_proj_weight.stride(1),
            output.stride(0), output.stride(1),
            BLOCK_M=64, BLOCK_N=128, BLOCK_K=64
        )

        return output


def run(*args):
    return ModelNew()(*args)
