import torch
import triton
import triton.language as tl

# Constants derived from the original code
NUM_ATTENTION_HEADS = 96
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
NUM_KEY_VALUE_GROUPS = 12
SCALING = 1.0 / (HEAD_DIM ** 0.5)  # 1/sqrt(128)
RMS_EPS = 1e-6

# 1) Triton GEMM for linear projection with bias: C[M, N] = A[M, K] @ B[N, K]^T + Bias[N]
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

# 2) Triton RMSNorm per head over last dim (HEAD_DIM) for Q and K: per-row normalization
@triton.jit
def rms_norm_lastdim_kernel(
    Input_ptr, Weight_ptr, Output_ptr,
    B, H, D,
    stride_ib, stride_ih, stride_id,   # Input strides: [B,H,D]
    stride_w,                                     # Weight stride: [D]
    stride_ob, stride_oh, stride_od,             # Output strides: [B,H,D]
    eps: tl.constexpr,                           # epsilon
    BLOCK_D: tl.constexpr
):
    # Each program handles one row: (b,h)
    pid = tl.program_id(0)
    total = B * H
    if pid >= total:
        return

    b = pid // H
    h = pid % H

    # Compute row pointer
    row_ptr = Input_ptr + b * stride_ib + h * stride_ih  # base pointer for this (b,h) row

    # Load row and compute variance over last dim
    sum_sq = 0.0
    for d0 in range(0, D, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        x = tl.load(row_ptr + d * stride_id, mask=d < D, other=0.0)
        sum_sq += tl.sum(x * x)
    mean_sq = sum_sq / D
    norm = tl.rsqrt(mean_sq + eps)  # 1/sqrt(var+eps)

    # Scale and apply learnable weight, store
    for d0 in range(0, D, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        x = tl.load(row_ptr + d * stride_id, mask=d < D, other=0.0)
        w = tl.load(Weight_ptr + d * stride_w, mask=d < D, other=1.0)
        y = x * norm * w
        out_ptr = Output_ptr + b * stride_ob + h * stride_oh + d * stride_od
        tl.store(out_ptr, y, mask=d < D)

# 3) Triton rotate_half kernel for Q and K: swap and negate second half (dims D/2=64)
@triton.jit
def rotate_half_kernel(
    Input_ptr, Output_ptr,
    D,                      # last dim length (128)
    stride_ib, stride_id,   # Input strides: [B,H,D]
    stride_ob, stride_od,   # Output strides: [B,H,D]
    BLOCK_D: tl.constexpr   # tile size along D
):
    pid = tl.program_id(0)
    total = B * H  # same as before
    if pid >= total:
        return

    b = pid // H
    h = pid % H

    in_row_ptr = Input_ptr + b * stride_ib + h * stride_ih
    out_row_ptr = Output_ptr + b * stride_ob + h * stride_oh

    # First half
    for d0 in range(0, D // 2, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        x1 = tl.load(in_row_ptr + d * stride_id, mask=d < (D // 2), other=0.0)
        tl.store(out_row_ptr + d * stride_od, x1, mask=d < (D // 2))

    # Second half: negate first half values and place into second half positions
    for d0 in range(0, D // 2, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        x2 = tl.load(in_row_ptr + (d + (D // 2)) * stride_id, mask=d < (D // 2), other=0.0)
        y = -x2
        tl.store(out_row_ptr + ((d + (D // 2)) * stride_od), y, mask=d < (D // 2))

# 4) Triton GQA expand K/V from 8 heads to 96 heads via NUM_KEY_VALUE_GROUPS=12
@triton.jit
def gqa_expand_kernel(
    Input_ptr, Output_ptr,
    B, S, H_in, D, G,               # Input: [B,S,H_in,D], H_in=NUM_KEY_VALUE_HEADS, D=HEAD_DIM, G=NUM_KEY_VALUE_GROUPS
    stride_ib, stride_is, stride_ih, stride_id,
    stride_ob, stride_os, stride_oh, stride_od,
    BLOCK_D: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h_out = tl.program_id(2)

    b = pid_b
    s = pid_s
    h_out = pid_h_out

    # Map output head to input head and group
    h_in = h_out // G
    g = h_out % G

    # Compute base pointers
    in_ptr = Input_ptr + b * stride_ib + s * stride_is + h_in * stride_ih
    out_ptr = Output_ptr + b * stride_ob + s * stride_os + h_out * stride_oh

    # Copy row from input to all repeated groups
    for d0 in range(0, D, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        x = tl.load(in_ptr + d * stride_id, mask=d < D, other=0.0)
        tl.store(out_ptr + d * stride_od, x, mask=d < D)

# 5) Triton attention scores kernel: per (b,h,i) compute S[j] = sum_k Q_norm[b,h,i,k] * K_exp[b,h,j,k] * SCALING
@triton.jit
def attn_scores_rowwise_kernel(
    Q_ptr, K_ptr, Out_ptr,
    B, H, S, D, SCALING,
    stride_qb, stride_qh, stride_qd,     # Q strides: [B,H,D]
    stride_kb, stride_kh, stride_kd,     # K strides after expansion: [B,H,S,D]
    stride_ob, stride_oh, stride_os, stride_od,  # Out strides: [B,H,S,S]
    BLOCK_D: tl.constexpr
):
    pid = tl.program_id(0)
    total = B * H * S
    if pid >= total:
        return

    b = pid // (H * S)
    h = (pid % (H * S)) // S
    i = pid % S

    q_row_ptr = Q_ptr + b * stride_qb + h * stride_qh
    out_row_ptr = Out_ptr + b * stride_ob + h * stride_oh + i * stride_os  # vector base for j dim

    # Accumulate S[j]
    S_vec = tl.zeros((S,), dtype=tl.float32)
    for d0 in range(0, D, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        q = tl.load(q_row_ptr + d * stride_qd, mask=d < D, other=0.0)  # [BLOCK_D]
        for j in range(0, S):
            k_row_ptr = K_ptr + b * stride_kb + h * stride_kh + j * stride_kd
            k = tl.load(k_row_ptr + d * stride_kd, mask=d < D, other=0.0)
            S_vec[j] += tl.sum(q * k)  # scalar accumulate for this j
    # Store
    for j in range(0, S):
        tl.store(out_row_ptr + j * stride_od, S_vec[j])

# 6) Triton softmax along last dim (columns) per row with causal mask (j > i)
@triton.jit
def softmax_rows_causal_kernel(
    In_ptr, Out_ptr,
    B, H, S,
    stride_ib, stride_ih, stride_is,      # In strides: [B,H,S]
    stride_ob, stride_oh, stride_os,      # Out strides: [B,H,S]
    BLOCK_S: tl.constexpr
):
    pid = tl.program_id(0)
    total = B * H * S
    if pid >= total:
        return

    b = pid // (H * S)
    h = (pid % (H * S)) // S
    i = pid % S

    in_row_ptr = In_ptr + b * stride_ib + h * stride_ih + i * stride_is
    out_row_ptr = Out_ptr + b * stride_ob + h * stride_oh + i * stride_os

    # Load row, apply causal mask: set j <= i to -inf
    max_val = -float('inf')
    for s0 in range(0, S, BLOCK_S):
        s = s0 + tl.arange(0, BLOCK_S)
        mask_s = s < S
        x = tl.load(in_row_ptr + s * stride_is, mask=mask_s, other=-float('inf'))
        # causal: for positions where s <= i, set to -inf
        causal = s <= i
        x = tl.where(causal, -float('inf'), x)
        max_val = tl.maximum(max_val, tl.max(tl.where(mask_s, x, -float('inf'))))

    sum_exp = 0.0
    for s0 in range(0, S, BLOCK_S):
        s = s0 + tl.arange(0, BLOCK_S)
        mask_s = s < S
        x = tl.load(in_row_ptr + s * stride_is, mask=mask_s, other=-float('inf'))
        x = tl.where(s <= i, -float('inf'), x)
        e = tl.exp(x - max_val)
        sum_exp += tl.sum(tl.where(mask_s, e, 0.0))

    for s0 in range(0, S, BLOCK_S):
        s = s0 + tl.arange(0, BLOCK_S)
        mask_s = s < S
        x = tl.load(in_row_ptr + s * stride_is, mask=mask_s, other=-float('inf'))
        x = tl.where(s <= i, -float('inf'), x)
        e = tl.exp(x - max_val)
        y = e / sum_exp
        tl.store(out_row_ptr + s * stride_os, y, mask=mask_s)

# 7) Triton output projection: [B,S,Dq] @ [Dq,Dq] -> [B,S,Dq] (no bias)
@triton.jit
def output_projection_kernel(
    Input_ptr, Weight_ptr, Output_ptr,
    M, N, K,                 # M=B*S, N=K=Hq*D=Dq (e.g., 12288), K=Dq
    stride_im, stride_in,    # Input strides: [M,N]
    stride_wk, stride_wn,    # Weight strides: [K,N]
    stride_om, stride_on,    # Output strides: [M,N]
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k  # [BLOCK_K]

        # Load A: Input[M,K] block
        A_ptrs = Input_ptr + offs_m[:, None] * stride_im + k[None, :] * stride_in
        A = tl.load(A_ptrs, mask=(offs_m[:, None] < M) & (k[None, :] < K), other=0.0)

        # Load B: Weight[K,N] block
        W_ptrs = Weight_ptr + k[:, None] * stride_wk + offs_n[None, :] * stride_wn
        B = tl.load(W_ptrs, mask=(k[:, None] < K) & (offs_n[None, :] < N), other=0.0)

        acc += tl.dot(A, B)  # [BM,BN]

    # Store result
    O_ptrs = Output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(O_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

# Entry point: ModelNew forward
class ModelNew(torch.nn.Module):
    def __init__(self, rms_eps: float = RMS_EPS):
        super().__init__()
        self.rms_eps = rms_eps

    def forward(self, hidden_states, q_proj_weight, q_proj_bias,
                k_proj_weight, k_proj_bias,
                v_proj_weight, v_proj_bias,
                o_proj_weight, q_norm_weight, k_norm_weight,
                cos, sin):
        # Ensure all inputs are on CUDA and dtype is float32 (original hidden_states dtype)
        assert hidden_states.is_cuda, "All tensors must be on CUDA device for Triton kernels."
        device = hidden_states.device
        dtype = hidden_states.dtype
        B, S, hidden_dim_in = hidden_states.shape

        # 1) Linear projections with bias
        Q = torch.empty((B, S, NUM_ATTENTION_HEADS * HEAD_DIM), device=device, dtype=dtype)
        K = torch.empty((B, S, NUM_KEY_VALUE_HEADS * HEAD_DIM), device=device, dtype=dtype)
        V = torch.empty((B, S, NUM_KEY_VALUE_HEADS * HEAD_DIM), device=device, dtype=dtype)

        # Launch linear_gemm_bias_kernel for Q
        M_q = B * S
        N_q = NUM_ATTENTION_HEADS * HEAD_DIM
        K_q = hidden_dim_in  # input hidden size (3*3*128 = 12288)
        grid_q = (triton.cdiv(M_q, 64), triton.cdiv(N_q, 128))
        linear_gemm_bias_kernel[grid_q](
            hidden_states, q_proj_weight, q_proj_bias, Q,
            M_q, N_q, K_q,
            hidden_states.stride(0), hidden_states.stride(1),
            q_proj_weight.stride(0), q_proj_weight.stride(1),
            Q.stride(0), Q.stride(1),
            BLOCK_M=64, BLOCK_N=128, BLOCK_K=64
        )

        # Launch linear_gemm_bias_kernel for K
        M_k = B * S
        N_k = NUM_KEY_VALUE_HEADS * HEAD_DIM
        K_k = hidden_dim_in
        grid_k = (triton.cdiv(M_k, 64), triton.cdiv(N_k, 128))
        linear_gemm_bias_kernel[grid_k](
            hidden_states, k_proj_weight, k_proj_bias, K,
            M_k, N_k, K_k,
            hidden_states.stride(0), hidden_states.stride(1),
            k_proj_weight.stride(0), k_proj_weight.stride(1),
            K.stride(0), K.stride(1),
            BLOCK_M=64, BLOCK_N=128, BLOCK_K=64
        )

        # Launch linear_gemm_bias_kernel for V
        M_v = B * S
        N_v = NUM_KEY_VALUE_HEADS * HEAD_DIM
        K_v = hidden_dim_in
        grid_v = (triton.cdiv(M_v, 64), triton.cdiv(N_v, 128))
        linear_gemm_bias_kernel[grid_v](
            hidden_states, v_proj_weight, v_proj_bias, V,
            M_v, N_v, K_v,
            hidden_states.stride(0), hidden_states.stride(1),
            v_proj_weight.stride(0), v_proj_weight.stride(1),
            V.stride(0), V.stride(1),
            BLOCK_M=64, BLOCK_N=128, BLOCK_K=64
        )

        # 2) Reshape to heads
        Q_heads = Q.view(B, S, NUM_ATTENTION_HEADS, HEAD_DIM)    # [B, S, 96, 128]
        K_heads = K.view(B, S, NUM_KEY_VALUE_HEADS, HEAD_DIM)   # [B, S, 8, 128]
        V_heads = V.view(B, S, NUM_KEY_VALUE_HEADS, HEAD_DIM)   # [B, S, 8, 128]

        # 3) RMSNorm per head for Q and K (over last dim)
        Q_norm = torch.empty_like(Q_heads, device=device, dtype=torch.float32)
        K_norm = torch.empty_like(K_heads, device=device, dtype=torch.float32)

        grid_rms = (B * NUM_ATTENTION_HEADS,)
        rms_norm_lastdim_kernel[grid_rms](
            Q_heads, q_norm_weight, Q_norm,
            B, NUM_ATTENTION_HEADS, HEAD_DIM,
            Q_heads.stride(0), Q_heads.stride(1), Q_heads.stride(3),
            q_norm_weight.stride(0),
            Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(3),
            eps=self.rms_eps, BLOCK_D=128
        )

        grid_rms_k = (B * NUM_KEY_VALUE_HEADS,)
        rms_norm_lastdim_kernel[grid_rms_k](
            K_heads, k_norm_weight, K_norm,
            B, NUM_KEY_VALUE_HEADS, HEAD_DIM,
            K_heads.stride(0), K_heads.stride(1), K_heads.stride(3),
            k_norm_weight.stride(0),
            K_norm.stride(0), K_norm.stride(1), K_norm.stride(3),
            eps=self.rms_eps, BLOCK_D=128
        )

        # 4) Rotate half for Q and K (swap and negate second half)
        Q_rot = torch.empty_like(Q_norm, device=device, dtype=torch.float32)
        K_rot = torch.empty_like(K_norm, device=device, dtype=torch.float32)

        grid_rotate = (B * NUM_ATTENTION_HEADS,)
        rotate_half_kernel[grid_rotate](
            Q_norm, Q_rot,
            HEAD_DIM,
            Q_norm.stride(0), Q_norm.stride(3),
            Q_rot.stride(0), Q_rot.stride(3),
            BLOCK_D=128
        )

        grid_rotate_k = (B * NUM_KEY_VALUE_HEADS,)
        rotate_half_kernel[grid_rotate_k](
            K_norm, K_rot,
            HEAD_DIM,
            K_norm.stride(0), K_norm.stride(3),
            K_rot.stride(0), K_rot.stride(3),
            BLOCK_D=128
        )

        # 5) GQA expand K and V from 8 heads to 96 heads via NUM_KEY_VALUE_GROUPS=12
        K_expanded = torch.empty((B, S, NUM_ATTENTION_HEADS, HEAD_DIM), device=device, dtype=torch.float32)
        V_expanded = torch.empty((B, S, NUM_ATTENTION_HEADS, HEAD_DIM), device=device, dtype=torch.float32)

        grid_gqa = (B, S, NUM_ATTENTION_HEADS)
        gqa_expand_kernel[grid_gqa](
            K_rot, K_expanded,
            B, S, NUM_KEY_VALUE_HEADS, HEAD_DIM, NUM_KEY_VALUE_GROUPS,
            K_rot.stride(0), K_rot.stride(1), K_rot.stride(3),
            K_expanded.stride(0), K_expanded.stride(1), K_expanded.stride(2), K_expanded.stride(3),
            BLOCK_D=128
        )

        gqa_expand_kernel[grid_gqa](
            V_rot, V_expanded,
            B, S, NUM_KEY_VALUE_HEADS, HEAD_DIM, NUM_KEY_VALUE_GROUPS,
            V_rot.stride(0), V_rot.stride(1), V_rot.stride(3),
            V_expanded.stride(0), V_expanded.stride(1), V_expanded.stride(2), V_expanded.stride(3),
            BLOCK_D=128
        )

        # For consistency, here V_rot isn't defined; we should have normalized V as well. Fix by normalizing V:
        V_rot = torch.empty_like(V_heads, device=device, dtype=torch.float32)
        grid_rotate_v = (B * NUM_KEY_VALUE_HEADS,)
        rotate_half_kernel[grid_rotate_v](
            V_norm, V_rot,
            HEAD_DIM,
            V_norm.stride(0), V_norm.stride(3),
            V_rot.stride(0), V_rot.stride(3),
            BLOCK_D=128
        )
        # Correct: we normalized Q and K above. V rotation is not used in attention weight, but we need expanded V.
        # So we can use K_rot as expanded K and V_norm as V (but original code rotates both). We already have V_rot
        # from V_norm rotated; we don't use it for attention score. The original attention uses rotated K, but not V.
        # However, original code applies rotation to both Q and K. V is used in output as value, no rotation applied.
        # To match original, we should not rotate V. Let's recompute V_expanded from original V_heads (no rotation).
        # But we don't have original V_heads untouched; we normalized V, then expanded. The original code rotates Q and K,
        # but does not rotate V. We can expand V_norm without rotation:
        V_expanded_correct = torch.empty((B, S, NUM_ATTENTION_HEADS, HEAD_DIM), device=device, dtype=torch.float32)
        grid_gqa_v = (B, S, NUM_ATTENTION_HEADS)
        gqa_expand_kernel[grid_gqa_v](
            V_norm, V_expanded_correct,
            B, S, NUM_KEY_VALUE_HEADS, HEAD_DIM, NUM_KEY_VALUE_GROUPS,
            V_norm.stride(0), V_norm.stride(1), V_norm.stride(3),
            V_expanded_correct.stride(0), V_expanded_correct.stride(1), V_expanded_correct.stride(2), V_expanded_correct.stride(3),
            BLOCK_D=128
        )

        # 6) Compute attention scores S[b,h,i,:] per row using Triton
        attn_scores = torch.empty((B, NUM_ATTENTION_HEADS, S, S), device=device, dtype=torch.float32)

        grid_as = (B * NUM_ATTENTION_HEADS * S,)
        attn_scores_rowwise_kernel[grid_as](
            Q_rot, K_expanded, attn_scores,
            B, NUM_ATTENTION_HEADS, S, HEAD_DIM, SCALING,
            Q_rot.stride(0), Q_rot.stride(1), Q_rot.stride(3),
            K_expanded.stride(0), K_expanded.stride(1), K_expanded.stride(2), K_expanded.stride(3),
            attn_scores.stride(0), attn_scores.stride(1), attn_scores.stride(2), attn_scores.stride(3),
            BLOCK_D=64
        )

        # 7) Softmax along last dim with causal mask: apply in Triton
        attn_out = torch.empty_like(attn_scores, device=device, dtype=torch.float32)

        grid_soft = (B * NUM_ATTENTION_HEADS * S,)
        softmax_rows_causal_kernel[grid_soft](
            attn_scores, attn_out,
            B, NUM_ATTENTION_HEADS, S,
            attn_scores.stride(0), attn_scores.stride(1), attn_scores.stride(2),
            attn_out.stride(0), attn_out.stride(1), attn_out.stride(2),
            BLOCK_S=64
        )

        # 8) Compute attention output: attn_output[b,i,:] = attn_out[b,:,i,:] (sum over heads). Shape [B,S,12288]
        attn_output = torch.empty((B, S, NUM_ATTENTION_HEADS * HEAD_DIM), device=device, dtype=torch.float32)

        # Triton kernel: for each (b,i), sum over h
        grid_out = (B * S,)
        attn_output_kernel[grid_out](
            attn_out, attn_output,
            B, S, NUM_ATTENTION_HEADS * HEAD_DIM,
            attn_out.stride(0), attn_out.stride(1), attn_out.stride(2),
            attn_output.stride(0), attn_output.stride(1), attn_output.stride(2),
            BLOCK_N=256
        )

        # 9) Output projection (no bias): attn_output [B,S,Dq] @ o_proj_weight [Dq,Dq] -> [B,S,Dq]
        # Here Dq = Hq * HEAD_DIM = 12288
        Output = torch.empty_like(attn_output, device=device, dtype=torch.float32)

        M = B * S
        N = NUM_ATTENTION_HEADS * HEAD_DIM  # 12288
        K = N  # Hq*D

        grid_proj = (triton.cdiv(M, 128), triton.cdiv(N, 128))
        output_projection_kernel[grid_proj](
            attn_output, o_proj_weight, Output,
            M, N, K,
            attn_output.stride(0), attn_output.stride(1),
            o_proj_weight.stride(0), o_proj_weight.stride(1),
            Output.stride(0), Output.stride(1),
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=128
        )

        # Return final output (float32, contiguous) with shape [B, S, 12288]
        return Output


def run(*args):
    return ModelNew()(*args)
