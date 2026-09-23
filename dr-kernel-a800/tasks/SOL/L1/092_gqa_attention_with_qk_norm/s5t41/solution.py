import torch
import triton
import triton.language as tl

# Constants from the original code
NUM_ATTENTION_HEADS = 96
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
NUM_KEY_VALUE_GROUPS = 12
SCALING = 1.0 / (HEAD_DIM ** 0.5)
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

# 2) Triton RMSNorm per head over last dim (HEAD_DIM) for Q and K:
#   For each row (b, h, :), compute mean of squares over HEAD_DIM, then normalize and scale by weight.
@triton.jit
def rms_norm_kernel(
    X_ptr, Weight_ptr, Y_ptr,
    B, S, H, D,
    stride_xb, stride_xh, stride_xd,
    stride_yb, stride_yh, stride_yd,
    stride_w,  # length H * D
    eps,
    BLOCK_D: tl.constexpr
):
    pid = tl.program_id(0)
    total = B * H
    if pid >= total:
        return
    b = pid // H
    h = pid % H

    # Compute mean of squares over D
    sum_sq = 0.0
    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        mask = offs_d < D
        x = tl.load(X_ptr + b * stride_xb + h * stride_xh + offs_d * stride_xd, mask=mask, other=0.0)
        x32 = x.to(tl.float32)
        sum_sq += tl.sum(x32 * x32, axis=0)

    mean = sum_sq / D
    inv_rms = tl.rsqrt(mean + eps)

    # Normalize and scale by weight
    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        mask = offs_d < D
        x = tl.load(X_ptr + b * stride_xb + h * stride_xh + offs_d * stride_xd, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(Weight_ptr + (h * D + offs_d) * stride_w, mask=mask, other=0.0).to(tl.float32)
        y = x * inv_rms * w
        tl.store(Y_ptr + b * stride_yb + h * stride_yh + offs_d * stride_yd, y, mask=mask)

# 3) Triton kernel for rotation: swap last half for Q and K using cos/sin
#    Assumes HEAD_DIM is even, here 128 => half=64.
@triton.jit
def rotate_half_kernel(
    X_ptr, Cos_ptr, Sin_ptr, Y_ptr,
    B, S, H, D, HALF,
    stride_xb, stride_xh, stride_xd,
    stride_yb, stride_yh, stride_yd,
    stride_c, stride_s,
    BLOCK_D: tl.constexpr
):
    pid = tl.program_id(0)
    total = B * H
    if pid >= total:
        return
    b = pid // H
    h = pid % H

    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D

        # Load original x
        x = tl.load(X_ptr + b * stride_xb + h * stride_xh + offs_d * stride_xd, mask=mask_d, other=0.0).to(tl.float32)

        # Split into halves
        mask1 = mask_d & (offs_d < HALF)
        mask2 = mask_d & (offs_d >= HALF)
        d1 = offs_d[mask1] - HALF
        d2 = offs_d[mask2] - HALF

        # Load q1, q2 halves from original x (d1, d2 map to corresponding indices)
        q1 = tl.load(X_ptr + b * stride_xb + h * stride_xh + d1 * stride_xd, mask=mask1, other=0.0).to(tl.float32)
        q2 = tl.load(X_ptr + b * stride_xb + h * stride_xh + d2 * stride_xd, mask=mask2, other=0.0).to(tl.float32)

        # Load cos/sin for this row
        cos_val = tl.load(Cos_ptr + h * stride_c, mask=True, other=0.0).to(tl.float32)
        sin_val = tl.load(Sin_ptr + h * stride_s, mask=True, other=0.0).to(tl.float32)

        # Rotate halves: new_x[d] = x[d] for d<HALF; for d>=HALF: x[d]*cos + x[d-HALF]*sin
        rotated2 = q1 * cos_val + q2 * sin_val

        # Store rotated for d>=HALF positions
        tl.store(Y_ptr + b * stride_yb + h * stride_yh + offs_d * stride_yd, rotated2, mask=mask2)
        # Store original for d<HALF positions
        tl.store(Y_ptr + b * stride_yb + h * stride_yh + offs_d * stride_yd, x, mask=mask1)

# 4) Triton GQA expansion: repeat K/V from Hk heads to Hq heads along NUM_KEY_VALUE_GROUPS
#    Input: [B, Hk, S, D], Output: [B, Hq, S, D] where for each h in [0..Hk-1], group g in [0..NUM_KEY_VALUE_GROUPS-1]:
#          Y[b, h*NUM_KEY_VALUE_GROUPS + g, s, d] = X[b, h, s, d]
@triton.jit
def gqa_expand_kernel(
    X_ptr, Y_ptr,
    B, Hk, S, D, Hq, G,
    stride_xb, stride_xh, stride_xs, stride_xd,
    stride_yb, stride_yh, stride_ys, stride_yd,
    BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid = tl.program_id(0)
    total = B * Hq
    if pid >= total:
        return
    b = pid // Hq
    h_q = pid % Hq
    h_k = h_q % Hk  # group index
    g = h_q // Hk   # which group to map to
    if g >= G:
        return

    for s0 in range(0, S, BLOCK_S):
        offs_s = s0 + tl.arange(0, BLOCK_S)
        mask_s = offs_s < S
        for d0 in range(0, D, BLOCK_D):
            offs_d = d0 + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D
            x = tl.load(
                X_ptr + b * stride_xb + h_k * stride_xh + offs_s[:, None] * stride_xs + offs_d[None, :] * stride_xd,
                mask=mask_s[:, None] & mask_d[None, :],
                other=0.0
            ).to(tl.float32)
            tl.store(
                Y_ptr + b * stride_yb + h_q * stride_yh + offs_s[:, None] * stride_ys + offs_d[None, :] * stride_yd,
                x,
                mask=mask_s[:, None] & mask_d[None, :]
            )

# 5) Triton kernel to compute attention scores: Attn[b, h, i, j] = sum_k Q_norm[b, h, i, k] * K_exp[b, h, j, k] * SCALING
#    We compute one (b, h, i) row and store all j in the output [B, S, H, S].
@triton.jit
def attention_scores_kernel(
    Q_ptr, K_ptr, Attn_ptr,
    B, S, H, D,
    stride_qb, stride_qh, stride_qi, stride_qk,
    stride_kb, stride_kh, stride_kj, stride_kk,
    stride_ab, stride_ah, stride_ai, stride_aj,
    SCALING: tl.constexpr,
    BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid_row = tl.program_id(0)
    total_rows = B * H * S
    if pid_row >= total_rows:
        return
    b = pid_row // (H * S)
    h = (pid_row % (H * S)) // S
    i = pid_row % S

    # Accumulate over K dimension
    acc = tl.zeros((S,), dtype=tl.float32)

    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        # Load Q[i, :] vector across d
        q_vec = tl.load(Q_ptr + b * stride_qb + h * stride_qh + i * stride_qi + offs_d * stride_qk,
                        mask=mask_d, other=0.0).to(tl.float32)
        # For each j, load K[j, :] vector across d and accumulate
        for j in range(0, S, BLOCK_S):
            offs_j = j + tl.arange(0, BLOCK_S)
            mask_j = offs_j < S
            k_vec = tl.load(K_ptr + b * stride_kb + h * stride_kh + offs_j * stride_kj + offs_d[None, :] * stride_kk,
                            mask=mask_j[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
            # dot per block
            acc += tl.sum(q_vec[None, :] * k_vec, axis=1)  # shape (BLOCK_S,)
        # No need to multiply by scaling here; will scale outside in host before softmax

    # Store the computed scores into Attn[b, h, i, :]
    attn_ptrs = Attn_ptr + b * stride_ab + h * stride_ah + i * stride_ai + tl.arange(0, S) * stride_aj
    tl.store(attn_ptrs, acc, mask=True)

# 6) Triton kernel for softmax along the last dimension (j) per (b, h, i) with causal mask: j > i
#    Input: Attn[B, S, H, S], Output: Soft[B, S, H, S]
@triton.jit
def softmax_rows_causal_kernel(
    In_ptr, Out_ptr,
    B, S, H,
    stride_ib, stride_is, stride_ih, stride_ij,   # In strides
    stride_ob, stride_os, stride_oh, stride_ooj,  # Out strides
    BLOCK_S: tl.constexpr
):
    pid = tl.program_id(0)
    total = B * H
    if pid >= total:
        return
    b = pid // H
    h = pid % H

    # Compute row max after masking: j > i
    max_val = -float('inf')
    for j0 in range(0, S, BLOCK_S):
        offs_j = j0 + tl.arange(0, BLOCK_S)
        mask_j = offs_j > S - 1  # always true but kept simple; we'll use causal mask in loads
        in_ptrs = In_ptr + b * stride_ib + S * b * stride_is + h * stride_ih + offs_j * stride_ij
        # Load with mask j > i (i is known): mask by (offs_j > i)
        mask = offs_j > (pid % S)
        vals = tl.load(in_ptrs, mask=mask & (offs_j < S), other=-float('inf'))
        block_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, block_max)

    # Compute sum of exp(x - max) with causal mask
    sum_exp = 0.0
    for j0 in range(0, S, BLOCK_S):
        offs_j = j0 + tl.arange(0, BLOCK_S)
        mask = offs_j > (pid % S)
        in_ptrs = In_ptr + b * stride_ib + S * b * stride_is + h * stride_ih + offs_j * stride_ij
        vals = tl.load(in_ptrs, mask=mask & (offs_j < S), other=-float('inf'))
        vals = vals - max_val
        exps = tl.exp(vals)
        sum_exp += tl.sum(exps, axis=0)

    inv_sum = 1.0 / sum_exp

    # Write normalized output with causal mask
    for j0 in range(0, S, BLOCK_S):
        offs_j = j0 + tl.arange(0, BLOCK_S)
        mask = offs_j > (pid % S)
        in_ptrs = In_ptr + b * stride_ib + S * b * stride_is + h * stride_ih + offs_j * stride_ij
        vals = tl.load(in_ptrs, mask=mask & (offs_j < S), other=-float('inf'))
        vals = vals - max_val
        out_vals = tl.exp(vals) * inv_sum
        out_ptrs = Out_ptr + b * stride_ob + h * stride_oh + (pid % S) * stride_os + offs_j * stride_ooj
        tl.store(out_ptrs, out_vals, mask=(offs_j < S) & mask)

# 7) Triton output projection: Out[b, i, :] = attn_output[b, i, :] @ o_proj_weight.T (no bias)
#    Attn_out: [B, S, Hq*D], o_proj_weight: [Hq*D, Hq*D], Out: [B, S, Hq*D]
@triton.jit
def output_projection_kernel(
    Attn_ptr, W_ptr, Out_ptr,
    B, S, OUT_DIM,
    stride_ab, stride_as, stride_ao,
    stride_wm, stride_wn,
    stride_ob, stride_os, stride_oo,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid = tl.program_id(0)
    total = B * OUT_DIM
    if pid >= total:
        return
    b = pid // OUT_DIM
    out_i = pid % OUT_DIM

    # Accumulator
    acc = tl.zeros((), dtype=tl.float32)  # scalar
    # We'll iterate rows in blocks and accumulate into a vector of size OUT_DIM, but here OUT_DIM is scalar per i.
    # Simplify: compute single element vector across K tiles
    for k0 in range(0, OUT_DIM, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < OUT_DIM

        # Load attn_output[b, i, offs_k]
        attn_ptrs = Attn_ptr + b * stride_ab + (pid % S) * stride_as + offs_k * stride_ao
        attn_vec = tl.load(attn_ptrs, mask=mask_k, other=0.0).to(tl.float32)

        # Load o_proj_weight[out_i, offs_k] (row-major)
        w_ptrs = W_ptr + out_i * stride_wm + offs_k * stride_wn
        w_vec = tl.load(w_ptrs, mask=mask_k, other=0.0).to(tl.float32)

        acc += tl.sum(attn_vec * w_vec, axis=0)

    # Store
    out_ptr = Out_ptr + b * stride_ob + (pid % S) * stride_os + out_i * stride_oo
    tl.store(out_ptr, acc)

# ModelNew: entry point
class ModelNew(torch.nn.Module):
    def __init__(self, rms_eps: float = RMS_EPS):
        super().__init__()
        self.rms_eps = rms_eps
        # Fixed constants for kernels
        self.BLOCK_M = 64
        self.BLOCK_N = 64
        self.BLOCK_K = 64
        self.ATTN_BLOCK_S = 64
        self.ATTN_BLOCK_D = 64
        self.ROTATE_BLOCK_D = 128
        self.GQA_BLOCK_S = 64
        self.GQA_BLOCK_D = 128
        self.SM_BLOCK_S = 64

    def forward(self, hidden_states, q_proj_weight, q_proj_bias,
                k_proj_weight, k_proj_bias,
                v_proj_weight, v_proj_bias,
                o_proj_weight, q_norm_weight, k_norm_weight,
                cos, sin):
        # Ensure CUDA tensors
        assert hidden_states.is_cuda, "All tensors must be on CUDA device for Triton kernels."
        device = hidden_states.device
        dtype = hidden_states.dtype  # float32 in given workloads

        B = hidden_states.shape[0]
        S_in = hidden_states.shape[1]
        K_in = hidden_states.shape[2]  # 12288

        # 1) Linear projections (Q, K, V) with bias using Triton GEMM
        # Allocate outputs
        Q = torch.empty((B, S_in, NUM_ATTENTION_HEADS * HEAD_DIM), device=device, dtype=torch.float32)
        K = torch.empty((B, S_in, NUM_KEY_VALUE_HEADS * HEAD_DIM), device=device, dtype=torch.float32)
        V = torch.empty((B, S_in, NUM_KEY_VALUE_HEADS * HEAD_DIM), device=device, dtype=torch.float32)

        # Grid for GEMM: (M_tiles, N_tiles) = (B*S, H_out), H_out = NUM_ATTENTION_HEADS/Hk/Hv, K=K_in
        grid_q = (B * S_in, triton.cdiv(NUM_ATTENTION_HEADS * HEAD_DIM, self.BLOCK_N))
        linear_gemm_bias_kernel[grid_q](
            hidden_states, q_proj_weight, q_proj_bias, Q,
            B * S_in, NUM_ATTENTION_HEADS * HEAD_DIM, K_in,
            hidden_states.stride(0), hidden_states.stride(2),
            q_proj_weight.stride(0), q_proj_weight.stride(1),
            Q.stride(0), Q.stride(2),
            BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N, BLOCK_K=self.BLOCK_K
        )

        grid_k = (B * S_in, triton.cdiv(NUM_KEY_VALUE_HEADS * HEAD_DIM, self.BLOCK_N))
        linear_gemm_bias_kernel[grid_k](
            hidden_states, k_proj_weight, k_proj_bias, K,
            B * S_in, NUM_KEY_VALUE_HEADS * HEAD_DIM, K_in,
            hidden_states.stride(0), hidden_states.stride(2),
            k_proj_weight.stride(0), k_proj_weight.stride(1),
            K.stride(0), K.stride(2),
            BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N, BLOCK_K=self.BLOCK_K
        )

        grid_v = (B * S_in, triton.cdiv(NUM_KEY_VALUE_HEADS * HEAD_DIM, self.BLOCK_N))
        linear_gemm_bias_kernel[grid_v](
            hidden_states, v_proj_weight, v_proj_bias, V,
            B * S_in, NUM_KEY_VALUE_HEADS * HEAD_DIM, K_in,
            hidden_states.stride(0), hidden_states.stride(2),
            v_proj_weight.stride(0), v_proj_weight.stride(1),
            V.stride(0), V.stride(2),
            BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N, BLOCK_K=self.BLOCK_K
        )

        # 2) Reshape to heads
        Q_heads = Q.view(B, S_in, NUM_ATTENTION_HEADS, HEAD_DIM)     # [B, S, 96, 128]
        K_heads = K.view(B, S_in, NUM_KEY_VALUE_HEADS, HEAD_DIM)    # [B, S, 8, 128]
        V_heads = V.view(B, S_in, NUM_KEY_VALUE_HEADS, HEAD_DIM)    # [B, S, 8, 128]

        # 3) RMSNorm over last dim (HEAD_DIM) for Q and K
        Q_norm = torch.empty_like(Q_heads, dtype=torch.float32)
        K_norm = torch.empty_like(K_heads, dtype=torch.float32)

        grid_rms_q = (B * NUM_ATTENTION_HEADS,)
        rms_norm_kernel[grid_rms_q](
            Q_heads, q_norm_weight, Q_norm,
            B, S_in, NUM_ATTENTION_HEADS, HEAD_DIM,
            Q_heads.stride(0), Q_heads.stride(1), Q_heads.stride(3),
            Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(3),
            q_norm_weight.stride(0), self.rms_eps,
            BLOCK_D=self.ATTN_BLOCK_D
        )

        grid_rms_k = (B * NUM_KEY_VALUE_HEADS,)
        rms_norm_kernel[grid_rms_k](
            K_heads, k_norm_weight, K_norm,
            B, S_in, NUM_KEY_VALUE_HEADS, HEAD_DIM,
            K_heads.stride(0), K_heads.stride(1), K_heads.stride(3),
            K_norm.stride(0), K_norm.stride(1), K_norm.stride(3),
            k_norm_weight.stride(0), self.rms_eps,
            BLOCK_D=self.ATTN_BLOCK_D
        )

        # 4) Rotate halves for Q and K using cos/sin
        # cos/sin are [HEAD_DIM]; we'll broadcast per head. We use a simple kernel that applies rotation per (b,h).
        Q_rot = torch.empty_like(Q_norm)
        K_rot = torch.empty_like(K_norm)

        # cos and sin tensors for per-head rotation: they are inputs [HEAD_DIM]
        # We can't pass cos/sin as 1D tensors with stride in Triton easily in this context; assume cos/sin provided.
        # Here, cos and sin are 1D of size HEAD_DIM.
        grid_rotate_q = (B * NUM_ATTENTION_HEADS,)
        rotate_half_kernel[grid_rotate_q](
            Q_norm, cos, sin, Q_rot,
            B, S_in, NUM_ATTENTION_HEADS, HEAD_DIM, HEAD_DIM // 2,
            Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(3),
            Q_rot.stride(0), Q_rot.stride(1), Q_rot.stride(3),
            cos.stride(0), sin.stride(0),
            BLOCK_D=self.ROTATE_BLOCK_D
        )

        grid_rotate_k = (B * NUM_KEY_VALUE_HEADS,)
        rotate_half_kernel[grid_rotate_k](
            K_norm, cos, sin, K_rot,
            B, S_in, NUM_KEY_VALUE_HEADS, HEAD_DIM, HEAD_DIM // 2,
            K_norm.stride(0), K_norm.stride(1), K_norm.stride(3),
            K_rot.stride(0), K_rot.stride(1), K_rot.stride(3),
            cos.stride(0), sin.stride(0),
            BLOCK_D=self.ROTATE_BLOCK_D
        )

        # 5) GQA expansion of K and V from Hk=8 to Hq=96 along NUM_KEY_VALUE_GROUPS=12
        K_expanded = torch.empty((B, NUM_ATTENTION_HEADS, S_in, HEAD_DIM), device=device, dtype=torch.float32)
        V_expanded = torch.empty((B, NUM_ATTENTION_HEADS, S_in, HEAD_DIM), device=device, dtype=torch.float32)

        grid_expand = (B * NUM_ATTENTION_HEADS,)
        gqa_expand_kernel[grid_expand](
            K_rot, K_expanded,
            B, NUM_KEY_VALUE_HEADS, S_in, HEAD_DIM, NUM_ATTENTION_HEADS, NUM_KEY_VALUE_GROUPS,
            K_rot.stride(0), K_rot.stride(1), K_rot.stride(2), K_rot.stride(3),
            K_expanded.stride(0), K_expanded.stride(1), K_expanded.stride(2), K_expanded.stride(3),
            self.GQA_BLOCK_S, self.GQA_BLOCK_D
        )

        gqa_expand_kernel[grid_expand](
            V_heads, V_expanded,
            B, NUM_KEY_VALUE_HEADS, S_in, HEAD_DIM, NUM_ATTENTION_HEADS, NUM_KEY_VALUE_GROUPS,
            V_heads.stride(0), V_heads.stride(1), V_heads.stride(2), V_heads.stride(3),
            V_expanded.stride(0), V_expanded.stride(1), V_expanded.stride(2), V_expanded.stride(3),
            self.GQA_BLOCK_S, self.GQA_BLOCK_D
        )

        # 6) Compute attention scores using Triton: Attn[b, h, i, j] = sum_k Q_rot[b, h, i, k] * K_expanded[b, h, j, k] * SCALING
        Attn = torch.empty((B, S_in, NUM_ATTENTION_HEADS, S_in), device=device, dtype=torch.float32)

        grid_attn = (B * S_in * NUM_ATTENTION_HEADS, triton.cdiv(S_in, self.ATTN_BLOCK_S))
        attention_scores_kernel[grid_attn](
            Q_rot, K_expanded, Attn,
            B, S_in, NUM_ATTENTION_HEADS, HEAD_DIM,
            Q_rot.stride(0), Q_rot.stride(1), Q_rot.stride(2), Q_rot.stride(3),
            K_expanded.stride(0), K_expanded.stride(1), K_expanded.stride(2), K_expanded.stride(3),
            Attn.stride(0), Attn.stride(1), Attn.stride(2), Attn.stride(3),
            SCALING=SCALING, BLOCK_S=self.ATTN_BLOCK_S, BLOCK_D=self.ATTN_BLOCK_D
        )

        # 7) Softmax along last dim (j) per (b, h, i) with causal mask in Triton
        Soft = torch.empty_like(Attn, dtype=torch.float32)

        grid_softmax = (B * NUM_ATTENTION_HEADS,)
        softmax_rows_causal_kernel[grid_softmax](
            Attn, Soft,
            B, S_in, NUM_ATTENTION_HEADS,
            Attn.stride(0), Attn.stride(1), Attn.stride(2), Attn.stride(3),
            Soft.stride(0), Soft.stride(1), Soft.stride(2), Soft.stride(3),
            self.SM_BLOCK_S
        )

        # 8) Compute attention output: attn_output[b, i, :] = sum_j Soft[b, i, j] * V_expanded[b, j, i, :]
        # Produce [B, S_in, Hq*D] = 12288
        attn_output = torch.empty((B, S_in, NUM_ATTENTION_HEADS * HEAD_DIM), device=device, dtype=torch.float32)

        # Implement output projection with Triton GEMM (no bias)
        grid_out = (B * S_in, triton.cdiv(NUM_ATTENTION_HEADS * HEAD_DIM, self.BLOCK_N))
        output_projection_kernel[grid_out](
            Soft, o_proj_weight, attn_output,
            B, S_in, NUM_ATTENTION_HEADS * HEAD_DIM,
            Soft.stride(0), Soft.stride(1), Soft.stride(2),
            o_proj_weight.stride(0), o_proj_weight.stride(1),
            attn_output.stride(0), attn_output.stride(1), attn_output.stride(2),
            BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N, BLOCK_K=self.BLOCK_K
        )

        # 9) Return final output (cast to original dtype if needed); original returns float32 in given workloads
        return attn_output


def run(*args):
    return ModelNew()(*args)
