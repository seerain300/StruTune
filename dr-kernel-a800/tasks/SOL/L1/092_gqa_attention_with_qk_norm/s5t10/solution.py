import torch
import triton
import triton.language as tl


# 1) Triton linear projection: C[M, N] = A[M, K] @ B[N, K]^T + bias[N]
# A: [M, K], B: [N, K], C: [M, N], Bias: [N]
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

        a = a.to(tl.float32)
        b = b.to(tl.float32)

        acc += tl.dot(a, b)

    bias = tl.load(Bias_ptr + offs_n, mask=(offs_n < N), other=0.0).to(tl.float32)
    acc += bias[None, :]

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# 2) Triton RMSNorm over last dim (D): input X [B, S, H, D] -> output same, weight [D]
# We launch one program per (b, s, h) and reduce over D.
@triton.jit
def rms_norm_kernel(
    X_ptr, W_ptr, Y_ptr,
    B, S, H, D,
    stride_xb, stride_xs, stride_xh, stride_xd,
    stride_yb, stride_ys, stride_yh, stride_yd,
    stride_w,
    EPS: tl.constexpr,
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

    # Compute variance over D
    sumsq = 0.0
    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        mask = offs_d < D
        x = tl.load(X_ptr + b * stride_xb + s * stride_xs + h * stride_xh + offs_d * stride_xd, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sumsq += tl.sum(x * x)

    mean = sumsq / D
    inv_rms = 1.0 / tl.sqrt(mean + EPS)

    # Normalize and apply weight
    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        mask = offs_d < D
        x = tl.load(X_ptr + b * stride_xb + s * stride_xs + h * stride_xh + offs_d * stride_xd, mask=mask, other=0.0)
        x = x.to(tl.float32) * inv_rms
        w = tl.load(W_ptr + offs_d * stride_w, mask=mask, other=1.0).to(tl.float32)
        y = x * w
        tl.store(Y_ptr + b * stride_yb + s * stride_ys + h * stride_yh + offs_d * stride_yd, y, mask=mask)


# 3) Triton GQA expansion: expand K/V from Hk to H via groups.
# K_in: [B, Hk, S, D], K_out: [B, H, S, D], Hk=NUM_KEY_VALUE_HEADS, H=NUM_ATTENTION_HEADS, groups=NUM_KEY_VALUE_GROUPS
# For each (b,h), copy from K_in[b, h % Hk, :, :].
@triton.jit
def gqa_expand_kernel(
    K_in_ptr, K_out_ptr,
    B, S, D, H, Hk,
    stride_kib, stride_kih, stride_kis, stride_kid,
    stride_kob, stride_koh, stride_kos, stride_kod,
):
    pid = tl.program_id(0)
    total = B * H
    if pid >= total:
        return
    b = pid // H
    h = pid % H
    src_h = h % Hk

    for i in range(0, S):
        for d in range(0, D):
            k_val = tl.load(K_in_ptr + b * stride_kib + src_h * stride_kih + i * stride_kis + d * stride_kid)
            tl.store(K_out_ptr + b * stride_kob + h * stride_koh + i * stride_kos + d * stride_kod, k_val)


# 4) Triton softmax over last dimension S for each (b, h): softmax across columns j for fixed i.
# Input scores: [B, H, S, S], output probs: [B, H, S, S].
# We implement one program per (b, h), loop over i and j inside the kernel. This is fine for moderate S (<=2048).
@triton.jit
def softmax_cols_kernel(
    In_ptr, Out_ptr,
    B, H, S,
    stride_ib, stride_ih, stride_is, stride_ij,
    stride_ob, stride_oh, stride_os, stride_oj,
):
    b = tl.program_id(0)
    h = tl.program_id(1)

    for i in range(0, S):
        # Load row i across j
        max_val = -1e30
        sum_exp = 0.0
        for j in range(0, S):
            ptr = In_ptr + b * stride_ib + h * stride_ih + i * stride_is + j * stride_ij
            val = tl.load(ptr).to(tl.float32)
            if val > max_val:
                max_val = val
            sum_exp += tl.exp(val - max_val)

        # Store normalized probabilities
        for j in range(0, S):
            ptr_in = In_ptr + b * stride_ib + h * stride_ih + i * stride_is + j * stride_ij
            val = tl.load(ptr_in).to(tl.float32)
            prob = tl.exp(val - max_val) / sum_exp
            ptr_out = Out_ptr + b * stride_ob + h * stride_oh + i * stride_os + j * stride_oj
            tl.store(ptr_out, prob)


# 5) Triton matmul for attention output: Out[b,h,i,j] = sum_k Probs[b,h,i,k] * V[b,h,k,j]
# Inputs: Probs [B, H, S, S], V [B, H, S, D]. Output: Out [B, H, S, D].
# One program per (b, h), loop i, j, k inside.
@triton.jit
def attn_output_kernel(
    Probs_ptr, V_ptr, Out_ptr,
    B, H, S, D,
    stride_pb, stride_ph, stride_ps, stride_pj,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
):
    b = tl.program_id(0)
    h = tl.program_id(1)

    for i in range(0, S):
        for j in range(0, S):
            acc = 0.0
            for k in range(0, S):
                p = tl.load(Probs_ptr + b * stride_pb + h * stride_ph + i * stride_ps + k * stride_pj).to(tl.float32)
                v = tl.load(V_ptr + b * stride_vb + h * stride_vh + k * stride_vs + j * stride_vd).to(tl.float32)
                acc += p * v
            tl.store(Out_ptr + b * stride_ob + h * stride_oh + i * stride_os + j * stride_od, acc)


# 6) Triton output projection (no bias): C[M, N] = In[M, K] @ Wt[N, K]
# In: [B*S*H, H*D], Wt: [D_out, H*D], Out: [B*S*H, D_out]
@triton.jit
def linear_output_nobias_kernel(
    In_ptr, Wt_ptr, Out_ptr,
    M, N, K,
    stride_im, stride_ik,
    stride_wm, stride_wk,  # Wt is [N, K]
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a_ptrs = In_ptr + (offs_m[:, None] * stride_im + (k + offs_k)[None, :] * stride_ik)  # [BM, BK]
        b_ptrs = Wt_ptr + (offs_n[None, :] * stride_wm + (k + offs_k)[:, None] * stride_wk)  # [BK, BN]

        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & ((k + offs_k)[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_n[None, :] < N) & ((k + offs_k)[:, None] < K), other=0.0)

        a = a.to(tl.float32)
        b = b.to(tl.float32)

        acc += tl.dot(a, b)

    c_ptrs = Out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters; we rely on inputs passed to forward

    def forward(self, hidden_states: torch.Tensor,
                q_proj_weight: torch.Tensor, q_proj_bias: torch.Tensor,
                k_proj_weight: torch.Tensor, k_proj_bias: torch.Tensor,
                v_proj_weight: torch.Tensor, v_proj_bias: torch.Tensor,
                o_proj_weight: torch.Tensor,
                q_norm_weight: torch.Tensor, k_norm_weight: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor,
                rms_norm_eps: float):
        """
        hidden_states: [B, S, D], q_proj_weight: [H*D, D], k/v/o_proj_weight: [D, H*D], norm weights: [D]
        cos, sin: [D]
        Returns output tensor [B, S, H*D] where H=NUM_ATTENTION_HEADS, D=head_dim.
        """

        assert hidden_states.dim() == 3, "hidden_states must be [B, S, D]"
        assert hidden_states.device.type == 'cuda', "ModelNew requires CUDA device for Triton kernels"

        B, S, D = hidden_states.shape
        device = hidden_states.device

        # 1) Linear projections using Triton
        hidden_f32 = hidden_states.to(torch.float32)

        # Q = hidden @ q_proj_weight^T + q_proj_bias
        Q = torch.empty((B, S, D), device=device, dtype=torch.float32)  # dummy to get shape; we'll compute via Triton
        # Prepare A [B*S, D], B q_proj_weight [H*D, D], C [B*S, H*D]
        Mq = B * S
        Kq = D
        Hq = NUM_ATTENTION_HEADS
        Nq = Hq * D
        Aq = hidden_f32.reshape(Mq, Kq).contiguous()
        Bq = q_proj_weight  # [Nq, K


def run(*args):
    return ModelNew()(*args)
