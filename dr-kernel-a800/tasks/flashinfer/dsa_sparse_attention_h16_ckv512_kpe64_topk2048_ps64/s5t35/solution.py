import torch
import math
import triton
import triton.language as tl


# Generic fp32 matmul: A[M, K] @ B[K, N] -> C[M, N]
@triton.jit
def _matmul_fp32(A_ptr, B_ptr, C_ptr,
                 M, N, K,
                 stride_am, stride_ak,
                 stride_bk, stride_bn,
                 stride_cm, stride_cn,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# Kernel A: qn [H, 512] x Kc_gather [M, 512] -> logits_qn [H, M]
@triton.jit
def matmul_qn_kc_kernel(qn_ptr, Kc_ptr, logits_qn_ptr,
                        H: tl.constexpr, M,
                        stride_qn0, stride_qn1,
                        stride_kc0, stride_kc1,
                        stride_out0, stride_out1,
                        BLOCK_H: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_h = tl.program_id(0)  # tile along H
    pid_m = tl.program_id(1)  # tile along M

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    acc = tl.zeros((BLOCK_H, BLOCK_M), dtype=tl.float32)

    Kq = 512
    for k0 in range(0, Kq, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        q_ptrs = qn_ptr + (offs_h[:, None] * stride_qn0 + offs_k[None, :] * stride_qn1)
        a = tl.load(q_ptrs, mask=(offs_h[:, None] < H) & (offs_k[None, :] < Kq), other=0.0)
        kc_ptrs = Kc_ptr + (offs_m[None, :] * stride_kc0 + offs_k[:, None] * stride_kc1)
        b = tl.load(kc_ptrs, mask=(offs_k[:, None] < Kq) & (offs_m[None, :] < M), other=0.0)
        acc += tl.dot(a, b)  # (BLOCK_H, BLOCK_K) @ (BLOCK_K, BLOCK_M) -> (BLOCK_H, BLOCK_M)

    out_ptrs = logits_qn_ptr + (offs_h[:, None] * stride_out0 + offs_m[None, :] * stride_out1)
    tl.store(out_ptrs, acc, mask=(offs_h[:, None] < H) & (offs_m[None, :] < M))


# Kernel B: qp [H, 64] x Kp_gather [M, 64] -> logits_qp [H, M]
@triton.jit
def matmul_qp_kp_kernel(qp_ptr, Kp_ptr, logits_qp_ptr,
                        H: tl.constexpr, M,
                        stride_qp0, stride_qp1,
                        stride_kp0, stride_kp1,
                        stride_out0, stride_out1,
                        BLOCK_H: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_h = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    acc = tl.zeros((BLOCK_H, BLOCK_M), dtype=tl.float32)

    Kp = 64
    for k0 in range(0, Kp, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        q_ptrs = qp_ptr + (offs_h[:, None] * stride_qp0 + offs_k[None, :] * stride_qp1)
        a = tl.load(q_ptrs, mask=(offs_h[:, None] < H) & (offs_k[None, :] < Kp), other=0.0)
        kp_ptrs = Kp_ptr + (offs_m[None, :] * stride_kp0 + offs_k[:, None] * stride_kp1)
        b = tl.load(kp_ptrs, mask=(offs_k[:, None] < Kp) & (offs_m[None, :] < M), other=0.0)
        acc += tl.dot(a, b)

    out_ptrs = logits_qp_ptr + (offs_h[:, None] * stride_out0 + offs_m[None, :] * stride_out1)
    tl.store(out_ptrs, acc, mask=(offs_h[:, None] < H) & (offs_m[None, :] < M))


# Elementwise add and scale: C[H,M] = (A[H,M] + B[H,M]) * scale
@triton.jit
def add_scale_kernel(A_ptr, B_ptr, C_ptr,
                     H: tl.constexpr, M,
                     stride_a0, stride_a1,
                     stride_b0, stride_b1,
                     stride_c0, stride_c1,
                     scale: tl.float32):
    pid_h = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_h = pid_h * 16 + tl.arange(0, 16)
    offs_m = pid_m * 256 + tl.arange(0, 256)
    mask_h = offs_h < H
    mask_m = offs_m < M

    a_ptrs = A_ptr + (offs_h[:, None] * stride_a0 + offs_m[None, :] * stride_a1)
    b_ptrs = B_ptr + (offs_h[:, None] * stride_b0 + offs_m[None, :] * stride_b1)
    c_ptrs = C_ptr + (offs_h[:, None] * stride_c0 + offs_m[None, :] * stride_c1)

    a = tl.load(a_ptrs, mask=mask_h[:, None] & mask_m[None, :], other=0.0)
    b = tl.load(b_ptrs, mask=mask_h[:, None] & mask_m[None, :], other=0.0)
    c = a + b
    c = c * scale
    tl.store(c_ptrs, c, mask=mask_h[:, None] & mask_m[None, :])


# Per-row stable softmax: C[H,M] -> attn[H,M]
@triton.jit
def softmax_row_kernel(C_ptr, attn_ptr,
                       H: tl.constexpr, M,
                       stride_c0, stride_c1,
                       stride_a0, stride_a1):
    h = tl.program_id(0)  # one program per row
    # Pass 1: find row max
    row_max = -float("inf")
    chunk = 256
    for m0 in range(0, M, chunk):
        offs_m = m0 + tl.arange(0, chunk)
        c_ptrs = C_ptr + (h * stride_c0 + offs_m * stride_c1)
        x = tl.load(c_ptrs, mask=offs_m < M, other=-float("inf"))
        local_max = tl.max(x, axis=0)
        row_max = tl.maximum(row_max, local_max)

    # Pass 2: compute sum of exp(x - row_max)
    sum_exp = 0.0
    for m0 in range(0, M, chunk):
        offs_m = m0 + tl.arange(0, chunk)
        c_ptrs = C_ptr + (h * stride_c0 + offs_m * stride_c1)
        x = tl.load(c_ptrs, mask=offs_m < M, other=-float("inf"))
        sum_exp += tl.sum(tl.exp(x - row_max), axis=0)

    inv_sum = 1.0 / sum_exp

    # Pass 3: write normalized outputs
    for m0 in range(0, M, chunk):
        offs_m = m0 + tl.arange(0, chunk)
        c_ptrs = C_ptr + (h * stride_c0 + offs_m * stride_c1)
        x = tl.load(c_ptrs, mask=offs_m < M, other=-float("inf"))
        y = tl.exp(x - row_max) * inv_sum
        attn_ptrs = attn_ptr + (h * stride_a0 + offs_m * stride_a1)
        tl.store(attn_ptrs, y, mask=offs_m < M)


# Per-row logsumexp on C: lse[H] = logsumexp(C[h, :]) / ln(2)
@triton.jit
def lse_row_kernel(C_ptr, lse_ptr,
                   H: tl.constexpr, M,
                   stride_c0, stride_c1):
    h = tl.program_id(0)  # one program per row
    row_max = -float("inf")
    chunk = 256
    # Pass 1: row max
    for m0 in range(0, M, chunk):
        offs_m = m0 + tl.arange(0, chunk)
        c_ptrs = C_ptr + (h * stride_c0 + offs_m * stride_c1)
        x = tl.load(c_ptrs, mask=offs_m < M, other=-float("inf"))
        local_max = tl.max(x, axis=0)
        row_max = tl.maximum(row_max, local_max)

    # Pass 2: sum_exp
    sum_exp = 0.0
    for m0 in range(0, M, chunk):
        offs_m = m0 + tl.arange(0, chunk)
        c_ptrs = C_ptr + (h * stride_c0 + offs_m * stride_c1)
        x = tl.load(c_ptrs, mask=offs_m < M, other=-float("inf"))
        sum_exp += tl.sum(tl.exp(x - row_max), axis=0)

    ln2 = 0.6931471805599453
    lse_val = row_max + tl.log(sum_exp) / ln2
    tl.store(lse_ptr + h, lse_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
        device = q_nope.device
        num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
        assert num_qo_heads == 16
        assert head_dim_ckv == 512

        # Flatten caches to [num_tokens*64, dim] and move to float32 for compute
        num_pages = ckv_cache.shape[0]
        total_tokens = num_pages * 64
        Kc_all = ckv_cache.reshape(-1, head_dim_ckv).to(torch.float32).contiguous()  # [total_tokens, 512]


def run(*args):
    return ModelNew()(*args)
