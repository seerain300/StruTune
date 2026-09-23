import torch
import math
import triton
import triton.language as tl


# Kernel A: qn [H, 512] x Kc_gather.T [512,M] -> logits_qn [H, M], H=16
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
        a = tl.load(q_ptrs, mask=(offs_h[:, None] < H) & (offs_k[None, :] < Kq), other=0.0)  # [BLOCK_H, BLOCK_K]

        kc_ptrs = Kc_ptr + (offs_m[None, :] * stride_kc0 + offs_k[:, None] * stride_kc1)
        b = tl.load(kc_ptrs, mask=(offs_k[:, None] < Kq) & (offs_m[None, :] < M), other=0.0)  # [BLOCK_K, BLOCK_M]

        acc += tl.dot(a, b)  # (BLOCK_H, BLOCK_K) @ (BLOCK_K, BLOCK_M) -> (BLOCK_H, BLOCK_M)

    out_ptrs = logits_qn_ptr + (offs_h[:, None] * stride_out0 + offs_m[None, :] * stride_out1)
    tl.store(out_ptrs, acc, mask=(offs_h[:, None] < H) & (offs_m[None, :] < M))


# Kernel B: qp [H, 64] x Kp_gather.T [64,M] -> logits_qp [H, M], H=16
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
        a = tl.load(q_ptrs, mask=(offs_h[:, None] < H) & (offs_k[None, :] < Kp), other=0.0)  # [BLOCK_H, BLOCK_K]

        kp_ptrs = Kp_ptr + (offs_m[None, :] * stride_kp0 + offs_k[:, None] * stride_kp1)
        b = tl.load(kp_ptrs, mask=(offs_k[:, None] < Kp) & (offs_m[None, :] < M), other=0.0)  # [BLOCK_K, BLOCK_M]

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


# Per-row stable softmax along dim=1 for a matrix C[H,M] -> attn[H,M]
@triton.jit
def softmax_row_kernel(C_ptr, attn_ptr,
                       H: tl.constexpr, M,
                       stride_c0, stride_c1,
                       stride_a0, stride_a1):
    pid_h = tl.program_id(0)  # one program per row
    offs = tl.arange(0, 256)
    mask = offs < M

    # Load row h
    c_row_ptrs = C_ptr + pid_h * stride_c0 + offs * stride_c1
    row = tl.load(c_row_ptrs, mask=mask, other=-float("inf"))

    # Stable softmax: subtract max
    row_max = tl.max(row, axis=0)
    row = row - row_max
    exp_row = tl.exp(row)
    row_sum = tl.sum(exp_row, axis=0)
    attn_row = exp_row / row_sum

    attn_row_ptrs = attn_ptr + pid_h * stride_a0 + offs * stride_a1
    tl.store(attn_row_ptrs, attn_row, mask=mask)


# Per-row logsumexp along dim=1 for a matrix C[H,M] -> lse[H], base-2
@triton.jit
def lse_row_kernel(C_ptr, lse_ptr,
                   H: tl.constexpr, M,
                   stride_c0, stride_c1,
                   inv_ln2: tl.float32):
    pid_h = tl.program_id(0)  # one program per row
    offs = tl.arange(0, 256)
    mask = offs < M

    c_row_ptrs = C_ptr + pid_h * stride_c0 + offs * stride_c1
    row = tl.load(c_row_ptrs, mask=mask, other=-float("inf"))

    # Stable logsumexp: subtract max
    row_max = tl.max(row, axis=0)
    row_shifted = row - row_max
    row_sum = tl.sum(tl.exp(row_shifted), axis=0)
    lse_val = tl.log(row_sum) + row_max
    lse_val = lse_val * inv_ln2  # convert to base-2

    # Store scalar lse for row pid_h
    tl.store(lse_ptr + pid_h, lse_val)


# Generic fp32 matmul: A[M, K] @ B[K, N] -> C[M, N], used for attn @ Kc_gather -> [H, 512]
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


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
        # Prepare inputs: ensure dtype float32 and contiguous
        device = q_nope.device
        P = ckv_cache.shape[0] * ckv_cache.shape[1]
        Kc_all = ckv_cache.reshape(P, 512).to(torch.float32).contiguous()
        Kp_all = kpe_cache.reshape(P, 64).to(torch.float32).contiguous()

        num_tokens, H, _ = q_nope.shape
        output = torch.empty((num_tokens, H, 512), dtype=torch.float32, device=device)
        lse = torch.empty((num_tokens, H), dtype=torch.float32, device=device)

        for t in range(num_tokens):
            indices = sparse_indices[t]  # [topk]
            valid_mask = indices


def run(*args):
    return ModelNew()(*args)
