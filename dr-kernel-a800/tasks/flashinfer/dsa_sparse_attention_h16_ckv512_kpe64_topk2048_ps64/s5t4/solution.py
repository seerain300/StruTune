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


# Per-row stable softmax along M for a matrix C[H,M] -> attn[H,M]
@triton.jit
def softmax_row_kernel(C_ptr, attn_ptr,
                       H: tl.constexpr, M,
                       stride_c0, stride_c1,
                       stride_out0, stride_out1):
    h = tl.program_id(0)  # one program per row
    offs_m = tl.arange(0, 256)
    row_ptrs = C_ptr + h * stride_c0 + offs_m * stride_c1
    row = tl.load(row_ptrs, mask=offs_m < M, other=-float("inf"))
    row_max = tl.max(row, axis=0)
    row = row - row_max
    exp_row = tl.exp(row)
    row_sum = tl.sum(exp_row, axis=0)
    attn_row = exp_row / row_sum
    out_ptrs = attn_ptr + h * stride_out0 + offs_m * stride_out1
    tl.store(out_ptrs, attn_row, mask=offs_m < M)


# Per-row logsumexp along M for a matrix C[H,M] -> lse[H], divide by ln(2)
@triton.jit
def lse_row_kernel(C_ptr, lse_ptr,
                   H: tl.constexpr, M,
                   stride_c0, stride_c1,
                   inv_ln2: tl.float32):
    h = tl.program_id(0)
    offs_m = tl.arange(0, 256)
    row_ptrs = C_ptr + h * stride_c0 + offs_m * stride_c1
    row = tl.load(row_ptrs, mask=offs_m < M, other=-float("inf"))
    row_max = tl.max(row, axis=0)
    row = row - row_max
    exp_row = tl.exp(row)
    row_sum = tl.sum(exp_row, axis=0)
    lse_val = tl.log(row_sum) + row_max
    lse_val = lse_val / inv_ln2
    tl.store(lse_ptr + h, lse_val)


# Matmul: A[H,M] @ B[M,N] -> C[H,N], generic fp32
@triton.jit
def _matmul_fp32_AB(A_ptr, B_ptr, C_ptr,
                    H, M, N,
                    stride_am, stride_ak,
                    stride_bk, stride_bn,
                    stride_cm, stride_cn,
                    BLOCK_H: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_h = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_H, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, M, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = A_ptr + (offs_h[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        a = tl.load(a_ptrs, mask=(offs_h[:, None] < H) & (offs_k[None, :] < M), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < M) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)

    c_ptrs = C_ptr + (offs_h[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_h[:, None] < H) & (offs_n[None, :] < N))


# Specialized for attn[H,M] @ Kc[M,512] -> out[H,512]
@triton.jit
def matmul_attn_kc_kernel(attn_ptr, Kc_ptr, out_ptr,
                          H: tl.constexpr, M,
                          stride_a0, stride_a1,
                          stride_k0, stride_k1,
                          stride_out0, stride_out1,
                          BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_h = tl.program_id(0)  # tile along H
    pid_n = tl.program_id(1)  # tile along N (512)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_H, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, M, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = attn_ptr + (offs_h[:, None] * stride_a0 + offs_k[None, :] * stride_a1)  # [BH, BK]
        b_ptrs = Kc_ptr + (offs_k[:, None] * stride_k0 + offs_n[None, :] * stride_k1)    # [BK, BN]
        a = tl.load(a_ptrs, mask=(offs_h[:, None] < H) & (offs_k[None, :] < M), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < M) & (offs_n[None, :] < 512), other=0.0)
        acc += tl.dot(a, b)  # (BH, BK) @ (BK, BN) -> (BH, BN)

    out_ptrs = out_ptr + (offs_h[:, None] * stride_out0 + offs_n[None, :] * stride_out1)
    tl.store(out_ptrs, acc, mask=(offs_h[:, None] < H) & (offs_n[None, :] < 512))


def _next_power_of_2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << ((x - 1).bit_length())


def _ensure_2d_shape(shape):
    return shape[0], shape[1]


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
        # Flatten caches to [P, dim] where P = num_pages * 64
        P = ckv_cache.shape[0] * 64
        device = q_nope.device
        # Cast caches to fp32 and make contiguous
        Kc_all = ckv_cache.reshape(P, 512).to(torch.float32).contiguous()
        Kp_all = kpe_cache.reshape(P, 64).to(torch.float32).contiguous()

        # Prepare output and lse
        num_tokens, H, _ = q_nope.shape
        output = torch.empty((num_tokens, H, 512), dtype=torch.float32, device=device)
        lse = torch.empty((num_tokens, H), dtype=torch.float32, device=device)

        for t in range(num_tokens):
            # Build valid indices and tok_idx
            indices = sparse_indices[t]  # [topk]
            valid_mask = indices != -1
            tok_idx = indices[valid_mask].to(torch.long)  # [M]
            M = tok_idx.numel()

            if M == 0:
                output[t].zero_()
                lse[t].fill_(-float("inf"))
                continue

            # Gather Kc_gather and Kp_gather
            Kc_gather = Kc_all[tok_idx]  # [M, 512], fp32
            Kp_gather = Kp_all[tok_idx]  # [M, 64], fp32

            # Prepare queries as fp32 and contiguous
            qn = q_nope[t].to(torch.float32).contiguous()  # [H, 512]
            qp = q_pe[t].to(torch.float32).contiguous()   # [H, 64]

            # Allocate intermediates
            logits_qn = torch.empty((H, M), dtype=torch.float32, device=device)
            logits_qp = torch.empty((H, M), dtype=torch.float32, device=device)
            C = torch.empty((H, M), dtype=torch.float32, device=device)
            attn = torch.empty((H, M), dtype=torch.float32, device=device)

            # Launch matmul_qn_kc: qn @ Kc_gather.T -> logits_qn
            BLOCK_M = min(256, _next_power_of_2(M))
            grid_qn = (_ensure_2d_shape((H,))[0], triton.cdiv(M, BLOCK_M))
            matmul_qn_kc_kernel[grid_qn](
                qn, Kc_gather, logits_qn,
                H=16, M=M,
                stride_qn0=qn.stride(0), stride_qn1=qn.stride(1),
                stride_kc0=Kc_gather.stride(0), stride_kc1=Kc_gather.stride(1),
                stride_out0=logits_qn.stride(0), stride_out1=logits_qn.stride(1),
                BLOCK_H=16, BLOCK_M=BLOCK_M, BLOCK_K=64,
            )

            # Launch matmul_qp_kp: qp @ Kp_gather.T -> logits_qp
            grid_qp = (_ensure_2d_shape((H,))[0], triton.cdiv(M, BLOCK_M))
            matmul_qp_kp_kernel[grid_qp](
                qp, Kp_gather, logits_qp,
                H=16, M=M,
                stride_qp0=qp.stride(0), stride_qp1=qp.stride(1),
                stride_kp0=Kp_gather.stride(0), stride_kp1=Kp_gather.stride(1),
                stride_out0=logits_qp.stride(0), stride_out1=logits_qp.stride(1),
                BLOCK_H=16, BLOCK_M=BLOCK_M, BLOCK_K=32,
            )

            # Launch add_scale: C = (logits_qn + logits_qp) * sm_scale
            grid_add = (_ensure_2d_shape((H,))[0], triton.cdiv(M, BLOCK_M))
            add_scale_kernel[grid_add](
                logits_qn, logits_qp, C,
                H=16, M=M,
                stride_a0=logits_qn.stride(0), stride_a1=logits_qn.stride(1),
                stride_b0=logits_qp.stride(0), stride_b1=logits_qp.stride(1),
                stride_c0=C.stride(0), stride_c1=C.stride(1),
                scale=float(sm_scale),
            )

            # Launch softmax_row: attn = softmax(C, dim=1)
            grid_soft = (H,)
            softmax_row_kernel[grid_soft](
                C, attn,
                H=16, M=M,
                stride_c0=C.stride(0), stride_c1=C.stride(1),
                stride_out0=attn.stride(0), stride_out1=attn.stride(1),
            )

            # Launch matmul_attn_kc: output[t] = attn @ Kc_gather -> [H, 512]
            out_t = torch.empty((H, 512), dtype=torch.float32, device=device)
            BLOCK_N = 128
            grid_out = (_ensure_2d_shape((H,))[0], triton.cdiv(512, BLOCK_N))
            matmul_attn_kc_kernel[grid_out](
                attn, Kc_gather, out_t,
                H=16, M=M,
                stride_a0=attn.stride(0), stride_a1=attn.stride(1),
                stride_k0=Kc_gather.stride(0), stride_k1=Kc_gather.stride(1),
                stride_out0=out_t.stride(0), stride_out1=out_t.stride(1),
                BLOCK_H=16, BLOCK_N=BLOCK_N, BLOCK_K=64,
            )
            output[t] = out_t

            # Launch lse_row: lse[t] = logsumexp(C) / ln(2)
            inv_ln2 = 1.0 / math.log(2.0)
            lse_t = torch.empty((H,), dtype=torch.float32, device=device)
            lse_row_kernel[(H,)](
                C, lse_t,
                H=16, M=M,
                stride_c0=C.stride(0), stride_c1=C.stride(1),
                inv_ln2=inv_ln2,
            )
            lse[t] = lse_t

        # Cast output to bfloat16 to match original return type
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
