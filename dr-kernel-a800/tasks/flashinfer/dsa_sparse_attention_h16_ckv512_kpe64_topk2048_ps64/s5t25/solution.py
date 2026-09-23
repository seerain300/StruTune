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


# Per-row softmax along dim=1 (columns) for a matrix C[H,M] -> attn[H,M]
@triton.jit
def softmax_row_kernel(C_ptr, attn_ptr,
                       H: tl.constexpr, M,
                       stride_c0, stride_c1,
                       stride_a0, stride_a1):
    h = tl.program_id(0)
    offs_m = tl.arange(0, 256)  # we tile along M with chunk 256
    # First pass: compute row max
    row_max = -float("inf")
    for m0 in range(0, M, 256):
        offs = m0 + offs_m
        mask = offs < M
        c_ptrs = C_ptr + (h * stride_c0 + offs * stride_c1)
        c = tl.load(c_ptrs, mask=mask, other=-float("inf"))
        row_max = tl.maximum(row_max, tl.max(c, axis=0))

    # Second pass: compute sum of exp(c - row_max)
    sum_exp = 0.0
    for m0 in range(0, M, 256):
        offs = m0 + offs_m
        mask = offs < M
        c_ptrs = C_ptr + (h * stride_c0 + offs * stride_c1)
        c = tl.load(c_ptrs, mask=mask, other=-float("inf"))
        expv = tl.exp(c - row_max)
        sum_exp += tl.sum(expv, axis=0)

    # Third pass: write normalized attention
    for m0 in range(0, M, 256):
        offs = m0 + offs_m
        mask = offs < M
        c_ptrs = C_ptr + (h * stride_c0 + offs * stride_c1)
        c = tl.load(c_ptrs, mask=mask, other=-float("inf"))
        expv = tl.exp(c - row_max) / sum_exp
        attn_ptrs = attn_ptr + (h * stride_a0 + offs * stride_a1)
        tl.store(attn_ptrs, expv, mask=mask)


# Row-wise logsumexp along dim=1 for a matrix C[H,M] -> lse[H]
@triton.jit
def lse_row_kernel(C_ptr, lse_ptr,
                   H: tl.constexpr, M,
                   stride_c0, stride_c1,
                   inv_ln2: tl.float32):
    h = tl.program_id(0)
    offs_m = tl.arange(0, 256)
    # Compute max
    row_max = -float("inf")
    for m0 in range(0, M, 256):
        offs = m0 + offs_m
        mask = offs < M
        c_ptrs = C_ptr + (h * stride_c0 + offs * stride_c1)
        c = tl.load(c_ptrs, mask=mask, other=-float("inf"))
        row_max = tl.maximum(row_max, tl.max(c, axis=0))
    # Compute sum exp
    sum_exp = 0.0
    for m0 in range(0, M, 256):
        offs = m0 + offs_m
        mask = offs < M
        c_ptrs = C_ptr + (h * stride_c0 + offs * stride_c1)
        c = tl.load(c_ptrs, mask=mask, other=-float("inf"))
        sum_exp += tl.sum(tl.exp(c - row_max), axis=0)
    lse_val = row_max + tl.log(sum_exp) * inv_ln2
    tl.store(lse_ptr + h, lse_val)


# Kernel for attn [H, M] @ Kc [M, N] -> out [H, N]
@triton.jit
def matmul_attn_kc_kernel(attn_ptr, Kc_ptr, out_ptr,
                          H: tl.constexpr, N: tl.constexpr, M,
                          stride_a0, stride_a1,
                          stride_k0, stride_k1,
                          stride_out0, stride_out1,
                          BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_h = tl.program_id(0)  # tile along H
    pid_n = tl.program_id(1)  # tile along N

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_H, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, M, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = attn_ptr + (offs_h[:, None] * stride_a0 + offs_k[None, :] * stride_a1)
        k_ptrs = Kc_ptr + (offs_k[:, None] * stride_k0 + offs_n[None, :] * stride_k1)
        a = tl.load(a_ptrs, mask=(offs_h[:, None] < H) & (offs_k[None, :] < M), other=0.0)
        k = tl.load(k_ptrs, mask=(offs_k[:, None] < M) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, k)  # (BLOCK_H, BLOCK_K) @ (BLOCK_K, BLOCK_N) -> (BLOCK_H, BLOCK_N)

    out_ptrs = out_ptr + (offs_h[:, None] * stride_out0 + offs_n[None, :] * stride_out1)
    tl.store(out_ptrs, acc, mask=(offs_h[:, None] < H) & (offs_n[None, :] < N))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
        # Flatten caches to [P, dim]
        num_tokens, H, _ = q_nope.shape
        P = (ckv_cache.shape[0] * ckv_cache.shape[1])
        device = q_nope.device

        Kc_all = ckv_cache.reshape(P, 512).to(torch.float32).contiguous()
        Kp_all = kpe_cache.reshape(P, 64).to(torch.float32).contiguous()

        output = torch.empty((num_tokens, H, 512), dtype=torch.float32, device=device)
        lse = torch.empty((num_tokens, H), dtype=torch.float32, device=device)

        for t in range(num_tokens):
            indices = sparse_indices[t]  # [topk]
            valid_mask = indices != -1
            tok_idx = indices[valid_mask].to(torch.long)  # [M]

            M = tok_idx.numel()
            if M == 0:
                output[t].zero_()
                lse[t].fill_(-float("inf"))
                continue

            # Gather Kc_gather and Kp_gather
            Kc_gather = Kc_all[tok_idx]  # [M, 512]
            Kp_gather = Kp_all[tok_idx]  # [M, 64]

            # Prepare queries as float32
            qn = q_nope[t].to(torch.float32).contiguous()  # [H, 512]
            qp = q_pe[t].to(torch.float32).contiguous()   # [H, 64]

            # Intermediates
            logits_qn = torch.empty((H, M), dtype=torch.float32, device=device)
            logits_qp = torch.empty((H, M), dtype=torch.float32, device=device)
            C = torch.empty((H, M), dtype=torch.float32, device=device)
            attn = torch.empty((H, M), dtype=torch.float32, device=device)

            # 1) qn @ Kc_gather.T -> logits_qn
            grid_qn = (triton.cdiv(H, 16), triton.cdiv(M, 256))
            matmul_qn_kc_kernel[grid_qn](
                qn, Kc_gather, logits_qn,
                H=16, M=M,
                stride_qn0=qn.stride(0), stride_qn1=qn.stride(1),
                stride_kc0=Kc_gather.stride(0), stride_kc1=Kc_gather.stride(1),
                stride_out0=logits_qn.stride(0), stride_out1=logits_qn.stride(1),
                BLOCK_H=16, BLOCK_M=256, BLOCK_K=64,
            )

            # 2) qp @ Kp_gather.T -> logits_qp
            grid_qp = (triton.cdiv(H, 16), triton.cdiv(M, 256))
            matmul_qp_kp_kernel[grid_qp](
                qp, Kp_gather, logits_qp,
                H=16, M=M,
                stride_qp0=qp.stride(0), stride_qp1=qp.stride(1),
                stride_kp0=Kp_gather.stride(0), stride_kp1=Kp_gather.stride(1),
                stride_out0=logits_qp.stride(0), stride_out1=logits_qp.stride(1),
                BLOCK_H=16, BLOCK_M=256, BLOCK_K=32,
            )

            # 3) C = (logits_qn + logits_qp) * sm_scale
            grid_add = (triton.cdiv(H, 16), triton.cdiv(M, 256))
            add_scale_kernel[grid_add](
                logits_qn, logits_qp, C,
                H=16, M=M,
                stride_a0=logits_qn.stride(0), stride_a1=logits_qn.stride(1),
                stride_b0=logits_qp.stride(0), stride_b1=logits_qp.stride(1),
                stride_c0=C.stride(0), stride_c1=C.stride(1),
                scale=float(sm_scale),
            )

            # 4) attn = softmax(C, dim=1)
            grid_soft = (H,)
            softmax_row_kernel[grid_soft](
                C, attn,
                H=16, M=M,
                stride_c0=C.stride(0), stride_c1=C.stride(1),
                stride_a0=attn.stride(0), stride_a1=attn.stride(1),
            )

            # 5) output[t] = attn @ Kc_gather
            out_t = torch.empty((H, 512), dtype=torch.float32, device=device)
            grid_out = (triton.cdiv(H, 16), triton.cdiv(512, 128))
            matmul_attn_kc_kernel[grid_out](
                attn, Kc_gather, out_t,
                H=16, N=512, M=M,
                stride_a0=attn.stride(0), stride_a1=attn.stride(1),
                stride_k0=Kc_gather.stride(0), stride_k1=Kc_gather.stride(1),
                stride_out0=out_t.stride(0), stride_out1=out_t.stride(1),
                BLOCK_H=16, BLOCK_N=128, BLOCK_K=64,
            )
            output[t] = out_t

            # 6) lse[t] = logsumexp(C) / ln(2)
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
