import torch
import math
import triton
import triton.language as tl


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
    c = (a + b) * scale
    tl.store(c_ptrs, c, mask=mask_h[:, None] & mask_m[None, :])


# Per-row stable softmax: C[H,M] -> attn[H,M]
@triton.jit
def softmax_row_kernel(C_ptr, attn_ptr,
                       H: tl.constexpr, M,
                       stride_c0, stride_c1,
                       stride_a0, stride_a1):
    h = tl.program_id(0)
    # Pass 1: row_max
    row_max = -float("inf")
    for m0 in range(0, M, 256):
        offs_m = m0 + tl.arange(0, 256)
        c_ptrs = C_ptr + (h * stride_c0 + offs_m * stride_c1)
        x = tl.load(c_ptrs, mask=offs_m < M, other=-float("inf"))
        m_chunk = tl.max(x, axis=0)
        row_max = tl.maximum(row_max, m_chunk)

    # Pass 2: row_sum of exp(x - row_max)
    row_sum = 0.0
    for m0 in range(0, M, 256):
        offs_m = m0 + tl.arange(0, 256)
        c_ptrs = C_ptr + (h * stride_c0 + offs_m * stride_c1)
        x = tl.load(c_ptrs, mask=offs_m < M, other=-float("inf"))
        e = tl.exp(x - row_max)
        row_sum += tl.sum(e, axis=0)

    inv_row_sum = 1.0 / row_sum

    # Pass 3: write normalized outputs
    for m0 in range(0, M, 256):
        offs_m = m0 + tl.arange(0, 256)
        c_ptrs = C_ptr + (h * stride_c0 + offs_m * stride_c1)
        x = tl.load(c_ptrs, mask=offs_m < M, other=-float("inf"))
        y = tl.exp(x - row_max) * inv_row_sum
        attn_ptrs = attn_ptr + (h * stride_a0 + offs_m * stride_a1)
        tl.store(attn_ptrs, y, mask=offs_m < M)


# Per-row logsumexp on C[H,M] -> lse[H]
@triton.jit
def lse_row_kernel(C_ptr, lse_ptr,
                   H: tl.constexpr, M,
                   stride_c0, stride_c1):
    h = tl.program_id(0)
    row_max = -float("inf")
    for m0 in range(0, M, 256):
        offs_m = m0 + tl.arange(0, 256)
        c_ptrs = C_ptr + (h * stride_c0 + offs_m * stride_c1)
        x = tl.load(c_ptrs, mask=offs_m < M, other=-float("inf"))
        m_chunk = tl.max(x, axis=0)
        row_max = tl.maximum(row_max, m_chunk)

    sum_exp = 0.0
    for m0 in range(0, M, 256):
        offs_m = m0 + tl.arange(0, 256)
        c_ptrs = C_ptr + (h * stride_c0 + offs_m * stride_c1)
        x = tl.load(c_ptrs, mask=offs_m < M, other=-float("inf"))
        e = tl.exp(x - row_max)
        sum_exp += tl.sum(e, axis=0)

    lse_val = row_max + tl.log(sum_exp)  # logsumexp in natural log
    tl.store(lse_ptr + h, lse_val)


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


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
        device = q_nope.device

        # Flatten paged KV caches to token-level (float32)
        num_pages, num_rows, head_dim_ckv = ckv_cache.shape  # num_rows is 64 in the original code
        assert head_dim_ckv == 512
        Kc_all = ckv_cache.reshape(-1, head_dim_ckv).to(torch.float32).contiguous()  # [num_pages*num_rows, 512]
        Kp_all = kpe_cache.reshape(-1, head_dim_ckv).contiguous()  # actually [num_pages*num_rows, 64], per original code
        # Note: original asserts expect head_dim_kpe == 64 and kpe_cache shape [num_pages, 64, 64]. We keep this assumption.

        num_tokens, num_qo_heads, _ = q_nope.shape
        assert num_qo_heads == 16

        # Prepare outputs
        output = torch.empty((num_tokens, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.empty((num_tokens, num_qo_heads), dtype=torch.float32, device=device)

        for t in range(num_tokens):
            indices = sparse_indices[t]  # [topk]
            valid_mask = indices != -1
            valid_indices = indices[valid_mask].to(torch.long)  # [M]
            M = valid_indices.numel()

            if M == 0:
                # If no valid rows, output zeros and lse = -inf
                output[t].zero_()
                lse[t].fill_(-float("inf"))
                continue

            # Gather Kc_gather and Kp_gather [M, 512] and [M, 64]
            Kc_gather = Kc_all[valid_indices]  # [M, 512]
            Kp_gather = kpe_cache.reshape(-1, Kp_all.shape[1])[valid_indices]  # [M, 64]
            # Note: in the original example, kpe_cache has shape [num_pages, 64, 64]. We assume head_dim_kpe == 64, so
            # we reshape kpe_cache from [num_pages, 64, 64] to [num_pages*64, 64] and gather by tok_idx.

            # Prepare queries
            qn = q_nope[t].to(torch.float32).contiguous()  # [16, 512]
            qp = q_pe[t].to(torch.float32).contiguous()   # [16, 64]

            # Allocate intermediates
            logits_qn = torch.empty((16, M), dtype=torch.float32, device=device)
            logits_qp = torch.empty((16, M), dtype=torch.float32, device=device)
            C = torch.empty((16, M), dtype=torch.float32, device=device)
            attn = torch.empty((16, M), dtype=torch.float32, device=device)
            out_t = torch.empty((16, 512), dtype=torch.float32, device=device)
            lse_t = torch.empty((16,), dtype=torch.float32, device=device)

            # Launch matmul_qn_kc: qn @ Kc_gather.T -> logits_qn
            grid_qn = (triton.cdiv(16, 16), triton.cdiv(M, 256))
            matmul_qn_kc_kernel[grid_qn](
                qn, Kc_gather, logits_qn,
                H=16, M=M,
                stride_qn0=qn.stride(0), stride_qn1=qn.stride(1),
                stride_kc0=Kc_gather.stride(0), stride_kc1=Kc_gather.stride(1),
                stride_out0=logits_qn.stride(0), stride_out1=logits_qn.stride(1),
                BLOCK_H=16, BLOCK_M=256, BLOCK_K=64,
            )

            # Launch matmul_qp_kp: qp @ Kp_gather.T -> logits_qp
            grid_qp = (triton.cdiv(16, 16), triton.cdiv(M, 256))
            matmul_qp_kp_kernel[grid_qp](
                qp, Kp_gather, logits_qp,
                H=16, M=M,
                stride_qp0=qp.stride(0), stride_qp1=qp.stride(1),
                stride_kp0=Kp_gather.stride(0), stride_kp1=Kp_gather.stride(1),
                stride_out0=logits_qp.stride(0), stride_out1=logits_qp.stride(1),
                BLOCK_H=16, BLOCK_M=256, BLOCK_K=32,
            )

            # Launch add_scale: C = (logits_qn + logits_qp) * sm_scale
            grid_add = (triton.cdiv(16, 16), triton.cdiv(M, 256))
            add_scale_kernel[grid_add](
                logits_qn, logits_qp, C,
                H=16, M=M,
                stride_a0=logits_qn.stride(0), stride_a1=logits_qn.stride(1),
                stride_b0=logits_qp.stride(0), stride_b1=logits_qp.stride(1),
                stride_c0=C.stride(0), stride_c1=C.stride(1),
                scale=float(sm_scale),
            )

            # Launch softmax_row: attn = softmax(C, dim=1)
            grid_soft = (16,)
            softmax_row_kernel[grid_soft](
                C, attn,
                H=16, M=M,
                stride_c0=C.stride(0), stride_c1=C.stride(1),
                stride_a0=attn.stride(0), stride_a1=attn.stride(1),
            )

            # Launch _matmul_fp32: output[t] = attn @ Kc_gather -> [16, 512]
            grid_out = (triton.cdiv(16, 16), triton.cdiv(512, 128))
            _matmul_fp32[grid_out](
                attn, Kc_gather, out_t,
                M=M, N=512, K=M,  # attn [16,M] @ Kc_gather [M,512]
                stride_am=attn.stride(0), stride_ak=attn.stride(1),
                stride_bk=Kc_gather.stride(0), stride_bn=Kc_gather.stride(1),
                stride_cm=out_t.stride(0), stride_cn=out_t.stride(1),
                BLOCK_M=16, BLOCK_N=128, BLOCK_K=64,
            )
            output[t] = out_t

            # Launch lse_row: lse[t] = logsumexp(C) / ln(2)
            grid_lse = (16,)
            lse_row_kernel[grid_lse](
                C, lse_t,
                H=16, M=M,
                stride_c0=C.stride(0), stride_c1=C.stride(1),
            )
            lse[t] = lse_t / math.log(2.0)

        return output, lse


def run(*args):
    return ModelNew()(*args)
